import json
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(".env", override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=API_KEY)


def get_mime_type(image_path):
    mime, _ = mimetypes.guess_type(image_path)
    if mime:
        return mime

    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"

    return "image/jpeg"


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        cleaned = text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "", 1).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "", 1).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        try:
            return json.loads(cleaned)
        except Exception:
            return {
                "semantic_visual_type": "unknown",
                "visual_topic": "Unable to parse JSON",
                "visual_explanation": cleaned,
                "key_points": [],
                "module4_usage": "Use with caution because the output was not valid JSON.",
                "confidence": 0.3
            }


def analyze_visual_with_gemini(image_path, visual_type, ocr_text="", label=""):
    mime_type = get_mime_type(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = f"""
You are analyzing a lecture slide frame for an automated lecture video summarization system.

Important:
- Do not only repeat OCR text.
- Do not describe exact pixel positions.
- Explain what the diagram, table, graph, or mixed visual content is TALKING ABOUT educationally.
- The output will be used by Module 4 to select video summary segments.

Known metadata:
- Module 3 importance label: {label}
- Detected visual type: {visual_type}
- OCR text extracted from slide: {ocr_text}

Return ONLY valid JSON using this exact structure:

{{
  "semantic_visual_type": "text/table/graph/diagram/mixed/unknown",
  "visual_topic": "short topic of the visual content",
  "visual_explanation": "short explanation of what this visual content is communicating",
  "key_points": ["point 1", "point 2", "point 3"],
  "module4_usage": "how this frame can help final video summary selection",
  "confidence": 0.0
}}

Rules:
- If it is a graph, explain the trend, comparison, result, or relationship shown.
- If it is a table, explain what comparison or structured information the table presents.
- If it is a diagram, explain the process, relationship, architecture, or concept shown.
- If it is mixed, explain the text and visual content together.
- Keep it short, academic, and useful.
- confidence must be between 0 and 1.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config={
            "response_mime_type": "application/json"
        }
    )

    return safe_json_loads(response.text)




