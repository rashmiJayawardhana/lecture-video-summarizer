import itertools
import json
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(".env", override=True)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def _build_clients():
    """
    Ordered list of (label, genai.Client). Primary key from GEMINI_API_KEY,
    plus any comma-separated backup keys from GEMINI_API_KEY_FALLBACK - same
    convention as gemini_enrich.py's fallback support.
    """
    primary = os.getenv("GEMINI_API_KEY")
    fallback_raw = os.getenv("GEMINI_API_KEY_FALLBACK", "")
    fallback_keys = [k.strip() for k in fallback_raw.split(",") if k.strip()]

    all_keys = ([primary] if primary else []) + fallback_keys
    if not all_keys:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (and optionally "
            "GEMINI_API_KEY_FALLBACK for backup keys) in your .env file."
        )

    clients = []
    for idx, key in enumerate(all_keys):
        label = "primary" if idx == 0 else f"fallback_{idx}"
        clients.append((label, genai.Client(api_key=key)))
    return clients


_CLIENTS = _build_clients()
_round_robin = itertools.cycle(range(len(_CLIENTS)))


def num_configured_keys() -> int:
    """
    Lets callers scale their rate-limit sleep down proportionally: with N
    keys round-robined, each individual key is still only hit once every
    (sleep_seconds) on average, so the effective per-call wait can safely
    be sleep_seconds / N without increasing any single key's request rate.
    """
    return len(_CLIENTS)


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

    start = next(_round_robin)
    ordered_clients = _CLIENTS[start:] + _CLIENTS[:start]

    last_error = None
    for label, client in ordered_clients:
        try:
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
        except Exception as e:
            print(f"Gemini key '{label}' failed for this frame ({e}); trying next key if available.")
            last_error = e
            continue

    # All configured keys failed - raise so the caller's existing
    # except-Exception -> OCR-fallback path in enhanced_gemini_extractor.py
    # handles it exactly as it already does for a single-key failure.
    raise last_error




