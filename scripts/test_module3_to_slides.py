"""
scripts/test_module3_to_slides.py

CONFIRMATION TEST — not a permanent pipeline script.

Question being answered: can qwen2.5:7b-instruct take real Module 3 output
(lecture_021_module3_final_output.json) and convert it into the fused_slides
shape slideshow_video.py expects, using ONLY the facts already present in
Module 3's own semantic_analysis (no invented content)?

Module 2 is intentionally excluded here — its sample file only covers 0-139s
and is from an unrelated lecture, so there's no real overlapping transcript
data to test against yet.

Usage:
    venv\\Scripts\\python.exe scripts/test_module3_to_slides.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = (
    "You convert a lecture slide analysis record into a fixed JSON schema for "
    "a video-narration slide. Use ONLY facts present in the input — do not add "
    "any information, examples, or claims that are not already stated in the "
    "input fields. "
    "Output fields:\n"
    "  title: short slide title (derive from visual_topic)\n"
    "  summary: 1-3 sentence summary (derive from slide_summary)\n"
    "  key_concepts: list of short bullet strings (derive from key_points, "
    "shortened if needed, do not add new bullets)\n"
    "  code_example: if ocr_text contains actual code/syntax, extract it "
    "verbatim; otherwise return an empty string — never invent code\n"
    "  voiceover_script: 2-4 spoken sentences that a lecturer would say aloud, "
    "based only on slide_summary and key_points — rephrase for spoken delivery, "
    "but do not introduce any fact, number, or example not already present\n"
    "\n"
    "Return a single JSON object with EXACTLY these 5 keys: "
    "title, summary, key_concepts, code_example, voiceover_script. "
    "Do not omit any key, do not nest them under another key, do not add "
    "slide_number or timestamp, no explanation, no markdown fences."
)


def convert_slide(slide: dict) -> dict:
    sem = slide.get("semantic_analysis", {})
    source_facts = {
        "visual_topic": sem.get("visual_topic", ""),
        "key_points": sem.get("key_points", []),
        "slide_summary": slide.get("slide_summary", ""),
        "ocr_text": slide.get("ocr_text", ""),
    }

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(source_facts, indent=2)},
        ],
        format="json",
        options={"temperature": 0},
    )
    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    module3 = json.load(open("input/lecture_021_module3_final_output.json", encoding="utf-8"))

    # Pick 2 slides with substantive technical content (skip pure logistics slides)
    candidates = [s for s in module3 if s.get("semantic_analysis", {}).get("key_points")]
    test_slides = candidates[3:5]  # a couple slides past the intro/logistics ones

    for i, slide in enumerate(test_slides):
        print("=" * 70)
        print(f"  SOURCE SLIDE {i+1} (frame_time={slide['frame_time']}s)")
        print("=" * 70)
        print("visual_topic  :", slide["semantic_analysis"]["visual_topic"])
        print("key_points    :", slide["semantic_analysis"]["key_points"])
        print("slide_summary :", slide["slide_summary"])
        print("ocr_text      :", slide["ocr_text"][:150])

        result = convert_slide(slide)

        print("\n--- MODEL OUTPUT ---")
        print(json.dumps(result, indent=2))
        print()
