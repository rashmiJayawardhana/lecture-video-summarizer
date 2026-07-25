"""
scripts/generate_video_json_input.py

STEP A (content generation): converts Module 2 (audio script) + Module 3
(slide analysis) real outputs into video_json_input.json — the fused_slides
+ overall_summary shape create_slideshow_video() expects.

For each Module 3 slide:
    - Gathers Module 2 sentences whose timestamp overlaps that slide's window.
    - Asks Qwen to produce title/summary/key_concepts/code_example/
      voiceover_script using ONLY the slide's own semantic_analysis fields
      plus those overlapping sentences — no invented facts.

Then generates overall_summary (lecture_title, main_topic, intro_voiceover,
key_takeaways) from the set of all per-slide titles/summaries.

Usage:
    venv\\Scripts\\python.exe scripts/generate_video_json_input.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

from src.utils.json_schema import validate_fused_slides, validate_overall_summary

MODEL = "qwen2.5:7b-instruct"
MODULE2_PATH = "input/lecture_021_module2_audio_script.json"
MODULE3_PATH = "input/lecture_021_module3_final_output.json"
OUTPUT_PATH = "input/video_json_input.json"

SLIDE_SYSTEM_PROMPT = (
    "You convert a lecture slide analysis record into a fixed JSON schema for "
    "a video-narration slide. Use ONLY facts present in the input — do not add "
    "any information, examples, or claims that are not already stated in the "
    "input fields.\n"
    "Output fields:\n"
    "  title: short slide title (derive from visual_topic)\n"
    "  summary: 1-3 sentence summary (derive from slide_summary)\n"
    "  key_concepts: list of short bullet strings (derive from key_points, "
    "do not add new bullets)\n"
    "  code_example: if ocr_text contains actual code/syntax, extract it "
    "verbatim; otherwise return an empty string — never invent code\n"
    "  voiceover_script: 2-4 spoken sentences a lecturer would say, grounded "
    "in slide_summary, key_points, and the provided spoken_sentences (if any) "
    "— do not introduce any fact not already present\n"
    "\n"
    "Return a single JSON object with EXACTLY these 5 keys: "
    "title, summary, key_concepts, code_example, voiceover_script. "
    "Do not omit any key, do not nest under another key, no slide_number, "
    "no timestamp, no explanation, no markdown fences."
)

OVERALL_SYSTEM_PROMPT = (
    "You write a lecture-level summary from a list of per-slide titles and "
    "summaries. Use ONLY the facts given — do not add topics, names, or "
    "claims not already present in the provided slide list.\n"
    "Output fields:\n"
    "  lecture_title: short overall title\n"
    "  main_topic: 1 sentence describing the overall subject\n"
    "  intro_voiceover: 2-3 spoken sentences welcoming viewers and previewing "
    "what will be covered, based only on the slide titles/summaries given\n"
    "  key_takeaways: list of 4-6 short bullet strings summarizing the most "
    "important points across all slides\n"
    "\n"
    "Return a single JSON object with EXACTLY these 4 keys: "
    "lecture_title, main_topic, intro_voiceover, key_takeaways. "
    "No other keys, no explanation, no markdown fences."
)


def get_overlapping_sentences(module2: list, start: float, end: float) -> list:
    return [
        s["sentence"] for s in module2
        if s["timestamp_start"] < end and s["timestamp_end"] > start
    ]


def convert_slide(slide: dict, spoken_sentences: list) -> dict:
    sem = slide.get("semantic_analysis", {})
    source_facts = {
        "visual_topic": sem.get("visual_topic", ""),
        "key_points": sem.get("key_points", []),
        "slide_summary": slide.get("slide_summary", ""),
        "ocr_text": slide.get("ocr_text", ""),
        "spoken_sentences": spoken_sentences,
    }
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SLIDE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(source_facts, indent=2)},
        ],
        format="json",
        options={"temperature": 0},
    )
    return json.loads(response["message"]["content"])


def generate_overall_summary(fused_slides: list) -> dict:
    slide_digest = [
        {"title": s["summary"]["title"], "summary": s["summary"]["summary"]}
        for s in fused_slides
    ]
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": OVERALL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(slide_digest, indent=2)},
        ],
        format="json",
        options={"temperature": 0},
    )
    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    module2 = json.load(open(MODULE2_PATH, encoding="utf-8"))
    module3 = json.load(open(MODULE3_PATH, encoding="utf-8"))

    fused_slides = []
    for i, slide in enumerate(module3):
        start = slide["frame_time"]
        end = module3[i + 1]["frame_time"] if i + 1 < len(module3) else start + 20.0
        overlapping = get_overlapping_sentences(module2, start, end)

        print(f"[{i+1}/{len(module3)}] {slide['semantic_analysis'].get('visual_topic', '')}")
        summary = convert_slide(slide, overlapping)

        fused_slides.append({
            "slide_number": i + 1,
            "timestamp": start,
            "summary": summary,
        })

    print("Generating overall_summary...")
    overall_summary = generate_overall_summary(fused_slides)

    slide_errors = validate_fused_slides(fused_slides)
    summary_errors = validate_overall_summary(overall_summary)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"overall_summary": overall_summary, "fused_slides": fused_slides}, f, indent=2)

    print(f"\nWrote {len(fused_slides)} slides to {OUTPUT_PATH}")
    print("fused_slides validation errors :", slide_errors)
    print("overall_summary validation errors:", summary_errors)
