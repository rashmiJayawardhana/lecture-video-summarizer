"""
scripts/check_content_loss.py

STEP B (content/data-loss check): compares each generated fused_slides entry
in video_input/video_json_input.json against the ONLY real source material
for that slide (input/aligned_sources.json), via Ollama Cloud (hosted —
never local, per deployment constraints).

Fixes two kinds of problems, grounded ONLY in that slide's own source facts
— never inventing anything new:
  1. GROUNDING  — false attribution, e.g. "According to the instructor..."
     on a slide that has no audio_blocks at all (a real bug seen in Stage 1
     output: slide 5 has zero audio blocks yet claims "the instructor" said
     something — the model hallucinated a spoken-source frame it doesn't
     have).
  2. COMPLETENESS — a source key_point/fact that's completely absent from
     the generated summary/key_concepts gets added back in, using the
     source's own wording.

Usage:
    venv\\Scripts\\python.exe scripts/check_content_loss.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.module4_synthesis.ollama_cloud import chat_json
from src.utils.json_schema import validate_fused_slides, validate_overall_summary

VIDEO_JSON_PATH = "video_input/video_json_input.json"
ALIGNED_SOURCES_PATH = "input/aligned_sources.json"
OUTPUT_PATH = "video_input/video_json_input_checked.json"

CHECK_SYSTEM_PROMPT = (
    "You are checking a generated lecture-slide summary against the ONLY facts "
    "available for that slide. Fix two kinds of problems, using ONLY the given "
    "source facts — never invent anything new:\n"
    "1. GROUNDING: remove or rephrase any claim in the generated text that is not "
    "supported by the source facts — most importantly, if has_audio_transcript is "
    "false, there is NO spoken source at all, so any phrase attributing content to "
    "'the instructor', 'the speaker', 'according to the lecture', etc. is a "
    "fabrication and must be rewritten to not claim a spoken source.\n"
    "2. COMPLETENESS: if a source key_point or fact is completely missing from the "
    "generated summary/key_concepts, add it in — using the source's own wording, "
    "not new phrasing.\n"
    "If the generated content is already accurate and complete, return it unchanged.\n"
    "Return a single JSON object with EXACTLY these 5 keys: "
    "title, summary, key_concepts, code_example, voiceover_script. "
    "No other keys, no explanation, no markdown fences."
)


def check_slide(source_slide: dict, generated: dict) -> dict:
    payload = {
        "source_facts": {
            "has_audio_transcript": bool(source_slide.get("audio_blocks")),
            "visual": source_slide["visual"],
            "audio_blocks": source_slide.get("audio_blocks", []),
        },
        "generated": generated,
    }
    return chat_json(prompt=json.dumps(payload, indent=2), system=CHECK_SYSTEM_PROMPT)


if __name__ == "__main__":
    with open(VIDEO_JSON_PATH, "r", encoding="utf-8") as f:
        video_json = json.load(f)
    with open(ALIGNED_SOURCES_PATH, "r", encoding="utf-8") as f:
        aligned = json.load(f)

    source_by_slide_number = {s["slide_number"]: s for s in aligned["slides"]}

    changed_slides = []
    checked_fused_slides = []

    for entry in video_json["fused_slides"]:
        slide_number = entry["slide_number"]
        source_slide = source_by_slide_number.get(slide_number)
        if source_slide is None:
            print(f"[{slide_number}] no matching source in aligned_sources.json — leaving as-is")
            checked_fused_slides.append(entry)
            continue

        original = entry["summary"]
        corrected = check_slide(source_slide, original)

        if corrected != original:
            changed_slides.append(slide_number)
            print(f"[{slide_number}] changed")
        else:
            print(f"[{slide_number}] unchanged")

        checked_fused_slides.append({
            "slide_number": slide_number,
            "timestamp": entry["timestamp"],
            "summary": corrected,
        })

    video_json["fused_slides"] = checked_fused_slides

    slide_errors = validate_fused_slides(checked_fused_slides)
    summary_errors = validate_overall_summary(video_json["overall_summary"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(video_json, f, indent=2)

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"{len(changed_slides)}/{len(checked_fused_slides)} slides changed: {changed_slides}")
    print("fused_slides validation errors   :", slide_errors)
    print("overall_summary validation errors:", summary_errors)
