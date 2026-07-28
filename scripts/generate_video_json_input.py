"""
scripts/generate_video_json_input.py

STEP A (content generation): converts input/aligned_sources.json (built by
src/module4_synthesis/align_sources.py from the real Module 3 slide data +
real-but-sparse audio data) into video_input/video_json_input.json — the
fused_slides + overall_summary shape create_slideshow_video() expects.

Runs the fine-tuned PyTorch model locally (models/module4/stage1/) instead
of calling any Ollama model — no local or cloud LLM call happens in this
script at all.

*** STAGE 1 LIMITATION — read before trusting the output ***
The checkpoint in models/module4/stage1/ was only fine-tuned on QMSum
(generic "transcript segment -> summary" compression). It was NOT trained
on the fused_slides JSON schema, so this script can only use it for the
free-text `summary` field. Everything else is derived with plain rules,
not learned:
    - title           <- visual_topic (or slide_summary, truncated) as-is
    - key_concepts     <- key_points from the slide + any overlapping audio, deduped
    - code_example     <- ocr_text, only if it looks like code (crude keyword check);
                          otherwise "" (never invented)
    - voiceover_script <- reuses the same generated `summary` text (not a
                          distinct spoken-style rewrite yet)
Stage 2 (fine-tuning further on a schema-shaped example set) is what
replaces this rule-based scaffolding with the model actually producing
all 5 fields itself.

Usage:
    venv\\Scripts\\python.exe scripts/generate_video_json_input.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.json_schema import validate_fused_slides, validate_overall_summary

MODEL_DIR = "models/module4/stage1"
ALIGNED_SOURCES_PATH = "input/aligned_sources.json"
OUTPUT_PATH = "video_input/video_json_input.json"

MAX_INPUT_LEN = 1024
MAX_SUMMARY_LEN = 160

CODE_HINTS = ("def ", "import ", "class ", "print(", "==", "();", "SELECT ", "function ", "{}")


def looks_like_code(text: str) -> bool:
    return any(hint in text for hint in CODE_HINTS)


def build_segment_text(slide: dict) -> str:
    v = slide["visual"]
    lines = [f"Slide: {v['visual_topic']}. {v['slide_summary']}"]
    if v.get("ocr_text"):
        lines.append(f"On-screen text: {v['ocr_text']}")
    if v.get("key_points"):
        lines.append("Key points: " + "; ".join(v["key_points"]))
    for block in slide.get("audio_blocks", []):
        if block.get("summary"):
            lines.append(f"Instructor said: {block['summary']}")
    return "\n".join(lines)


def summarize(tokenizer, model, text: str, max_length: int = MAX_SUMMARY_LEN) -> str:
    prompt = (
        "Summarize this lecture/meeting segment in 2-4 sentences, "
        "capturing only what was actually discussed.\n\n"
        f"Transcript segment:\n{text}"
    )
    inputs = tokenizer(prompt, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=max_length, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_fused_slide(slide: dict, generated_summary: str) -> dict:
    v = slide["visual"]
    key_concepts = list(dict.fromkeys(
        v.get("key_points", [])
        + [kp for b in slide.get("audio_blocks", []) for kp in b.get("key_points", [])]
    ))
    ocr_text = v.get("ocr_text", "")
    code_example = ocr_text if looks_like_code(ocr_text) else ""

    return {
        "slide_number": slide["slide_number"],
        "timestamp": slide["frame_time"],
        "summary": {
            "title": v.get("visual_topic") or v.get("slide_summary", "")[:60],
            "summary": generated_summary,
            "key_concepts": key_concepts,
            "code_example": code_example,
            "voiceover_script": generated_summary,
        },
    }


def build_overall_summary(tokenizer, model, fused_slides: list) -> dict:
    combined = "\n".join(
        f"{s['summary']['title']}: {s['summary']['summary']}" for s in fused_slides
    )
    main_topic = summarize(tokenizer, model, combined, max_length=200)
    lecture_title = fused_slides[0]["summary"]["title"] if fused_slides else "Lecture Summary"

    return {
        "lecture_title": lecture_title,
        "main_topic": main_topic,
        "intro_voiceover": f"Welcome to this summary. {main_topic}",
        "key_takeaways": [s["summary"]["summary"] for s in fused_slides[:6]],
    }


if __name__ == "__main__":
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()

    with open(ALIGNED_SOURCES_PATH, "r", encoding="utf-8") as f:
        aligned = json.load(f)
    slides = aligned["slides"]

    fused_slides = []
    for slide in slides:
        print(f"[{slide['slide_number']}/{len(slides)}] {slide['visual']['visual_topic']}")
        segment_text = build_segment_text(slide)
        generated_summary = summarize(tokenizer, model, segment_text)
        fused_slides.append(build_fused_slide(slide, generated_summary))

    print("Generating overall_summary...")
    overall_summary = build_overall_summary(tokenizer, model, fused_slides)

    slide_errors = validate_fused_slides(fused_slides)
    summary_errors = validate_overall_summary(overall_summary)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"overall_summary": overall_summary, "fused_slides": fused_slides}, f, indent=2)

    print(f"\nWrote {len(fused_slides)} slides to {OUTPUT_PATH}")
    print("fused_slides validation errors  :", slide_errors)
    print("overall_summary validation errors:", summary_errors)
    print(
        "\nReminder: title/key_concepts/code_example/voiceover_script are "
        "rule-derived, not model-generated (Stage 1 limitation — see this "
        "file's docstring). Run the Ollama Cloud check-and-fill step next, "
        "then Stage 2 once built."
    )
