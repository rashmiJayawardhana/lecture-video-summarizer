"""
scripts/generate_synthetic_audio_script.py

Generates a SYNTHETIC Module-2-shaped audio script for lecture_021, since no
real ASR transcript exists for it yet. Qwen writes plausible spoken sentences
per slide, grounded ONLY in that slide's Module 3 semantic_analysis fields
(ocr_text, slide_summary, key_points) — no facts beyond what's already there.

THIS IS NOT A REAL TRANSCRIPT. It was never checked against actual lecture
audio. It exists purely so the Module 2 + Module 3 -> video_json_input.json
flow can be exercised end-to-end for lecture_021. Replace with real Module 2
output as soon as it's available.

Usage:
    venv\\Scripts\\python.exe scripts/generate_synthetic_audio_script.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

MODEL = "qwen2.5:7b-instruct"
MODULE3_PATH = "input/lecture_021_module3_final_output.json"
OUTPUT_PATH = "input/lecture_021_module2_audio_script.json"

SYSTEM_PROMPT = (
    "You write plausible spoken lecture sentences for a single slide. "
    "Use ONLY the facts given in slide_summary, key_points, and ocr_text — "
    "do not add any new fact, number, name, or example that isn't already "
    "present in those fields. "
    "Write 2-4 short sentences a lecturer might say while presenting this "
    "slide, in natural spoken style. "
    "Return a JSON object with one key 'sentences': a list of plain strings. "
    "No other keys, no explanation, no markdown fences."
)


def generate_sentences(slide: dict) -> list:
    sem = slide.get("semantic_analysis", {})
    source_facts = {
        "slide_summary": slide.get("slide_summary", ""),
        "key_points": sem.get("key_points", []),
        "ocr_text": slide.get("ocr_text", ""),
    }
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(source_facts, indent=2)},
        ],
        format="json",
        options={"temperature": 0.3},
    )
    result = json.loads(response["message"]["content"])
    return result.get("sentences", [])


def build_audio_script(module3: list) -> list:
    entries = []
    for i, slide in enumerate(module3):
        start = slide["frame_time"]
        end = module3[i + 1]["frame_time"] if i + 1 < len(module3) else start + 20.0
        sentences = generate_sentences(slide)
        if not sentences:
            continue

        span = max(end - start, 1.0)
        step = span / len(sentences)
        for j, sentence in enumerate(sentences):
            s_start = start + j * step
            s_end = min(start + (j + 1) * step, end)
            entries.append({
                "sentence": sentence,
                "timestamp_start": round(s_start, 2),
                "timestamp_end": round(s_end, 2),
                "is_important": True,
                "importance_ratio_T": 0.7,
                "synthetic": True,
            })
    return entries


if __name__ == "__main__":
    module3 = json.load(open(MODULE3_PATH, encoding="utf-8"))
    print(f"Generating synthetic audio script for {len(module3)} slides...")

    entries = build_audio_script(module3)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} synthetic sentences to {OUTPUT_PATH}")
