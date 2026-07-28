"""
scripts/test_schema_repair.py

Quick manual test for the Module 4 schema repair step.
Feeds intentionally broken overall_summary / fused_slides data through
src.module4_synthesis.schema_repair and prints before/after + validation errors.

Requires Ollama running locally with the target model pulled, e.g.:
    ollama pull qwen2.5:7b-instruct

Usage:
    python scripts/test_schema_repair.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.json_schema import validate_overall_summary, validate_fused_slides
from src.module4_synthesis.schema_repair import repair_overall_summary, repair_fused_slides

# ─── Intentionally broken sample data ──────────────────────────────────────

broken_overall_summary = {
    "lecture_title": "Arrays in Python",
    # missing "main_topic"
    "intro_voiceover": "Welcome to this summary on Arrays in Python.",
    "key_takeaways": "Python lists act as dynamic arrays.",  # wrong type: str instead of list
}

broken_fused_slides = [
    {
        "slide_number": "1",  # wrong type: str instead of int
        "timestamp": 0,
        "summary": {
            "title": "What is an Array?",
            "summary": "An array is an ordered collection of elements.",
            "key_concepts": ["Ordered collection of elements", "Zero-based indexing"],
            # missing "code_example"
            "voiceover_script": "An array is an ordered collection of elements accessed by index.",
        },
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print("  BEFORE repair")
    print("=" * 60)
    print("overall_summary errors:", validate_overall_summary(broken_overall_summary))
    print("fused_slides errors   :", validate_fused_slides(broken_fused_slides))

    fixed_summary, summary_errors = repair_overall_summary(broken_overall_summary)
    fixed_slides, slide_errors = repair_fused_slides(broken_fused_slides)

    print("\n" + "=" * 60)
    print("  AFTER repair")
    print("=" * 60)
    print(json.dumps(fixed_summary, indent=2))
    print(json.dumps(fixed_slides, indent=2))

    print("\nRemaining overall_summary errors:", summary_errors)
    print("Remaining fused_slides errors   :", slide_errors)

    if not summary_errors and not slide_errors:
        print("\nAll data repaired successfully.")
    else:
        print("\nSome errors could not be repaired — see above.")
