"""
Module 4 — Schema Repair

Repairs malformed overall_summary / fused_slides input before it reaches
slideshow_video.create_slideshow_video(). Uses a local Ollama model to fix
STRUCTURE ONLY (missing fields, wrong types, wrong nesting) — it must not
invent new lecture content. If the input already validates, no model call
is made at all.

Requires: `ollama pull <model>` and the Ollama app running locally.
"""

import json
from typing import Any, Dict, List, Tuple

import ollama

from src.utils.json_schema import (
    validate_fused_slides,
    validate_overall_summary,
    FUSED_SLIDE_SCHEMA,
    SLIDE_SUMMARY_SCHEMA,
    OVERALL_SUMMARY_SCHEMA,
)

DEFAULT_MODEL = "qwen2.5:7b-instruct"

REPAIR_SYSTEM_PROMPT = (
    "You are a data repair tool. You will be given a JSON value that is supposed "
    "to match a target schema, plus a list of validation errors. "
    "Fix ONLY the structure: rename/add/remove fields, fix types, fix nesting, "
    "so the result matches the schema and the errors go away. "
    "Preserve every existing piece of text and every value exactly as given wherever possible. "
    "Do NOT invent new facts, titles, code examples, or narration text. "
    "If a required field has no source value to recover it from, use an empty string "
    "(or empty list for list fields) rather than making content up. "
    "Return ONLY the corrected JSON value — no explanation, no markdown fences."
)


def _call_ollama_json(payload: Dict[str, Any], errors: List[str], schema_hint: Dict[str, Any], model: str) -> Any:
    user_prompt = (
        f"Target schema (field: expected type):\n{json.dumps(schema_hint, default=str, indent=2)}\n\n"
        f"Validation errors found:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
        f"Broken JSON value to fix:\n{json.dumps(payload, indent=2)}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    return json.loads(response["message"]["content"])


def repair_overall_summary(
    overall_summary: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    max_attempts: int = 2,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate overall_summary; if invalid, ask the local model to repair structure.

    Returns:
        (possibly-repaired dict, remaining validation errors — empty if fully valid)
    """
    errors = validate_overall_summary(overall_summary)
    if not errors:
        return overall_summary, []

    current = overall_summary
    for _ in range(max_attempts):
        current = _call_ollama_json(current, errors, OVERALL_SUMMARY_SCHEMA, model)
        errors = validate_overall_summary(current)
        if not errors:
            break

    return current, errors


def repair_fused_slides(
    fused_slides: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_attempts: int = 2,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate fused_slides; if invalid, ask the local model to repair structure
    one slide at a time (keeps prompts small and failures isolated per slide).

    Returns:
        (possibly-repaired list, remaining validation errors — empty if fully valid)
    """
    errors = validate_fused_slides(fused_slides)
    if not errors:
        return fused_slides, []

    repaired_slides = []
    all_errors = []

    for i, slide in enumerate(fused_slides):
        slide_errors = validate_fused_slides([slide])
        current = slide
        for _ in range(max_attempts):
            if not slide_errors:
                break
            schema_hint = {**FUSED_SLIDE_SCHEMA, "summary": SLIDE_SUMMARY_SCHEMA}
            current = _call_ollama_json(current, slide_errors, schema_hint, model)
            slide_errors = validate_fused_slides([current])

        repaired_slides.append(current)
        if slide_errors:
            all_errors.extend(f"Slide {i}: {e}" for e in slide_errors)

    return repaired_slides, all_errors


def ensure_valid_input(
    overall_summary: Dict[str, Any],
    fused_slides: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Validate and, if needed, repair both inputs before video generation.

    Raises:
        ValueError if repair could not bring the data into a valid shape.
    """
    fixed_summary, summary_errors = repair_overall_summary(overall_summary, model=model)
    fixed_slides, slide_errors = repair_fused_slides(fused_slides, model=model)

    remaining = summary_errors + slide_errors
    if remaining:
        raise ValueError(
            "Could not repair input into a valid schema after model correction:\n"
            + "\n".join(remaining)
        )

    return fixed_summary, fixed_slides
