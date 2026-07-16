"""
Export Module 4 Handoff - Project Integra (Module 1)

Turns an annotations JSON (module1_annotations.json) into the exact JSON schema
Module 4 expects from Module 1:

    {"segment_id": ..., "timestamp_start": ..., "timestamp_end": ..., "score_V": ...}

Two modes, controlled by --source:

  ai_only        Use the Gemini `ai_score` for every segment. This is the fast
                 "Part 0" placeholder used to unblock Module 4's integration
                 testing before human annotation + training are finished. It is
                 an internal engineering stub only - never present this as
                 Module 1's real output in the report, demo, or evaluation.

  human_priority Use the human `normalized_score` wherever a human label exists,
                 falling back to the AI score only for segments not yet
                 hand-annotated. This is the mode to use once real annotation is
                 underway, and again once the trained BiLSTM's own inference
                 output replaces this file entirely.

Usage:
  python scripts/export_module4_handoff.py --source ai_only
  python scripts/export_module4_handoff.py --file module1_annotations.json --source human_priority --out module1_scores_for_module4.json
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import load_annotations, ok, info, warn, MAX_SCORE


def normalize(raw):
    return round(raw / float(MAX_SCORE), 4)


def export(file_name, out_file, source, dry_run=False):
    ann_path = project_root / file_name
    if not ann_path.exists():
        warn(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    annotations = load_annotations(str(ann_path))
    info(f"Loaded {len(annotations)} segment(s) from {file_name}. Source mode: {source}")

    records = []
    from_human = 0
    from_ai = 0
    missing = 0

    for seg_id, rec in annotations.items():
        if rec.get("skipped", False):
            continue

        score_v = None
        origin = None

        if source == "human_priority" and rec.get("normalized_score") is not None:
            score_v = rec["normalized_score"]
            origin = "human"
        elif rec.get("ai_score") is not None:
            score_v = normalize(rec["ai_score"])
            origin = "ai"
        elif source == "human_priority" and rec.get("raw_score") is not None:
            score_v = normalize(rec["raw_score"])
            origin = "human"

        if score_v is None:
            missing += 1
            continue

        if origin == "human":
            from_human += 1
        else:
            from_ai += 1

        records.append({
            "segment_id": rec["segment_id"],
            "video_id": rec["video_id"],
            "timestamp_start": rec["timestamp_start"],
            "timestamp_end": rec["timestamp_end"],
            "score_V": score_v,
        })

    records.sort(key=lambda r: (r["video_id"], r["timestamp_start"]))

    info(f"Segments with a score: {len(records)}  (from human: {from_human}, from AI: {from_ai})")
    if missing:
        warn(f"{missing} segment(s) have no human or AI score yet and were excluded.")

    if dry_run:
        warn("[DRY RUN] Would write "
             f"{len(records)} record(s) to {out_file}. Re-run without --dry-run to save.")
        return

    out_path = project_root / out_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    ok(f"Wrote {len(records)} record(s) to {out_file}.")
    if source == "ai_only":
        warn("Reminder: this is an AI-baseline placeholder for Module 4's integration "
             "testing only. It must not appear as Module 1's real output in the report, "
             "demo, or evaluation. Replace it once human-trained BiLSTM inference is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Module 1 -> Module 4 handoff JSON")
    parser.add_argument("--file", type=str, default="module1_annotations.json", help="Input annotations JSON")
    parser.add_argument("--out", type=str, default="module1_scores_for_module4.json", help="Output handoff JSON")
    parser.add_argument("--source", choices=["ai_only", "human_priority"], default="human_priority",
                         help="Which score to export per segment")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving changes")
    args = parser.parse_args()
    export(file_name=args.file, out_file=args.out, source=args.source, dry_run=args.dry_run)
