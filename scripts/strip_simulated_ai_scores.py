"""
Strip Simulated AI Scores - Project Integra

Removes any AI scores that were injected by inject_simulated_ai_scores.py
(identified by the "[Simulated]" prefix in ai_reasoning) from the personal
annotation file. Use this to clean up before running --review-disagreements
with real AI scores from the backfill script.

Usage:
  python scripts/strip_simulated_ai_scores.py --file annotations_rashmi.json
"""

import sys
import argparse
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import load_annotations, save_annotations, ok, info, warn


def strip_simulated(file_name, dry_run=False):
    ann_path = project_root / file_name
    if not ann_path.exists():
        print(f"[ERROR] File not found: {ann_path}")
        sys.exit(1)

    annotations = load_annotations(str(ann_path))
    info(f"Loaded {len(annotations)} segments from {file_name}.")

    stripped_count = 0
    kept_real_count = 0

    for seg_id, rec in annotations.items():
        reasoning = rec.get("ai_reasoning", "")
        has_ai = "ai_score" in rec

        if has_ai and isinstance(reasoning, str) and reasoning.startswith("[Simulated]"):
            # Remove the simulated AI fields
            rec.pop("ai_score", None)
            rec.pop("ai_reasoning", None)
            rec.pop("reviewed", None)
            stripped_count += 1
        elif has_ai:
            kept_real_count += 1

    if dry_run:
        warn(f"[DRY RUN] Would strip {stripped_count} simulated scores. Would keep {kept_real_count} real scores.")
        warn("Re-run without --dry-run to apply changes.")
        return

    save_annotations(str(ann_path), annotations)
    ok(f"Stripped {stripped_count} simulated AI scores. Kept {kept_real_count} real AI scores.")
    info(f"Saved to: {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip simulated AI scores from annotations")
    parser.add_argument("--file", type=str, default="annotations_rashmi.json", help="Annotations file to clean")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving changes")
    args = parser.parse_args()
    strip_simulated(file_name=args.file, dry_run=args.dry_run)
