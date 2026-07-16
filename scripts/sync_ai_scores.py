"""
Sync AI Scores - Project Integra (Module 1)

Copies `ai_score` / `ai_reasoning` from a source annotations JSON (e.g. the
module1_annotations.json produced by the Colab AI backfill) into your own
personal annotator file (e.g. annotations_rashmi.json), WITHOUT ever
overwriting a segment you've already hand-scored.

This lets the "AI opinion" overlay in annotate_module1.py show Gemini's
suggested score next to segments you haven't scored yet - useful as a quick
cross-reference while you do the real human annotation locally, without
needing your own local Gemini API key.

Rule: a segment is only updated if your own file either doesn't have it yet,
or has it but with raw_score still null (i.e. you haven't scored it). Any
segment you've already scored (raw_score is not None) is left completely
untouched.

Usage:
  python scripts/sync_ai_scores.py --source module1_annotations.json --target annotations_rashmi.json
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import load_annotations, save_annotations, ok, info, warn


def sync(source_file, target_file, dry_run=False):
    source_path = project_root / source_file
    target_path = project_root / target_file

    if not source_path.exists():
        warn(f"Source file not found: {source_path}")
        sys.exit(1)

    source = load_annotations(str(source_path))
    target = load_annotations(str(target_path))
    info(f"Loaded {len(source)} segment(s) from {source_file} (source).")
    info(f"Loaded {len(target)} segment(s) from {target_file} (target).")

    added = 0
    updated = 0
    skipped_scored = 0
    skipped_no_ai = 0

    for seg_id, src_rec in source.items():
        if src_rec.get("ai_score") is None:
            skipped_no_ai += 1
            continue

        if seg_id in target:
            if target[seg_id].get("raw_score") is not None:
                skipped_scored += 1
                continue
            target[seg_id]["ai_score"] = src_rec["ai_score"]
            target[seg_id]["ai_reasoning"] = src_rec.get("ai_reasoning")
            updated += 1
        else:
            target[seg_id] = {
                "video_id": src_rec["video_id"],
                "segment_index": src_rec.get("segment_index"),
                "segment_id": seg_id,
                "timestamp_start": src_rec["timestamp_start"],
                "timestamp_end": src_rec["timestamp_end"],
                "middle_frame_time": src_rec.get("middle_frame_time"),
                "raw_score": None,
                "normalized_score": None,
                "skipped": False,
                "annotator": None,
                "ai_score": src_rec["ai_score"],
                "ai_reasoning": src_rec.get("ai_reasoning"),
            }
            added += 1

    info(f"Would add {added} new segment(s), update {updated} existing unscored segment(s).")
    info(f"Skipped {skipped_scored} already hand-scored segment(s) (untouched, as intended).")
    info(f"Skipped {skipped_no_ai} source segment(s) with no ai_score yet.")

    if dry_run:
        warn("[DRY RUN] No changes saved. Re-run without --dry-run to apply.")
        return

    save_annotations(str(target_path), target)
    ok(f"Synced AI scores into {target_file}. Total segments in file now: {len(target)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync AI scores from a source annotations JSON into your own file")
    parser.add_argument("--source", type=str, default="module1_annotations.json", help="Source file (e.g. from Colab)")
    parser.add_argument("--target", type=str, default="annotations_rashmi.json", help="Your personal annotator file")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving changes")
    args = parser.parse_args()
    sync(source_file=args.source, target_file=args.target, dry_run=args.dry_run)
