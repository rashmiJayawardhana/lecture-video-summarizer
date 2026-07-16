"""
Seed Segments - Project Integra (Module 1)

Builds the empty segment skeleton (video_id, segment_id, timestamps) for every
10-second segment of every video in VIDEO_DIR, without requiring any human
annotation first. This lets scripts/backfill_ai_scores.py fill in `ai_score` for
the entire dataset immediately, which scripts/export_module4_handoff.py can then
turn into a schema-correct placeholder JSON for Module 4 (see the "Part 0" step
of the Module 1 completion plan).

Existing segments already present in the output file (e.g. a video someone has
already hand-annotated) are left untouched - this script only adds segments that
are missing, it never overwrites real annotation data.

Usage:
  python scripts/seed_segments.py
  python scripts/seed_segments.py --video-dir videos --out module1_annotations.json
"""

import sys
import glob
import argparse
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import (
    build_segments,
    load_annotations,
    save_annotations,
    ok,
    info,
    warn,
)

VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm"]


def find_videos(video_dir):
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(video_dir, "*" + ext)))
    return sorted(paths)


def seed(video_dir, out_file, dry_run=False):
    video_paths = find_videos(video_dir)
    if not video_paths:
        warn(f"No video files found in '{video_dir}'.")
        sys.exit(1)

    out_path = project_root / out_file
    annotations = load_annotations(str(out_path))
    existing_count = len(annotations)
    info(f"Loaded {existing_count} existing segment(s) from {out_file}.")

    added = 0
    skipped_existing = 0
    videos_seeded = 0

    for video_path in video_paths:
        segs = build_segments(video_path)
        if not segs:
            warn(f"Could not read video info for {os.path.basename(video_path)}. Skipping.")
            continue

        new_for_video = 0
        for seg in segs:
            seg_id = seg["segment_id"]
            if seg_id in annotations:
                skipped_existing += 1
                continue
            annotations[seg_id] = {
                "video_id": seg["video_id"],
                "segment_index": seg["segment_index"],
                "segment_id": seg_id,
                "timestamp_start": seg["timestamp_start"],
                "timestamp_end": seg["timestamp_end"],
                "middle_frame_time": seg["middle_frame_time"],
                "raw_score": None,
                "normalized_score": None,
                "skipped": False,
                "annotator": None,
            }
            new_for_video += 1
            added += 1

        if new_for_video > 0:
            videos_seeded += 1
        info(f"{os.path.basename(video_path)}: {len(segs)} segments, {new_for_video} newly seeded.")

    if dry_run:
        warn(f"[DRY RUN] Would add {added} new segment(s) across {videos_seeded} video(s). "
             f"{skipped_existing} segment(s) already present were left untouched.")
        return

    save_annotations(str(out_path), annotations)
    ok(f"Seeded {added} new segment(s) across {videos_seeded} video(s) into {out_file}.")
    if skipped_existing:
        info(f"Left {skipped_existing} already-existing segment(s) untouched (no overwrite).")
    info(f"Total segments in file now: {len(annotations)}.")
    info("Next step: python scripts/backfill_ai_scores.py --file " + out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the segment skeleton for every video in VIDEO_DIR")
    parser.add_argument("--video-dir", type=str, default="videos", help="Folder containing lecture videos")
    parser.add_argument("--out", type=str, default="module1_annotations.json", help="Annotations JSON to seed into")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving changes")
    args = parser.parse_args()
    seed(video_dir=args.video_dir, out_file=args.out, dry_run=args.dry_run)
