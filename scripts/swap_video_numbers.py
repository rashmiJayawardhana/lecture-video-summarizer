"""
Swap Video Numbers - Project Integra (Module 1)

Swaps the "LecVideo NNN" number prefix of two videos throughout an
annotations JSON file - rewriting both `video_id` and `segment_id` for every
matching segment, so any already-completed human annotation work travels
with the video into its new train/val/test position.

Only rewrites JSON keys/fields. You must separately rename the two actual
video files (swap their "LecVideo NNN" prefixes) so train.py's file-based
split still finds them under the right number.

Usage:
  python scripts/swap_video_numbers.py --file annotations_rashmi.json --a "LecVideo 008" --b "LecVideo 055" --dry-run
  python scripts/swap_video_numbers.py --file annotations_rashmi.json --a "LecVideo 008" --b "LecVideo 055"
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import load_annotations, save_annotations, ok, info, warn


def swap(file_name, prefix_a, prefix_b, dry_run=False):
    path = project_root / file_name
    if not path.exists():
        warn(f"File not found: {path}")
        sys.exit(1)

    data = load_annotations(str(path))
    info(f"Loaded {len(data)} segment(s) from {file_name}.")

    new_data = {}
    moved_a_to_b = 0
    moved_b_to_a = 0

    for seg_id, rec in data.items():
        vid = rec.get("video_id", "")
        if vid.startswith(prefix_a):
            new_vid = prefix_b + vid[len(prefix_a):]
            moved_a_to_b += 1
        elif vid.startswith(prefix_b):
            new_vid = prefix_a + vid[len(prefix_b):]
            moved_b_to_a += 1
        else:
            new_data[seg_id] = rec
            continue

        rec = dict(rec)
        rec["video_id"] = new_vid
        # segment_id is "{video_id}__seg_XXXX" - rebuild it from the new video_id
        suffix = seg_id.split("__seg_", 1)[1]
        new_seg_id = f"{new_vid}__seg_{suffix}"
        rec["segment_id"] = new_seg_id
        new_data[new_seg_id] = rec

    info(f"Would rename {moved_a_to_b} segment(s) from '{prefix_a}' -> '{prefix_b}'.")
    info(f"Would rename {moved_b_to_a} segment(s) from '{prefix_b}' -> '{prefix_a}'.")

    if dry_run:
        warn("[DRY RUN] No changes saved. Re-run without --dry-run to apply.")
        return

    save_annotations(str(path), new_data)
    ok(f"Swapped video number prefixes in {file_name}.")
    info("Remember: also rename the two actual .mp4 files to swap their number prefixes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swap LecVideo number prefixes throughout an annotations JSON")
    parser.add_argument("--file", type=str, required=True, help="Annotations JSON to modify")
    parser.add_argument("--a", type=str, required=True, help='First video prefix, e.g. "LecVideo 008"')
    parser.add_argument("--b", type=str, required=True, help='Second video prefix, e.g. "LecVideo 055"')
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving changes")
    args = parser.parse_args()
    swap(file_name=args.file, prefix_a=args.a, prefix_b=args.b, dry_run=args.dry_run)
