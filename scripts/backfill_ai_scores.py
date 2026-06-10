"""
Backfill AI Scores and Reasoning 

Loops through an annotation JSON file, finds segments missing 'ai_score',
extracts their frames from the corresponding video file, calls the configured
LLM (Gemini/Claude/GPT-4o), and saves the results.

Usage:
  python scripts/backfill_ai_scores.py --file annotations_rashmi.json --delay 4.5 --limit 10
"""

import os
import sys
import time
import argparse
import json
import cv2
from pathlib import Path

# Add project root and scripts directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

# Import functions from annotate_module1
try:
    from annotate_module1 import (
        load_dotenv,
        load_annotations,
        save_annotations,
        grab_frame,
        get_ai_score_and_reasoning,
        C,
        info,
        ok,
        warn,
        err,
    )
except ImportError as e:
    print(f"Error importing from annotate_module1: {e}")
    sys.exit(1)


def find_video_path(video_id, video_dir="videos"):
    """Find the correct video file path on disk matching the video_id."""
    video_dir_path = project_root / video_dir
    if not video_dir_path.exists():
        return None
    
    # Check common extensions
    for ext in [".mp4", ".mkv", ".avi", ".mov", ".webm"]:
        candidate = video_dir_path / f"{video_id}{ext}"
        if candidate.exists():
            return str(candidate)
            
    # Try normalized matching for unicode character differences (e.g. checkmark ✓ vs \u2713)
    for entry in video_dir_path.iterdir():
        if entry.is_file():
            name = entry.stem
            if (name == video_id or 
                name.replace("✓", "\u2713") == video_id or 
                name == video_id.replace("\u2713", "✓")):
                return str(entry)
                
    return None


def backfill(file_name, delay, limit=None):
    """Backfill missing AI scores in the specified annotation file."""
    # Load dotenv to load any API keys
    load_dotenv()
    
    # Check for available API Keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if anthropic_key:
        info("Using Claude API for scoring.")
    elif openai_key:
        info("Using GPT-4o API for scoring.")
    elif gemini_key:
        info("Using Gemini 2.0 Flash API (FREE) for scoring.")
    else:
        err("No API key found in environment variables (GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY).")
        sys.exit(1)

    ann_path = project_root / file_name
    if not ann_path.exists():
        err(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    annotations = load_annotations(str(ann_path))
    info(f"Loaded {len(annotations)} annotation segments from {file_name}.")

    # Filter segments that need backfilling
    to_backfill = []
    for seg_id, rec in annotations.items():
        if rec.get("skipped", False):
            continue
        if "ai_score" not in rec or rec.get("ai_score") is None:
            to_backfill.append(seg_id)

    total_missing = len(to_backfill)
    if total_missing == 0:
        ok("All segments already have AI scores. Nothing to backfill!")
        return

    info(f"Found {total_missing} segments missing AI scores.")
    if limit:
        to_backfill = to_backfill[:limit]
        info(f"Limiting backfill to the first {len(to_backfill)} segments.")

    # Open video capture cache
    current_video_path = None
    cap = None
    processed_count = 0

    try:
        for i, seg_id in enumerate(to_backfill, 1):
            rec = annotations[seg_id]
            video_id = rec["video_id"]
            video_path = find_video_path(video_id)

            if not video_path:
                warn(f"Could not find video file for '{video_id}'. Skipping segment.")
                continue

            # Cache video capture
            if current_video_path != video_path:
                if cap is not None:
                    cap.release()
                info(f"Opening video file: {os.path.basename(video_path)}")
                cap = cv2.VideoCapture(video_path)
                current_video_path = video_path

            # Grab frames
            t_start = rec["timestamp_start"]
            t_end = rec["timestamp_end"]
            t_mid = rec.get("middle_frame_time", (t_start + t_end) / 2.0)

            start_f = grab_frame(cap, t_start)
            mid_f = grab_frame(cap, t_mid)
            end_f = grab_frame(cap, max(t_end - 0.2, t_start))

            if start_f is None or mid_f is None or end_f is None:
                warn(f"Could not extract frames for segment '{seg_id}'. Skipping.")
                continue

            print(f"[{i}/{len(to_backfill)}] Calling LLM for segment {seg_id}...", end="", flush=True)

            # Call AI
            ai_score, ai_reasoning = get_ai_score_and_reasoning(start_f, mid_f, end_f)

            if ai_score is not None:
                annotations[seg_id]["ai_score"] = ai_score
                annotations[seg_id]["ai_reasoning"] = ai_reasoning
                # Save progress immediately so we don't lose anything if interrupted
                save_annotations(str(ann_path), annotations)
                print(f" {C.GRN}Score: {ai_score}{C.R}")
                processed_count += 1
            else:
                print(f" {C.RED}Error: {ai_reasoning}{C.R}")

            # Sleep to respect API rate limits (unless it's the last iteration)
            if i < len(to_backfill) and delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print()
        warn("Backfill process interrupted by user. Progress saved.")
    finally:
        if cap is not None:
            cap.release()

    ok(f"Backfill finished. Successfully scored {processed_count} segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing AI scores in annotations JSON")
    parser.add_argument("--file", type=str, default="annotations_rashmi.json", help="Path to annotations JSON file")
    parser.add_argument("--delay", type=float, default=4.5, help="Delay in seconds between API calls to avoid rate limits")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of segments to process")
    
    args = parser.parse_args()
    backfill(file_name=args.file, delay=args.delay, limit=args.limit)
