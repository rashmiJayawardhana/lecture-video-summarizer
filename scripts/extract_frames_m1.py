"""
Module 1 — Frame Extraction & Segment Annotation Script
========================================================
Week 1 HIGH priority task (Project Plan):
  "Write OpenCV frame extraction script — extract one frame per second
   from a sample video. Annotate 10 videos: rate each 10-second segment
   0–10 for instructional importance."

Owner : Rashmi (214093E) — Module 1
Usage :
    # Extract frames from a single video
    python scripts/extract_frames_m1.py --video data/raw/lecture_001.mp4

    # Extract from all videos in the raw folder
    python scripts/extract_frames_m1.py --all

    # Interactive annotation mode (rate segments 0-10)
    python scripts/extract_frames_m1.py --video data/raw/lecture_001.mp4 --annotate

Output:
    data/processed/frames/lecture_001/
        seg_001_t0000_t0010/
            frame_000.jpg  frame_001.jpg  ... frame_029.jpg
        seg_002_t0010_t0020/
            ...
    data/annotations/module1_annotations.json
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── Configuration ────────────────────────────────────────────
SEGMENT_DURATION_SEC = 10     # Each segment = 10 seconds (project spec)
FRAMES_PER_SEGMENT   = 30     # 30 frames per segment (3 fps sampling)
FRAME_SIZE           = (224, 224)  # ResNet-50 input size
RAW_DIR              = Path("data/raw")
FRAMES_DIR           = Path("data/processed/frames")
ANNOTATIONS_FILE     = Path("data/annotations/module1_annotations.json")


# ─── Annotation criteria (from project report Section 4.3) ────
ANNOTATION_GUIDE = """
=================================================================
  MODULE 1 — Segment Annotation Guide
=================================================================
Rate each 10-second segment from 0 to 10 based on:

  CRITERION 1 — New concept introduced?
    (new diagram, definition, algorithm, data structure)

  CRITERION 2 — Formula or equation displayed?
    (mathematical notation, pseudocode, code snippet)

  CRITERION 3 — Worked example being solved?
    (step-by-step walkthrough, calculation being done live)

  CRITERION 4 — Visual emphasis cues?
    (lecturer points/circles something, animation, highlighting)

SCORING:
  0–2  = Not important (transitions, pauses, repeated content)
  3–5  = Somewhat important (background/context information)
  6–8  = Important (explains a key concept or technique)
  9–10 = Critical (core concept, formula definition, key diagram)

KEYBOARD SHORTCUTS:
  0-9  = Score (press the digit)
  s    = Skip this segment (mark as needs_review)
  q    = Quit and save progress
=================================================================
"""


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    segment_sec: int = SEGMENT_DURATION_SEC,
    frames_per_seg: int = FRAMES_PER_SEGMENT,
    verbose: bool = True,
) -> list[dict]:
    """
    Extract frames from a video, organised into 10-second segments.

    Returns a list of segment metadata dicts:
        {
            "segment_id":      "seg_001",
            "video_name":      "lecture_001",
            "timestamp_start": 0.0,
            "timestamp_end":   10.0,
            "frame_dir":       "data/processed/frames/lecture_001/seg_001_...",
            "frame_count":     30,
            "score_V":         null   ← filled in during annotation
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    video_name = video_path.stem
    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Video : {video_name}")
        print(f"  FPS   : {fps:.1f}")
        print(f"  Frames: {total_frames:,}")
        print(f"  Length: {duration_sec/60:.1f} min ({duration_sec:.0f} s)")
        n_segments = int(duration_sec // segment_sec)
        print(f"  Segments: {n_segments} × {segment_sec}s")
        print(f"{'─'*60}")

    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    segments         = []
    frames_per_sec   = fps
    sample_every_n   = max(1, int(fps * segment_sec / frames_per_seg))
    segment_idx      = 0
    t_start          = 0.0

    while t_start + segment_sec <= duration_sec:
        t_end      = t_start + segment_sec
        seg_id     = f"seg_{segment_idx+1:03d}"
        seg_label  = f"{seg_id}_t{int(t_start):04d}_t{int(t_end):04d}"
        seg_dir    = video_output_dir / seg_label
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Collect frames for this segment
        frame_start = int(t_start * fps)
        frame_end   = int(t_end   * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        saved          = 0
        frame_pos      = frame_start
        sample_indices = np.linspace(frame_start, frame_end - 1,
                                     frames_per_seg, dtype=int)

        for target_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, FRAME_SIZE)
            out_path = seg_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1

        segments.append({
            "segment_id":      seg_id,
            "video_name":      video_name,
            "timestamp_start": round(t_start, 2),
            "timestamp_end":   round(t_end,   2),
            "frame_dir":       str(seg_dir.relative_to(Path("."))),
            "frame_count":     saved,
            "score_V":         None,   # filled during annotation
            "needs_review":    False,
            "annotated_at":    None,
        })

        if verbose and (segment_idx + 1) % 10 == 0:
            pct = (t_end / duration_sec * 100)
            print(f"  Extracted {segment_idx+1} segments ({pct:.0f}%)...")

        t_start     += segment_sec
        segment_idx += 1

    cap.release()

    if verbose:
        print(f"\n  ✓ Extracted {len(segments)} segments → {video_output_dir}")

    return segments


def annotate_segments_interactive(segments: list[dict]) -> list[dict]:
    """
    Interactive terminal annotation for a list of segments.
    Shows the segment representative frame and asks for a 0-10 score.
    """
    print(ANNOTATION_GUIDE)

    unannotated = [s for s in segments if s["score_V"] is None
                   and not s["needs_review"]]
    print(f"  Segments to annotate: {len(unannotated)}")

    for i, seg in enumerate(unannotated):
        frame_dir = Path(seg["frame_dir"])

        # Show the middle frame of the segment in an OpenCV window
        mid_frame_path = frame_dir / f"frame_{FRAMES_PER_SEGMENT//2:03d}.jpg"
        if mid_frame_path.exists():
            img = cv2.imread(str(mid_frame_path))
            if img is not None:
                display = cv2.resize(img, (640, 480))
                cv2.putText(
                    display,
                    f"{seg['segment_id']}  {seg['timestamp_start']:.0f}s – {seg['timestamp_end']:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.imshow("Segment Frame — Press 0-9 to score, S to skip, Q to quit", display)

        progress = f"[{i+1}/{len(unannotated)}]"
        print(f"\n  {progress} {seg['video_name']} | "
              f"{seg['segment_id']} | "
              f"{seg['timestamp_start']:.0f}s – {seg['timestamp_end']:.0f}s")
        print("  Score (0-10): ", end="", flush=True)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            print("\n  Saving and quitting...")
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            seg["needs_review"] = True
            print("skipped (marked for review)")
            continue
        elif chr(key).isdigit():
            score_str = chr(key)
            # Allow typing two digits (e.g., "10")
            print(score_str, end="", flush=True)
            key2 = cv2.waitKey(1500) & 0xFF
            if chr(key2).isdigit():
                score_str += chr(key2)
                print(chr(key2))
            else:
                print()
            score = int(score_str)
            if score > 10:
                score = 10
            seg["score_V"]      = round(score / 10.0, 2)  # normalise to [0,1]
            seg["annotated_at"] = datetime.now().isoformat()
            print(f"  → Saved score {score}/10  (normalised: {seg['score_V']})")

    cv2.destroyAllWindows()
    return segments


def load_annotations() -> list[dict]:
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return []


def save_annotations(segments: list[dict]) -> None:
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(segments, f, indent=2)

    annotated = sum(1 for s in segments if s["score_V"] is not None)
    total     = len(segments)
    print(f"\n  ✓ Saved {annotated}/{total} annotated segments → {ANNOTATIONS_FILE}")


def print_annotation_stats(segments: list[dict]) -> None:
    total      = len(segments)
    annotated  = sum(1 for s in segments if s["score_V"] is not None)
    skipped    = sum(1 for s in segments if s["needs_review"])
    remaining  = total - annotated - skipped

    print(f"\n{'─'*50}")
    print(f"  Annotation Progress")
    print(f"{'─'*50}")
    print(f"  Total segments : {total}")
    print(f"  Annotated      : {annotated}")
    print(f"  Needs review   : {skipped}")
    print(f"  Remaining      : {remaining}")
    if annotated > 0:
        scores  = [s["score_V"] for s in segments if s["score_V"] is not None]
        avg     = sum(scores) / len(scores)
        top20pct = sum(1 for s in scores if s >= 0.8)
        print(f"  Avg score      : {avg:.2f}")
        print(f"  Top 20% (≥0.8) : {top20pct} segments → candidate for summary")
    print(f"{'─'*50}\n")


# ─── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 1 — Frame extraction and segment annotation"
    )
    parser.add_argument("--video", type=str,
                        help="Path to a single video file")
    parser.add_argument("--all", action="store_true",
                        help="Process all .mp4 files in data/raw/")
    parser.add_argument("--annotate", action="store_true",
                        help="Launch interactive annotation after extraction")
    parser.add_argument("--stats", action="store_true",
                        help="Print annotation progress stats and exit")
    args = parser.parse_args()

    # Stats-only mode
    if args.stats:
        segments = load_annotations()
        print_annotation_stats(segments)
        raise SystemExit(0)

    # Collect videos to process
    videos = []
    if args.video:
        videos = [Path(args.video)]
    elif args.all:
        videos = sorted(RAW_DIR.glob("*.mp4"))
        if not videos:
            print(f"  No .mp4 files found in {RAW_DIR}")
            raise SystemExit(1)
    else:
        parser.print_help()
        raise SystemExit(0)

    # Load existing annotations (so we don't lose previous work)
    all_segments = load_annotations()
    existing_videos = {s["video_name"] for s in all_segments}

    for video_path in videos:
        if video_path.stem in existing_videos:
            print(f"  Skipping {video_path.stem} (already extracted)")
            continue
        new_segments = extract_frames_from_video(
            video_path, FRAMES_DIR, verbose=True
        )
        all_segments.extend(new_segments)

    # Save after extraction (even without annotation)
    save_annotations(all_segments)

    # Annotation mode
    if args.annotate:
        all_segments = annotate_segments_interactive(all_segments)
        save_annotations(all_segments)

    print_annotation_stats(all_segments)
