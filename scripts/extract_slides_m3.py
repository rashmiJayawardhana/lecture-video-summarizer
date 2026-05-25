"""
Module 3 — Slide Frame Extraction Script
=========================================
Week 1 HIGH priority task (Project Plan):
  "Write OpenCV script to extract all unique slide frames from 5 sample
   videos (using frame difference threshold). Annotate 100 slide images
   as Critical / Important / Skip."

Owner : Fazly (214008C) — Module 3
Usage :
    # Extract slides from a single video
    python scripts/extract_slides_m3.py --video data/raw/lecture_001.mp4

    # Extract from all videos in data/raw/
    python scripts/extract_slides_m3.py --all

    # Interactive annotation (label Critical / Important / Skip)
    python scripts/extract_slides_m3.py --video data/raw/lecture_001.mp4 --annotate

    # Print annotation progress
    python scripts/extract_slides_m3.py --stats

Output:
    data/processed/slides/
        lecture_001/
            slide_0042.5s.jpg
            slide_0091.3s.jpg
            ...
    data/annotations/module3_annotations.json
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── Configuration ────────────────────────────────────────────
DIFF_THRESHOLD    = 0.15      # SSIM drop below this = new slide (tune if needed)
MIN_SLIDE_GAP_SEC = 2.0       # Ignore slides within 2 seconds of last one
SLIDE_SIZE        = (224, 224)  # ViT-base input size
LABELS            = ["Critical", "Important", "Skip"]
LABEL_KEYS        = {"c": "Critical", "i": "Important", "s": "Skip"}

RAW_DIR           = Path("data/raw")
SLIDES_DIR        = Path("data/processed/slides")
ANNOTATIONS_FILE  = Path("data/annotations/module3_annotations.json")


# ─── Annotation criteria (from project report Section 4.5) ────
ANNOTATION_GUIDE = """
=================================================================
  MODULE 3 — Slide Annotation Guide
=================================================================
Label each extracted slide frame as one of three categories:

  [C] CRITICAL
      • Key diagrams (architecture, flow charts, ERDs, network maps)
      • Important formulas or equations
      • Algorithm pseudocode or code listings
      • Summary/conclusion slides
      • Important tables or comparison matrices

  [I] IMPORTANT
      • Definition slides (term + explanation)
      • Step-by-step procedure slides
      • Worked examples or case studies
      • Topic-related bullet point lists
      • Useful supporting diagrams

  [S] SKIP
      • Title slides (just a title, no content)
      • Blank or near-blank frames
      • Repeated/duplicate slides (same content as previous)
      • Lecturer-only frames (camera on person, no slide shown)
      • Transition frames (brief visual between slides)
      • Table of contents or agenda slides

KEYBOARD SHORTCUTS:
  C  = Critical
  I  = Important
  S  = Skip
  B  = Back (undo last annotation)
  Q  = Quit and save progress
=================================================================
"""


def frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compute mean absolute pixel difference between two grayscale frames.
    Returns a value in [0, 1] where 0 = identical, 1 = completely different.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(float)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(float)
    diff  = np.abs(gray1 - gray2).mean() / 255.0
    return diff


def is_blank_frame(frame: np.ndarray, threshold: float = 0.03) -> bool:
    """Return True if frame is mostly white/black (blank slide or transition)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    return gray.std() < threshold


def extract_slides_from_video(
    video_path: Path,
    output_dir: Path,
    diff_threshold: float = DIFF_THRESHOLD,
    min_gap_sec: float = MIN_SLIDE_GAP_SEC,
    verbose: bool = True,
) -> list[dict]:
    """
    Extract unique slide frames from a video using frame-difference detection.

    A new slide is saved when:
      1. The pixel difference from the previous saved slide exceeds diff_threshold
      2. At least min_gap_sec seconds have passed since the last saved slide

    Returns a list of slide metadata dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    video_name   = video_path.stem
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Video : {video_name}")
        print(f"  Length: {duration_sec/60:.1f} min  |  FPS: {fps:.1f}")
        print(f"  Diff threshold: {diff_threshold}  |  Min gap: {min_gap_sec}s")
        print(f"{'─'*60}")

    video_slide_dir = output_dir / video_name
    video_slide_dir.mkdir(parents=True, exist_ok=True)

    slides           = []
    prev_frame       = None
    last_saved_time  = -min_gap_sec  # Allow saving from the very start
    frame_idx        = 0

    # Sample at 2 fps to speed up processing (slides don't change faster)
    sample_every_n = max(1, int(fps / 2))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        if frame_idx % sample_every_n == 0:
            if is_blank_frame(frame):
                prev_frame = frame
                frame_idx += 1
                continue

            if prev_frame is None:
                diff = diff_threshold + 1  # Force save first frame
            else:
                diff = frame_difference(prev_frame, frame)

            time_since_last = timestamp - last_saved_time

            if diff >= diff_threshold and time_since_last >= min_gap_sec:
                # Save this slide
                slide_filename = f"slide_{timestamp:07.1f}s.jpg"
                slide_path     = video_slide_dir / slide_filename
                resized = cv2.resize(frame, SLIDE_SIZE)
                cv2.imwrite(str(slide_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])

                slides.append({
                    "slide_id":   f"{video_name}_{len(slides)+1:04d}",
                    "video_name": video_name,
                    "frame_time": round(timestamp, 2),
                    "image_path": str(slide_path.relative_to(Path("."))),
                    "label":      None,   # filled during annotation
                    "ocr_text":   None,   # filled by TrOCR in Module 3 pipeline
                    "diff_score": round(float(diff), 4),
                    "annotated_at": None,
                })

                prev_frame      = frame
                last_saved_time = timestamp

        frame_idx += 1

        if verbose and frame_idx % int(fps * 60) == 0:
            mins = timestamp / 60
            pct  = timestamp / duration_sec * 100
            print(f"  Processing: {mins:.1f} min  ({pct:.0f}%)  |  "
                  f"Slides so far: {len(slides)}")

    cap.release()

    if verbose:
        print(f"\n  ✓ Extracted {len(slides)} unique slides → {video_slide_dir}")

    return slides


def annotate_slides_interactive(slides: list[dict]) -> list[dict]:
    """
    Interactive terminal + OpenCV annotation for slide images.
    """
    print(ANNOTATION_GUIDE)

    unannotated = [s for s in slides if s["label"] is None]
    print(f"  Slides to annotate: {len(unannotated)}")

    history = []  # For undo support

    i = 0
    while i < len(unannotated):
        slide = unannotated[i]
        img_path = Path(slide["image_path"])

        if img_path.exists():
            img     = cv2.imread(str(img_path))
            display = cv2.resize(img, (800, 600))
            cv2.putText(
                display,
                f"{slide['video_name']}  |  {slide['frame_time']:.1f}s  "
                f"[{i+1}/{len(unannotated)}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                display, "C=Critical  I=Important  S=Skip  B=Back  Q=Quit",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
            )
            cv2.imshow("Slide Annotation", display)

        print(f"\n  [{i+1}/{len(unannotated)}] {slide['video_name']} | "
              f"t={slide['frame_time']:.1f}s")
        print("  Label (C/I/S/B/Q): ", end="", flush=True)

        key = cv2.waitKey(0) & 0xFF
        ch  = chr(key).lower()

        if ch == 'q':
            print("\n  Saving and quitting...")
            break
        elif ch == 'b' and history:
            # Undo last annotation
            prev_slide = history.pop()
            prev_slide["label"]        = None
            prev_slide["annotated_at"] = None
            i = max(0, i - 1)
            print("undo")
            continue
        elif ch in LABEL_KEYS:
            label = LABEL_KEYS[ch]
            slide["label"]        = label
            slide["annotated_at"] = datetime.now().isoformat()
            history.append(slide)
            print(label)
            i += 1
        else:
            print("(invalid key — use C, I, S, B, or Q)")

    cv2.destroyAllWindows()
    return slides


def load_annotations() -> list[dict]:
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return []


def save_annotations(slides: list[dict]) -> None:
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(slides, f, indent=2)

    labeled   = sum(1 for s in slides if s["label"] is not None)
    print(f"\n  ✓ Saved {labeled}/{len(slides)} labeled slides → {ANNOTATIONS_FILE}")


def print_stats(slides: list[dict]) -> None:
    total    = len(slides)
    labeled  = sum(1 for s in slides if s["label"] is not None)
    remaining = total - labeled

    print(f"\n{'─'*50}")
    print(f"  Module 3 Annotation Progress")
    print(f"{'─'*50}")
    print(f"  Total slides   : {total}")
    print(f"  Labeled        : {labeled}")
    print(f"  Remaining      : {remaining}")
    for lbl in LABELS:
        count = sum(1 for s in slides if s["label"] == lbl)
        bar   = "█" * (count // 5)
        print(f"  {lbl:<12} : {count:>4}  {bar}")
    targets = {"training": 600, "validation": 100, "test": 200}
    print(f"\n  Target dataset: 800–1,000 slides")
    print(f"  Progress to 800: {'✓' if total >= 800 else f'{total}/800 ({total/8:.0f}%)'}")
    print(f"{'─'*50}\n")


# ─── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 3 — Slide extraction and annotation"
    )
    parser.add_argument("--video",    type=str, help="Path to a single video file")
    parser.add_argument("--all",      action="store_true",
                        help="Process all .mp4 files in data/raw/")
    parser.add_argument("--annotate", action="store_true",
                        help="Launch interactive annotation after extraction")
    parser.add_argument("--stats",    action="store_true",
                        help="Print annotation progress and exit")
    parser.add_argument("--threshold", type=float, default=DIFF_THRESHOLD,
                        help=f"Frame difference threshold (default: {DIFF_THRESHOLD})")
    args = parser.parse_args()

    if args.stats:
        slides = load_annotations()
        print_stats(slides)
        raise SystemExit(0)

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

    # Load existing (don't overwrite previous annotation work)
    all_slides    = load_annotations()
    existing_vids = {s["video_name"] for s in all_slides}

    for video_path in videos:
        if video_path.stem in existing_vids:
            print(f"  Skipping {video_path.stem} (already extracted)")
            continue
        new_slides = extract_slides_from_video(
            video_path, SLIDES_DIR, diff_threshold=args.threshold, verbose=True
        )
        all_slides.extend(new_slides)

    save_annotations(all_slides)

    if args.annotate:
        all_slides = annotate_slides_interactive(all_slides)
        save_annotations(all_slides)

    print_stats(all_slides)
