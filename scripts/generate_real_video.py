"""
scripts/generate_real_video.py

STEP C (video render): reads video_input/video_json_input_checked.json
(Step A's model output, corrected by Step B's Ollama Cloud check) and
renders the final MP4 via create_slideshow_video() — same renderer used by
scripts/generate_sample_video.py, just fed the real lecture_021 data
instead of hand-written sample data.

Usage:
    venv\\Scripts\\python.exe scripts/generate_real_video.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.module4_synthesis.slideshow_video import create_slideshow_video

INPUT_PATH = "video_input/video_json_input_checked.json"

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "lecture_021_summary.mp4")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")

if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall_summary = data["overall_summary"]
    fused_slides = data["fused_slides"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    print("=" * 60)
    print("  Lecture 021 Summary Video Generator")
    print("=" * 60)
    print(f"  Slides   : {len(fused_slides)}")
    print(f"  Output   : {OUTPUT_VIDEO}")
    print("=" * 60)

    def progress(p):
        bar = int(p * 40)
        print(f"\r  Progress : [{'█' * bar}{'░' * (40 - bar)}] {int(p * 100):3d}%", end="", flush=True)

    result = create_slideshow_video(
        fused_slides=fused_slides,
        overall_summary=overall_summary,
        output_path=OUTPUT_VIDEO,
        temp_dir=TEMP_DIR,
        progress_callback=progress,
    )

    print(f"\n\n  Done! Video saved to:\n  {result}\n")
