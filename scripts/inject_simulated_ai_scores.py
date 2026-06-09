"""
Inject Simulated AI Scores - Project Integra

Generates realistic AI scores and reasoning for all annotated segments in an
annotation JSON file, following the exact statistical distribution of the 
inter-annotator calibration study (62% exact, 12% diff=1, 6% diff=2, 20% diff>=3).

This allows developers/users to fully test the review-disagreements tool,
the merge tool, and the evaluation baseline metrics without requiring a working 
LLM API key or incurring any API usage costs.

Usage:
  python scripts/inject_simulated_ai_scores.py --file annotations_rashmi.json
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.annotate_module1 import load_annotations, save_annotations, C, ok, info, warn, err


def generate_reasoning(score):
    """Generate a realistic rubric-aligned reasoning sentence based on the score."""
    if score >= 8:
        reasons = [
            "Core flow chart and architectural block diagram are shown on screen with formula.",
            "Complete mathematical derivation and algorithm steps are displayed with lecturer writing annotations.",
            "Detailed database schema diagram and system sequence diagrams are fully presented.",
        ]
    elif score == 7:
        reasons = [
            "Clear slide content showing a code snippet in Python along with partial worked examples.",
            "Substantial diagram showing component relationships and list of key specifications.",
        ]
    elif score == 6:
        reasons = [
            "A small table and partial worked example are visible, lecturer is gesturing.",
            "Simple block diagram and summary comparison table are displayed on the slide.",
        ]
    elif score == 5:
        reasons = [
            "Full paragraph containing a definition of the core concept and bulleted list.",
            "Explanation slide with a dense paragraph explaining the design pattern definition.",
        ]
    elif score == 4:
        reasons = [
            "A short list of three bullet points detailing project requirements.",
            "Slide contains a few bullet points about administration details.",
        ]
    elif score == 3:
        reasons = [
            "Single heading and one simple bullet point visible on the screen.",
            "Minimal visual information with a single sentence description.",
        ]
    elif score == 2:
        reasons = [
            "Title heading only with no additional explanation bullets or diagrams.",
            "One very short bullet point under the main section title.",
        ]
    elif score == 1:
        reasons = [
            "Lecturer visible on camera, slide is empty or contains decorative graphics only.",
            "Empty slides with lecturer presenting in front of a green screen.",
        ]
    else:  # score == 0
        reasons = [
            "Blank black screen, intro logo, transition slide, or table of contents cover.",
            "TOC, title slide, or admin placeholder screen with no educational content.",
        ]
    return random.choice(reasons)


def inject_scores(file_name, overwrite=False):
    """Inject simulated AI scores and reasoning based on human scores."""
    ann_path = project_root / file_name
    if not ann_path.exists():
        err(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    annotations = load_annotations(str(ann_path))
    info(f"Loaded {len(annotations)} annotation segments from {file_name}.")

    random.seed(42)  # Set seed for reproducible simulation results

    injected_count = 0
    updated_count = 0
    skipped_count = 0

    for seg_id, rec in annotations.items():
        if rec.get("skipped", False):
            skipped_count += 1
            continue

        human_score = rec.get("raw_score")
        if human_score is None:
            continue

        # Check if segment already has ai_score
        has_ai = "ai_score" in rec and rec["ai_score"] is not None
        if has_ai and not overwrite:
            continue

        # Generate realistic AI score based on pilot study distribution:
        # - 62% exact match (diff = 0)
        # - 12% minor difference (diff = 1)
        # - 6% moderate difference (diff = 2)
        # - 20% critical divergence (diff >= 3)
        r = random.random()
        if r < 0.62:
            ai_score = human_score
        elif r < 0.74:
            ai_score = human_score + random.choice([-1, 1])
        elif r < 0.80:
            ai_score = human_score + random.choice([-2, 2])
        else:
            ai_score = human_score + random.choice([-4, -3, 3, 4])

        # Clamp AI score to [0, 10] range
        ai_score = max(0, min(10, ai_score))

        rec["ai_score"] = ai_score
        rec["ai_reasoning"] = f"[Simulated] {generate_reasoning(ai_score)}"
        rec["reviewed"] = False  # Reset reviewed flag to allow testing review-disagreements

        if has_ai:
            updated_count += 1
        else:
            injected_count += 1

    save_annotations(str(ann_path), annotations)
    
    ok(f"Simulated scoring finished:")
    print(f"  - Successfully injected AI scores into {injected_count} segments.")
    print(f"  - Updated {updated_count} existing AI scores.")
    print(f"  - Skipped {skipped_count} skipped/unannotated segments.")
    print(f"  - Saved changes to: {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject simulated AI scores for offline testing")
    parser.add_argument("--file", type=str, default="annotations_rashmi.json", help="Path to annotations JSON file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing AI scores")
    
    args = parser.parse_args()
    inject_scores(file_name=args.file, overwrite=args.overwrite)
