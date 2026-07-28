"""
Evaluate Trained Module 1 Model

Computes precision, recall, F1-score, Mean Absolute Error (MAE),
Mean Squared Error (MSE), and Pearson correlation metrics for the TRAINED
BiLSTM model's score_V predictions compared to human ground-truth annotations,
on the held-out TEST split (videos numbered > 50).

Mirrors evaluate_ai_baseline.py's methodology (same metrics, same thresholds,
same 0-10 scale) so the two can be directly compared in the report.

Usage:
  python scripts/evaluate_trained_model.py --annotations module1_annotations.json --model best_module1_model.pt --threshold 5
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.module1_importance.inference import run_inference
from src.evaluation.metrics import compute_segment_precision_recall

try:
    from scripts.annotate_module1 import C, ok, info, warn, err, banner, hr
except ImportError:
    class C:
        R = B = D = RED = GRN = YEL = BLU = MAG = CYN = GRY = ""
    def info(msg): print(f"[i] {msg}")
    def ok(msg): print(f"[OK] {msg}")
    def warn(msg): print(f"[!] {msg}")
    def err(msg): print(f"[ERROR] {msg}")
    def banner(lines):
        print("\n" + "="*50)
        for l in lines: print(l)
        print("="*50)
    def hr(): print("-"*50)


def compute_correlations(x, y):
    """Compute Pearson and Spearman correlation coefficients."""
    if len(x) < 2:
        return 0.0, 0.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    pearson = num / den if den > 0 else 0.0

    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))
    mean_rx, mean_ry = np.mean(x_ranks), np.mean(y_ranks)
    num_r = np.sum((x_ranks - mean_rx) * (y_ranks - mean_ry))
    den_r = np.sqrt(np.sum((x_ranks - mean_rx) ** 2) * np.sum((y_ranks - mean_ry) ** 2))
    spearman = num_r / den_r if den_r > 0 else 0.0

    return pearson, spearman


def get_test_video_ids(features_dir):
    """Video number > 50 = test split, matching train.py's dataset split logic."""
    features_dir = Path(features_dir)
    test_ids = []
    for feat_path in sorted(features_dir.glob("*_features.npy")):
        video_id = feat_path.stem.replace("_features", "")
        try:
            vid_num = int(video_id.split()[1])
        except (IndexError, ValueError):
            continue
        if vid_num > 50:
            test_ids.append(video_id)
    return test_ids


def evaluate_trained_model(
    annotations_path,
    model_path,
    features_dir,
    target_threshold,
    output_dir="outputs/module1_test_inference",
):
    """Run the trained model on the test split and compare against human annotations."""
    ann_path = Path(annotations_path)
    if not ann_path.exists():
        err(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    features_dir = Path(features_dir)
    test_video_ids = get_test_video_ids(features_dir)
    if not test_video_ids:
        err(f"No test-split videos (video number > 50) found in {features_dir}")
        sys.exit(1)

    info(f"Found {len(test_video_ids)} test-split videos: {', '.join(test_video_ids)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    human_scores = []
    model_scores = []
    segment_ids = []

    for video_id in test_video_ids:
        features_path = features_dir / f"{video_id}_features.npy"
        timestamps_path = features_dir / f"{video_id}_timestamps.npy"
        if not features_path.exists() or not timestamps_path.exists():
            warn(f"Missing features for {video_id}, skipping")
            continue

        out_json = output_dir / f"{video_id}_score_V.json"
        predictions = run_inference(
            features_path=str(features_path),
            timestamps_path=str(timestamps_path),
            model_path=str(model_path),
            output_json_path=str(out_json),
            device="cuda",
        )

        for pred in predictions:
            seg_id = pred["segment_id"]
            rec = annotations.get(seg_id)
            if rec is None:
                continue
            if rec.get("skipped") or rec.get("raw_score") is None:
                continue

            human_scores.append(rec["raw_score"])
            model_scores.append(pred["score_V"] * 10.0)  # scale 0-1 -> 0-10 to match human scale
            segment_ids.append(seg_id)

    total_evaluated = len(human_scores)
    if total_evaluated == 0:
        err("No matching test-split segments found between model predictions and annotations.")
        sys.exit(1)

    h_arr = np.array(human_scores, dtype=float)
    m_arr = np.array(model_scores, dtype=float)

    mae = np.mean(np.abs(h_arr - m_arr))
    mse = np.mean((h_arr - m_arr) ** 2)
    rmse = np.sqrt(mse)
    pearson, spearman = compute_correlations(h_arr, m_arr)

    banner([
        "MODULE 1 - TRAINED MODEL EVALUATION (TEST SPLIT)",
        f"  Model checkpoint : {model_path}",
        f"  Test videos      : {len(test_video_ids)}",
        f"  Total segments   : {total_evaluated}",
    ])

    print()
    print(f"  {C.B}Continuous Scoring Agreement Metrics:{C.R}")
    print(f"    Mean Absolute Error (MAE)  : {mae:.4f}  (scale 0-10)")
    print(f"    Mean Squared Error (MSE)   : {mse:.4f}")
    print(f"    Root Mean Squared Error    : {rmse:.4f}")
    print(f"    Pearson Correlation (r)    : {pearson:.4f}")
    print(f"    Spearman Rank Correlation  : {spearman:.4f}")
    print()

    print(f"  {C.B}Segment Selection Performance by Threshold:{C.R}")
    print()
    print(f"    {'Threshold':<11} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'Selected':<8} | {'Target Met?':<12}")
    print(f"    {'-' * 75}")

    for threshold in range(3, 9):
        gt_segments = {segment_ids[i] for i in range(total_evaluated) if human_scores[i] >= threshold}
        sel_segments = {segment_ids[i] for i in range(total_evaluated) if model_scores[i] >= threshold}

        metrics = compute_segment_precision_recall(sel_segments, gt_segments)

        if metrics["precision"] >= 0.75 and metrics["recall"] >= 0.75:
            met = f"{C.GRN}YES{C.R}"
        else:
            reasons = []
            if metrics["precision"] < 0.75:
                reasons.append("P")
            if metrics["recall"] < 0.75:
                reasons.append("R")
            met = f"{C.RED}No ({'/'.join(reasons)}){C.R}"

        bold_prefix = C.B if threshold == target_threshold else ""
        bold_suffix = C.R if threshold == target_threshold else ""

        print(f"    {bold_prefix}Score >= {threshold}{bold_suffix:<4} | "
              f"{bold_prefix}{metrics['precision']:.4f}{bold_suffix:<9} | "
              f"{bold_prefix}{metrics['recall']:.4f}{bold_suffix:<8} | "
              f"{bold_prefix}{metrics['f1']:.4f}{bold_suffix:<8} | "
              f"{len(sel_segments):<8} | "
              f"{met}")

    print()
    hr()
    print(f"  {C.D}Note: Target thresholds are Precision > 0.75, Recall > 0.75 per the project plan.{C.R}")
    print(f"  {C.D}Compare against scripts/evaluate_ai_baseline.py's numbers at the same thresholds.{C.R}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained Module 1 model against human annotations on the test split")
    parser.add_argument("--annotations", type=str, default="module1_annotations.json", help="Path to merged annotations JSON file")
    parser.add_argument("--model", type=str, default="best_module1_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--features_dir", type=str, default="data/processed/features", help="Path to extracted ResNet-50 features")
    parser.add_argument("--threshold", type=int, default=5, help="Visual importance score threshold [0-10] to highlight")
    parser.add_argument("--output_dir", type=str, default="outputs/module1_test_inference", help="Where to save per-video score_V predictions")

    args = parser.parse_args()
    evaluate_trained_model(
        annotations_path=args.annotations,
        model_path=args.model,
        features_dir=args.features_dir,
        target_threshold=args.threshold,
        output_dir=args.output_dir,
    )
