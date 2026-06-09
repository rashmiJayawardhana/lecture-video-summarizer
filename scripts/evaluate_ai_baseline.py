"""
Evaluate AI Baseline - Project Integra

Computes precision, recall, F1-score, Mean Absolute Error (MAE),
Mean Squared Error (MSE), and Pearson correlation metrics for zero-training
direct AI scoring (Gemini/Claude/GPT-4o) compared to human ground-truth annotations.

This script produces the metrics needed for Table 6.1 of the final report.

Usage:
  python scripts/evaluate_ai_baseline.py --file module1_annotations.json --threshold 5
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

try:
    from src.evaluation.metrics import compute_segment_precision_recall
except ImportError as e:
    print(f"Error importing from src.evaluation.metrics: {e}")
    sys.exit(1)

try:
    from scripts.annotate_module1 import C, ok, info, warn, err, banner, hr
except ImportError:
    # Fallback formatting if annotate_module1 cannot be imported
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
    
    # Pearson Correlation
    mean_x, mean_y = np.mean(x), np.mean(y)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    pearson = num / den if den > 0 else 0.0

    # Spearman Correlation (simple rank correlation)
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))
    mean_rx, mean_ry = np.mean(x_ranks), np.mean(y_ranks)
    num_r = np.sum((x_ranks - mean_rx) * (y_ranks - mean_ry))
    den_r = np.sqrt(np.sum((x_ranks - mean_rx)**2) * np.sum((y_ranks - mean_ry)**2))
    spearman = num_r / den_r if den_r > 0 else 0.0

    return pearson, spearman


def evaluate_baseline(file_path, target_threshold):
    """Load annotations and run baseline evaluations."""
    ann_path = Path(file_path)
    if not ann_path.exists():
        err(f"Annotation file not found: {ann_path}")
        sys.exit(1)

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter segments that have both a human score and an AI score
    human_scores = []
    ai_scores = []
    segment_ids = []

    for seg_id, rec in data.items():
        if rec.get("skipped", False):
            continue
        h_score = rec.get("raw_score")
        a_score = rec.get("ai_score")
        if h_score is not None and a_score is not None:
            human_scores.append(h_score)
            ai_scores.append(a_score)
            segment_ids.append(seg_id)

    total_evaluated = len(human_scores)
    if total_evaluated == 0:
        err("No segments found containing both human and AI scores. Run backfill/simulation first!")
        sys.exit(1)

    h_arr = np.array(human_scores, dtype=float)
    a_arr = np.array(ai_scores, dtype=float)

    # 1. Regression Metrics
    mae = np.mean(np.abs(h_arr - a_arr))
    mse = np.mean((h_arr - a_arr)**2)
    rmse = np.sqrt(mse)
    pearson, spearman = compute_correlations(h_arr, a_arr)

    banner([
        "MODULE 1 - AI DIRECT SCORING BASELINE EVALUATION",
        f"  Annotation file  : {ann_path.name}",
        f"  Total segments   : {total_evaluated}"
    ])

    print()
    print(f"  {C.B}Continuous Scoring Agreement Metrics:{C.R}")
    print(f"    Mean Absolute Error (MAE)  : {mae:.4f}  (scale 0-10)")
    print(f"    Mean Squared Error (MSE)   : {mse:.4f}")
    print(f"    Root Mean Squared Error    : {rmse:.4f}")
    print(f"    Pearson Correlation (r)    : {pearson:.4f}")
    print(f"    Spearman Rank Correlation  : {spearman:.4f}")
    print()

    # 2. Classification Metrics across thresholds (3 to 8)
    print(f"  {C.B}Segment Selection Performance by Threshold:{C.R}")
    print()
    print(f"    {'Threshold':<11} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8} | {'Selected':<8} | {'Target Met?':<12}")
    print(f"    {'-'*75}")

    for threshold in range(3, 9):
        # Human selected
        gt_segments = {segment_ids[i] for i in range(total_evaluated) if human_scores[i] >= threshold}
        # AI selected
        sel_segments = {segment_ids[i] for i in range(total_evaluated) if ai_scores[i] >= threshold}

        metrics = compute_segment_precision_recall(sel_segments, gt_segments)
        
        # Check target met (target is > 0.75 for precision and recall)
        met = "No"
        if metrics["precision"] >= 0.75 and metrics["recall"] >= 0.75:
            met = f"{C.GRN}YES{C.R}"
        else:
            reasons = []
            if metrics["precision"] < 0.75: reasons.append("P")
            if metrics["recall"] < 0.75: reasons.append("R")
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
    print(f"  {C.D}The highlighted threshold ({target_threshold}) is the currently configured default.{C.R}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI zero-training baseline against human annotations")
    parser.add_argument("--file", type=str, default="module1_annotations.json", help="Path to annotations JSON file")
    parser.add_argument("--threshold", type=int, default=5, help="Visual importance score threshold [0-10]")
    
    args = parser.parse_args()
    evaluate_baseline(file_path=args.file, target_threshold=args.threshold)
