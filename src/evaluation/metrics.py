"""
Evaluation Metrics for INTEGRA

Implements all quantitative evaluation metrics specified in the project plan:
    - ROUGE-L Score (target: > 0.40)
    - Segment Precision (target: > 0.75)
    - Segment Recall (target: > 0.75)
    - ViT Classification Accuracy (target: > 75%)
    - BERT F1 Score (target: > 0.70)
    - ASR Word Error Rate (target: < 15%)
    - Video Duration Constraint (≤ 600 seconds)
"""

import numpy as np
from typing import List, Dict, Any, Set, Tuple

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)


def compute_rouge_l(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute ROUGE-L score between predicted and reference summaries.
    Target: > 0.40
    
    Args:
        predictions: List of auto-generated summary texts.
        references: List of human-written reference summaries.
    
    Returns:
        Dict with 'precision', 'recall', 'fmeasure' for ROUGE-L.
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score required. Run: pip install rouge-score")
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    scores = {"precision": [], "recall": [], "fmeasure": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["precision"].append(result["rougeL"].precision)
        scores["recall"].append(result["rougeL"].recall)
        scores["fmeasure"].append(result["rougeL"].fmeasure)
    
    return {k: round(np.mean(v), 4) for k, v in scores.items()}


def compute_segment_precision_recall(
    selected_segments: Set[str],
    ground_truth_segments: Set[str],
) -> Dict[str, float]:
    """
    Compute precision and recall for segment selection.
    Target: Precision > 0.75, Recall > 0.75
    
    Args:
        selected_segments: Set of segment IDs selected by the system.
        ground_truth_segments: Set of segment IDs marked as important by humans.
    
    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    if not selected_segments:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    true_positives = selected_segments & ground_truth_segments
    
    precision = len(true_positives) / len(selected_segments) if selected_segments else 0.0
    recall = len(true_positives) / len(ground_truth_segments) if ground_truth_segments else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_word_error_rate(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    Compute ASR Word Error Rate using jiwer.
    Target: < 15% (0.15)
    
    Args:
        predictions: List of ASR-generated transcripts.
        references: List of human-written reference transcripts.
    
    Returns:
        Word Error Rate as a float (0.0 = perfect).
    """
    if not JIWER_AVAILABLE:
        raise ImportError("jiwer required. Run: pip install jiwer")
    
    wer = jiwer.wer(references, predictions)
    return round(wer, 4)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics (for BERT F1 or ViT accuracy).
    Targets: BERT F1 > 0.70, ViT Accuracy > 75%
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional label names for the report.
    
    Returns:
        Dict with 'accuracy', 'f1_weighted', 'precision', 'recall', 'report'.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted"), 4),
        "report": classification_report(y_true, y_pred, target_names=labels),
    }


def check_duration_constraint(
    video_duration: float,
    max_duration: float = 600.0,
) -> Dict[str, Any]:
    """
    Check if video meets the duration constraint.
    Target: ≤ 600 seconds (10 minutes)
    
    Args:
        video_duration: Actual video duration in seconds.
        max_duration: Maximum allowed duration.
    
    Returns:
        Dict with 'passed', 'duration', 'max_duration'.
    """
    return {
        "passed": video_duration <= max_duration,
        "duration_seconds": round(video_duration, 2),
        "max_duration_seconds": max_duration,
        "duration_formatted": f"{int(video_duration // 60)}:{int(video_duration % 60):02d}",
    }


# ─── Evaluation Targets (from project plan Section 8) ─────────────────
EVALUATION_TARGETS = {
    "rouge_l": {"target": 0.40, "metric": "ROUGE-L F-measure", "direction": ">"},
    "segment_precision": {"target": 0.75, "metric": "Segment Precision", "direction": ">"},
    "segment_recall": {"target": 0.75, "metric": "Segment Recall", "direction": ">"},
    "vit_accuracy": {"target": 0.75, "metric": "ViT Classification Accuracy", "direction": ">"},
    "bert_f1": {"target": 0.70, "metric": "BERT F1 Score", "direction": ">"},
    "asr_wer": {"target": 0.15, "metric": "ASR Word Error Rate", "direction": "<"},
    "video_duration": {"target": 600, "metric": "Video Duration (seconds)", "direction": "<="},
}


if __name__ == "__main__":
    print("=== INTEGRA Evaluation Targets ===\n")
    for key, info in EVALUATION_TARGETS.items():
        print(f"  {info['metric']}: {info['direction']} {info['target']}")
