"""
Weighted Score Fusion for Module 4

Implements the fusion formula: S = w1·V + w2·T + w3·L
    - w1·V: Module 1 visual importance score
    - w2·T: Module 2 text importance ratio
    - w3·L: Module 3 slide label score

Ranks all 10-second segments and selects top-scoring segments
for video synthesis.

What does this file do?

    Imagine 3 judges rating a cooking show:
      - Judge 1 (Module 1): "This clip looks visually interesting" → score_V = 0.8
      - Judge 2 (Module 2): "The chef said something important here" → score_T = 0.6
      - Judge 3 (Module 3): "There's a critical recipe card on screen" → score_L = 1.0

    This file COMBINES all three judges' scores into one final score:
      S = 0.33 × 0.8 + 0.33 × 0.6 + 0.34 × 1.0 = 0.802

    Then it picks the top-scoring segments that fit within 10 minutes.

Next :
    - Test this script with mock JSON (already has sample data)
    - Replace mock JSON with REAL JSON from Modules 1, 2, 3
    - Build the full ranked segment list for Pipeline A
    - Grid search weights (try different w1, w2, w3 values)
======================
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# BEGINNER: This maps Module 3's text labels to numbers.
# "Critical" = 1.0 (most important), "Important" = 0.6, "Skip" = 0.0
LABEL_SCORE_MAP = {
    "Critical": 1.0,
    "Important": 0.6,
    "Skip": 0.0,
}


class ScoreFusion:
    """
    Fuse importance scores from Modules 1, 2, and 3.
    
    Formula: S = w1 * score_V + w2 * importance_ratio_T + w3 * score_L
    
    Usage:
        # Step 1: Create fusion with weights
        fusion = ScoreFusion(w1=0.33, w2=0.33, w3=0.34)
        
        # Step 2: Load JSON from each module
        module1_data = json.load(open('module1_output.json'))
        module2_data = json.load(open('module2_output.json'))
        module3_data = json.load(open('module3_output.json'))
        
        # Step 3: Fuse scores
        segments = fusion.fuse(module1_data, module2_data, module3_data)
        
        # Step 4: Select best segments that fit in 5 minutes
        selected = fusion.select_segments(segments, max_duration=300)
        
        # Step 5: Pass 'selected' to Pipeline A or Pipeline B
    """
    
    def __init__(
        self,
        w1: float = 0.33,
        w2: float = 0.33,
        w3: float = 0.34,
        min_score_threshold: float = 0.3,
        segment_duration: float = 10.0,
    ):
        """
        Args:
            w1: Weight for Module 1 visual importance score (score_V).
            w2: Weight for Module 2 text importance ratio (importance_ratio_T).
            w3: Weight for Module 3 slide label score (converted from label).
            min_score_threshold: Segments with S < this are excluded (too boring).
            segment_duration: Duration of each segment in seconds (10s).
        
        BEGINNER: Weights must add up to 1.0 (or very close).
        In Week 7, you'll try different weight combinations to find the best one.
        """
        total = w1 + w2 + w3
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"
        
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_score_threshold = min_score_threshold
        self.segment_duration = segment_duration
    
    def fuse(
        self,
        module1_data: List[dict],
        module2_data: List[dict],
        module3_data: List[dict],
    ) -> List[dict]:
        """
        Fuse scores from all three modules for each 10-second segment.
        
        BEGINNER: This function:
            1. Goes through each segment from Module 1 (each has a time range)
            2. Finds matching sentences from Module 2 in that time range
            3. Finds matching slides from Module 3 in that time range
            4. Combines all scores using the formula: S = w1*V + w2*T + w3*L
            5. Returns segments sorted by score (best first)
        
        Args:
            module1_data: List of dicts from Module 1 JSON output.
                Example: [{"segment_id": "seg_001", "timestamp_start": 0.0,
                           "timestamp_end": 10.0, "score_V": 0.82}, ...]
            module2_data: List of dicts from Module 2 JSON output.
                Example: [{"sentence": "...", "importance_ratio_T": 0.75}, ...]
            module3_data: List of dicts from Module 3 JSON output.
                Example: [{"frame_time": 14.5, "label": "Critical"}, ...]
        
        Returns:
            List of segment dicts with fused score S, sorted by score (best first).
        """
        fused_segments = []
        
        for seg in module1_data:
            seg_start = seg["timestamp_start"]
            seg_end = seg["timestamp_end"]
            score_V = seg["score_V"]
            
            # Find text importance for sentences spoken during this segment
            score_T = self._get_text_score(module2_data, seg_start, seg_end)
            
            # Find slide importance for frames shown during this segment
            score_L = self._get_slide_score(module3_data, seg_start, seg_end)
            
            # THE FUSION FORMULA
            S = self.w1 * score_V + self.w2 * score_T + self.w3 * score_L
            
            fused_segments.append({
                "segment_id": seg["segment_id"],
                "timestamp_start": seg_start,
                "timestamp_end": seg_end,
                "score_V": round(score_V, 4),
                "score_T": round(score_T, 4),
                "score_L": round(score_L, 4),
                "fused_score": round(S, 4),
            })
        
        # Sort by fused score — best segments first
        fused_segments.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return fused_segments
    
    def select_segments(
        self,
        fused_segments: List[dict],
        max_duration: float = 300.0,
        min_topic_sections: int = 3,
    ) -> List[dict]:
        """
        Select top segments that fit within the time limit.
        
        BEGINNER: This is like filling a 10-minute playlist:
            1. Start with the highest-scoring segment
            2. Add the next best, then the next...
            3. Stop when you've filled 5 minutes (or 10 minutes)
            4. Skip any segment below the quality threshold (S < 0.3)
        
        Args:
            fused_segments: Output from self.fuse(), sorted by score.
            max_duration: Maximum total duration in seconds (300 = 5 min).
            min_topic_sections: Minimum number of distinct topic sections.
        
        Returns:
            Selected segments, sorted by TIMESTAMP (for video assembly).
        """
        selected = []
        total_duration = 0.0
        
        for seg in fused_segments:
            # Skip boring segments
            if seg["fused_score"] < self.min_score_threshold:
                continue
            
            seg_duration = seg["timestamp_end"] - seg["timestamp_start"]
            
            # Add if we still have room
            if total_duration + seg_duration <= max_duration:
                selected.append(seg)
                total_duration += seg_duration
        
        # IMPORTANT: Sort by timestamp so the video plays in chronological order
        selected.sort(key=lambda x: x["timestamp_start"])
        
        return selected
    
    def _get_text_score(
        self,
        module2_data: List[dict],
        seg_start: float,
        seg_end: float,
    ) -> float:
        """
        Get average importance_ratio_T for sentences spoken during a segment.
        
        BEGINNER: If 3 sentences were spoken during this 10-second segment,
        and their importance scores are [0.8, 0.2, 0.9], the average is 0.633.
        """
        scores = []
        for sent in module2_data:
            # Check if this sentence overlaps with our segment's time range
            if (sent["timestamp_start"] < seg_end and
                    sent["timestamp_end"] > seg_start):
                scores.append(sent["importance_ratio_T"])
        
        return np.mean(scores) if scores else 0.0
    
    def _get_slide_score(
        self,
        module3_data: List[dict],
        seg_start: float,
        seg_end: float,
    ) -> float:
        """
        Get max slide label score for frames shown during a segment.
        
        BEGINNER: If 2 frames appeared during this segment:
            - Frame at 12.5s: "Important" → 0.6
            - Frame at 14.0s: "Critical" → 1.0
        We take the MAX = 1.0 (because if ANY critical slide appears, the segment matters).
        """
        scores = []
        for frame in module3_data:
            if seg_start <= frame["frame_time"] < seg_end:
                label_score = LABEL_SCORE_MAP.get(frame["label"], 0.0)
                scores.append(label_score)
        
        return max(scores) if scores else 0.0
    
    def save_fused_output(
        self,
        segments: List[dict],
        output_path: str,
    ) -> None:
        """Save fused segment scores to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2)
        
        print(f"Fused scores saved to: {path}")


# ─── DEMO: Test with sample data ──────────────────────────────────────
if __name__ == "__main__":
    # These are SAMPLE outputs — same format each module will produce
    sample_module1 = [
        {"segment_id": "seg_001", "timestamp_start": 0.0, "timestamp_end": 10.0, "score_V": 0.45},
        {"segment_id": "seg_002", "timestamp_start": 10.0, "timestamp_end": 20.0, "score_V": 0.82},
        {"segment_id": "seg_003", "timestamp_start": 20.0, "timestamp_end": 30.0, "score_V": 0.15},
    ]
    
    sample_module2 = [
        {"sentence": "A binary search tree maintains sorted order.",
         "timestamp_start": 12.0, "timestamp_end": 15.5,
         "is_important": True, "importance_ratio_T": 0.75},
        {"sentence": "Let me take a sip of water.",
         "timestamp_start": 5.0, "timestamp_end": 7.0,
         "is_important": False, "importance_ratio_T": 0.05},
    ]
    
    sample_module3 = [
        {"frame_time": 3.0, "label": "Skip", "ocr_text": ""},
        {"frame_time": 14.5, "label": "Critical", "ocr_text": "BST: O(log n)"},
        {"frame_time": 25.0, "label": "Important", "ocr_text": "Example code"},
    ]
    
    # Run fusion
    fusion = ScoreFusion(w1=0.33, w2=0.33, w3=0.34)
    fused = fusion.fuse(sample_module1, sample_module2, sample_module3)
    
    print("=== Fused Segment Scores (sorted best → worst) ===\n")
    for seg in fused:
        print(f"  {seg['segment_id']}: S = {seg['fused_score']:.3f}")
        print(f"    Visual={seg['score_V']:.2f}  Text={seg['score_T']:.2f}  Slide={seg['score_L']:.2f}")
        above = "✅ SELECTED" if seg["fused_score"] >= 0.3 else "❌ Below threshold"
        print(f"    → {above}\n")
    
    # Select top segments
    selected = fusion.select_segments(fused, max_duration=20)
    print(f"=== Selected {len(selected)} segments for the output video ===")
    for seg in selected:
        print(f"  {seg['segment_id']}: {seg['timestamp_start']}s – {seg['timestamp_end']}s "
              f"(S = {seg['fused_score']:.3f})")
