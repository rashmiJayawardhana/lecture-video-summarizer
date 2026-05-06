"""
Shared JSON Schema Definitions & Validators

These schemas define the inter-module communication format for INTEGRA.
Agreed in Week 1, frozen after Week 5 integration meeting.

Schema Reference:
    Module 1 → { segment_id, timestamp_start, timestamp_end, score_V }
    Module 2 → { sentence, timestamp_start, timestamp_end, is_important, importance_ratio_T }
    Module 3 → { frame_time, label, ocr_text }
    Fusion   → S = w1·V + w2·T + w3·L
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


# ─── Module 1 JSON Schema ─────────────────────────────────────────────
MODULE1_SCHEMA = {
    "segment_id": str,       # e.g., "seg_001"
    "timestamp_start": float, # seconds from video start
    "timestamp_end": float,   # seconds from video start
    "score_V": float,         # importance score [0.0, 1.0]
}

# ─── Module 2 JSON Schema ─────────────────────────────────────────────
MODULE2_SCHEMA = {
    "sentence": str,             # transcribed sentence text
    "timestamp_start": float,    # seconds from video start
    "timestamp_end": float,      # seconds from video start
    "is_important": bool,        # True if sentence is important
    "importance_ratio_T": float, # importance ratio [0.0, 1.0]
}

# ─── Module 3 JSON Schema ─────────────────────────────────────────────
MODULE3_SCHEMA = {
    "frame_time": float, # seconds from video start
    "label": str,        # "Critical", "Important", or "Skip"
    "ocr_text": str,     # extracted text from slide (empty if Skip)
}

# Valid labels for Module 3
MODULE3_VALID_LABELS = {"Critical", "Important", "Skip"}


def validate_module1_output(data: List[Dict[str, Any]]) -> List[str]:
    """
    Validate Module 1 JSON output against the agreed schema.
    
    Args:
        data: List of segment dictionaries from Module 1.
    
    Returns:
        List of error messages. Empty list means valid.
    """
    errors = []
    for i, segment in enumerate(data):
        prefix = f"Segment {i}"
        
        for key, expected_type in MODULE1_SCHEMA.items():
            if key not in segment:
                errors.append(f"{prefix}: Missing required field '{key}'")
            elif not isinstance(segment[key], expected_type):
                errors.append(
                    f"{prefix}: Field '{key}' expected {expected_type.__name__}, "
                    f"got {type(segment[key]).__name__}"
                )
        
        # Validate score range
        if "score_V" in segment:
            if not (0.0 <= segment["score_V"] <= 1.0):
                errors.append(f"{prefix}: score_V={segment['score_V']} out of range [0, 1]")
        
        # Validate timestamps
        if "timestamp_start" in segment and "timestamp_end" in segment:
            if segment["timestamp_start"] >= segment["timestamp_end"]:
                errors.append(f"{prefix}: timestamp_start >= timestamp_end")
    
    return errors


def validate_module2_output(data: List[Dict[str, Any]]) -> List[str]:
    """
    Validate Module 2 JSON output against the agreed schema.
    
    Args:
        data: List of sentence dictionaries from Module 2.
    
    Returns:
        List of error messages. Empty list means valid.
    """
    errors = []
    for i, sentence in enumerate(data):
        prefix = f"Sentence {i}"
        
        for key, expected_type in MODULE2_SCHEMA.items():
            if key not in sentence:
                errors.append(f"{prefix}: Missing required field '{key}'")
            elif not isinstance(sentence[key], expected_type):
                errors.append(
                    f"{prefix}: Field '{key}' expected {expected_type.__name__}, "
                    f"got {type(sentence[key]).__name__}"
                )
        
        # Validate importance ratio range
        if "importance_ratio_T" in sentence:
            if not (0.0 <= sentence["importance_ratio_T"] <= 1.0):
                errors.append(
                    f"{prefix}: importance_ratio_T={sentence['importance_ratio_T']} "
                    f"out of range [0, 1]"
                )
    
    return errors


def validate_module3_output(data: List[Dict[str, Any]]) -> List[str]:
    """
    Validate Module 3 JSON output against the agreed schema.
    
    Args:
        data: List of frame dictionaries from Module 3.
    
    Returns:
        List of error messages. Empty list means valid.
    """
    errors = []
    for i, frame in enumerate(data):
        prefix = f"Frame {i}"
        
        for key, expected_type in MODULE3_SCHEMA.items():
            if key not in frame:
                errors.append(f"{prefix}: Missing required field '{key}'")
            elif not isinstance(frame[key], expected_type):
                errors.append(
                    f"{prefix}: Field '{key}' expected {expected_type.__name__}, "
                    f"got {type(frame[key]).__name__}"
                )
        
        # Validate label value
        if "label" in frame:
            if frame["label"] not in MODULE3_VALID_LABELS:
                errors.append(
                    f"{prefix}: label='{frame['label']}' not in "
                    f"{MODULE3_VALID_LABELS}"
                )
    
    return errors


def load_and_validate(json_path: str, module: int) -> tuple:
    """
    Load a JSON file and validate it against the appropriate module schema.
    
    Args:
        json_path: Path to the JSON file.
        module: Module number (1, 2, or 3).
    
    Returns:
        Tuple of (data, errors). data is the loaded JSON, errors is a list.
    """
    path = Path(json_path)
    if not path.exists():
        return None, [f"File not found: {json_path}"]
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    validators = {
        1: validate_module1_output,
        2: validate_module2_output,
        3: validate_module3_output,
    }
    
    if module not in validators:
        return data, [f"No validator for module {module}"]
    
    errors = validators[module](data)
    return data, errors


# ─── Sample JSON for Testing (Mid-Week 3 checkpoint) ──────────────────

SAMPLE_MODULE1_OUTPUT = [
    {
        "segment_id": "seg_001",
        "timestamp_start": 0.0,
        "timestamp_end": 10.0,
        "score_V": 0.45
    },
    {
        "segment_id": "seg_002",
        "timestamp_start": 10.0,
        "timestamp_end": 20.0,
        "score_V": 0.82
    },
]

SAMPLE_MODULE2_OUTPUT = [
    {
        "sentence": "A binary search tree is a data structure that maintains sorted order.",
        "timestamp_start": 12.0,
        "timestamp_end": 15.5,
        "is_important": True,
        "importance_ratio_T": 0.75
    },
    {
        "sentence": "Let me take a sip of water.",
        "timestamp_start": 15.5,
        "timestamp_end": 17.0,
        "is_important": False,
        "importance_ratio_T": 0.05
    },
]

SAMPLE_MODULE3_OUTPUT = [
    {
        "frame_time": 14.5,
        "label": "Critical",
        "ocr_text": "Binary Search Tree: O(log n) average case"
    },
    {
        "frame_time": 30.0,
        "label": "Skip",
        "ocr_text": ""
    },
]


if __name__ == "__main__":
    # Self-test: validate sample outputs
    print("=== Validating Sample Module Outputs ===\n")
    
    for name, sample, module_num in [
        ("Module 1", SAMPLE_MODULE1_OUTPUT, 1),
        ("Module 2", SAMPLE_MODULE2_OUTPUT, 2),
        ("Module 3", SAMPLE_MODULE3_OUTPUT, 3),
    ]:
        validators = {
            1: validate_module1_output,
            2: validate_module2_output,
            3: validate_module3_output,
        }
        errors = validators[module_num](sample)
        if errors:
            print(f"❌ {name}: {len(errors)} error(s)")
            for err in errors:
                print(f"   - {err}")
        else:
            print(f"✅ {name}: Valid ({len(sample)} records)")
    
    print("\n=== Schema Validation Complete ===")
