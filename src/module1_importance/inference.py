"""
Inference Script for Module 1 - Importance Scoring
Loads a trained model and runs predictions on pre-extracted video features
to produce the final segment importance scores (score_V) in the required JSON format.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.module1_importance.model import VideoImportanceScorer
from src.utils.json_schema import validate_module1_output


def run_inference(
    features_path: str,
    timestamps_path: str,
    model_path: str,
    output_json_path: str,
    device: str = "cuda"
):
    """
    Run inference on pre-extracted features and save importance scores to JSON.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------- 1. Load model --------
    print(f"Loading model from {model_path}...")
    model = VideoImportanceScorer()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Model weights file {model_path} not found. Running with random weights (dry run).")
    
    model = model.to(device)
    model.eval()

    # -------- 2. Load features and timestamps --------
    print(f"Loading features from {features_path}...")
    features = np.load(features_path)
    timestamps = np.load(timestamps_path)
    print(f"Loaded feature shape: {features.shape} | Timestamps count: {len(timestamps)}")

    # -------- 3. Process segments --------
    # A segment is 10 seconds (10 frames at 1 FPS). We step by 10 frames.
    video_id = Path(features_path).stem.replace("_features", "")
    output_data = []
    
    sequence_length = 10
    total_frames = len(features)
    num_segments = int(np.ceil(total_frames / sequence_length))
    
    print(f"Processing {num_segments} segments (10s each)...")
    
    for i in range(num_segments):
        start_idx = i * sequence_length
        end_idx = min(start_idx + sequence_length, total_frames)
        
        # Slice features
        seg_features = features[start_idx:end_idx]
        
        # If segment is shorter than sequence_length, pad with zeros
        if len(seg_features) < sequence_length:
            padding = np.zeros((sequence_length - len(seg_features), 2048), dtype=seg_features.dtype)
            seg_features = np.vstack([seg_features, padding])
            
        # Reshape to (batch=1, sequence_length=10, features=2048)
        input_tensor = torch.tensor(seg_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get frame-level predictions: shape (1, 10)
            predictions = model(input_tensor)
            # Average frame predictions to get a single segment-level score (matches training loss average)
            segment_score = float(predictions.mean().cpu().item())
            
        # Timestamps corresponding to the segment
        t_start = float(timestamps[start_idx])
        # For the end timestamp, either use the actual timestamp of the last frame or start + 10.0
        t_end = float(timestamps[end_idx - 1] + 1.0) if end_idx > start_idx else float(t_start + 10.0)
        
        segment_id = f"{video_id}__seg_{i:04d}"
        
        output_data.append({
            "segment_id": segment_id,
            "timestamp_start": round(t_start, 2),
            "timestamp_end": round(t_end, 2),
            "score_V": round(segment_score, 4)
        })

    # -------- 4. Validate output JSON against shared schema --------
    print("Validating output schema...")
    errors = validate_module1_output(output_data)
    if errors:
        print(f"[FAILED] Schema validation failed with {len(errors)} error(s):")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  - ... and {len(errors) - 5} more.")
    else:
        print("[OK] Schema validation passed successfully!")

    # -------- 5. Save output JSON --------
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved segment scores to: {output_json_path}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for Module 1 Importance Scoring")
    parser.add_argument("--features", type=str, required=True, help="Path to *_features.npy file")
    parser.add_argument("--timestamps", type=str, required=True, help="Path to *_timestamps.npy file")
    parser.add_argument("--model", type=str, default="best_module1_model.pt", help="Path to trained model weights checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device to use")
    
    args = parser.parse_args()
    
    run_inference(
        features_path=args.features,
        timestamps_path=args.timestamps,
        model_path=args.model,
        output_json_path=args.output,
        device=args.device
    )
