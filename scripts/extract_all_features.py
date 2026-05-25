import os
import torch
import numpy as np
from pathlib import Path
from src.module1_importance.feature_extractor import FrameFeatureExtractor

# Initialize extractor
extractor = FrameFeatureExtractor(device='cuda')

raw_dir = Path("data/raw")
features_dir = Path("data/processed/features")
features_dir.mkdir(parents=True, exist_ok=True)

# Loop through all 35 raw videos
for video_path in sorted(raw_dir.glob("*.mp4")):
    output_feat_path = features_dir / f"{video_path.stem}_features.npy"
    output_time_path = features_dir / f"{video_path.stem}_timestamps.npy"
    
    # Checkpoint check: if files exist, skip this video!
    if output_feat_path.exists() and output_time_path.exists():
        print(f"Skipping {video_path.name} (Features already extracted ✓)")
        continue
        
    print(f"Extracting features for {video_path.name}...")
    
    # Extract features using pre-trained ResNet-50
    # (Extracts 2048-dim feature vector per frame at 1 FPS)
    features, timestamps = extractor.extract_from_video(str(video_path), fps=1)
    
    # Save the features as a single clean binary file on your Drive
    np.save(output_feat_path, features)
    np.save(output_time_path, np.array(timestamps))
    
    print(f"Saved: {video_path.stem}_features.npy | Shape: {features.shape}")
