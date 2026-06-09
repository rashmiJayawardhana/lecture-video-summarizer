"""
Unit Tests for Module 1 - Keyframe Detection & Importance Scoring

To run tests:
    pytest tests/
"""

import os
import sys
import json
import tempfile
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.module1_importance.model import VideoImportanceScorer
from src.module1_importance.train import LectureFeatureDataset
from src.module1_importance.inference import run_inference


def test_video_importance_scorer_model():
    """
    Test VideoImportanceScorer neural network dimensions and shapes.
    """
    model = VideoImportanceScorer(hidden_size=256, num_lstm_layers=2, dropout=0.3)
    
    # Fake batch of features: batch_size=4, sequence_length=10, feature_dim=2048
    dummy_input = torch.randn(4, 10, 2048)
    
    # Run forward pass
    output = model(dummy_input)
    
    # Assert output shape is (batch_size, sequence_length) -> (4, 10)
    assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
    
    # Assert output values are strictly in [0, 1] range (enforced by Sigmoid)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output scores must be bounded between 0 and 1"
    
    # Check that model parameters are trainable
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 0, "Model should have trainable parameters"


def test_lecture_feature_dataset_loading_and_augmentation():
    """
    Test LectureFeatureDataset loading, padding, and data augmentation.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        
        # 1. Create a dummy annotations JSON file
        dummy_annotations = {
            "LecVideo 001 - Intro to Programming__seg_0000": {
                "video_id": "LecVideo 001 - Intro to Programming",
                "segment_index": 0,
                "segment_id": "LecVideo 001 - Intro to Programming__seg_0000",
                "timestamp_start": 0.0,
                "timestamp_end": 10.0,
                "raw_score": 5,
                "normalized_score": 0.5,
                "skipped": False,
                "annotator": "TestAnnotator"
            },
            "LecVideo 001 - Intro to Programming__seg_0001": {
                "video_id": "LecVideo 001 - Intro to Programming",
                "segment_index": 1,
                "segment_id": "LecVideo 001 - Intro to Programming__seg_0001",
                "timestamp_start": 10.0,
                "timestamp_end": 20.0,
                "raw_score": 8,
                "normalized_score": 0.8,
                "skipped": False,
                "annotator": "TestAnnotator"
            }
        }
        
        ann_file = tmp_dir_path / "module1_annotations.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(dummy_annotations, f)
            
        # 2. Create a dummy features .npy file for LecVideo 001
        # Video is 20 seconds long -> 20 frames at 1 FPS
        dummy_features = np.random.randn(20, 2048).astype(np.float32)
        np.save(tmp_dir_path / "LecVideo 001 - Intro to Programming_features.npy", dummy_features)
        
        # 3. Test dataset without augmentation
        dataset = LectureFeatureDataset(
            features_dir=tmp_dir,
            annotations_json=str(ann_file),
            split="train",
            augment=False
        )
        
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        features, score = dataset[0]
        assert features.shape == (10, 2048), f"Expected shape (10, 2048), got {features.shape}"
        assert score == 0.5, f"Expected score 0.5, got {score}"
        
        # 4. Test dataset with augmentation (temporal jitter & frame dropping)
        dataset_aug = LectureFeatureDataset(
            features_dir=tmp_dir,
            annotations_json=str(ann_file),
            split="train",
            augment=True
        )
        
        # Seed NumPy for deterministic augmentation testing
        np.random.seed(42)
        
        features_aug, score_aug = dataset_aug[0]
        assert features_aug.shape == (10, 2048)
        assert score_aug == 0.5
        
        # Check that one frame in augmented features was zeroed out (random frame dropping)
        zero_rows = np.all(features_aug.numpy() == 0.0, axis=1)
        assert np.any(zero_rows), "Expected at least one frame to be zeroed out in augmented sequence"


def test_inference_pipeline_and_validation():
    """
    Test the complete inference execution and JSON validation.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        
        # Create mock features and timestamps
        mock_features = np.random.randn(25, 2048).astype(np.float32)
        mock_timestamps = np.arange(0, 25, 1).astype(float)
        
        feat_path = tmp_dir_path / "LecVideo_001_features.npy"
        time_path = tmp_dir_path / "LecVideo_001_timestamps.npy"
        out_path = tmp_dir_path / "scores.json"
        
        np.save(feat_path, mock_features)
        np.save(time_path, mock_timestamps)
        
        # Create a dummy model weights checkpoint
        model = VideoImportanceScorer()
        model_path = tmp_dir_path / "mock_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Run inference
        output = run_inference(
            features_path=str(feat_path),
            timestamps_path=str(time_path),
            model_path=str(model_path),
            output_json_path=str(out_path),
            device="cpu"
        )
        
        # Assert output is created and has correct size
        assert len(output) == 3, f"Expected 3 segments for 25 frames (sequence_length=10), got {len(output)}"
        assert out_path.exists(), "Output JSON file was not saved"
        
        # Load and verify JSON structure
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data[0]["segment_id"] == "LecVideo_001__seg_0000"
        assert data[0]["timestamp_start"] == 0.0
        assert data[0]["timestamp_end"] == 10.0
        assert "score_V" in data[0]
        assert 0.0 <= data[0]["score_V"] <= 1.0
