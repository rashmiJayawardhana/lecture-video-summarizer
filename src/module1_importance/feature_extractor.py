"""
ResNet-50 Feature Extraction for Module 1

Extracts 2048-dimensional visual features from video frames using
a pre-trained ResNet-50 backbone (ImageNet weights).

Owner: Rashmi (214093E)
Reference: Rahman et al. 2020
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class FrameFeatureExtractor:
    """
    Extracts ResNet-50 features from video frames.
    
    Usage:
        extractor = FrameFeatureExtractor(device='cuda')
        features = extractor.extract_from_video('lecture_001.mp4')
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet-50, remove classification head
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 2048-dim feature vector from a single frame.
        
        Args:
            frame: BGR image (H, W, 3) from OpenCV.
        
        Returns:
            Feature vector of shape (2048,).
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features.squeeze().cpu().numpy()
    
    def extract_from_video(
        self, 
        video_path: str, 
        fps: int = 1,
        max_frames: Optional[int] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Extract features from a video at the specified frame rate.
        
        Args:
            video_path: Path to the input video file.
            fps: Frames per second to extract (default: 1 fps).
            max_frames: Maximum number of frames to extract.
        
        Returns:
            Tuple of (features, timestamps):
                features: np.ndarray of shape (num_frames, 2048)
                timestamps: list of float timestamps in seconds
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        features_list = []
        timestamps = []
        frame_idx = 0
        
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_features = total_video_frames // frame_interval if total_video_frames > 0 else "?"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                feat = self.extract_from_frame(frame)
                features_list.append(feat)
                timestamps.append(frame_idx / video_fps)
                
                # Print progress every 100 extracted frames
                extracted_count = len(features_list)
                if extracted_count % 100 == 0:
                    print(f"    ... extracted {extracted_count}/{expected_features} features ({extracted_count/expected_features*100:.1f}%)")
                
                if max_frames and len(features_list) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        features = np.stack(features_list) if features_list else np.array([])
        return features, timestamps


if __name__ == "__main__":
    # Quick test
    extractor = FrameFeatureExtractor(device='cpu')
    
    # Test with a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    features = extractor.extract_from_frame(dummy_frame)
    print(f"Feature shape: {features.shape}")  # Expected: (2048,)
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
