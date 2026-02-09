"""
VideoImportanceScorer Model
Combines ResNet50 for visual features and LSTM for temporal modeling
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoImportanceScorer(nn.Module):
    """
    Neural network model to score video segment importance.
    
    Architecture:
    - ResNet50 backbone for visual feature extraction
    - LSTM for temporal pattern modeling
    - Linear classifier for importance score prediction (0-1)
    """
    
    def __init__(self, hidden_size=512, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        # Visual encoder: Pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers (optional - can fine-tune later)
        for param in list(self.visual_encoder.parameters())[:-10]:
            param.requires_grad = False
        
        # Temporal modeling: LSTM
        self.lstm = nn.LSTM(
            input_size=2048,  # ResNet50 output dimension
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Importance score predictor
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output: importance score 0-1
        )
    
    def forward(self, video_frames):
        """
        Forward pass
        
        Args:
            video_frames: Tensor of shape (batch, sequence_length, 3, 224, 224)
        
        Returns:
            importance_scores: Tensor of shape (batch, sequence_length, 1)
        """
        batch_size, seq_len, c, h, w = video_frames.shape
        
        # Reshape for ResNet: (batch * seq_len, 3, 224, 224)
        frames = video_frames.view(batch_size * seq_len, c, h, w)
        
        # Extract visual features
        with torch.set_grad_enabled(self.training):
            visual_features = self.visual_encoder(frames)  # (batch * seq_len, 2048, 1, 1)
        
        visual_features = visual_features.view(batch_size, seq_len, -1)  # (batch, seq_len, 2048)
        
        # Temporal modeling with LSTM
        temporal_features, _ = self.lstm(visual_features)  # (batch, seq_len, hidden_size)
        
        # Predict importance scores
        importance_scores = self.classifier(temporal_features)  # (batch, seq_len, 1)
        
        return importance_scores.squeeze(-1)  # (batch, seq_len)


if __name__ == "__main__":
    # Test the model
    model = VideoImportanceScorer()
    
    # Dummy input: batch of 2, sequence of 10 frames, 224x224 RGB
    dummy_input = torch.randn(2, 10, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample scores: {output[0]}")
