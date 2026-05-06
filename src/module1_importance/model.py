"""
VideoImportanceScorer Model
Combines ResNet-50 for spatial features and BiLSTM for temporal modeling.

Architecture (validated by Rahman et al. 2020, Zhang et al. 2016, Lin et al. 2022):
  - ResNet-50 backbone: extracts 2048-dim visual features per frame
  - Bidirectional LSTM: models temporal context across 30-frame sequences
  - Linear classifier: predicts importance score (0-1) per segment

Owner: Rashmi (214093E) — Module 1: Keyframe Detection & Importance Scoring

=== BEGINNER GUIDE ===
How this model works (step by step):

    1. You give it a batch of video clips. Each clip is 30 frames (30 screenshots).
    2. ResNet-50 looks at each frame and converts it into 2048 numbers (a "feature vector").
       - Think of it like: "this frame has a diagram (feature #45 = high), 
         a person talking (feature #120 = high), text on screen (feature #800 = high)..."
    3. BiLSTM reads these features IN ORDER (frame 1 → frame 30 AND frame 30 → frame 1)
       to understand patterns over time (e.g., "the lecturer paused on this slide longer").
    4. A simple classifier takes the BiLSTM output and predicts: "How important is this
       segment?" as a number between 0 (not important) and 1 (very important).

What you need to do next:
    - Write train.py: Load annotated data, feed it through this model, compute loss, update weights
    - Write inference.py: Load a trained model, feed in a new video, get importance scores
======================
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoImportanceScorer(nn.Module):
    """
    Neural network model to score video segment importance.
    
    Architecture:
    - ResNet-50 backbone for visual feature extraction (pretrained on ImageNet)
    - Bidirectional LSTM for temporal pattern modeling across 30-frame sequences
    - Linear classifier for importance score prediction (0-1)
    
    Input:  (batch, sequence_length, 3, 224, 224) — video frame sequences
    Output: (batch, sequence_length) — importance scores per frame
    
    Example:
        model = VideoImportanceScorer()
        frames = torch.randn(2, 30, 3, 224, 224)  # 2 clips, 30 frames each
        scores = model(frames)                      # shape: (2, 30)
        # scores[0] = [0.1, 0.3, 0.8, ...] — importance of each frame in clip 1
    """
    
    def __init__(self, hidden_size=512, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        # ─── Step 1: Visual Encoder (Pre-trained ResNet-50) ─────────
        # BEGINNER: ResNet-50 is a model trained on 14 million images.
        # We download it already trained and reuse its "knowledge" of images.
        # We remove its last layer (which classified 1000 ImageNet categories)
        # because we don't need to classify cats/dogs — we just want the features.
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # BEGINNER: "Freezing" means we DON'T update these layers during training.
        # The early layers already know basic things (edges, shapes, colors).
        # We only fine-tune the last few layers for our specific task.
        for param in list(self.visual_encoder.parameters())[:-10]:
            param.requires_grad = False
        
        # ─── Step 2: Temporal Modeling (Bidirectional LSTM) ─────────
        # BEGINNER: LSTM reads frames one-by-one and "remembers" what it saw.
        # "Bidirectional" means it reads FORWARD (frame 1→30) AND BACKWARD (30→1).
        # This helps because sometimes you need future context to understand
        # if the current frame is important.
        self.lstm = nn.LSTM(
            input_size=2048,       # Each frame becomes 2048 numbers from ResNet
            hidden_size=hidden_size,  # LSTM internal memory size
            num_layers=num_lstm_layers,  # 2 stacked LSTMs for deeper learning
            batch_first=True,      # Input format: (batch, sequence, features)
            bidirectional=True,    # BiLSTM: reads forward AND backward
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # ─── Step 3: Importance Score Predictor ─────────────────────
        # BEGINNER: This is a simple neural network that takes the LSTM output
        # and predicts a single number between 0 and 1.
        # hidden_size * 2 because BiLSTM outputs forward + backward = double size.
        # Sigmoid at the end squashes any number into the range [0, 1].
        lstm_output_size = hidden_size * 2  # bidirectional doubles the output
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),  # Compress: 1024 → 256
            nn.ReLU(),                          # Activation function (removes negatives)
            nn.Dropout(dropout),                # Randomly turn off neurons (prevents overfitting)
            nn.Linear(256, 1),                  # Compress: 256 → 1 (single score)
            nn.Sigmoid()                        # Squash to [0, 1] range
        )
    
    def forward(self, video_frames):
        """
        Forward pass: takes video frames → returns importance scores.
        
        This is called automatically when you do: scores = model(frames)
        
        Args:
            video_frames: Tensor of shape (batch, sequence_length, 3, 224, 224)
                          - batch: how many clips you process at once (e.g., 8)
                          - sequence_length: frames per clip (30)
                          - 3: RGB color channels (red, green, blue)
                          - 224, 224: image height and width in pixels
        
        Returns:
            importance_scores: Tensor of shape (batch, sequence_length)
                               Each value is a float in [0, 1].
                               Higher = more important.
        """
        batch_size, seq_len, c, h, w = video_frames.shape
        
        # BEGINNER: ResNet expects individual images, not sequences.
        # So we reshape: (2 clips × 30 frames) = 60 individual images
        frames = video_frames.view(batch_size * seq_len, c, h, w)
        
        # Step 1: Extract visual features from each frame
        with torch.set_grad_enabled(self.training):
            visual_features = self.visual_encoder(frames)  # → (60, 2048, 1, 1)
        
        # Reshape back to sequences: (2 clips, 30 frames, 2048 features)
        visual_features = visual_features.view(batch_size, seq_len, -1)
        
        # Step 2: BiLSTM reads the sequence of features
        temporal_features, _ = self.lstm(visual_features)  # → (2, 30, 1024)
        
        # Step 3: Predict importance score for each frame
        importance_scores = self.classifier(temporal_features)  # → (2, 30, 1)
        
        return importance_scores.squeeze(-1)  # → (2, 30) — remove last dimension


# ─── Quick Test (run this file directly to verify it works) ────────────
if __name__ == "__main__":
    print("Testing VideoImportanceScorer...\n")
    
    # Create the model
    model = VideoImportanceScorer(hidden_size=512, num_lstm_layers=2, dropout=0.3)
    
    # Create fake input: 2 clips, each with 30 frames of 224x224 RGB images
    dummy_input = torch.randn(2, 30, 3, 224, 224)
    
    # Run the model
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")  # (2, 30, 3, 224, 224)
    print(f"Output shape: {output.shape}")        # (2, 30)
    print(f"Sample scores for clip 1: {output[0][:5].detach()}")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print("\n✅ Model works correctly!")
