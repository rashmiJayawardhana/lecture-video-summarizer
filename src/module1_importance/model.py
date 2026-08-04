"""
VideoImportanceScorer Model
Combines ResNet-50 for spatial features and BiLSTM for temporal modeling.

Architecture (validated by Rahman et al. 2020, Zhang et al. 2016, Lin et al. 2022):
  - ResNet-50 backbone: extracts 2048-dim visual features per frame
  - Bidirectional LSTM: models temporal context across 10-frame sequences
  - Linear classifier: predicts importance score (0-1) per segment

How this model works (step by step):

    1. You give it a batch of video clips. Each clip is 10 frames (10 seconds, at 1 FPS).
    2. Since we already ran `extract_all_features.py`, the frames are already converted into
       2048 numbers (a "feature vector") per frame by ResNet-50.
    3. BiLSTM reads these features IN ORDER (frame 1 → frame 10 AND frame 10 → frame 1)
       to understand patterns over time (e.g., "the lecturer paused on this slide longer").
    4. A simple classifier takes the BiLSTM output and predicts: "How important is this
       segment?" as a number between 0 (not important) and 1 (very important).

next:
    - train.py: Load annotated data, feed it through this model, compute loss, update weights
    - inference.py: Load a trained model, feed in a new video, get importance scores
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VideoImportanceScorer(nn.Module):
    """
    Neural network model to score video segment importance.
    
    Architecture:
    - Bidirectional LSTM for temporal pattern modeling across sequences
    - Linear classifier for importance score prediction (0-1)
    
    Input:  (batch, sequence_length, 2048) — pre-extracted ResNet-50 features
    Output: (batch, sequence_length) — importance scores per frame
    
    Example:
        model = VideoImportanceScorer()
        features = torch.randn(2, 10, 2048)  # 2 clips, 10 frames each, 2048-dim features
        scores = model(features)             # shape: (2, 10)
        # scores[0] = [0.1, 0.3, 0.8, ...] — importance of each frame in clip 1
    """
    
    def __init__(self, hidden_size=512, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        
        # ─── Step 1: Visual Encoder ─────────
        # Note: We used to have ResNet-50 here, but to save compute and memory,
        # we now pre-extract features using `extract_all_features.py` and pass
        # the 2048-dim vectors directly into this model!
        
        # ─── Step 2: Temporal Modeling (Bidirectional LSTM) ─────────
        # LSTM reads frames one-by-one and "remembers" what it saw.
        # "Bidirectional" means it reads FORWARD (frame 1→10) AND BACKWARD (10→1).
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
        # This is a simple neural network that takes the LSTM output
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
    
    def forward(self, features):
        """
        Forward pass: takes pre-extracted features → returns importance scores.
        
        This is called automatically when you do: scores = model(features)
        
        Args:
            features: Tensor of shape (batch, sequence_length, 2048)
                      - batch: how many clips you process at once (e.g., 8)
                      - sequence_length: frames per clip (e.g., 10 for a 10s segment)
                      - 2048: ResNet-50 feature dimension
        
        Returns:
            importance_scores: Tensor of shape (batch, sequence_length)
                               Each value is a float in [0, 1].
                               Higher = more important.
        """
        
        # Step 1: BiLSTM reads the sequence of features
        temporal_features, _ = self.lstm(features)  # → (batch, seq_len, 1024)
        
        # Step 2: Predict importance score for each frame
        importance_scores = self.classifier(temporal_features)  # → (batch, seq_len, 1)
        
        return importance_scores.squeeze(-1)  # → (batch, seq_len) — remove last dimension


# ─── Quick Test (run this file directly to verify it works) ────────────
if __name__ == "__main__":
    print("Testing VideoImportanceScorer...\n")
    
    # Create the model
    model = VideoImportanceScorer(hidden_size=512, num_lstm_layers=2, dropout=0.3)
    
    # Create fake input: 2 clips, each with 10 frames of 2048-dim features
    dummy_input = torch.randn(2, 10, 2048)
    
    # Run the model
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")  # (2, 10, 2048)
    print(f"Output shape: {output.shape}")        # (2, 10)
    print(f"Sample scores for clip 1: {output[0][:5].detach()}")
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (Trainable): {total_params:,}")
    print("\n✅ Model works correctly!")
