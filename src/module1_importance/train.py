import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.module1_importance.model import VideoImportanceScorer

class LectureFeatureDataset(Dataset):
    """
    Loads pre-extracted ResNet-50 features and their corresponding annotations.
    """
    def __init__(self, features_dir, annotations_json, split="train"):
        """
        Args:
            features_dir: Directory containing *_features.npy files.
            annotations_json: Path to the merged module1_annotations.json file.
            split: "train" (videos 1-45), "val" (videos 46-50), or "test" (videos 51-60)
        """
        self.features_dir = Path(features_dir)
        self.split = split
        self.sequence_length = 10  # 10 seconds = 10 frames at 1 FPS
        
        # Load annotations
        with open(annotations_json, 'r', encoding='utf-8') as f:
            all_annotations = json.load(f)
            
        self.samples = []
        self.video_cache = {}
        
        # Filter samples based on the requested split
        print(f"Loading {split} dataset...")
        for seg_id, data in all_annotations.items():
            if data.get("skipped", False) or data.get("normalized_score") is None:
                continue
                
            video_name = data["video_id"]
            
            # Extract video number (assuming format "LecVideo 001...")
            try:
                # E.g. "LecVideo 046 - ... " -> 46
                vid_num = int(video_name.split()[1])
            except (IndexError, ValueError):
                continue
                
            # Check if this video belongs in the current split
            if split == "train" and vid_num > 45:
                continue
            if split == "val" and (vid_num <= 45 or vid_num > 50):
                continue
            if split == "test" and vid_num <= 50:
                continue
                
            self.samples.append(data)
            
            # Cache the .npy file in memory if not already loaded (they are small enough)
            if video_name not in self.video_cache:
                npy_path = self.features_dir / f"{video_name}_features.npy"
                if npy_path.exists():
                    self.video_cache[video_name] = np.load(npy_path)
                else:
                    print(f"Warning: Missing features for {video_name}")
                    
        print(f"Loaded {len(self.samples)} annotated segments for {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        video_name = data["video_id"]
        
        # We need a 10-frame sequence (since FPS=1, timestamp_start matches the frame index)
        start_idx = int(data["timestamp_start"])
        end_idx = int(data["timestamp_end"])
        
        # Get score
        score = float(data["normalized_score"])
        
        # Fetch features
        video_features = self.video_cache.get(video_name)
        if video_features is None:
            # Fallback to zeros if file was missing
            return torch.zeros((self.sequence_length, 2048), dtype=torch.float32), torch.tensor(score, dtype=torch.float32)
            
        # Slice the 10-second window
        segment_features = video_features[start_idx:end_idx]
        
        # Pad with zeros if it's the very end of the video and we got less than 10 frames
        if len(segment_features) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(segment_features), 2048), dtype=segment_features.dtype)
            segment_features = np.vstack([segment_features, padding])
            
        # Keep exactly sequence_length frames (just in case)
        segment_features = segment_features[:self.sequence_length]
        
        return torch.tensor(segment_features, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)


def train_model(data_dir, annotations_path, epochs=10, batch_size=32, lr=1e-4):
    """
    Main training loop for Module 1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Datasets and DataLoaders
    train_dataset = LectureFeatureDataset(data_dir, annotations_path, split="train")
    val_dataset = LectureFeatureDataset(data_dir, annotations_path, split="val")
    
    if len(train_dataset) == 0:
        print("Error: No training data found. Ensure you have run annotate_module1.py and merged the JSON.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = VideoImportanceScorer().to(device)
    
    # We use Mean Squared Error because we want to predict the exact score (regression)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print("\nStarting Training...")
    for epoch in range(epochs):
        # ─── Training Phase ───
        model.train()
        train_loss = 0.0
        
        for features, scores in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            features = features.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            predictions = model(features)
            
            # The model outputs a score per frame: (batch, 10)
            # We want one score per segment, so we average the frame scores
            segment_predictions = predictions.mean(dim=1)
            
            # Calculate loss
            loss = criterion(segment_predictions, scores)
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # ─── Validation Phase ───
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, scores in val_loader:
                features = features.to(device)
                scores = scores.to(device)
                
                predictions = model(features)
                segment_predictions = predictions.mean(dim=1)
                
                loss = criterion(segment_predictions, scores)
                val_loss += loss.item() * features.size(0)
                
        if len(val_loader.dataset) > 0:
            val_loss /= len(val_loader.dataset)
        else:
            val_loss = 0.0
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss and len(val_loader.dataset) > 0:
            best_val_loss = val_loss
            save_path = "best_module1_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")
            

if __name__ == "__main__":
    # Example usage:
    # Set these to your Drive paths when running on Colab
    FEATURES_DIR = "../../data/raw" # Or /content/drive/MyDrive/INTEGRA_Data/raw
    ANNOTATIONS_FILE = "../../data/module1_annotations.json"
    
    # To run this, you need the merged module1_annotations.json
    if Path(ANNOTATIONS_FILE).exists():
        train_model(FEATURES_DIR, ANNOTATIONS_FILE)
    else:
        print(f"Please run the annotation tool first and create {ANNOTATIONS_FILE}")
