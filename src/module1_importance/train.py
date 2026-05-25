import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class LectureFeatureDataset(Dataset):
    def __init__(self, features_dir, annotations_json):
        # Load your annotations and the matching .npy files
        self.features = {}
        for npy_file in Path(features_dir).glob("*_features.npy"):
            video_name = npy_file.name.replace("_features.npy", "")
            self.features[video_name] = np.load(npy_file) # Loads instantly!
            
    def __getitem__(self, idx):
        # 1. Fetch the segment timestamp (e.g. 10s to 20s)
        # 2. Slice the features array (e.g. indices 10 to 20)
        # 3. Feed the sliced sequence of shape (10, 2048) into the BiLSTM!
        pass
