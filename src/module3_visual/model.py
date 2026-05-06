"""
ViT-base Slide Importance Classifier for Module 3

Fine-tuned Vision Transformer (ViT-base) that classifies extracted
slide frames into three categories:
    1. Critical  — Key diagrams, important formulas, summary slides
    2. Important — Definitions, step explanations, worked examples
    3. Skip      — Title slides, duplicates, blank frames

Owner: Fazly (214008C)
Reference: Biswas et al. 2025
Fallback: If accuracy < 70%, switch to ResNet-50 + linear head
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor
from typing import List, Dict, Any
from PIL import Image


# Label mapping
LABEL_MAP = {0: "Critical", 1: "Important", 2: "Skip"}
LABEL_TO_ID = {"Critical": 0, "Important": 1, "Skip": 2}


class SlideImportanceClassifier(nn.Module):
    """
    ViT-base classifier for lecture slide importance.
    
    Architecture:
        ViT-base-patch16-224 → [CLS] token → Linear(768, 3) → Softmax
    
    Input:  Slide image (224 x 224 RGB)
    Output: Classification into Critical / Important / Skip
    """
    
    def __init__(
        self,
        base_model: str = "google/vit-base-patch16-224",
        num_labels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        """
        Forward pass.
        
        Args:
            pixel_values: Image tensor (batch, 3, 224, 224).
        
        Returns:
            logits: Classification logits (batch, num_labels).
        """
        outputs = self.vit(pixel_values=pixel_values)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_label(
        self,
        images: List[Image.Image],
        feature_extractor: ViTFeatureExtractor,
        device: str = "cuda",
    ) -> List[Dict[str, Any]]:
        """
        Predict importance label for a list of slide images.
        
        Args:
            images: List of PIL Image objects.
            feature_extractor: ViT feature extractor from Hugging Face.
            device: Device for inference.
        
        Returns:
            List of dicts with 'label' and 'confidence'.
        """
        self.eval()
        results = []
        
        for img in images:
            inputs = feature_extractor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            with torch.no_grad():
                logits = self.forward(pixel_values)
                probs = torch.softmax(logits, dim=-1)
                pred_id = probs.argmax(dim=-1).item()
                confidence = probs[0, pred_id].item()
            
            results.append({
                "label": LABEL_MAP[pred_id],
                "confidence": round(confidence, 4),
            })
        
        return results


class SlideImportanceClassifierFallback(nn.Module):
    """
    Fallback: ResNet-50 + linear classification head.
    Use if ViT accuracy < 70% (validated by Biswas et al. 2025).
    """
    
    def __init__(self, num_labels: int = 3, dropout: float = 0.3):
        super().__init__()
        
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_labels),
        )
    
    def forward(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.classifier(features)


if __name__ == "__main__":
    model = SlideImportanceClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SlideImportanceClassifier (ViT-base) initialized")
    print(f"Total parameters: {total_params:,}")
    print(f"Labels: {LABEL_MAP}")
