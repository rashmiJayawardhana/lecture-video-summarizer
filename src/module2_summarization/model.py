"""
BERT-base Sentence Importance Classifier for Module 2

Fine-tuned BERT-base model that classifies lecture sentences as
Important (1) or Not Important (0) based on three criteria:
    1. Concept introduction
    2. Term definition
    3. Key explanation

Owner: Ravindu (214095L)
Reference: Gonzalez et al. 2023
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Any


class LectureSentenceClassifier(nn.Module):
    """
    BERT-base classifier for lecture sentence importance.
    
    Architecture:
        BERT-base-uncased → [CLS] token → Linear(768, 2) → Softmax
    
    Input:  Tokenized sentence (max_length=512)
    Output: Binary classification (Important / Not Important)
    """
    
    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).
            token_type_ids: Optional token type IDs.
        
        Returns:
            logits: Classification logits (batch, num_labels).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict_importance(
        self,
        sentences: List[str],
        tokenizer: BertTokenizer,
        device: str = "cuda",
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Predict importance for a list of sentences.
        
        Args:
            sentences: List of sentence strings.
            tokenizer: BERT tokenizer instance.
            device: Device for inference.
            threshold: Classification threshold.
        
        Returns:
            List of dicts with 'sentence', 'is_important', 'importance_ratio_T'.
        """
        self.eval()
        results = []
        
        for sentence in sentences:
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)
                importance_score = probs[0, 1].item()  # P(Important)
            
            results.append({
                "sentence": sentence,
                "is_important": importance_score >= threshold,
                "importance_ratio_T": round(importance_score, 4),
            })
        
        return results


if __name__ == "__main__":
    model = LectureSentenceClassifier()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LectureSentenceClassifier initialized")
    print(f"Total parameters: {total_params:,}")
