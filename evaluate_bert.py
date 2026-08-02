import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# ── Settings ──────────────────────────────────────────────
BERT_MODEL_PATH = "src/module2_summarization/bert_model_v2"
LABELS_FILE     = "src/module2_summarization/labelled_sentences.json"
# ──────────────────────────────────────────────────────────

print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.eval()
print("BERT loaded!\n")

# IT Keywords (for boost feature)
IT_KEYWORDS = [
    "algorithm", "database", "sql", "query", "protocol",
    "security", "encryption", "authentication", "authorization",
    "confidentiality", "integrity", "availability", "network",
    "software", "engineering", "process", "methodology",
    "agile", "waterfall", "requirements", "design", "testing",
    "risk", "threat", "vulnerability", "attack", "firewall",
    "system", "data", "information", "computer", "programming"
]

DEFINITION_PATTERNS = [
    "is defined as", "stands for", "is a type of", "refers to",
    "is called", "is known as", "means that", "is the process of"
]

def check_keyword_boost(sentence):
    return any(kw in sentence.lower() for kw in IT_KEYWORDS)

def check_definition(sentence):
    return any(p in sentence.lower() for p in DEFINITION_PATTERNS)

def classify_with_features(sentence, all_sentences):
    """Classify sentence and return confidence score (0-1) scaled to (0-10)"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, 
                       max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()
    
    # Apply novelty boosts
    if check_keyword_boost(sentence):
        confidence = min(confidence + 0.10, 1.0)
    if check_definition(sentence):
        confidence = min(confidence + 0.15, 1.0)
    
    # Scale 0-1 to 0-10 (like Module 1)
    score = confidence * 10
    return score

# Load labeled data
print("Loading labels...")
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total labeled sentences: {len(data)}\n")

# Get all sentences for repetition analysis
all_sentences = [d["sentence"] for d in data]

# Classify all sentences
print("Classifying sentences with BERT + Novel Features...")
print("This will take 5-10 minutes...\n")

predictions = []
true_labels = []

for i, item in enumerate(data):
    if i % 100 == 0:
        print(f"Processed {i}/{len(data)}...")
    
    sentence = item["sentence"]
    true_label = item["is_important"]
    
    # Get score (0-10)
    score = classify_with_features(sentence, all_sentences)
    
    predictions.append(score)
    true_labels.append(true_label)

print(f"\nAll {len(data)} sentences processed!\n")

# ── EVALUATE AT DIFFERENT THRESHOLDS ──────────────────────
print("=" * 80)
print("EVALUATION RESULTS — BERT + 4 Novel Features")
print("=" * 80)
print(f"{'Threshold':<15} {'Precision':<12} {'Recall':<10} {'F1-score':<12} {'Selected':<12} {'Target Met?':<15}")
print("-" * 80)

thresholds = [3, 4, 5, 6, 7, 8]
results = []

for threshold in thresholds:
    # Predict important (1) if score >= threshold
    pred_labels = [1 if p >= threshold else 0 for p in predictions]
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    # Count selected
    selected = sum(pred_labels)
    
    # Check target (precision > 0.85 AND recall > 0.75)
    target_met = "YES" if (precision >= 0.85 and recall >= 0.75) else "No (R)"
    
    print(f"Score >= {threshold:<7} {precision:<12.4f} {recall:<10.4f} {f1:<12.4f} {selected:<12} {target_met:<15}")
    
    results.append({
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "selected": selected,
        "target_met": target_met
    })

print("=" * 80)

# Save results
with open("src/module2_summarization/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: src/module2_summarization/evaluation_results.json")
print("\nBest threshold: Look for high F1 with 'YES' target met!")