import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification

# ── Settings ──────────────────────────────────────────────
BERT_MODEL_PATH = "src/module2_summarization/bert_model_v2"
LABELS_FILE     = "src/module2_summarization/labelled_sentences.json"
# ──────────────────────────────────────────────────────────

print("=" * 70)
print("MODULE 2 — BERT EVALUATION (80/20 Train-Test Split)")
print("=" * 70)

# Load model
print("\nLoading BERT model...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.eval()
print("BERT loaded!")

# Load labels
print("\nLoading labels...")
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)[["sentence", "is_important"]]
print(f"Total labels: {len(df)}")
print(f"Important (1): {sum(df['is_important'] == 1)}")
print(f"Not important (0): {sum(df['is_important'] == 0)}")

# 80/20 split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["is_important"]
)
print(f"\nTrain set: {len(train_df)} sentences")
print(f"Test set:  {len(test_df)} sentences")

# IT keywords for novelty
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

def classify_sentence(sentence):
    """Classify with BERT + novel features. Returns (is_important, confidence)."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, 
                       max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()
    
    # Novel feature: IT keyword boost
    if any(kw in sentence.lower() for kw in IT_KEYWORDS):
        confidence = min(confidence + 0.10, 1.0)
    
    # Novel feature: Definition pattern (forces True)
    is_definition = any(p in sentence.lower() for p in DEFINITION_PATTERNS)
    if is_definition:
        confidence = min(confidence + 0.15, 1.0)
        is_important = True
    else:
        # Single cutoff at 0.5
        is_important = confidence > 0.5
    
    return is_important, confidence

# Evaluate on test set
print(f"\nClassifying {len(test_df)} test sentences...")
print("This will take 2-3 minutes...\n")

y_true = []
y_pred = []
confidences = []

for i, row in enumerate(test_df.itertuples()):
    if i % 100 == 0:
        print(f"Processed {i}/{len(test_df)}...")
    
    is_pred, conf = classify_sentence(row.sentence)
    y_true.append(row.is_important)
    y_pred.append(int(is_pred))
    confidences.append(conf)

print(f"\nAll {len(test_df)} test sentences classified!")

# Calculate metrics
print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='binary', pos_label=1, zero_division=0
)

# Also get per-class metrics
report = classification_report(
    y_true, y_pred,
    target_names=["Not Important", "Important"],
    digits=4
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nSingle Binary Classification at cutoff = 0.5")
print(f"With novel feature: Definition patterns force True\n")

print(f"{'Metric':<20} {'Value':<15}")
print("-" * 35)
print(f"{'Precision':<20} {precision:.4f}")
print(f"{'Recall':<20} {recall:.4f}")
print(f"{'F1 Score':<20} {f1:.4f}")
print(f"{'Accuracy':<20} {(tp + tn) / (tp + tn + fp + fn):.4f}")

print(f"\nDetailed Classification Report:")
print(report)

print(f"\nConfusion Matrix:")
print(f"                 Predicted Not    Predicted Important")
print(f"Actual Not       {tn:<15} {fp:<15}")
print(f"Actual Important {fn:<15} {tp:<15}")

# Target check
print("\n" + "=" * 70)
print("TARGET ACHIEVEMENT")
print("=" * 70)
target_f1 = 0.70
print(f"Research target F1: {target_f1}")
print(f"Achieved F1:        {f1:.4f}")
if f1 >= target_f1:
    print(f"STATUS: TARGET EXCEEDED by {((f1 - target_f1) / target_f1) * 100:.1f}%")
else:
    print(f"STATUS: Below target")

# Save results
results = {
    "dataset_size": len(df),
    "train_size": len(train_df),
    "test_size": len(test_df),
    "cutoff_threshold": 0.5,
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "target_f1": target_f1,
    "target_met": bool(f1 >= target_f1)
}

with open("src/module2_summarization/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: src/module2_summarization/evaluation_results.json")
print("\n" + "=" * 70)