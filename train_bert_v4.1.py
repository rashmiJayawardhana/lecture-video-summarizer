import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset

print("=" * 70)
print("BERT v4 TRAINING — 3061 Labels (FIXED)")
print("=" * 70)

# Load labels
with open("src/module2_summarization/labelled_sentences.json", "r") as f:
    data = json.load(f)

print(f"\nTotal labels: {len(data)}")
df = pd.DataFrame(data)[["sentence", "is_important"]]
df.columns = ["text", "label"]

# Same split as evaluation (random_state=42)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model
print("\nLoading base BERT...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./bert_temp_v4",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=30,
    learning_rate=1e-5,
    warmup_steps=100,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nTraining BERT v4 (FIXED) with 3061 labels...")
print("Key fixes: lower LR (1e-5), early stopping, best model saved")
print("Expected time: 30-60 minutes on CPU")
print("Please leave running!\n")

trainer.train()

# Evaluate
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

precision, recall, f1, _ = precision_recall_fscore_support(
    labels, preds, average="binary", pos_label=1, zero_division=0
)

print(
    classification_report(
        labels, preds, target_names=["Not Important", "Important"], digits=4
    )
)

print(f"\nFinal Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

if f1 >= 0.70:
    print(f"\n✅ TARGET MET! ({f1:.4f} >= 0.70)")
else:
    print(f"\n❌ Below target (need 0.70, got {f1:.4f})")

# Save model
output_dir = "src/module2_summarization/bert_model_v4"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nBERT v4 saved to: {output_dir}")