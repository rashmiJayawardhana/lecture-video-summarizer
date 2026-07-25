# Module 4 — Stage 1: domain-adaptation fine-tune (BART-base on QMSum)
#
# Run this in Google Colab (Runtime > Change runtime type > T4 GPU).
# Paste each "# %% CELL" block into its own Colab cell, in order.
#
# What this teaches the model: compress a chunk of spoken transcript into a
# short factual summary (QMSum's specific_query_list = local segment -> local
# summary, structurally the closest public analogue to "slide's audio window
# -> slide summary"). It does NOT yet teach the fused_slides JSON schema
# (title/summary/key_concepts/code_example/voiceover_script) or use any real
# lecture_021 data — that's Stage 2, once a small bootstrapped example set
# exists from the real lecture files.

# %% CELL 1 — install packages
# !pip install -q "transformers>=4.41.0" "datasets>=2.19.0" "accelerate>=0.30.0" "evaluate>=0.4.0" "rouge_score" "sentencepiece"

# %% CELL 2 — GPU check
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "CPU only — go to Runtime > Change runtime type > T4 GPU")

# %% CELL 3 — get QMSum (public repo, no auth needed)
# !git clone https://github.com/Yale-LILY/QMSum.git

# %% CELL 4 — parse train/val jsonl into (input, target) pairs
import json, os

DATA_DIR = "QMSum/data/ALL"
files_in_dir = os.listdir(DATA_DIR)

def find_file(keyword):
    matches = [f for f in files_in_dir if keyword in f.lower() and f.endswith((".jsonl", ".json"))]
    if not matches:
        raise FileNotFoundError(f"No file containing '{keyword}' found in {DATA_DIR}: {files_in_dir}")
    return os.path.join(DATA_DIR, matches[0])

TRAIN_FILE = find_file("train")
VAL_FILE = find_file("val")
print("train file:", TRAIN_FILE)
print("val file:", VAL_FILE)

def load_meetings(path):
    meetings = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meetings.append(json.loads(line))
    return meetings

def spans_to_text(transcript, spans):
    parts = []
    for span in spans:
        start, end = int(span[0]), int(span[1])
        for turn in transcript[start:end + 1]:
            speaker = (turn.get("speaker") or "").strip()
            content = (turn.get("content") or "").strip()
            if content:
                parts.append(f"{speaker}: {content}" if speaker else content)
    return "\n".join(parts)

def build_examples(meetings):
    examples = []
    for meeting in meetings:
        transcript = meeting.get("meeting_transcripts", [])
        for q in meeting.get("specific_query_list", []):
            answer = (q.get("answer") or "").strip()
            spans = q.get("relevant_text_span", [])
            if not answer or not spans:
                continue
            segment_text = spans_to_text(transcript, spans)
            if not segment_text:
                continue
            input_text = (
                "Summarize this lecture/meeting segment in 2-4 sentences, "
                "capturing only what was actually discussed.\n\n"
                f"Transcript segment:\n{segment_text}"
            )
            examples.append({"input": input_text, "target": answer})
    return examples

train_meetings = load_meetings(TRAIN_FILE)
val_meetings = load_meetings(VAL_FILE)

train_examples = build_examples(train_meetings)
val_examples = build_examples(val_meetings)

print(f"train examples: {len(train_examples)}")
print(f"val examples:   {len(val_examples)}")
print("\n--- sample ---")
print(train_examples[0]["input"][:500])
print("---")
print(train_examples[0]["target"])

# %% CELL 5 — build HF datasets
from datasets import Dataset, DatasetDict

raw_datasets = DatasetDict({
    "train": Dataset.from_list(train_examples),
    "validation": Dataset.from_list(val_examples),
})
raw_datasets

# %% CELL 6 — tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 160

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(batch):
    model_inputs = tokenizer(batch["input"], max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=["input", "target"])

# %% CELL 7 — ROUGE metric for eval
import numpy as np
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# %% CELL 8 — train
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/module4_stage1_checkpoints",
    eval_strategy="epoch",          # older transformers: rename to evaluation_strategy
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %% CELL 9 — save, zip, download
SAVE_DIR = "/content/module4_stage1_model"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

import shutil
shutil.make_archive("/content/module4_stage1_model", "zip", SAVE_DIR)

from google.colab import files
files.download("/content/module4_stage1_model.zip")
