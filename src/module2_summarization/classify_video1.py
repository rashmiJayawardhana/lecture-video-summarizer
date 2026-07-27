import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ── Settings ──────────────────────────────────────────────
BERT_MODEL_PATH = "src/module2_summarization/bert_model_v2"
INPUT_FILE      = "src/module2_summarization/json_output/LecVideo_001_-_#01_-_Relational_Model_&_Algebra_(CMU_Intro_to_Database_Systems).json"
OUTPUT_FILE     = "src/module2_summarization/final_output/LecVideo_001_module2_final.json"
# ──────────────────────────────────────────────────────────

os.makedirs("src/module2_summarization/final_output", exist_ok=True)

print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.eval()
print("✅ BERT loaded!")

IT_KEYWORDS = [
    "algorithm", "database", "sql", "table", "schema", "storage",
    "buffer", "cache", "transaction", "join", "index", "query",
    "record", "tuple", "column", "row", "relation", "attribute",
    "primary key", "foreign key", "normalization", "constraint",
    "b-tree", "hash", "page", "block", "disk", "memory", "file"
]

def check_keyword_boost(sentence):
    sentence_lower = sentence.lower()
    found = [kw for kw in IT_KEYWORDS if kw in sentence_lower]
    return len(found) > 0

DEFINITION_PATTERNS = [
    "is defined as", "stands for", "is a type of", "refers to",
    "is called", "is known as", "means that", "is the process of"
]

def check_definition(sentence):
    sentence_lower = sentence.lower()
    return any(p in sentence_lower for p in DEFINITION_PATTERNS)

def build_repetition_map(all_sentences):
    word_count = {}
    for sent in all_sentences:
        for word in sent.lower().split():
            if len(word) > 5:
                word_count[word] = word_count.get(word, 0) + 1
    return word_count

def check_repetition(sentence, word_count):
    words = sentence.lower().split()
    return any(len(w) > 5 and word_count.get(w, 0) > 3 for w in words)

def classify_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()
    return confidence > 0.5, round(confidence, 3)

# Load Video 001
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = json.load(f)

print(f"\nProcessing Video 001 - Relational Model & Algebra")
print(f"Total sentences: {len(sentences)}")
print("This will take ~10-15 minutes...\n")

all_texts = [s["sentence"] for s in sentences]
word_count = build_repetition_map(all_texts)

output = []
segment_size = 10.0
max_time = max(s["timestamp_end"] for s in sentences)
num_segments = int(max_time // segment_size) + 1

for seg_idx in range(num_segments):
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    seg_sentences = [s for s in sentences
                    if s["timestamp_start"] >= seg_start
                    and s["timestamp_end"] <= seg_end]

    if not seg_sentences:
        continue

    seg_results = []
    for sent in seg_sentences:
        text = sent["sentence"]
        kw_boost = check_keyword_boost(text)
        def_match = check_definition(text)
        rep_boost = check_repetition(text, word_count)
        is_important, confidence = classify_sentence(text)

        if kw_boost:
            confidence = min(confidence + 0.10, 1.0)
        if def_match:
            confidence = min(confidence + 0.15, 1.0)
            is_important = True
        if rep_boost:
            confidence = min(confidence + 0.05, 1.0)

        seg_results.append({
            "sentence": text,
            "timestamp_start": sent["timestamp_start"],
            "timestamp_end": sent["timestamp_end"],
            "is_important": is_important,
            "confidence": round(confidence, 3),
            "keyword_boost": kw_boost,
            "definition_match": def_match,
            "repetition_boost": rep_boost
        })

    important_count = sum(1 for s in seg_results if s["is_important"])
    importance_ratio = round(important_count / len(seg_results), 3)

    output.append({
        "segment_id": f"seg_{seg_idx:03d}",
        "timestamp_start": seg_start,
        "timestamp_end": seg_end,
        "importance_ratio_T": importance_ratio,
        "sentences": seg_results
    })

    # Progress indicator every 30 segments
    if seg_idx % 30 == 0:
        print(f"Processed segment {seg_idx}/{num_segments}")

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

important_segs = sum(1 for s in output if s["importance_ratio_T"] > 0.5)
print(f"\n✅ DONE!")
print(f"Total segments: {len(output)}")
print(f"Important segments: {important_segs}")
print(f"Saved to: {OUTPUT_FILE}")