import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ── Settings ──────────────────────────────────────────────
BERT_MODEL_PATH  = "src/module2_summarization/bert_model"
JSON_FOLDER      = "src/module2_summarization/json_output"
OUTPUT_FOLDER    = "src/module2_summarization/final_output"
# ──────────────────────────────────────────────────────────

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Load BERT ──────────────────────────────────────────────
print("Loading BERT model...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.eval()
print("✅ BERT loaded!")

# ── Novel Feature 1 — IT Keyword Boosting ─────────────────
IT_KEYWORDS = [
    "algorithm", "network", "protocol", "function",
    "model", "neural", "data", "system", "policy",
    "reward", "classification", "training", "deep learning",
    "machine learning", "database", "api", "http", "tcp",
    "encryption", "recursion", "complexity", "gradient",
    "backpropagation", "transformer", "attention", "layer",
    "optimization", "loss", "accuracy", "precision", "recall",
    "architecture", "deployment", "server", "client", "query",
    "index", "cluster", "pipeline", "feature", "vector",
    "matrix", "probability", "distribution", "inference"
]

def check_keyword_boost(sentence):
    sentence_lower = sentence.lower()
    found = [kw for kw in IT_KEYWORDS if kw in sentence_lower]
    return len(found) > 0, found

# ── Novel Feature 2 — Definition Pattern Detection ────────
DEFINITION_PATTERNS = [
    "is defined as", "stands for", "is a type of",
    "refers to", "is called", "is known as",
    "can be defined", "is essentially", "means that",
    "is described as", "is characterized by",
    "is the process of", "is a method", "is an approach",
    "is a technique", "is a framework"
]

def check_definition(sentence):
    sentence_lower = sentence.lower()
    for pattern in DEFINITION_PATTERNS:
        if pattern in sentence_lower:
            return True, pattern
    return False, None

# ── Novel Feature 3 — Repetition Scoring ─────────────────
def build_repetition_map(all_sentences):
    word_count = {}
    for sent in all_sentences:
        words = sent.lower().split()
        for word in words:
            if len(word) > 5:  # only meaningful words
                word_count[word] = word_count.get(word, 0) + 1
    return word_count

def check_repetition(sentence, word_count):
    words = sentence.lower().split()
    repeated = [w for w in words if len(w) > 5 and word_count.get(w, 0) > 3]
    return len(repeated) > 0

# ── Novel Feature 4 — BERT Classification with Confidence ─
def classify_sentence(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=1)
    confidence = probs[0][1].item()  # probability of "important"
    is_important = confidence > 0.5

    return is_important, round(confidence, 3)

# ── Process All JSON Files ─────────────────────────────────
json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]
print(f"\nFound {len(json_files)} transcript files to process\n")

for json_file in json_files:
    path = os.path.join(JSON_FOLDER, json_file)
    with open(path, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"Processing: {json_file}")
    print(f"Total sentences: {len(sentences)}")

    # Build repetition map for this video
    all_texts    = [s["sentence"] for s in sentences]
    word_count   = build_repetition_map(all_texts)

    # Process each segment (10-second buckets)
    output       = []
    segment_size = 10.0
    max_time     = max(s["timestamp_end"] for s in sentences)
    num_segments = int(max_time // segment_size) + 1

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end   = seg_start + segment_size

        # Get sentences in this segment
        seg_sentences = [
            s for s in sentences
            if s["timestamp_start"] >= seg_start
            and s["timestamp_end"] <= seg_end
        ]

        if not seg_sentences:
            continue

        # Process each sentence with all 4 features
        seg_results = []
        for sent in seg_sentences:
            text = sent["sentence"]

            # Feature 1 — IT Keywords
            kw_boost, keywords_found = check_keyword_boost(text)

            # Feature 2 — Definition
            def_match, def_pattern = check_definition(text)

            # Feature 3 — Repetition
            rep_boost = check_repetition(text, word_count)

            # Feature 4 — BERT + Confidence
            is_important, confidence = classify_sentence(text)

            # Boost confidence based on novel features
            if kw_boost:
                confidence = min(confidence + 0.10, 1.0)
            if def_match:
                confidence = min(confidence + 0.15, 1.0)
                is_important = True
            if rep_boost:
                confidence = min(confidence + 0.05, 1.0)

            seg_results.append({
                "sentence"        : text,
                "timestamp_start" : sent["timestamp_start"],
                "timestamp_end"   : sent["timestamp_end"],
                "is_important"    : is_important,
                "confidence"      : round(confidence, 3),
                "keyword_boost"   : kw_boost,
                "definition_match": def_match,
                "repetition_boost": rep_boost
            })

        # Calculate importance_ratio_T for segment
        important_count  = sum(1 for s in seg_results if s["is_important"])
        importance_ratio = round(important_count / len(seg_results), 3)

        # Add segment summary
        output.append({
            "segment_id"       : f"seg_{seg_idx:03d}",
            "timestamp_start"  : seg_start,
            "timestamp_end"    : seg_end,
            "importance_ratio_T": importance_ratio,
            "sentences"        : seg_results
        })

    # Save final output
    out_name = json_file.replace(".json", "_module2_final.json")
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    important_segs = sum(1 for s in output if s["importance_ratio_T"] > 0.5)
    print(f"✅ Done! Segments: {len(output)} | Important: {important_segs}")
    print(f"   Saved: {out_path}\n")

print("=" * 50)
print("ALL FILES PROCESSED!")
print(f"Final JSON files in: {OUTPUT_FOLDER}")