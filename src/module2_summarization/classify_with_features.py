import argparse
import json
import os
from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ── Settings ──────────────────────────────────────────────
DEFAULT_MODEL_PATH = "src/module2_summarization/bert_model_v2"
JSON_FOLDER         = "src/module2_summarization/json_output"
OUTPUT_FOLDER        = "src/module2_summarization/final_output"
# ──────────────────────────────────────────────────────────

tokenizer = None
model = None

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


def load_bert(model_path: str) -> None:
    """Load the BERT tokenizer/model into the module-level tokenizer/model globals."""
    global tokenizer, model

    weights_path = Path(model_path) / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Module 2 BERT checkpoint not found at {weights_path} — "
            f"train the classifier and place model.safetensors there before running."
        )

    print(f"Loading BERT model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("✅ BERT loaded!")


def classify_sentences(sentences: list) -> list:
    """Classify an in-memory list of {sentence, timestamp_start, timestamp_end} records
    and bucket them into 10s segments. Pure function, no file I/O -- reused by process_file()
    and by the API's in-process /classify endpoint."""
    print(f"Total sentences: {len(sentences)}")

    # Build repetition map for this video
    all_texts  = [s["sentence"] for s in sentences]
    word_count = build_repetition_map(all_texts)

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

    important_segs = sum(1 for s in output if s["importance_ratio_T"] > 0.5)
    print(f"✅ Done! Segments: {len(output)} | Important: {important_segs}")

    return output


def process_file(input_path: str, output_path: str) -> list:
    """Classify every sentence in input_path and write the segmented result to output_path."""
    with open(input_path, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"Processing: {input_path}")
    output = classify_sentences(sentences)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   Saved: {out_path}\n")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify transcript sentences by importance using BERT + rule-based features."
    )
    parser.add_argument("--input_json", help="Single transcript JSON to classify (single-file mode).")
    parser.add_argument("--output_json", help="Output path for the classified JSON (single-file mode).")
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the BERT checkpoint directory (default: {DEFAULT_MODEL_PATH}).",
    )
    args = parser.parse_args()

    load_bert(args.model_path)

    if args.input_json and args.output_json:
        # Single-file mode (used by the backend)
        process_file(args.input_json, args.output_json)
        return

    # Batch mode (default): process every transcript in JSON_FOLDER -> OUTPUT_FOLDER
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]
    print(f"\nFound {len(json_files)} transcript files to process\n")

    for json_file in json_files:
        path = os.path.join(JSON_FOLDER, json_file)
        out_name = json_file.replace(".json", "_module2_final.json")
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        process_file(path, out_path)

    print("=" * 50)
    print("ALL FILES PROCESSED!")
    print(f"Final JSON files in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
