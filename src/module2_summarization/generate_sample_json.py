import json
import os

# Load transcription JSON
JSON_FOLDER = "src/module2_summarization/json_output"
json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

# Take first file
first_file = os.path.join(JSON_FOLDER, json_files[0])
with open(first_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Create sample output — first 20 sentences
sample_output = []
for i, item in enumerate(data[:20]):
    # Add novel features manually for sample
    is_important = i % 2 == 0  # alternating for demo

    # IT keyword check
    it_keywords = [
        "algorithm", "network", "protocol", "function",
        "model", "learning", "neural", "data", "system",
        "policy", "reward", "classification", "training"
    ]
    keyword_boost = any(
        kw in item["sentence"].lower() for kw in it_keywords
    )

    # Definition pattern check
    definition_patterns = [
        "is defined as", "stands for", "is a type of",
        "refers to", "means", "is called", "is known as"
    ]
    definition_match = any(
        p in item["sentence"].lower() for p in definition_patterns
    )

    # Confidence score
    confidence = 0.90 if keyword_boost else 0.65
    if definition_match:
        confidence = 0.95
        is_important = True

    sample_output.append({
        "sentence"           : item["sentence"],
        "timestamp_start"    : item["timestamp_start"],
        "timestamp_end"      : item["timestamp_end"],
        "is_important"       : is_important,
        "importance_ratio_T" : round(confidence, 2),
        "keyword_boost"      : keyword_boost,
        "definition_match"   : definition_match,
        "confidence"         : confidence
    })

# Save sample JSON
output_path = "src/module2_summarization/module2_sample_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sample_output, f, indent=2, ensure_ascii=False)

print(f"✅ Sample JSON created!")
print(f"   File: {output_path}")
print(f"   Records: {len(sample_output)}")
print(f"\n--- PREVIEW (first 3 records) ---")
for item in sample_output[:3]:
    print(json.dumps(item, indent=2))