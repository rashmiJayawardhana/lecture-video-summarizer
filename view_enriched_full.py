import json

with open("src/module2_summarization/final_output/LecVideo_001_enriched.json", "r") as f:
    data = json.load(f)

print(f"Total blocks: {len(data)}\n")

# Show first block RAW
print("=== FIRST BLOCK RAW ===")
print(json.dumps(data[0], indent=2))