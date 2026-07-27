import json

with open("src/module2_summarization/final_output/LecVideo_001_enriched.json", "r") as f:
    data = json.load(f)

print(f"Total blocks: {len(data)}")
print()
print("=== First block ===")
print(json.dumps(data[0], indent=2))