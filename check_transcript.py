import json

# Check Video 002 - Modern SQL
path = "src/module2_summarization/json_output/LecVideo_002_-_#02_-_Modern_SQL_✸_dbt_Database_Talk_(CMU_Intro_to_Database_Systems).json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total sentences: {len(data)}")
print()
print("=== FIRST 5 SENTENCES ===")
for i, item in enumerate(data[:5]):
    print(f"[{item['timestamp_start']}s] {item['sentence']}")

print()
print("=== SENTENCES 100-105 ===")
for item in data[100:105]:
    print(f"[{item['timestamp_start']}s] {item['sentence']}")