import json
import os

folder = "src/module2_summarization/json_output"
files = os.listdir(folder)

total = 0
print("=== VIDEOS TRANSCRIBED ===")
for f in files:
    if f.endswith(".json"):
        path = os.path.join(folder, f)
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        name = f.replace("LecVideo_", "").split("_-_")[0]
        print(f"Video {name}: {len(data)} sentences")
        total += len(data)

print(f"\nGRAND TOTAL: {total} sentences")