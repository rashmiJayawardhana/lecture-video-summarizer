import json
import os
import glob
from collections import Counter

INPUT_PATTERN = "outputs/module3/lecture_*_visual_anchors_with_ocr.json"
OUTPUT_FILE = "outputs/module3/module3_output.json"

combined = []

files = sorted(glob.glob(INPUT_PATTERN))

if not files:
    print("No OCR output files found.")
    print("Run OCR first.")
    raise SystemExit(1)

print("OCR files found:", len(files))

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    combined.extend(data)
    print(f"Added {len(data)} records from {path}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=4, ensure_ascii=False)

print("\nFinal Module 3 output saved:", OUTPUT_FILE)
print("Total records:", len(combined))
print("Label distribution:", Counter(item.get("label") for item in combined))
print("Lecture distribution:", Counter(item.get("lecture_id") for item in combined))
print("Records with OCR text:", sum(1 for item in combined if item.get("ocr_text", "").strip()))
