import json
import glob
import os
import subprocess
import sys

INPUT_PATTERN = "outputs/module3/lecture_*_visual_anchors.json"

files = sorted(glob.glob(INPUT_PATTERN))

print("Visual anchor files found:", len(files))

total_anchors = 0

for input_file in files:
    lecture = os.path.basename(input_file).replace("_visual_anchors.json", "")
    output_file = f"outputs/module3/{lecture}_visual_anchors_with_ocr.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchor_count = len(data)
    total_anchors += anchor_count

    if anchor_count == 0:
        print(f"Skipping OCR for {lecture}: 0 anchors")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        continue

    print(f"\nRunning OCR for {lecture}: {anchor_count} anchors")

    subprocess.run([
        sys.executable,
        "src/module3_visual/ocr.py",
        "--input_json", input_file,
        "--output_json", output_file,
        "--threshold", "0.70"
    ], check=True)

print("\nOCR process completed.")
print("Total visual anchors:", total_anchors)
