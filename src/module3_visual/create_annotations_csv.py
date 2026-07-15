import os
import csv
import re

ANNOTATION_DIR = "data/annotations/module3"
OUTPUT_CSV = "data/annotations/module3/labels.csv"

LABEL_FOLDERS = {
    "critical": "Critical",
    "important": "Important",
    "skip": "Skip"
}


def extract_lecture_id(filename):
    match = re.search(r"(lecture_\d+)", filename)
    return match.group(1) if match else "unknown"


def extract_time_from_filename(filename):
    """
    Example:
    lecture_001_frame_00012_01m00s.jpg
    Output:
    60.0
    """
    match = re.search(r"_(\d+)m(\d+)s", filename)

    if not match:
        return ""

    minutes = int(match.group(1))
    seconds = int(match.group(2))

    return float(minutes * 60 + seconds)


rows = []

for folder_name, label in LABEL_FOLDERS.items():
    folder_path = os.path.join(ANNOTATION_DIR, folder_name)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)

            rows.append({
                "image_path": image_path,
                "lecture_id": extract_lecture_id(filename),
                "frame_time": extract_time_from_filename(filename),
                "label": label
            })

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["image_path", "lecture_id", "frame_time", "label"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print(f"CSV created: {OUTPUT_CSV}")
print(f"Total labelled images: {len(rows)}")

label_counts = {}

for row in rows:
    label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

print("Label counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")