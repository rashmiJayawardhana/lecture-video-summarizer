import json
import os
import random

# ── Settings ──────────────────────────────────────────────
JSON_FOLDER = "src/module2_summarization/json_output"
LABELS_FILE = "src/module2_summarization/labelled_sentences.json"
# ──────────────────────────────────────────────────────────

# Load existing labels
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = []

labeled_sentences = {l["sentence"] for l in labels}
print(f"Existing labels: {len(labels)}\n")

# Show video list
files = sorted([f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])
print("=" * 60)
print("AVAILABLE VIDEOS:")
print("=" * 60)
for i, f in enumerate(files):
    short_name = f.split("_-_")[0].replace("LecVideo_", "Video ")
    print(f"{i+1:2d}. {short_name}")
print()

# Ask which video
choice = input("Enter video number to label (1-16): ").strip()
try:
    idx = int(choice) - 1
    if idx < 0 or idx >= len(files):
        print("Invalid choice!")
        exit()
except:
    print("Invalid input!")
    exit()

video_file = files[idx]
print(f"\n📄 Selected: {video_file}\n")

# Load video
with open(os.path.join(JSON_FOLDER, video_file), "r", encoding="utf-8") as f:
    sentences = json.load(f)

# Filter unlabeled
unlabeled = [s for s in sentences if s["sentence"].strip() not in labeled_sentences]
print(f"Total sentences  : {len(sentences)}")
print(f"Already labeled  : {len(sentences) - len(unlabeled)}")
print(f"To label         : {len(unlabeled)}")

# Ask target count
target = input("\nHow many to label this session? (default 50): ").strip()
target = int(target) if target.isdigit() else 50

print("\n" + "=" * 60)
print("LABELING INSTRUCTIONS:")
print("=" * 60)
print("1 = Important (concepts, definitions, explanations)")
print("0 = Not important (filler, transitions, greetings)")
print("s = Skip this sentence")
print("q = Quit and save")
print("=" * 60)
print()

count = 0
important_added = 0
video_name = video_file.split("_-_")[0]

for sent in unlabeled:
    if count >= target:
        break

    text = sent["sentence"].strip()
    if not text:
        continue

    mins = int(sent['timestamp_start'] // 60)
    secs = int(sent['timestamp_start'] % 60)
    
    print(f"[{count+1}/{target}]  [{mins:02d}:{secs:02d}]")
    print(f"💬 {text[:120]}")

    while True:
        ans = input("Label (1/0/s/q): ").strip().lower()
        if ans in ["1", "0", "s", "q"]:
            break
        print("Please type 1, 0, s, or q")

    if ans == "q":
        break
    if ans == "s":
        print("Skipped\n")
        continue

    labels.append({
        "sentence"    : text,
        "is_important": int(ans),
        "source"      : video_name
    })
    labeled_sentences.add(text)

    count += 1
    if ans == "1":
        important_added += 1

    print(f"✓ Labeled as {ans}\n")

    # Auto-save every 25
    if count % 25 == 0:
        with open(LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"💾 Auto-saved! ({count} new labels this session)\n")

# Final save
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

print("\n" + "=" * 60)
print("✅ SESSION COMPLETE!")
print("=" * 60)
print(f"New labels this session: {count}")
print(f"Important (1) added: {important_added}")
print(f"Not important (0) added: {count - important_added}")
print(f"\nTotal labels overall: {len(labels)}")