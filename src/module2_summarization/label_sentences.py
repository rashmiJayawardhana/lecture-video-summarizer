import json
import os

# ── Settings ─────────────────────────────────────────────
JSON_FOLDER   = "src/module2_summarization/json_output"
LABELLED_FILE = "src/module2_summarization/labelled_sentences.json"
# ─────────────────────────────────────────────────────────

# Load existing labels if any
if os.path.exists(LABELLED_FILE):
    with open(LABELLED_FILE, "r", encoding="utf-8") as f:
        labelled = json.load(f)
else:
    labelled = []

already_done = set(
    f"{d['source']}_{d['timestamp_start']}" for d in labelled
)

print("=" * 60)
print("SENTENCE LABELLING TOOL")
print("=" * 60)
print("For each sentence type:")
print("  1  = Important (new concept / definition / key explanation)")
print("  0  = Not important (filler / transition / story)")
print("  s  = Skip this sentence for now")
print("  q  = Quit and save progress")
print("=" * 60)

json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

total_labelled = 0

for json_file in json_files:
    path = os.path.join(JSON_FOLDER, json_file)
    with open(path, "r", encoding="utf-8") as f:
        sentences = json.load(f)

    print(f"\n📄 File: {json_file}")
    print(f"   Total sentences: {len(sentences)}\n")

    for sent in sentences:
        key = f"{json_file}_{sent['timestamp_start']}"

        # Skip already labelled
        if key in already_done:
            continue

        # Skip very short sentences (likely noise)
        if len(sent["sentence"]) < 10:
            continue

        print(f"  ⏱  {sent['timestamp_start']}s - {sent['timestamp_end']}s")
        print(f"  💬 {sent['sentence']}")
        print()

        while True:
            choice = input("  Label (1 / 0 / s / q): ").strip().lower()
            if choice in ["0", "1", "s", "q"]:
                break
            print("  Please type 1, 0, s, or q")

        if choice == "q":
            print("\n💾 Saving progress and quitting...")
            with open(LABELLED_FILE, "w", encoding="utf-8") as f:
                json.dump(labelled, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(labelled)} labels to {LABELLED_FILE}")
            exit()

        if choice == "s":
            continue

        labelled.append({
            "source"          : json_file,
            "sentence"        : sent["sentence"],
            "timestamp_start" : sent["timestamp_start"],
            "timestamp_end"   : sent["timestamp_end"],
            "is_important"    : int(choice)
        })
        already_done.add(key)
        total_labelled += 1

        # Auto save every 50 labels
        if total_labelled % 50 == 0:
            with open(LABELLED_FILE, "w", encoding="utf-8") as f:
                json.dump(labelled, f, indent=2, ensure_ascii=False)
            print(f"\n  💾 Auto saved ({len(labelled)} total labels so far)\n")

# Final save
with open(LABELLED_FILE, "w", encoding="utf-8") as f:
    json.dump(labelled, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"✅ DONE! Total labels saved: {len(labelled)}")
print(f"   Saved to: {LABELLED_FILE}")