import json

path = "src/module2_summarization/final_output/LecVideo_001_module2_final.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"=== VIDEO 001 - Relational Model & Algebra ===")
print(f"Total segments: {len(data)}")
print()

important = [s for s in data if s["importance_ratio_T"] >= 0.8]
print(f"High-importance segments: {len(important)}")
print()

print("=== TOP 10 MOST IMPORTANT SEGMENTS ===\n")

for seg in important[:10]:
    mins = int(seg["timestamp_start"] // 60)
    secs = int(seg["timestamp_start"] % 60)

    print(f"[{mins:02d}:{secs:02d}] {seg['segment_id']} — Importance: {seg['importance_ratio_T']}")

    for s in seg["sentences"]:
        if s["is_important"]:
            tags = []
            if s["keyword_boost"]: tags.append("KEYWORD")
            if s["definition_match"]: tags.append("DEFINITION")
            if s["repetition_boost"]: tags.append("REPETITION")
            tag_str = " ".join(f"[{t}]" for t in tags)

            print(f"   ✓ {tag_str} {s['sentence'][:80]}")
            print(f"     Confidence: {s['confidence']}")
    print()