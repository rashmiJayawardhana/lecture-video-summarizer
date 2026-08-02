import json
import time
from google import genai

INPUT_FILE  = "src/module2_summarization/final_output/LecVideo_058_module2_final.json"
OUTPUT_FILE = "src/module2_summarization/final_output/LecVideo_058_enriched.json"
KEY_FILE    = "C:/Users/asus/Desktop/Gemini_key.txt"
MIN_RATIO   = 0.7
MODEL_NAME  = "gemini-flash-latest"

with open(KEY_FILE, "r") as f:
    api_key = f.read().strip()

client = genai.Client(api_key=api_key)
print("Gemini configured!")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    segments = json.load(f)

important = [s for s in segments if s["importance_ratio_T"] >= MIN_RATIO]
print(f"High-importance segments: {len(important)}")

def group_segments(segs, gap=30.0):
    if not segs:
        return []
    blocks = [[segs[0]]]
    for seg in segs[1:]:
        if seg["timestamp_start"] - blocks[-1][-1]["timestamp_end"] <= gap:
            blocks[-1].append(seg)
        else:
            blocks.append([seg])
    return blocks

blocks = group_segments(important)
print(f"Topic blocks: {len(blocks)}\n")

enriched = []
for i, block in enumerate(blocks):
    sentences = []
    for seg in block:
        for s in seg["sentences"]:
            if s["is_important"]:
                sentences.append(s["sentence"])
    
    text = " ".join(sentences)
    start = block[0]["timestamp_start"]
    end = block[-1]["timestamp_end"]
    mins = int(start // 60)
    secs = int(start % 60)
    
    print(f"[{i+1}/{len(blocks)}] [{mins:02d}:{secs:02d}] Processing...")
    
    prompt = f"""Analyze this Human-Computer Interaction lecture excerpt. Return ONLY valid JSON with:
- topic (string, max 8 words)
- summary (one clean sentence)
- key_points (array of 2-4 strings)

Text: {text}

Return only JSON, no markdown."""
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            result = response.text.strip()
            
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            
            analysis = json.loads(result.strip())
            print(f"   Topic: {analysis.get('topic', 'N/A')}")
            break
            
        except Exception as e:
            error_msg = str(e)[:80]
            if "429" in error_msg:
                print(f"   Rate limit, waiting 60 sec...")
                time.sleep(60)
                continue
            print(f"   ERROR: {error_msg}")
            analysis = {"topic": "Unknown", "summary": "Error", "key_points": []}
            break
    else:
        analysis = {"topic": "Failed", "summary": "Max retries", "key_points": []}
    
    enriched.append({
        "block_id": f"block_{i+1:03d}",
        "timestamp_start": start,
        "timestamp_end": end,
        "segments_count": len(block),
        "segment_ids": [s["segment_id"] for s in block],
        "semantic_analysis": analysis
    })
    
    time.sleep(6)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print(f"\nDONE! Saved {len(enriched)} blocks")
print(f"File: {OUTPUT_FILE}")
