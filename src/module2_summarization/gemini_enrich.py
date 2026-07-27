import json
import os
import time
from google import genai

# ── Settings ──────────────────────────────────────────────
INPUT_FILE  = "src/module2_summarization/final_output/LecVideo_001_module2_final.json"
OUTPUT_FILE = "src/module2_summarization/final_output/LecVideo_001_enriched.json"
KEY_FILE    = "C:/Users/asus/Desktop/Gemini_key.txt"
MIN_RATIO   = 0.7
MODEL_NAME  = "gemini-2.5-flash"
# ──────────────────────────────────────────────────────────

# Load API key
with open(KEY_FILE, "r") as f:
    api_key = f.read().strip()

client = genai.Client(api_key=api_key)
print("✅ Gemini configured!")

# Load Video 001 output
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    segments = json.load(f)

print(f"Total segments: {len(segments)}")

important_segments = [s for s in segments if s["importance_ratio_T"] >= MIN_RATIO]
print(f"High-importance segments: {len(important_segments)}")
print()

# Group segments into topic blocks
def group_segments(segs, gap_threshold=30.0):
    if not segs:
        return []
    blocks = [[segs[0]]]
    for seg in segs[1:]:
        last_end = blocks[-1][-1]["timestamp_end"]
        if seg["timestamp_start"] - last_end <= gap_threshold:
            blocks[-1].append(seg)
        else:
            blocks.append([seg])
    return blocks

blocks = group_segments(important_segments)
print(f"Grouped into {len(blocks)} topic blocks")
print()

def analyze_block(block):
    all_sentences = []
    for seg in block:
        for s in seg["sentences"]:
            if s["is_important"]:
                all_sentences.append(s["sentence"])
    
    text = " ".join(all_sentences)
    
    prompt = f"""Analyze this lecture excerpt from an IT/database course. Return ONLY a valid JSON object with these fields:
- topic: short topic name (max 8 words)
- summary: one clean sentence describing the concept
- key_points: array of 2-4 short bullet points

Lecture text:
{text}

Return ONLY the JSON object, no markdown, no code blocks, no explanation."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        result_text = response.text.strip()
        
        # Clean markdown if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        return json.loads(result_text.strip())
    except Exception as e:
        return {
            "topic": "Unknown",
            "summary": "Error processing block",
            "key_points": [],
            "error": str(e)
        }

enriched = []
for i, block in enumerate(blocks):
    block_start = block[0]["timestamp_start"]
    block_end   = block[-1]["timestamp_end"]
    
    mins = int(block_start // 60)
    secs = int(block_start % 60)
    
    print(f"[{i+1}/{len(blocks)}] [{mins:02d}:{secs:02d}] Analyzing {len(block)} segments...")
    
    analysis = analyze_block(block)
    
    enriched.append({
        "block_id"        : f"block_{i+1:03d}",
        "timestamp_start" : block_start,
        "timestamp_end"   : block_end,
        "segments_count"  : len(block),
        "segment_ids"     : [s["segment_id"] for s in block],
        "semantic_analysis": analysis
    })
    
    print(f"   Topic: {analysis.get('topic', 'N/A')}")
    print()
    
    time.sleep(4.5)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(enriched, f, indent=2, ensure_ascii=False)

print("=" * 50)
print("✅ GEMINI ENRICHMENT COMPLETE!")
print(f"Total blocks: {len(enriched)}")
print(f"Saved to: {OUTPUT_FILE}")