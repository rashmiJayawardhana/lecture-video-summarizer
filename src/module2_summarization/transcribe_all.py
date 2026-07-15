import whisper
import json
import os

# ── Settings ─────────────────────────────────────────────────
VIDEOS_FOLDER      = "videos"
TRANSCRIPTS_FOLDER = "src/module2_summarization/transcripts"
JSON_FOLDER        = "src/module2_summarization/json_output"
WHISPER_MODEL      = "base"  # use "large-v3" later for final run
# ─────────────────────────────────────────────────────────────

print("Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)
print("Model loaded!\n")

# Get all mp4 files from videos folder
video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(".mp4")]

# Remove duplicates by name similarity
seen = []
unique_videos = []
for v in video_files:
    key = v[:12]
    if key not in seen:
        seen.append(key)
        unique_videos.append(v)

print(f"Found {len(unique_videos)} unique videos to transcribe\n")

# Track results
skipped  = []
failed   = []
success  = []

# Process each video
for i, video_file in enumerate(unique_videos):
    video_path = os.path.join(VIDEOS_FOLDER, video_file)

    # Create output filenames
    base_name       = video_file.replace(".mp4", "").replace(" ", "_")
    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{base_name}.txt")
    json_path       = os.path.join(JSON_FOLDER, f"{base_name}.json")

    # Skip if already done
    if os.path.exists(json_path):
        print(f"[{i+1}/{len(unique_videos)}] SKIPPING (already done): {video_file}")
        skipped.append(video_file)
        continue

    print(f"[{i+1}/{len(unique_videos)}] Transcribing: {video_file}")
    print("Please wait...\n")

    # ── NEW: try/except so one failure does not stop the whole batch ──
    try:
        result = model.transcribe(video_path)

        # Build JSON output
        output = []
        for segment in result["segments"]:
            output.append({
                "sentence"           : segment["text"].strip(),
                "timestamp_start"    : round(segment["start"], 2),
                "timestamp_end"      : round(segment["end"], 2),
                "is_important"       : False,
                "importance_ratio_T" : 0.0
            })

        # Save transcript text file
        with open(transcript_path, "w", encoding="utf-8") as f:
            for item in output:
                f.write(f"{item['timestamp_start']}s - {item['timestamp_end']}s : {item['sentence']}\n")

        # Save JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ Done! Sentences found: {len(output)}")
        print(f"   Saved: {json_path}\n")
        success.append(video_file)

    except Exception as e:
        print(f"❌ FAILED: {video_file}")
        print(f"   Reason: {e}")
        print(f"   Skipping and continuing...\n")
        failed.append(video_file)

# ── Summary at the end ────────────────────────────────────────
print("=" * 50)
print("BATCH COMPLETE — SUMMARY")
print("=" * 50)
print(f"✅ Successfully transcribed : {len(success)}")
print(f"⏭️  Already done (skipped)  : {len(skipped)}")
print(f"❌ Failed (no audio/error)  : {len(failed)}")
if failed:
    print("\nFailed videos (tell your team these need replacing):")
    for f in failed:
        print(f"   - {f}")
print(f"\nTranscripts saved in : {TRANSCRIPTS_FOLDER}")
print(f"JSON files saved in  : {JSON_FOLDER}")