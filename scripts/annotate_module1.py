"""
Module 1 Annotation Helper - Project Integra
Automated Lecture Video Summarization
 
Fast keyboard-driven annotation tool for scoring 10-second lecture
segments on a 0 to 10 visual importance scale.
 
HOW IT WORKS
  The tool walks through every 10-second segment of every video in your
  VIDEO_DIR. For each segment it shows the middle frame (large) plus a small
  filmstrip of the start, middle and end frames for context. You press one
  key to score the segment and it auto-advances and auto-saves.
 
KEYS
  0 1 2 3 4 5 6 7 8 9   score the segment 0 to 9
  t                     score the segment 10  (t = ten)
  s                     skip this segment (unsure / not relevant)
  b                     go back to the previous segment to fix it
  q                     save and quit (you can resume later)
 
SETUP (run once)
  pip install opencv-python numpy
 
USAGE
  1. Put your assigned lecture videos in the folder named in VIDEO_DIR below.
  2. Change ANNOTATOR to your own name and OUTPUT_FILE to your own file.
  3. Run:  python annotate_module1.py
  4. To combine everyone's files at the end:  python annotate_module1.py --merge
 
The output is a JSON file of ground-truth labels used to train the
ResNet-50 + BiLSTM model in Module 1.
"""

import cv2
import os
import sys
import json
import glob
import datetime
import numpy as np
 
# ----------------------------------------------------------------------
# CONFIGURATION  (each group member edits these three lines)
# ----------------------------------------------------------------------
VIDEO_DIR = "videos"                      # folder containing your lecture videos
ANNOTATOR = "rashmi"                      # your name (goes into every record)
OUTPUT_FILE = "annotations_rashmi.json"   # your personal output file
 
# Fixed settings (do not change - keeps the whole team consistent)
SEGMENT_LENGTH = 10                       # seconds per segment
MAX_SCORE = 10                            # scale is 0 to 10
DISPLAY_WIDTH = 960                       # main frame display width in pixels
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
# ----------------------------------------------------------------------
 
 
def load_annotations(path):
    """Load existing annotations so we can resume without redoing work."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Warning: could not read", path, "- starting fresh.")
            return {}
    return {}
 
 
def save_annotations(path, data):
    """Write annotations to disk after every keystroke (crash-safe)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # atomic replace so the file is never half-written
 
 
def get_video_info(path):
    """Return (fps, duration_seconds, frame_count)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return fps, duration, frame_count
 
 
def grab_frame(cap, time_sec):
    """Grab a single frame at the given time in seconds."""
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None
 
 
def resize_keep_aspect(img, width):
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)))
 
 
def build_segments(video_path):
    """Return a list of segment dicts for one video (no frames yet)."""
    info = get_video_info(video_path)
    if info is None:
        return []
    fps, duration, _ = info
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    n = int(duration // SEGMENT_LENGTH)  # whole 10s segments only
    segments = []
    for i in range(n):
        start = i * SEGMENT_LENGTH
        end = start + SEGMENT_LENGTH
        segments.append({
            "video_path": video_path,
            "video_id": video_id,
            "segment_index": i,
            "segment_id": "%s__seg_%04d" % (video_id, i),
            "timestamp_start": float(start),
            "timestamp_end": float(end),
            "middle_frame_time": float((start + end) / 2.0),
        })
    return segments
 
 
def make_canvas(main_frame, thumbs, info_lines):
    """Compose the main frame, a 3-thumbnail filmstrip and a text panel."""
    main = resize_keep_aspect(main_frame, DISPLAY_WIDTH)
 
    # filmstrip: three small frames side by side
    strip_w = DISPLAY_WIDTH // 3
    strip_imgs = []
    for t in thumbs:
        if t is None:
            t = np.zeros((10, 10, 3), dtype=np.uint8)
        strip_imgs.append(resize_keep_aspect(t, strip_w - 4))
    strip_h = max(s.shape[0] for s in strip_imgs)
    strip = np.zeros((strip_h, DISPLAY_WIDTH, 3), dtype=np.uint8)
    x = 0
    for s in strip_imgs:
        strip[0:s.shape[0], x:x + s.shape[1]] = s
        x += strip_w
 
    # text panel
    panel_h = 26 * len(info_lines) + 20
    panel = np.zeros((panel_h, DISPLAY_WIDTH, 3), dtype=np.uint8)
    y = 26
    for line in info_lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
 
    canvas = np.vstack([main, strip, panel])
    return canvas
 
 
def normalize(raw):
    return round(raw / float(MAX_SCORE), 4)
 
 
def merge_all():
    """Combine all annotations_*.json files into module1_annotations.json."""
    combined = {}
    files = glob.glob("annotations_*.json")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                part = json.load(f)
            combined.update(part)
            print("Merged %d records from %s" % (len(part), fp))
        except (json.JSONDecodeError, IOError):
            print("Skipped unreadable file", fp)
    with open("module1_annotations.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print("Total %d records written to module1_annotations.json" % len(combined))
 
 
def run():
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, "*" + ext)))
    video_paths.sort()
    if not video_paths:
        print("No videos found in folder:", VIDEO_DIR)
        return
 
    annotations = load_annotations(OUTPUT_FILE)
 
    # build the flat list of all segments across all videos
    all_segments = []
    for vp in video_paths:
        all_segments.extend(build_segments(vp))
    total = len(all_segments)
    done = sum(1 for s in all_segments if s["segment_id"] in annotations)
    print("Found %d videos, %d segments total, %d already done." %
          (len(video_paths), total, done))
 
    idx = 0
    current_cap = None
    current_video = None
 
    while 0 <= idx < total:
        seg = all_segments[idx]
 
        # skip already-annotated segments when moving forward
        if seg["segment_id"] in annotations and seg.get("_visited") is not True:
            idx += 1
            continue
 
        # open the right video file
        if current_video != seg["video_path"]:
            if current_cap is not None:
                current_cap.release()
            current_cap = cv2.VideoCapture(seg["video_path"])
            current_video = seg["video_path"]
 
        mid = grab_frame(current_cap, seg["middle_frame_time"])
        if mid is None:
            idx += 1
            continue
        start_f = grab_frame(current_cap, seg["timestamp_start"])
        end_f = grab_frame(current_cap, max(seg["timestamp_end"] - 0.2,
                                            seg["timestamp_start"]))
 
        done = sum(1 for s in all_segments if s["segment_id"] in annotations)
        prev = annotations.get(seg["segment_id"], {}).get("raw_score", "-")
        info_lines = [
            "Video: %s   Segment %d   (%.0f-%.0f s)" % (
                seg["video_id"], seg["segment_index"],
                seg["timestamp_start"], seg["timestamp_end"]),
            "Progress: %d / %d done (%.1f%%)   Current score: %s" % (
                done, total, 100.0 * done / total, str(prev)),
            "0-9 = score  |  t = 10  |  s = skip  |  b = back  |  q = save+quit",
            "8-10 key concept/formula/diagram   4-7 useful   0-3 filler/blank",
        ]
 
        canvas = make_canvas(mid, [start_f, mid, end_f], info_lines)
        cv2.imshow("Module 1 Annotation - Integra", canvas)
        key = cv2.waitKey(0) & 0xFF
 
        if key == ord('q'):
            break
        elif key == ord('b'):
            # step back to the previous real segment
            j = idx - 1
            while j >= 0:
                all_segments[j]["_visited"] = True
                idx = j
                break
            if j < 0:
                idx = 0
            continue
        elif key == ord('s'):
            annotations[seg["segment_id"]] = {
                "video_id": seg["video_id"],
                "segment_index": seg["segment_index"],
                "segment_id": seg["segment_id"],
                "timestamp_start": seg["timestamp_start"],
                "timestamp_end": seg["timestamp_end"],
                "middle_frame_time": seg["middle_frame_time"],
                "raw_score": None,
                "normalized_score": None,
                "skipped": True,
                "annotator": ANNOTATOR,
                "annotated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            save_annotations(OUTPUT_FILE, annotations)
            seg.pop("_visited", None)
            idx += 1
            continue
 
        score = None
        if ord('0') <= key <= ord('9'):
            score = key - ord('0')
        elif key == ord('t'):
            score = 10
 
        if score is not None:
            annotations[seg["segment_id"]] = {
                "video_id": seg["video_id"],
                "segment_index": seg["segment_index"],
                "segment_id": seg["segment_id"],
                "timestamp_start": seg["timestamp_start"],
                "timestamp_end": seg["timestamp_end"],
                "middle_frame_time": seg["middle_frame_time"],
                "raw_score": score,
                "normalized_score": normalize(score),
                "skipped": False,
                "annotator": ANNOTATOR,
                "annotated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            save_annotations(OUTPUT_FILE, annotations)
            seg.pop("_visited", None)
            idx += 1
        # any other key: do nothing, redisplay same segment
 
    if current_cap is not None:
        current_cap.release()
    cv2.destroyAllWindows()
    save_annotations(OUTPUT_FILE, annotations)
    done = sum(1 for s in all_segments if s["segment_id"] in annotations)
    print("Saved %d annotations to %s" % (done, OUTPUT_FILE))
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_all()
    else:
        run()