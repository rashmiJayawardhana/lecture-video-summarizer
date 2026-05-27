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
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

# Base window sizing (will be adjusted dynamically depending on screen size)
DISPLAY_WIDTH = 960                       # default canvas width in pixels
MAX_CANVAS_HEIGHT = 680                   # default canvas height

# Try to automatically detect low-res / scaled screens (e.g. 1536x864 with 150% scaling)
try:
    import tkinter as tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    if screen_h <= 900:
        # Screen height is 900 or smaller.
        # We auto-switch to a gorgeous compact mode to guarantee no cut-offs!
        DISPLAY_WIDTH = 800
        MAX_CANVAS_HEIGHT = 480
        print(f"[UI] Small/scaled screen detected ({screen_w}x{screen_h}). Automatically using Compact UI: {DISPLAY_WIDTH}x{MAX_CANVAS_HEIGHT}")
    else:
        print(f"[UI] Large screen detected ({screen_w}x{screen_h}). Using Standard UI: {DISPLAY_WIDTH}x{MAX_CANVAS_HEIGHT}")
except Exception:
    pass
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
 
 
def resize_keep_aspect(img, width, max_height=None):
    """Resize image to a target width, preserving aspect ratio.
    If max_height is set, further shrink so the height does not exceed it."""
    h, w = img.shape[:2]
    scale = width / float(w)
    new_w, new_h = width, int(h * scale)
    if max_height and new_h > max_height:
        scale2 = max_height / float(new_h)
        new_w = int(new_w * scale2)
        new_h = max_height
    return cv2.resize(img, (new_w, new_h))
 
 
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
 
 
def make_canvas(main_frame, thumbs, seg, done, total, prev_score, last_action=""):
    """
    Build a highly polished, modern, color-coded annotation canvas that fits on screen.
    """
    W = DISPLAY_WIDTH
    H_MAX = MAX_CANVAS_HEIGHT

    # ── Colors (BGR) ──────────────────────────────────────────────────
    BG            = (28, 24, 24)       # Ultra-deep modern slate background
    ACCENT        = (245, 160, 20)     # Vibrant electric gold/amber
    WHITE         = (245, 245, 245)    # Off-white for high readability
    MUTED         = (160, 155, 155)    # Sleek dark gray
    BORDER_COLOR  = (55, 50, 50)       # Glassmorphic gray line divider
    STRIP_BG      = (20, 18, 18)       # Deeper contrast black for filmstrip
    BAR_BG        = (42, 38, 38)       # Progress bar background tracker
    
    # Glow/outline indicator accents
    GREEN_LIGHT   = (100, 220, 80)
    GREEN_DARK    = (20, 60, 20)
    YELLOW_LIGHT  = (240, 180, 60)
    YELLOW_DARK   = (20, 50, 60)
    RED_LIGHT     = (240, 90, 90)
    RED_DARK      = (20, 20, 50)
    GRAY_LIGHT    = (150, 150, 150)
    GRAY_DARK     = (40, 40, 40)

    FONT          = cv2.FONT_HERSHEY_SIMPLEX
    FONT_S        = cv2.FONT_HERSHEY_PLAIN

    # ── Dynamic Sizing ─────────────────────────────────────────────────
    if H_MAX < 550:
        # Compact heights to fit small or scaled screens perfectly
        HDR_H     = 28
        THUMB_H   = 55
        LABEL_H   = 14
        STRIP_H   = THUMB_H + LABEL_H + 4
        BAR_H     = 18
        SIG_H     = 18
        GUIDE_H   = 34
        KB_H      = 20
        font_scale_f = 0.36
        font_scale_s = 0.8
    else:
        # Standard heights for larger screens
        HDR_H     = 36
        THUMB_H   = 80
        LABEL_H   = 18
        STRIP_H   = THUMB_H + LABEL_H + 6
        BAR_H     = 24
        SIG_H     = 24
        GUIDE_H   = 48
        KB_H      = 28
        font_scale_f = 0.42
        font_scale_s = 0.95

    PANELS_H  = HDR_H + STRIP_H + BAR_H + SIG_H + GUIDE_H + KB_H
    MAX_MAIN_H = H_MAX - PANELS_H

    parts = []

    # ── 1. HEADER BAR ─────────────────────────────────────────────────
    hdr = np.full((HDR_H, W, 3), BG, dtype=np.uint8)
    # Sleek left edge indicator
    cv2.rectangle(hdr, (0, 0), (5, HDR_H), ACCENT, -1)
    cv2.line(hdr, (0, HDR_H - 1), (W, HDR_H - 1), BORDER_COLOR, 1)

    vid_label = seg["video_id"]
    if len(vid_label) > 55:
        vid_label = vid_label[:52] + "..."
    
    cv2.putText(hdr, vid_label, (12, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f + 0.05, WHITE, 1, cv2.LINE_AA)

    # Dynamic center action status (using ASCII separator to avoid ?? bug)
    if last_action:
        act_text = f"|  {last_action}"
        cv2.putText(hdr, act_text, (W // 2 - 40, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    ts_label = "Seg %d  |  %d:%02d - %d:%02d" % (
        seg["segment_index"],
        int(seg["timestamp_start"]) // 60, int(seg["timestamp_start"]) % 60,
        int(seg["timestamp_end"]) // 60, int(seg["timestamp_end"]) % 60)
    ts_sz = cv2.getTextSize(ts_label, FONT, font_scale_f, 1)[0]
    cv2.putText(hdr, ts_label, (W - ts_sz[0] - 12, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    parts.append(hdr)

    # ── 2. MAIN FRAME (scaled to fit remaining height) ─────────────────
    main = resize_keep_aspect(main_frame, W, max_height=MAX_MAIN_H)
    if main.shape[1] < W:
        pad_left = (W - main.shape[1]) // 2
        pad_right = W - main.shape[1] - pad_left
        main = cv2.copyMakeBorder(main, 0, 0, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=BG)
    
    # Sleek bottom border
    cv2.line(main, (0, main.shape[0] - 1), (W, main.shape[0] - 1), BORDER_COLOR, 1)
    parts.append(main)

    # ── 3. FILMSTRIP ───────────────────────────────────────────────────
    labels = ["START (0s)", "MIDDLE (5s)", "END (10s)"]
    thumb_w = W // 3
    thumb_imgs = []
    for t in thumbs:
        if t is None:
            t = np.zeros((40, 60, 3), dtype=np.uint8)
        thumb_imgs.append(resize_keep_aspect(t, thumb_w - 8, max_height=THUMB_H))

    strip = np.full((STRIP_H, W, 3), STRIP_BG, dtype=np.uint8)
    x = 0
    for i, (img, lbl) in enumerate(zip(thumb_imgs, labels)):
        x_off = x + (thumb_w - img.shape[1]) // 2
        y_off = LABEL_H + 3
        ih = min(img.shape[0], STRIP_H - y_off - 1)
        iw = min(img.shape[1], W - x_off)
        strip[y_off:y_off + ih, x_off:x_off + iw] = img[:ih, :iw]
        
        # Elegant outline around thumbnail
        cv2.rectangle(strip, (x_off, y_off), (x_off + iw - 1, y_off + ih - 1), BORDER_COLOR, 1)
        
        # Center middle frame overlay indicator
        if i == 1:
            cv2.rectangle(strip, (x_off, y_off), (x_off + iw - 1, y_off + ih - 1), ACCENT, 1)

        # Draw label above
        lbl_sz = cv2.getTextSize(lbl, FONT_S, font_scale_s, 1)[0]
        lbl_x = x + (thumb_w - lbl_sz[0]) // 2
        cv2.putText(strip, lbl, (lbl_x, LABEL_H - 3), FONT_S, font_scale_s, MUTED, 1, cv2.LINE_AA)
        
        # Vertical divider
        if i < 2:
            cv2.line(strip, (x + thumb_w - 1, 0), (x + thumb_w - 1, STRIP_H), BORDER_COLOR, 1)
        x += thumb_w
    
    cv2.line(strip, (0, STRIP_H - 1), (W, STRIP_H - 1), BORDER_COLOR, 1)
    parts.append(strip)

    # ── 4. PROGRESS BAR ───────────────────────────────────────────────
    bar = np.full((BAR_H, W, 3), BG, dtype=np.uint8)
    pct = done / max(total, 1)
    pct_text = "%d / %d segments rated (%.1f%%)" % (done, total, pct * 100)
    
    bx0, by0, bx1, by1 = 12, 5, W - 12, BAR_H - 5
    cv2.rectangle(bar, (bx0, by0), (bx1, by1), BAR_BG, -1)
    fill_x = bx0 + int((bx1 - bx0) * pct)
    if fill_x > bx0:
        cv2.rectangle(bar, (bx0, by0), (fill_x, by1), GREEN_LIGHT, -1)
        
    psz = cv2.getTextSize(pct_text, FONT, font_scale_f - 0.04, 1)[0]
    cv2.putText(bar, pct_text, ((W - psz[0]) // 2 + 1, BAR_H - int(BAR_H * 0.32) + 1), FONT, font_scale_f - 0.04, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, pct_text, ((W - psz[0]) // 2, BAR_H - int(BAR_H * 0.32)), FONT, font_scale_f - 0.04, WHITE, 1, cv2.LINE_AA)
    parts.append(bar)

    # ── 4b. SIGNALS REMINDER BAR ──────────────────────────────────────
    sig = np.full((SIG_H, W, 3), BG, dtype=np.uint8)
    sig_text = "SIGNALS: [1] New Concept   |   [2] Formula/Eq   |   [3] Worked Example   |   [4] Emphasis Cue"
    ssz = cv2.getTextSize(sig_text, FONT, font_scale_f - 0.05, 1)[0]
    cv2.putText(sig, sig_text, ((W - ssz[0]) // 2, SIG_H - int(SIG_H * 0.32)), FONT, font_scale_f - 0.05, WHITE, 1, cv2.LINE_AA)
    cv2.line(sig, (0, 0), (W, 0), BORDER_COLOR, 1)
    cv2.line(sig, (0, SIG_H - 1), (W, SIG_H - 1), BORDER_COLOR, 1)
    parts.append(sig)

    # ── 5. SCORE GUIDE (Sleek SaaS Cards - 4 Bands) ───────────────────
    guide = np.full((GUIDE_H, W, 3), BG, dtype=np.uint8)
    bands = [
        ("8-10 CRITICAL", "Diag/formula/example", GREEN_DARK, GREEN_LIGHT),
        ("4-7  USEFUL",   "Bullet pts/definitions", YELLOW_DARK, YELLOW_LIGHT),
        ("1-3  LOW VALUE", "Title/transition/talk", RED_DARK, RED_LIGHT),
        ("0    SKIP/FILLER", "Blank/logo/admin/dup", GRAY_DARK, GRAY_LIGHT),
    ]
    col_w = W // 4
    for i, (title, desc, bg_col, acc_col) in enumerate(bands):
        x0 = i * col_w + 4
        # Card Background
        cv2.rectangle(guide, (x0, 2), (x0 + col_w - 8, GUIDE_H - 2), bg_col, -1)
        # Left Accent highlight strip
        cv2.rectangle(guide, (x0, 2), (x0 + 4, GUIDE_H - 2), acc_col, -1)
        # Border
        cv2.rectangle(guide, (x0, 2), (x0 + col_w - 8, GUIDE_H - 2), BORDER_COLOR, 1)
        
        cv2.putText(guide, title, (x0 + 10, int(GUIDE_H * 0.42)), FONT, font_scale_f, WHITE, 1, cv2.LINE_AA)
        cv2.putText(guide, desc,  (x0 + 10, int(GUIDE_H * 0.80)), FONT_S, font_scale_s, MUTED, 1, cv2.LINE_AA)
    parts.append(guide)

    # ── 6. KEYBOARD HINT BAR ──────────────────────────────────────────
    kb = np.full((KB_H, W, 3), BG, dtype=np.uint8)
    keys_text = "[0-9] Score   |   [T] Ten   |   [S] Skip   |   [B] Back   |   [Q] Save & Quit"
    ksz = cv2.getTextSize(keys_text, FONT, font_scale_f, 1)[0]
    cv2.putText(kb, keys_text, ((W - ksz[0]) // 2, KB_H - int(KB_H * 0.35)), FONT, font_scale_f, MUTED, 1, cv2.LINE_AA)
    
    if prev_score is not None and prev_score != "-":
        ps_text = f"Prev: {prev_score}"
        cv2.putText(kb, ps_text, (12, KB_H - int(KB_H * 0.35)), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)
    
    cv2.line(kb, (0, 0), (W, 0), BORDER_COLOR, 1)
    kb[KB_H - 2:KB_H, :] = ACCENT
    parts.append(kb)

    return np.vstack(parts)
 
 
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
 
    # Create a named window that is fully resizable and scaled beautifully
    WIN_NAME = "Module 1 Annotation - Integra"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_WIDTH, MAX_CANVAS_HEIGHT)

    idx = 0
    current_cap = None
    current_video = None
    last_action = "Welcome!"
 
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
 
        canvas = make_canvas(mid, [start_f, mid, end_f], seg, done, total, prev, last_action)
        cv2.imshow(WIN_NAME, canvas)
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
            last_action = "Moved Back"
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
            last_action = "Skipped"
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
            last_action = f"Saved: {score}"
            idx += 1
        else:
            if key not in (ord('q'), ord('b'), ord('s')):
                last_action = "Invalid Key"
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