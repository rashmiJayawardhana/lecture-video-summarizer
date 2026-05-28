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
 
 
def make_canvas(main_frame, thumbs, seg, done, total, prev_score, last_action="", active_thumb_idx=1):
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
        HDR_H     = 28
        THUMB_H   = 55
        LABEL_H   = 14
        STRIP_H   = THUMB_H + LABEL_H + 4
        BAR_H     = 18
        L_W       = 140
        R_W       = 140
        font_scale_f = 0.36
        font_scale_s = 0.8
    else:
        HDR_H     = 36
        THUMB_H   = 80
        LABEL_H   = 18
        STRIP_H   = THUMB_H + LABEL_H + 6
        BAR_H     = 24
        L_W       = 160
        R_W       = 160
        font_scale_f = 0.42
        font_scale_s = 0.95

    C_W = W - L_W - R_W
    COL_H = H_MAX - HDR_H

    # ── 1. HEADER BAR (Full Width) ────────────────────────────────────
    hdr = np.full((HDR_H, W, 3), BG, dtype=np.uint8)
    cv2.rectangle(hdr, (0, 0), (5, HDR_H), ACCENT, -1)
    cv2.line(hdr, (0, HDR_H - 1), (W, HDR_H - 1), BORDER_COLOR, 1)

    vid_label = seg["video_id"]
    if len(vid_label) > 40:
        vid_label = vid_label[:37] + "..."
    cv2.putText(hdr, vid_label, (12, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f + 0.05, WHITE, 1, cv2.LINE_AA)

    if last_action:
        act_text = f"|  {last_action}"
        cv2.putText(hdr, act_text, (W // 2 - 40, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    ts_label = "Seg %d  |  %d:%02d - %d:%02d" % (
        seg["segment_index"],
        int(seg["timestamp_start"]) // 60, int(seg["timestamp_start"]) % 60,
        int(seg["timestamp_end"]) // 60, int(seg["timestamp_end"]) % 60)
    ts_sz = cv2.getTextSize(ts_label, FONT, font_scale_f, 1)[0]
    cv2.putText(hdr, ts_label, (W - ts_sz[0] - 12, HDR_H - int(HDR_H * 0.32)), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    if prev_score == "-":
        badge_text, badge_bg, badge_fg = "UNRATED", (42, 38, 38), MUTED
    elif prev_score is None:
        badge_text, badge_bg, badge_fg = "SKIPPED", (20, 50, 60), ACCENT
    else:
        badge_text, badge_bg, badge_fg = f"SCORE: {prev_score}", (20, 60, 20), GREEN_LIGHT

    badge_sz = cv2.getTextSize(badge_text, FONT, font_scale_f - 0.05, 1)[0]
    badge_w = badge_sz[0] + 16
    badge_h = int(HDR_H * 0.7)
    bx = W - ts_sz[0] - 12 - badge_w - 15
    by = (HDR_H - badge_h) // 2
    
    cv2.rectangle(hdr, (bx, by), (bx + badge_w, by + badge_h), badge_bg, -1)
    cv2.rectangle(hdr, (bx, by), (bx + badge_w, by + badge_h), BORDER_COLOR, 1)
    
    tx = bx + (badge_w - badge_sz[0]) // 2
    ty = by + badge_h - int(badge_h * 0.3)
    cv2.putText(hdr, badge_text, (tx, ty), FONT, font_scale_f - 0.05, badge_fg, 1, cv2.LINE_AA)

    # ── 2. LEFT COLUMN (Pass 1: Content) ──────────────────────────────
    col_l = np.full((COL_H, L_W, 3), BG, dtype=np.uint8)
    cv2.line(col_l, (L_W - 1, 0), (L_W - 1, COL_H), BORDER_COLOR, 1)

    p1_label = "PASS 1: CONTENT"
    p1_sz = cv2.getTextSize(p1_label, FONT, font_scale_f, 1)[0]
    cv2.putText(col_l, p1_label, ((L_W - p1_sz[0]) // 2, 20), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)
    
    content_cells = [
        (0, "Blank/Logo",   GRAY_DARK,   GRAY_LIGHT),
        (1, "Title/Cover",  RED_DARK,    RED_LIGHT),
        (2, "Lect.Focus",   RED_DARK,    RED_LIGHT),
        (3, "Head+1Bull",   RED_DARK,    RED_LIGHT),
        (4, "Bullet List",  YELLOW_DARK, YELLOW_LIGHT),
        (5, "Definition",   YELLOW_DARK, YELLOW_LIGHT),
        (6, "SmallDiag",    YELLOW_DARK, YELLOW_LIGHT),
        (7, "BigDiag/Code", GREEN_DARK,  GREEN_LIGHT),
        (8, "CoreFormula",  GREEN_DARK,  GREEN_LIGHT),
    ]
    
    cell_h = (COL_H - 40) // 9
    cell_w = L_W - 16
    pad_x = 8
    
    for i, (score_val, desc, bg_c, acc_c) in enumerate(content_cells):
        cx = pad_x
        cy = 30 + i * cell_h + 4
        cw = cell_w
        ch = cell_h - 4
        
        cv2.rectangle(col_l, (cx, cy), (cx + cw, cy + ch), bg_c, -1)
        cv2.rectangle(col_l, (cx, cy), (cx + 4, cy + ch), acc_c, -1)
        cv2.rectangle(col_l, (cx, cy), (cx + cw, cy + ch), BORDER_COLOR, 1)

        cv2.putText(col_l, str(score_val), (cx + 12, cy + ch - int(ch*0.25)), FONT, font_scale_f + 0.1, WHITE, 1, cv2.LINE_AA)
        cv2.putText(col_l, desc, (cx + 36, cy + ch - int(ch*0.35)), FONT_S, font_scale_s - 0.15, MUTED, 1, cv2.LINE_AA)

    # ── 3. RIGHT COLUMN (Pass 2 + Keys) ───────────────────────────────
    col_r = np.full((COL_H, R_W, 3), BG, dtype=np.uint8)
    cv2.line(col_r, (0, 0), (0, COL_H), BORDER_COLOR, 1)

    p2_label = "PASS 2: MOVE"
    p2_sz = cv2.getTextSize(p2_label, FONT, font_scale_f, 1)[0]
    cv2.putText(col_r, p2_label, ((R_W - p2_sz[0]) // 2, 20), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    mod_cells = [
        ("+0", "Still",     GRAY_DARK,   MUTED),
        ("+1", "Pointing",  YELLOW_DARK, YELLOW_LIGHT),
        ("+2", "Annotate",  GREEN_DARK,  GREEN_LIGHT),
    ]

    mod_cell_h = 45 if H_MAX < 550 else 60
    
    for i, (mod_val, desc, bg_c, acc_c) in enumerate(mod_cells):
        cx = pad_x
        cy = 30 + i * mod_cell_h + 4
        cw = cell_w
        ch = mod_cell_h - 4
        
        cv2.rectangle(col_r, (cx, cy), (cx + cw, cy + ch), bg_c, -1)
        cv2.rectangle(col_r, (cx, cy), (cx + 4, cy + ch), acc_c, -1)
        cv2.rectangle(col_r, (cx, cy), (cx + cw, cy + ch), BORDER_COLOR, 1)

        cv2.putText(col_r, mod_val, (cx + 12, cy + ch - int(ch*0.3)), FONT, font_scale_f + 0.05, acc_c, 1, cv2.LINE_AA)
        cv2.putText(col_r, desc, (cx + 40, cy + ch - int(ch*0.35)), FONT_S, font_scale_s - 0.1, WHITE, 1, cv2.LINE_AA)

    key_label = "SHORTCUTS"
    key_y = 30 + 3 * mod_cell_h + 20
    cv2.putText(col_r, key_label, (pad_x, key_y), FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    keys = ["0-9/T: Score", "S: Skip", "B: Back", "Space: Keep", "Q: Quit"]
    for i, k in enumerate(keys):
        ky = key_y + 18 + i * 20 if H_MAX < 550 else key_y + 24 + i * 26
        cv2.putText(col_r, k, (pad_x, ky), FONT_S, font_scale_s - 0.1, MUTED, 1, cv2.LINE_AA)

    # ── 4. CENTER COLUMN (Video + Strip + Bar) ────────────────────────
    col_c = np.full((COL_H, C_W, 3), BG, dtype=np.uint8)
    C_MAIN_H = COL_H - STRIP_H - BAR_H

    # a. Main Video
    main = resize_keep_aspect(main_frame, C_W, max_height=C_MAIN_H)
    pad_t = max(0, (C_MAIN_H - main.shape[0]) // 2)
    pad_b = max(0, C_MAIN_H - main.shape[0] - pad_t)
    pad_l = max(0, (C_W - main.shape[1]) // 2)
    pad_r = max(0, C_W - main.shape[1] - pad_l)
    main_padded = cv2.copyMakeBorder(main, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=BG)
    cv2.line(main_padded, (0, C_MAIN_H - 1), (C_W, C_MAIN_H - 1), BORDER_COLOR, 1)
    
    col_c[:C_MAIN_H, :] = main_padded

    # b. Filmstrip
    strip = np.full((STRIP_H, C_W, 3), STRIP_BG, dtype=np.uint8)
    labels = ["START (0s)", "MIDDLE (5s)", "END (10s)"]
    thumb_w = C_W // 3
    
    for i, (t_frame, lbl) in enumerate(zip(thumbs, labels)):
        if t_frame is None:
            t_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        img = resize_keep_aspect(t_frame, thumb_w - 8, max_height=THUMB_H)
        
        tx = i * thumb_w
        x_off = tx + (thumb_w - img.shape[1]) // 2
        y_off = LABEL_H + 3
        ih = min(img.shape[0], STRIP_H - y_off - 1)
        iw = min(img.shape[1], C_W - x_off)
        
        strip[y_off:y_off + ih, x_off:x_off + iw] = img[:ih, :iw]
        
        cv2.rectangle(strip, (x_off, y_off), (x_off + iw - 1, y_off + ih - 1), BORDER_COLOR, 1)
        if i == active_thumb_idx:
            cv2.rectangle(strip, (x_off, y_off), (x_off + iw - 1, y_off + ih - 1), ACCENT, 2)

        lbl_sz = cv2.getTextSize(lbl, FONT_S, font_scale_s, 1)[0]
        lbl_x = tx + (thumb_w - lbl_sz[0]) // 2
        cv2.putText(strip, lbl, (lbl_x, LABEL_H - 3), FONT_S, font_scale_s, MUTED, 1, cv2.LINE_AA)
        
        if i < 2:
            cv2.line(strip, (tx + thumb_w - 1, 0), (tx + thumb_w - 1, STRIP_H), BORDER_COLOR, 1)
            
    cv2.line(strip, (0, STRIP_H - 1), (C_W, STRIP_H - 1), BORDER_COLOR, 1)
    col_c[C_MAIN_H:C_MAIN_H + STRIP_H, :] = strip

    # c. Progress Bar
    bar = np.full((BAR_H, C_W, 3), BG, dtype=np.uint8)
    pct = done / max(total, 1)
    pct_text = "%d / %d segments rated (%.1f%%)" % (done, total, pct * 100)
    
    bx0, by0, bx1, by1 = 12, 5, C_W - 12, BAR_H - 5
    cv2.rectangle(bar, (bx0, by0), (bx1, by1), BAR_BG, -1)
    fill_x = bx0 + int((bx1 - bx0) * pct)
    if fill_x > bx0:
        cv2.rectangle(bar, (bx0, by0), (fill_x, by1), GREEN_LIGHT, -1)
        
    psz = cv2.getTextSize(pct_text, FONT, font_scale_f - 0.04, 1)[0]
    cv2.putText(bar, pct_text, ((C_W - psz[0]) // 2 + 1, BAR_H - int(BAR_H * 0.32) + 1), FONT, font_scale_f - 0.04, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, pct_text, ((C_W - psz[0]) // 2, BAR_H - int(BAR_H * 0.32)), FONT, font_scale_f - 0.04, WHITE, 1, cv2.LINE_AA)
    
    col_c[C_MAIN_H + STRIP_H:, :] = bar

    # ── 5. ASSEMBLE ───────────────────────────────────────────────────
    cols = np.hstack([col_l, col_c, col_r])
    canvas = np.vstack([hdr, cols])

    strip_y = HDR_H + C_MAIN_H
    return canvas, strip_y, STRIP_H, L_W, thumb_w
 
 
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

    reannotate_video_id = None
    reannotate_all = False

    while True:
        print("\n" + "="*60)
        print(" Welcome to Module 1 Annotation Helper (Integra)")
        print("="*60)
        print(f"Progress: {done} / {total} segments annotated ({done/total*100:.1f}%)")
        print("\nChoose an option:")
        if done > 0:
            print("  [1] Continue from where you left off (default)")
            print("  [2] Re-annotate / Review a specific video")
            print("  [3] Re-annotate / Review ALL videos")
            print("  [4] View Annotation Guidelines & Rubrics")
            print("  [5] Start completely fresh (WARNING: clears all your progress)")
        else:
            print("  [1] Start annotating (default)")
            print("  [2] View Annotation Guidelines & Rubrics")
        
        choice = input("\nEnter choice: ").strip()
        
        if not choice:
            choice = '1'
            
        if choice == '1':
            break
        elif done > 0 and choice == '2':
            annotated_video_ids = sorted(list(set(
                ann.get("video_id") for ann in annotations.values() if isinstance(ann, dict) and "video_id" in ann
            )))
            if not annotated_video_ids:
                print("No annotated videos found to re-annotate.")
                continue
            
            print("\nAlready Annotated Videos:")
            for idx_vid, v_id in enumerate(annotated_video_ids, 1):
                print(f"  [{idx_vid}] {v_id}")
            try:
                vid_choice = input(f"\nSelect a video number (1-{len(annotated_video_ids)}): ").strip()
                vid_idx = int(vid_choice) - 1
                if 0 <= vid_idx < len(annotated_video_ids):
                    reannotate_video_id = annotated_video_ids[vid_idx]
                    print(f"\n>>> Mode: Re-annotating video '{reannotate_video_id}'")
                    print(">>> TIP: Press SPACE, ENTER, or 'N' to KEEP the previous score and advance!")
                    break
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid selection.")
        elif done > 0 and choice == '3':
            reannotate_all = True
            print("\n>>> Mode: Re-annotating ALL videos")
            print(">>> TIP: Press SPACE, ENTER, or 'N' to KEEP the previous score and advance!")
            break
        elif (done > 0 and choice == '4') or (done == 0 and choice == '2'):
            print("\n" + "="*80)
            print("         INTEGRA - MODULE 1 ANNOTATION GUIDELINES & RUBRICS")
            print("="*80)
            print("\n[1] THE TWO-PASS MENTAL SHORTCUT")
            print("  Pass 1: Content Only (Base Score)")
            print("    * 8 - Core diagram, formula, algorithm")
            print("    * 7 - Big diagram, code, partial example")
            print("    * 6 - Small diagram or table")
            print("    * 5 - Definition or full text")
            print("    * 4 - Bullet list (a few points)")
            print("    * 3 - Heading + one bullet only")
            print("    * 2 - Lecturer camera shot, slide is decorative")
            print("    * 1 - Cover slide, title slide, empty slide")
            print("  Pass 2: Lecturer Movement (Adjust Base Score)")
            print("    * Standing still           -> Keep Base")
            print("    * Pointing/gesturing       -> +1 Point")
            print("    * Actively annotating/writing -> +2 Points (Capped at 10)")
            print("\n[2] SCORING BANDS SUMMARY")
            print("  * 8-10 CRITICAL   - Core diagram/formula/full worked example/annotations")
            print("  * 4-7  USEFUL     - Normal content, definitions, bullet points")
            print("  * 1-3  LOW VALUE  - Covers, titles, lecturer-focus shots")
            print("  * 0    SKIP/FILLER- Blank screens, logos, transition blurs")
            print("\n[3] TIE-BREAKER DECISION LADDER (for 1, 2, 3 confusion)")
            print("  Ask in order, stop at first 'yes':")
            print("    1. Is slide essentially empty? -> 1")
            print("    2. Is lecturer the focus with slide adding nothing? -> 2")
            print("    3. Small content (heading + one bullet) & standing still? -> 3")
            print("="*80 + "\n")
            input("Press Enter to return to menu...")
        elif done > 0 and choice == '5':
            confirm = input("\nWARNING: Are you absolutely sure you want to clear ALL annotations? (y/N): ").strip().lower()
            if confirm == 'y':
                annotations = {}
                save_annotations(OUTPUT_FILE, annotations)
                done = 0
                print("\n>>> All annotations cleared! Starting completely fresh.")
                break
            else:
                print("\nClear cancelled.")
        else:
            print("Invalid option. Please try again.")
 
    # Create a named window that is fully resizable and scaled beautifully
    WIN_NAME = "Module 1 Annotation - Integra"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_WIDTH, MAX_CANVAS_HEIGHT)

    mouse_state = {"clicked": False, "x": -1, "y": -1}
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["clicked"] = True
            mouse_state["x"] = x
            mouse_state["y"] = y
    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    idx = 0
    current_cap = None
    current_video = None
    last_action = "Welcome!"
 
    while 0 <= idx < total:
        seg = all_segments[idx]
 
        # skip already-annotated segments when moving forward
        if seg["segment_id"] in annotations and seg.get("_visited") is not True:
            if not reannotate_all and (reannotate_video_id is None or seg["video_id"] != reannotate_video_id):
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
 
        active_thumb_idx = 1
        force_redraw = True
        key = -1

        while True:
            if force_redraw:
                main_f = [start_f, mid, end_f][active_thumb_idx]
                canvas, strip_y, strip_h, L_W, thumb_w = make_canvas(main_f, [start_f, mid, end_f], seg, done, total, prev, last_action, active_thumb_idx)
                cv2.imshow(WIN_NAME, canvas)
                force_redraw = False
            
            key = cv2.waitKey(50) & 0xFF
            
            if key != 255:
                break

            if mouse_state["clicked"]:
                mx, my = mouse_state["x"], mouse_state["y"]
                mouse_state["clicked"] = False
                
                # Check if click is inside the filmstrip area
                if strip_y <= my <= strip_y + strip_h:
                    if mx >= L_W and mx <= DISPLAY_WIDTH - (140 if MAX_CANVAS_HEIGHT < 550 else 160):
                        rel_x = mx - L_W
                        if rel_x < thumb_w:
                            active_thumb_idx = 0
                        elif rel_x < 2 * thumb_w:
                            active_thumb_idx = 1
                        else:
                            active_thumb_idx = 2
                        force_redraw = True

        # Quick "keep score" shortcut for re-annotations
        is_kept = False
        if key in (13, 10, 32, ord('n')) and prev != "-":
            if prev is None:
                key = ord('s')  # Convert to a skip action
            else:
                score = prev
                is_kept = True
  
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
            if is_kept:
                last_action = f"Kept Score: {score}"
            else:
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