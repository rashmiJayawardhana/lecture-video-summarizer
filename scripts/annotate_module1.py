"""
Module 1 Annotation Helper - Project Integra
Automated Lecture Video Summarization

Fast keyboard-driven annotation tool for scoring 10-second lecture
segments on a 0 to 10 visual importance scale.

HOW IT WORKS
  The tool walks through every 10-second segment of every video in your
  VIDEO_DIR. For each segment it shows the middle frame (large) plus a small
  filmstrip of the start, middle and end frames for context. You press one
  key to score the segment and the tool auto-advances and auto-saves.

KEYS (during annotation)
  0 1 2 3 4 5 6 7 8 9   score the segment 0 to 9
  t                     score the segment 10  (t = ten)
  s                     skip this segment (transition blur / unsure)
  b                     go back to the previous segment to fix it
  q                     save and quit (resumes next time you run)
  Space / Enter / N     keep the previous score and move on
                        (only useful when reviewing / re-annotating)

SETUP (run once)
  pip install opencv-python numpy

USAGE
  1. Put your assigned lecture videos in the folder named in VIDEO_DIR below.
  2. Change ANNOTATOR to your own name and OUTPUT_FILE to match.
  3. Run:  python annotate_module1.py
  4. To combine everyone's files at the end:
            python annotate_module1.py --merge

The output is a JSON file of ground-truth labels used to train the
ResNet-50 + BiLSTM model in Module 1.
"""

import cv2
import os
import sys
import json
import glob
import time
import shutil
import datetime
import numpy as np
import threading
import base64

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def draw_wrapped_text(img, text, x, y, max_w, font, scale, color, line_h=16):
    """Draw text wrapped to fit within max_w pixels."""
    words = str(text).split(" ")
    lines = []
    curr_line = ""
    for word in words:
        test_line = curr_line + (" " if curr_line else "") + word
        sz = cv2.getTextSize(test_line, font, scale, 1)[0][0]
        if sz > max_w:
            lines.append(curr_line)
            curr_line = word
        else:
            curr_line = test_line
    if curr_line:
        lines.append(curr_line)
    
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_h), font, scale, color, 1, cv2.LINE_AA)
    return len(lines) * line_h


def encode_image_base64(frame):
    """Encode an OpenCV BGR frame to base64 JPEG string."""
    if frame is None:
        return ""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def get_ai_score_and_reasoning(start_f, mid_f, end_f):
    """
    Call either Claude (Anthropic) or GPT-4o (OpenAI) depending on which API key
    is present, with the 10-second segment filmstrip frames and return (ai_score, ai_reasoning).
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Encode images to base64
    start_b64 = encode_image_base64(start_f)
    mid_b64 = encode_image_base64(mid_f)
    end_b64 = encode_image_base64(end_f)

    if not start_b64 or not mid_b64 or not end_b64:
        return None, "Error encoding frames to base64."

    system_prompt = """You are an expert AI annotator for university IT lecture videos.
Your task is to assign a visual importance score (0 to 10) to a 10-second video segment.
You are given 3 frames from the segment: the start frame, middle frame, and end frame.

Please follow this strict 4-point visual importance scoring rubric:
1. Base score (look at the slide content only):
   - Score 8: Core diagram, complete formula, full algorithm.
   - Score 7: Clear diagram, code snippet, partial worked example.
   - Score 6: Small diagram, table, or partial example.
   - Score 5: Definition or full paragraph of explanation.
   - Score 4: A few bullet points or a short list.
   - Score 3: Heading + one bullet, or a single sentence.
   - Score 2: Heading only, or one very short bullet.
   - Score 1: Lecturer on camera, slide empty or decorative.
   - Score 0: Blank, logo, transition, title, section cover, TOC, thank-you, admin slide, or a repeated slide.

2. Lecturer movement (adjust the base score):
   - Standing still / just talking: +0
   - Pointing or gesturing at the slide: +1
   - Annotating / writing / circling: +2
   - The +1/+2 bonus ONLY applies if the lecturer is visibly interacting with content
     that is actually part of the on-screen slide. If there is no slide visible at all,
     no bonus applies no matter how expressive the gesture is - generic talking hand
     movement with no slide always gets +0.
   - A physical object held up to the camera (a book, a printed sheet/handout, a prop,
     any real-world item) is NEVER slide content, even if the lecturer points at it,
     holds it toward the camera, or it has readable text/diagrams on it. Score the base
     as if no slide is present (0-1), and the movement bonus is always +0 for it.

The final score is the base score + movement adjustment, capped at a maximum of 10.
Important: If the lecturer is speaking with no slide content, score it low (0-2).
Important: You are given the start, middle, and end frame of this segment. If the
meaningful slide content is only visible in one of the three frames (e.g. only at the
start, with the lecturer alone for the rest), treat the segment as if the content was
NOT sustained - score it lower than you would if the same content were visible in all
three frames.
Return your response as a JSON object with keys:
"score": <integer from 0 to 10>
"reasoning": "<short sentence explaining the score based on the rubric features present in the frames>"
"""

    if anthropic_key and ANTHROPIC_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            claude_system_prompt = system_prompt + "\nReturn ONLY the JSON object, no other text."
            user_content = [
                {"type": "text", "text": "Analyze these three frames (start, middle, end) from a 10-second lecture segment and output the score and reasoning in the requested JSON format."}
            ]

            for label, img_b64 in [("Start Frame", start_b64),
                                    ("Middle Frame", mid_b64),
                                    ("End Frame", end_b64)]:
                user_content.append({"type": "text", "text": f"{label}:"})
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    }
                })

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.0,
                system=claude_system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )

            response_text = response.content[0].text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            res_json = json.loads(response_text)
            ai_score = int(res_json.get("score"))
            ai_reasoning = res_json.get("reasoning", "")
            ai_score = max(0, min(10, ai_score))
            return ai_score, ai_reasoning
        except Exception as e:
            return None, f"Claude AI generation error: {str(e)}"

    elif openai_key and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=openai_key)
            user_content = [
                {"type": "text", "text": "Analyze these three frames (start, middle, end) from a 10-second segment and output the score and reasoning in the requested JSON format."}
            ]

            for i, img_b64 in enumerate([start_b64, mid_b64, end_b64], 1):
                name = ["Start Frame", "Middle Frame", "End Frame"][i-1]
                user_content.append({"type": "text", "text": f"{name}:"})
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low"
                    }
                })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200
            )

            res_json = json.loads(response.choices[0].message.content)
            ai_score = int(res_json.get("score"))
            ai_reasoning = res_json.get("reasoning", "")
            ai_score = max(0, min(10, ai_score))
            return ai_score, ai_reasoning
        except Exception as e:
            return None, f"OpenAI GPT-4o generation error: {str(e)}"

    # --- Google Gemini (FREE TIER) ---
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key and GEMINI_AVAILABLE:
        try:
            client = genai_new.Client(api_key=gemini_key)

            # Build content parts using the new SDK format
            parts = [
                genai_types.Part.from_text(
                    text=system_prompt + "\nReturn ONLY the JSON object, no other text.\n"
                    "Analyze these three frames (start, middle, end) from a 10-second lecture segment "
                    "and output the score and reasoning in the requested JSON format."
                )
            ]
            for label, img_b64 in [("Start Frame", start_b64),
                                    ("Middle Frame", mid_b64),
                                    ("End Frame", end_b64)]:
                img_bytes = base64.b64decode(img_b64)
                parts.append(genai_types.Part.from_text(text=f"{label}:"))
                parts.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

            gemini_model = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
            response = client.models.generate_content(
                model=gemini_model,
                contents=parts,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=300,
                ),
            )

            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            res_json = json.loads(response_text)
            ai_score = int(res_json.get("score"))
            ai_reasoning = res_json.get("reasoning", "")
            ai_score = max(0, min(10, ai_score))
            return ai_score, ai_reasoning
        except Exception as e:
            return None, f"Gemini AI generation error: {str(e)}"

    else:
        reasons = []
        if not anthropic_key and not openai_key and not gemini_key:
            reasons.append("No API key found (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY).")
        if not ANTHROPIC_AVAILABLE and not OPENAI_AVAILABLE and not GEMINI_AVAILABLE:
            reasons.append("No AI packages installed (anthropic, openai, or google-generativeai).")
        return None, " ".join(reasons)


def fetch_ai_opinion_async(seg_id, start_f, mid_f, end_f, annotations_dict, output_file):
    """Fetch AI score and reasoning in the background so it doesn't block the human annotator."""
    def worker():
        try:
            # Check if already has ai_score
            if "ai_score" in annotations_dict.get(seg_id, {}):
                return
            sf = start_f.copy() if start_f is not None else None
            mf = mid_f.copy() if mid_f is not None else None
            ef = end_f.copy() if end_f is not None else None
            
            ai_score, ai_reasoning = get_ai_score_and_reasoning(sf, mf, ef)
            if ai_score is not None:
                if seg_id in annotations_dict:
                    annotations_dict[seg_id]["ai_score"] = ai_score
                    annotations_dict[seg_id]["ai_reasoning"] = ai_reasoning
                    save_annotations(output_file, annotations_dict)
        except Exception:
            pass
            
    threading.Thread(target=worker, daemon=True).start()

# =====================================================================
# CONFIGURATION  -  each group member edits the next TWO lines.
# Leave VIDEO_DIR set to "videos".
# =====================================================================
VIDEO_DIR   = "videos"                      # folder containing your lecture videos
ANNOTATOR   = "rashmi"                      # your name (goes into every record)
OUTPUT_FILE = "annotations_rashmi.json"     # your personal output file

# Fixed settings (do NOT change - keeps the whole team consistent)
SEGMENT_LENGTH   = 10                       # seconds per segment
MAX_SCORE        = 10                       # scale is 0 to 10
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

# Base annotation-window sizing (adjusted at runtime for small screens)
DISPLAY_WIDTH     = 960
MAX_CANVAS_HEIGHT = 680


# =====================================================================
# TERMINAL UI HELPERS  -  ANSI colours, with safe fallbacks
# =====================================================================
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass

def _enable_ansi_colours():
    """Enable ANSI colour codes on Windows 10+ and detect support."""
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            mode = ctypes.c_uint32()
            handle = kernel32.GetStdHandle(-11)
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
                return True
            return False
        except Exception:
            return False
    return True


_ANSI = _enable_ansi_colours()
_UNICODE = "utf" in (sys.stdout.encoding or "").lower()


class C:
    """ANSI colour codes (empty strings if the terminal cannot show them)."""
    R   = "\033[0m"  if _ANSI else ""  # reset
    B   = "\033[1m"  if _ANSI else ""  # bold
    D   = "\033[2m"  if _ANSI else ""  # dim
    RED = "\033[31m" if _ANSI else ""
    GRN = "\033[32m" if _ANSI else ""
    YEL = "\033[33m" if _ANSI else ""
    BLU = "\033[34m" if _ANSI else ""
    MAG = "\033[35m" if _ANSI else ""
    CYN = "\033[36m" if _ANSI else ""
    GRY = "\033[90m" if _ANSI else ""


# Unicode glyphs with ASCII fallbacks
TICK    = "\u2713" if _UNICODE else "[OK]"
CROSS   = "\u2717" if _UNICODE else "[X] "
INFOSIG = "i"      if _UNICODE else "[i]"
WARNSIG = "!"      if _UNICODE else "[!]"
ARROW   = "\u2192" if _UNICODE else "->"
BULLET  = "\u2022" if _UNICODE else "*"
HLINE   = "\u2500" if _UNICODE else "-"
if _UNICODE:
    BOX_TL, BOX_TR, BOX_BL, BOX_BR = "\u2554", "\u2557", "\u255A", "\u255D"
    BOX_H,  BOX_V                   = "\u2550", "\u2551"
else:
    BOX_TL = BOX_TR = BOX_BL = BOX_BR = "+"
    BOX_H, BOX_V                      = "=", "|"


def info(msg): print(f"{C.CYN}{INFOSIG}{C.R}  {msg}")
def ok(msg):   print(f"{C.GRN}{TICK}{C.R}  {msg}")
def warn(msg): print(f"{C.YEL}{WARNSIG}{C.R}  {msg}")
def err(msg):  print(f"{C.RED}{CROSS}{C.R}  {msg}")


def hr(width=70):
    print(C.GRY + HLINE * width + C.R)


def banner(lines, width=70):
    """Print a double-line box around the given title lines."""
    inner = width - 2
    print()
    print(C.BLU + BOX_TL + BOX_H * inner + BOX_TR + C.R)
    for line in lines:
        print(f"{C.BLU}{BOX_V}{C.R} {line.ljust(inner - 1)}{C.BLU}{BOX_V}{C.R}")
    print(C.BLU + BOX_BL + BOX_H * inner + BOX_BR + C.R)


# =====================================================================
# SCREEN-SIZE AUTO-DETECTION
# =====================================================================
try:
    import tkinter as _tk
    _root = _tk.Tk()
    _screen_w = _root.winfo_screenwidth()
    _screen_h = _root.winfo_screenheight()
    _root.destroy()
    if _screen_h <= 900:
        DISPLAY_WIDTH     = 800
        MAX_CANVAS_HEIGHT = 480
        _ui_msg = f"Compact UI ({DISPLAY_WIDTH}x{MAX_CANVAS_HEIGHT}) for screen {_screen_w}x{_screen_h}"
    else:
        _ui_msg = f"Standard UI ({DISPLAY_WIDTH}x{MAX_CANVAS_HEIGHT}) for screen {_screen_w}x{_screen_h}"
except Exception:
    _ui_msg = f"UI {DISPLAY_WIDTH}x{MAX_CANVAS_HEIGHT} (screen size not detected)"


# =====================================================================
# ANNOTATION FILE I/O
# =====================================================================
def load_annotations(path):
    """Load existing annotations. If corrupted, back the file up and start fresh."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        backup = path + ".corrupted." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            shutil.copy2(path, backup)
            warn(f"Could not read {path} ({e}).")
            warn(f"A backup was saved to {backup}. Starting fresh.")
        except Exception:
            warn(f"Could not read {path} and could not back it up. Starting fresh.")
        return {}


def save_annotations(path, data):
    """Atomic write so the file is never half-written if you Ctrl-C.

    Retries the final rename a few times: on Windows, antivirus/search
    indexing/OneDrive can briefly hold a lock on a just-written file and
    cause a transient PermissionError.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    last_err = None
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    raise last_err


# =====================================================================
# VIDEO HELPERS
# =====================================================================
def get_video_info(path):
    """Return (fps, duration_seconds, frame_count) or None on failure."""
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
    ok_flag, frame = cap.read()
    return frame if ok_flag else None


def resize_keep_aspect(img, width, max_height=None):
    """Resize to a target width, preserving aspect ratio; optionally cap height."""
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
    inf = get_video_info(video_path)
    if inf is None:
        return []
    _, duration, _ = inf
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    n = int(duration // SEGMENT_LENGTH)
    segments = []
    for i in range(n):
        start = i * SEGMENT_LENGTH
        end = start + SEGMENT_LENGTH
        segments.append({
            "video_path":         video_path,
            "video_id":           video_id,
            "segment_index":      i,
            "segment_id":         f"{video_id}__seg_{i:04d}",
            "timestamp_start":    float(start),
            "timestamp_end":      float(end),
            "middle_frame_time":  float((start + end) / 2.0),
        })
    return segments


# =====================================================================
# ANNOTATION CANVAS  (OpenCV-rendered window)
# =====================================================================
def _fmt_eta(seconds):
    if seconds is None or seconds <= 0:
        return "--"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def make_canvas(main_frame, thumbs, seg, done, total, prev_score,
                last_action="", active_thumb_idx=1,
                session_scored=0, session_skipped=0, eta_sec=None,
                review_mode=False, ai_score=None, ai_reasoning=None):
    """
    Build the modern, colour-coded annotation canvas.
    Returns: canvas, strip_y, strip_h, left_w, thumb_w
    """
    W = DISPLAY_WIDTH
    H_MAX = MAX_CANVAS_HEIGHT

    # ---- Colour palette (BGR) ----
    BG           = (28, 24, 24)
    ACCENT       = (245, 160, 20)        # gold/amber
    WHITE        = (245, 245, 245)
    MUTED        = (160, 155, 155)
    BORDER_COLOR = (55, 50, 50)
    STRIP_BG     = (20, 18, 18)
    BAR_BG       = (42, 38, 38)

    GREEN_LIGHT  = (100, 220, 80);  GREEN_DARK  = (20, 60, 20)
    YELLOW_LIGHT = (240, 180, 60);  YELLOW_DARK = (20, 50, 60)
    RED_LIGHT    = (240, 90, 90);   RED_DARK    = (20, 20, 50)
    BLUE_LIGHT   = (240, 180, 80);  BLUE_DARK   = (60, 30, 20)
    GRAY_LIGHT   = (150, 150, 150); GRAY_DARK   = (40, 40, 40)

    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    FONT_S = cv2.FONT_HERSHEY_PLAIN

    # ---- Dynamic sizing ----
    if H_MAX < 550:
        HDR_H, THUMB_H, LABEL_H = 28, 55, 14
        BAR_H, L_W, R_W         = 18, 140, 140
        font_scale_f, font_scale_s = 0.36, 0.8
    else:
        HDR_H, THUMB_H, LABEL_H = 36, 80, 18
        BAR_H, L_W, R_W         = 24, 160, 160
        font_scale_f, font_scale_s = 0.42, 0.95

    STRIP_H = THUMB_H + LABEL_H + 6
    C_W   = W - L_W - R_W
    COL_H = H_MAX - HDR_H

    # =====================================================
    # 1. HEADER BAR
    # =====================================================
    hdr = np.full((HDR_H, W, 3), BG, dtype=np.uint8)
    cv2.rectangle(hdr, (0, 0), (5, HDR_H), ACCENT, -1)
    cv2.line(hdr, (0, HDR_H - 1), (W, HDR_H - 1), BORDER_COLOR, 1)

    # Video label (left)
    vid_label = seg["video_id"]
    if len(vid_label) > 36:
        vid_label = vid_label[:33] + "..."
    cv2.putText(hdr, vid_label, (12, HDR_H - int(HDR_H * 0.32)),
                FONT, font_scale_f + 0.05, WHITE, 1, cv2.LINE_AA)

    # Last action (centre-ish, colour-coded)
    if last_action:
        if   last_action.startswith("Saved"):   act_col = GREEN_LIGHT
        elif last_action.startswith("Kept"):    act_col = ACCENT
        elif last_action.startswith("Skip"):    act_col = BLUE_LIGHT
        elif last_action.startswith("Moved"):   act_col = YELLOW_LIGHT
        elif last_action.startswith("Invalid"): act_col = RED_LIGHT
        else:                                   act_col = MUTED
        cv2.putText(hdr, f"|  {last_action}",
                    (W // 2 - 50, HDR_H - int(HDR_H * 0.32)),
                    FONT, font_scale_f, act_col, 1, cv2.LINE_AA)

    # Segment timestamp (right)
    ts_label = "Seg %d  |  %d:%02d - %d:%02d" % (
        seg["segment_index"],
        int(seg["timestamp_start"]) // 60, int(seg["timestamp_start"]) % 60,
        int(seg["timestamp_end"])   // 60, int(seg["timestamp_end"])   % 60)
    ts_sz = cv2.getTextSize(ts_label, FONT, font_scale_f, 1)[0]
    cv2.putText(hdr, ts_label,
                (W - ts_sz[0] - 12, HDR_H - int(HDR_H * 0.32)),
                FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    # Previous-score badge
    if prev_score == "-":
        badge_text, badge_bg, badge_fg = "UNRATED", (42, 38, 38), MUTED
    elif prev_score is None:
        badge_text, badge_bg, badge_fg = "SKIPPED", (20, 50, 60), ACCENT
    else:
        badge_text, badge_bg, badge_fg = f"SCORE {prev_score}", (20, 60, 20), GREEN_LIGHT
    badge_sz = cv2.getTextSize(badge_text, FONT, font_scale_f, 1)[0]
    badge_w  = badge_sz[0] + 18
    badge_h  = int(HDR_H * 0.78)
    bx = W - ts_sz[0] - 12 - badge_w - 15
    by = (HDR_H - badge_h) // 2
    cv2.rectangle(hdr, (bx, by), (bx + badge_w, by + badge_h), badge_bg, -1)
    cv2.rectangle(hdr, (bx, by), (bx + badge_w, by + badge_h), BORDER_COLOR, 1)
    tx = bx + (badge_w - badge_sz[0]) // 2
    ty = by + badge_h - int(badge_h * 0.3)
    cv2.putText(hdr, badge_text, (tx, ty), FONT, font_scale_f, badge_fg, 1, cv2.LINE_AA)

    # =====================================================
    # 2. LEFT COLUMN - PASS 1: Content
    # =====================================================
    col_l = np.full((COL_H, L_W, 3), BG, dtype=np.uint8)
    cv2.line(col_l, (L_W - 1, 0), (L_W - 1, COL_H), BORDER_COLOR, 1)

    p1_label = "PASS 1: CONTENT"
    p1_sz = cv2.getTextSize(p1_label, FONT, font_scale_f, 1)[0]
    cv2.putText(col_l, p1_label, ((L_W - p1_sz[0]) // 2, 20),
                FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

    content_cells = [
        (0, ["Blank, title,",     "filler, repeat"],     GRAY_DARK,   GRAY_LIGHT),
        (1, ["Lecturer shot,",    "decorative slide"],   RED_DARK,    RED_LIGHT),
        (2, ["Heading only or",   "1 short bullet"],     RED_DARK,    RED_LIGHT),
        (3, ["Heading+1 bullet",  "or 1 sentence"],      RED_DARK,    RED_LIGHT),
        (4, ["Few bullets,",      "short list"],         YELLOW_DARK, YELLOW_LIGHT),
        (5, ["Definition or",     "full paragraph"],     YELLOW_DARK, YELLOW_LIGHT),
        (6, ["Small diagram,",    "table, partial ex"],  YELLOW_DARK, YELLOW_LIGHT),
        (7, ["Big diagram/code,", "partial worked ex"],  GREEN_DARK,  GREEN_LIGHT),
        (8, ["Core diagram,",     "formula, algorithm"], GREEN_DARK,  GREEN_LIGHT),
    ]

    cell_h = (COL_H - 40) // 9
    cell_w = L_W - 16
    pad_x  = 8
    for i, (score_val, desc, bg_c, acc_c) in enumerate(content_cells):
        cx, cy = pad_x, 30 + i * cell_h + 4
        cw, ch = cell_w, cell_h - 4
        cv2.rectangle(col_l, (cx, cy), (cx + cw, cy + ch), bg_c, -1)
        cv2.rectangle(col_l, (cx, cy), (cx + 4, cy + ch), acc_c, -1)
        cv2.rectangle(col_l, (cx, cy), (cx + cw, cy + ch), BORDER_COLOR, 1)
        cv2.putText(col_l, str(score_val), (cx + 10, cy + ch - int(ch * 0.25)),
                    FONT, font_scale_f + 0.1, WHITE, 1, cv2.LINE_AA)
        line_h  = 12 if H_MAX < 550 else 16
        start_y = cy + 18 if len(desc) == 2 else cy + ch - int(ch * 0.35)
        for j, line in enumerate(desc):
            cv2.putText(col_l, line, (cx + 32, start_y + j * line_h),
                        FONT_S, font_scale_s - 0.2, MUTED, 1, cv2.LINE_AA)

    # =====================================================
    # 3. RIGHT COLUMN - PASS 2: Movement + Shortcuts OR Review Info
    # =====================================================
    col_r = np.full((COL_H, R_W, 3), BG, dtype=np.uint8)
    cv2.line(col_r, (0, 0), (0, COL_H), BORDER_COLOR, 1)

    if review_mode:
        p2_label = "DISAGREEMENT"
        p2_sz = cv2.getTextSize(p2_label, FONT, font_scale_f, 1)[0]
        cv2.putText(col_r, p2_label, ((R_W - p2_sz[0]) // 2, 20),
                    FONT, font_scale_f, RED_LIGHT, 1, cv2.LINE_AA)
        
        cv2.putText(col_r, f"Human Score: {prev_score}", (pad_x, 45),
                    FONT, font_scale_f, WHITE, 1, cv2.LINE_AA)
        cv2.putText(col_r, f"AI Score: {ai_score}", (pad_x, 65),
                    FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)
        
        diff = abs(int(prev_score) - int(ai_score)) if prev_score != "-" and ai_score is not None else 0
        cv2.putText(col_r, f"Diff: {diff}", (pad_x, 85),
                    FONT, font_scale_f, RED_LIGHT, 1, cv2.LINE_AA)
        
        cv2.putText(col_r, "AI REASONING:", (pad_x, 115),
                    FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)
        
        reason_text = ai_reasoning if ai_reasoning else "No reasoning provided."
        draw_wrapped_text(col_r, reason_text, pad_x, 135, R_W - 16,
                          FONT_S, font_scale_s - 0.25, MUTED, line_h=13)
        
        # Instruction at the bottom
        inst_y = COL_H - 75
        cv2.putText(col_r, "Enter Final Score:", (pad_x, inst_y),
                    FONT_S, font_scale_s - 0.1, WHITE, 1, cv2.LINE_AA)
        cv2.putText(col_r, "[0-9/T] : New score", (pad_x, inst_y + 18),
                    FONT_S, font_scale_s - 0.25, MUTED, 1, cv2.LINE_AA)
        cv2.putText(col_r, "Space/Enter: Keep", (pad_x, inst_y + 33),
                    FONT_S, font_scale_s - 0.25, MUTED, 1, cv2.LINE_AA)
        cv2.putText(col_r, "B: Back  Q: Save/Quit", (pad_x, inst_y + 48),
                    FONT_S, font_scale_s - 0.25, MUTED, 1, cv2.LINE_AA)
    else:
        p2_label = "PASS 2: MOVE"
        p2_sz = cv2.getTextSize(p2_label, FONT, font_scale_f, 1)[0]
        cv2.putText(col_r, p2_label, ((R_W - p2_sz[0]) // 2, 20),
                    FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)

        mod_cells = [
            ("+0", "Still",     GRAY_DARK,   MUTED),
            ("+1", "Pointing",  YELLOW_DARK, YELLOW_LIGHT),
            ("+2", "Annotate",  GREEN_DARK,  GREEN_LIGHT),
        ]
        mod_cell_h = 45 if H_MAX < 550 else 60
        for i, (mod_val, desc, bg_c, acc_c) in enumerate(mod_cells):
            cx, cy = pad_x, 30 + i * mod_cell_h + 4
            cw, ch = cell_w, mod_cell_h - 4
            cv2.rectangle(col_r, (cx, cy), (cx + cw, cy + ch), bg_c, -1)
            cv2.rectangle(col_r, (cx, cy), (cx + 4, cy + ch), acc_c, -1)
            cv2.rectangle(col_r, (cx, cy), (cx + cw, cy + ch), BORDER_COLOR, 1)
            cv2.putText(col_r, mod_val, (cx + 12, cy + ch - int(ch * 0.3)),
                        FONT, font_scale_f + 0.05, acc_c, 1, cv2.LINE_AA)
            cv2.putText(col_r, desc, (cx + 40, cy + ch - int(ch * 0.35)),
                        FONT_S, font_scale_s - 0.1, WHITE, 1, cv2.LINE_AA)

        # Shortcuts section
        key_y = 30 + 3 * mod_cell_h + 20
        cv2.putText(col_r, "SHORTCUTS", (pad_x, key_y),
                    FONT, font_scale_f, ACCENT, 1, cv2.LINE_AA)
        keys = ["0-9/T : Score",
                "S     : Skip",
                "B     : Back",
                "Spc/N : Keep",
                "Q     : Quit"]
        line_step = 20 if H_MAX < 550 else 26
        for i, k in enumerate(keys):
            cv2.putText(col_r, k, (pad_x, key_y + 18 + i * line_step),
                        FONT_S, font_scale_s - 0.1, MUTED, 1, cv2.LINE_AA)

    # =====================================================
    # 4. CENTRE COLUMN  -  Main frame + filmstrip + progress
    # =====================================================
    col_c = np.full((COL_H, C_W, 3), BG, dtype=np.uint8)
    C_MAIN_H = COL_H - STRIP_H - BAR_H

    # 4a. Main frame
    main = resize_keep_aspect(main_frame, C_W, max_height=C_MAIN_H)
    pad_t = max(0, (C_MAIN_H - main.shape[0]) // 2)
    pad_b = max(0, C_MAIN_H - main.shape[0] - pad_t)
    pad_l = max(0, (C_W - main.shape[1]) // 2)
    pad_r = max(0, C_W - main.shape[1] - pad_l)
    main_padded = cv2.copyMakeBorder(main, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=BG)
    cv2.line(main_padded, (0, C_MAIN_H - 1), (C_W, C_MAIN_H - 1),
             BORDER_COLOR, 1)
    col_c[:C_MAIN_H, :] = main_padded

    # 4b. Filmstrip
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
        cv2.rectangle(strip, (x_off, y_off),
                      (x_off + iw - 1, y_off + ih - 1), BORDER_COLOR, 1)
        if i == active_thumb_idx:
            cv2.rectangle(strip, (x_off, y_off),
                          (x_off + iw - 1, y_off + ih - 1), ACCENT, 2)
        lbl_sz = cv2.getTextSize(lbl, FONT_S, font_scale_s, 1)[0]
        cv2.putText(strip, lbl,
                    (tx + (thumb_w - lbl_sz[0]) // 2, LABEL_H - 3),
                    FONT_S, font_scale_s, MUTED, 1, cv2.LINE_AA)
        if i < 2:
            cv2.line(strip, (tx + thumb_w - 1, 0),
                     (tx + thumb_w - 1, STRIP_H), BORDER_COLOR, 1)
    cv2.line(strip, (0, STRIP_H - 1), (C_W, STRIP_H - 1), BORDER_COLOR, 1)
    col_c[C_MAIN_H:C_MAIN_H + STRIP_H, :] = strip

    # 4c. Progress bar with ETA + session stats
    bar = np.full((BAR_H, C_W, 3), BG, dtype=np.uint8)
    pct = done / max(total, 1)
    bx0, by0, bx1, by1 = 12, 5, C_W - 12, BAR_H - 5
    cv2.rectangle(bar, (bx0, by0), (bx1, by1), BAR_BG, -1)
    fill_x = bx0 + int((bx1 - bx0) * pct)
    if fill_x > bx0:
        cv2.rectangle(bar, (bx0, by0), (fill_x, by1), GREEN_LIGHT, -1)

    parts = [f"{done}/{total} ({pct * 100:.1f}%)"]
    if eta_sec is not None:
        parts.append(f"ETA ~{_fmt_eta(eta_sec)}")
    parts.append(f"Session: +{session_scored} / -{session_skipped}")
    bar_text = "   |   ".join(parts)

    psz = cv2.getTextSize(bar_text, FONT, font_scale_f - 0.04, 1)[0]
    tx = (C_W - psz[0]) // 2
    ty = BAR_H - int(BAR_H * 0.32)
    cv2.putText(bar, bar_text, (tx + 1, ty + 1),
                FONT, font_scale_f - 0.04, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, bar_text, (tx, ty),
                FONT, font_scale_f - 0.04, WHITE, 1, cv2.LINE_AA)
    col_c[C_MAIN_H + STRIP_H:, :] = bar

    # =====================================================
    # 5. ASSEMBLE
    # =====================================================
    cols   = np.hstack([col_l, col_c, col_r])
    canvas = np.vstack([hdr, cols])
    strip_y = HDR_H + C_MAIN_H
    return canvas, strip_y, STRIP_H, L_W, thumb_w


def normalize(raw):
    """Convert a 0-10 raw score to a 0-1 normalized value for training."""
    return round(raw / float(MAX_SCORE), 4)


# =====================================================================
# MERGE (run with --merge once everyone is done)
# =====================================================================
def merge_all():
    """Combine all annotations_*.json into module1_annotations.json."""
    banner(["MERGE  -  combining everyone's annotation files"])
    files = sorted(glob.glob("annotations_*.json"))
    if not files:
        warn("No 'annotations_*.json' files found in the current folder.")
        return

    combined  = {}
    owner     = {}    # segment_id -> annotator who first wrote it
    conflicts = []    # list of (segment_id, first_annotator, later_annotator)
    per_file  = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                part = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            warn(f"Skipped unreadable file '{fp}': {e}")
            continue
        for sid, rec in part.items():
            this_ann = rec.get("annotator", "?")
            if sid in owner and owner[sid] != this_ann:
                conflicts.append((sid, owner[sid], this_ann))
            owner[sid] = this_ann
        combined.update(part)
        per_file.append((fp, len(part)))
        ok(f"Merged {len(part):>5d} records from {fp}")

    if conflicts:
        print()
        warn(f"{len(conflicts)} segment(s) were annotated by more than one person")
        warn("  (last file in alphabetical order wins):")
        for sid, a, b in conflicts[:5]:
            print(f"    {C.GRY}{sid}{C.R}: {a}  {ARROW}  {b}")
        if len(conflicts) > 5:
            print(f"    {C.GRY}... and {len(conflicts) - 5} more{C.R}")
        print()

    with open("module1_annotations.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    hr()
    ok(f"Wrote {len(combined)} records to module1_annotations.json")

    # ─── Calculate Krippendorff's Alpha for Calibration (LecVideo 045) ───
    calib_segments = set()
    annotator_scores = {}
    
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                part = json.load(f)
        except Exception:
            continue
        for sid, rec in part.items():
            vid_id = rec.get("video_id", "")
            if vid_id.startswith("LecVideo 045"):
                calib_segments.add(sid)
                ann = rec.get("annotator", "")
                if ann:
                    if ann not in annotator_scores:
                        annotator_scores[ann] = {}
                    if not rec.get("skipped", False) and rec.get("raw_score") is not None:
                        annotator_scores[ann][sid] = float(rec["raw_score"])
                        
    if calib_segments and len(annotator_scores) >= 2:
        sorted_sids = sorted(list(calib_segments))
        sorted_anns = sorted(list(annotator_scores.keys()))
        
        matrix = []
        for ann in sorted_anns:
            row = []
            for sid in sorted_sids:
                row.append(annotator_scores[ann].get(sid, np.nan))
            matrix.append(row)
            
        matrix = np.array(matrix, dtype=float)
        valid_cols = np.sum(~np.isnan(matrix), axis=0) >= 2
        num_valid = np.sum(valid_cols)
        
        if num_valid > 0:
            try:
                import krippendorff
                alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='interval')
                print()
                banner([
                    "CALIBRATION ACCURACY  -  LecVideo 045 Inter-Annotator Agreement",
                    f"  Annotators (coders)    : {', '.join(sorted_anns)}",
                    f"  Segments in agreement  : {num_valid} of {len(sorted_sids)}",
                    f"  Krippendorff's alpha   : {alpha:.4f}  (Interval metric)"
                ])
                if alpha >= 0.7:
                    ok(f"Krippendorff's alpha is {alpha:.4f} (>= 0.7 target met! ✓)")
                else:
                    warn(f"Krippendorff's alpha is {alpha:.4f} (< 0.7 target. Review discrepancies!)")
                print()
            except Exception as e:
                warn(f"Could not compute Krippendorff's alpha: {e}")


# =====================================================================
# WELCOME / GUIDELINES / MENU
# =====================================================================
def show_welcome(video_count, total, done):
    pct = (done / total * 100.0) if total else 0.0
    banner([
        "PROJECT INTEGRA  -  Module 1 Annotation Helper",
        "Automated Lecture Video Summarization",
    ])
    print()
    print(f"  Annotator    : {C.B}{ANNOTATOR}{C.R}")
    print(f"  Videos folder: {C.B}{VIDEO_DIR}{C.R}")
    print(f"  Output file  : {C.B}{OUTPUT_FILE}{C.R}")
    print(f"  Display      : {C.D}{_ui_msg}{C.R}")
    print()
    print(f"  Found {C.B}{video_count}{C.R} video(s), "
          f"{C.B}{total}{C.R} segments total, "
          f"{C.GRN}{done}{C.R} already done ({pct:.1f}%).")
    print()


def show_guidelines():
    """Print the doc-aligned scoring guidelines to the terminal."""
    print()
    hr(80)
    print(f"{C.B}INTEGRA  -  Module 1 Annotation Guidelines & Rubrics{C.R}")
    hr(80)

    print(f"\n{C.BLU}[1] THE TWO-STEP METHOD{C.R}")
    print(f"\n  {C.B}Step 1{C.R} - Base score (look at the slide content only):")
    print(f"    {C.GRN}8{C.R}  Core diagram, complete formula, full algorithm")
    print(f"    {C.GRN}7{C.R}  Clear diagram, code snippet, partial worked example")
    print(f"    {C.YEL}6{C.R}  Small diagram, table, or partial example")
    print(f"    {C.YEL}5{C.R}  Definition or full paragraph of explanation")
    print(f"    {C.YEL}4{C.R}  A few bullet points or a short list")
    print(f"    {C.RED}3{C.R}  Heading + one bullet, or a single sentence")
    print(f"    {C.RED}2{C.R}  Heading only, or one very short bullet")
    print(f"    {C.RED}1{C.R}  Lecturer on camera, slide empty or decorative")
    print(f"    {C.GRY}0{C.R}  Blank, logo, transition, title, section cover,")
    print(f"       TOC, thank-you, admin slide, or a repeated slide")

    print(f"\n  {C.B}Step 2{C.R} - Lecturer movement (adjust the base score):")
    print(f"    Standing still / just talking          {ARROW}  +0")
    print(f"    Pointing or gesturing at the slide     {ARROW}  +1")
    print(f"    Annotating / writing / circling        {ARROW}  +2  (cap at 10)")
    print(f"\n  {C.B}Final score{C.R} = Step 1 base + Step 2 adjustment, max 10.")

    print(f"\n{C.BLU}[2] SCORING BANDS (how Module 4 will treat them){C.R}")
    print(f"    {C.GRN}8-10{C.R}  CRITICAL    A student must see this. Almost always kept.")
    print(f"    {C.YEL}4-7 {C.R}  USEFUL      Helpful to see. Kept if there is room.")
    print(f"    {C.RED}1-3 {C.R}  LOW VALUE   A little real content only. Usually cut.")
    print(f"    {C.GRY} 0  {C.R}  FILLER      No teaching content (incl. titles). Always cut.")

    print(f"\n{C.BLU}[3] EDGE CASE  -  no slide content, lecturer just talking{C.R}")
    print(f"    Score it {C.B}low (0-2){C.R}. What the lecturer says never raises the score.")
    print(f"      Black/blank/title/cover/TOC/admin              {ARROW}  0")
    print(f"      Lecturer visible, slide empty or decorative    {ARROW}  1")
    print(f"      Same as above + lecturer pointing at slide     {ARROW}  2")
    print(f"      Slide has only a heading or one short bullet   {ARROW}  2")
    print(f"    {C.D}Mute test: if muting makes you want to skip past it,{C.R}")
    print(f"    {C.D}it is a low score. Always.{C.R}")

    print(f"\n{C.BLU}[4] CONSISTENCY RULES{C.R}")
    print(f"    {BULLET} Calibration first: all annotate LecVideo 045, then compare.")
    print(f"    {BULLET} Mute the audio: score only what is visible on screen.")
    print(f"    {BULLET} If a frame is genuinely unclear, press S to skip it.")

    hr(80)
    input(f"{C.D}Press Enter to return to the menu...{C.R}")


def show_main_menu(done, total):
    """Print the main menu. Returns the user's choice as a string."""
    pct = (done / total * 100.0) if total else 0.0
    print()
    hr()
    print(f"  {C.B}What would you like to do?{C.R}")
    print(f"  {C.D}Progress: {done} / {total} segments ({pct:.1f}%){C.R}")
    print()
    if done > 0:
        print(f"   {C.CYN}[1]{C.R} Continue from where you left off       {C.D}(default){C.R}")
        print(f"   {C.CYN}[2]{C.R} Re-annotate or review a specific video")
        print(f"   {C.CYN}[3]{C.R} Re-annotate or review ALL videos")
        print(f"   {C.CYN}[4]{C.R} View annotation guidelines & rubrics")
        print(f"   {C.CYN}[5]{C.R} Start fresh                            {C.YEL}(clears your progress){C.R}")
    else:
        print(f"   {C.CYN}[1]{C.R} Start annotating                       {C.D}(default){C.R}")
        print(f"   {C.CYN}[2]{C.R} View annotation guidelines & rubrics")
    hr()
    return input(f"  Enter choice [1]: ").strip()


# =====================================================================
# MAIN ANNOTATION LOOP
# =====================================================================
def run(review_mode=False):
    # -------- 0. Validate config --------
    if not ANNOTATOR or " " in ANNOTATOR:
        err("ANNOTATOR is empty or contains spaces. Edit the CONFIGURATION block at the top of this file.")
        return
    if not OUTPUT_FILE.startswith("annotations_") or not OUTPUT_FILE.endswith(".json"):
        warn(f"OUTPUT_FILE '{OUTPUT_FILE}' does not match the team pattern 'annotations_<name>.json'.")
        warn("The --merge command will skip it. Continuing anyway.")

    # Check for available API Keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if anthropic_key and ANTHROPIC_AVAILABLE:
        print(f"{C.GRN}{TICK}{C.R}  [AI Opinion] ANTHROPIC_API_KEY found. Running in Claude AI-ASSISTED mode.")
    elif openai_key and OPENAI_AVAILABLE:
        print(f"{C.GRN}{TICK}{C.R}  [AI Opinion] OPENAI_API_KEY found. Running in GPT-4o AI-ASSISTED mode.")
    elif gemini_key and GEMINI_AVAILABLE:
        print(f"{C.GRN}{TICK}{C.R}  [AI Opinion] GEMINI_API_KEY found. Running in Gemini 2.0 Flash AI-ASSISTED mode (FREE).")
    else:
        print(f"{C.YEL}{WARNSIG}{C.R}  [AI Opinion] No valid API key found. Running in MANUAL-ONLY mode.")
        print(f"{C.D}     Tip: Get a FREE Gemini API key at https://aistudio.google.com/apikey{C.R}")
        print(f"{C.D}     Then add GEMINI_API_KEY=your_key to your .env file.{C.R}")

    # -------- 1. Discover videos --------
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, "*" + ext)))
    video_paths.sort()
    if not video_paths:
        err(f"No video files found in folder '{VIDEO_DIR}'.")
        info("Put your assigned .mp4/.mkv/.avi/.mov/.webm files there and run again.")
        return

    annotations = load_annotations(OUTPUT_FILE)
    all_segments = []
    for vp in video_paths:
        all_segments.extend(build_segments(vp))

    reannotate_video_id = None
    reannotate_all      = False

    if review_mode:
        banner(["REVIEW DISAGREEMENTS MODE", "Looking for segments where |human - AI| >= 3"])
        # Filter all_segments to only include the ones that need review
        review_segments = []
        for s in all_segments:
            sid = s["segment_id"]
            if sid in annotations:
                rec = annotations[sid]
                h_score = rec.get("raw_score")
                a_score = rec.get("ai_score")
                if h_score is not None and a_score is not None and abs(h_score - a_score) >= 3:
                    if not rec.get("reviewed", False):
                        review_segments.append(s)
        
        if not review_segments:
            print()
            banner(["REVIEW MODE", "No disagreements (|human - AI| >= 3) found to review! ✓"])
            return
            
        ok(f"Starting review mode: {len(review_segments)} disagreements found.")
        all_segments = review_segments
        total = len(all_segments)
        done = 0
    else:
        total = len(all_segments)
        if total == 0:
            err("Could not extract any segments. Are the video files readable?")
            return
        done = sum(1 for s in all_segments if s["segment_id"] in annotations)

        # -------- 2. Welcome --------
        show_welcome(len(video_paths), total, done)

        # -------- 3. Main menu --------
        while True:
            choice = show_main_menu(done, total) or "1"

            if choice == "1":
                break

            # Option 2 means different things depending on progress
            if choice == "2" and done == 0:
                show_guidelines()
                continue

            if choice == "2" and done > 0:
                # Pick a specific already-annotated video
                annotated_ids = sorted({
                    r.get("video_id") for r in annotations.values()
                    if isinstance(r, dict) and "video_id" in r
                })
                if not annotated_ids:
                    warn("No annotated videos found.")
                    continue
                print()
                info("Already annotated videos:")
                for i, vid in enumerate(annotated_ids, 1):
                    print(f"   {C.CYN}[{i}]{C.R} {vid}")
                print()
                try:
                    vc = input(f"  Select a video number (1-{len(annotated_ids)}): ").strip()
                    vi = int(vc) - 1
                    if 0 <= vi < len(annotated_ids):
                        reannotate_video_id = annotated_ids[vi]
                        ok(f"Re-annotation mode: '{reannotate_video_id}'")
                        info("Press SPACE / ENTER / N to KEEP a previous score and advance.")
                        break
                    err("Invalid selection.")
                except ValueError:
                    err("Please enter a number.")
                continue

            if choice == "3" and done > 0:
                reannotate_all = True
                ok("Re-annotation mode: ALL videos")
                info("Press SPACE / ENTER / N to KEEP a previous score and advance.")
                break

            if choice == "4" and done > 0:
                show_guidelines()
                continue

            if choice == "5" and done > 0:
                print()
                warn("This will clear ALL your annotations. This cannot be undone.")
                confirm = input(f"  Type {C.B}YES{C.R} to confirm, anything else cancels: ").strip()
                if confirm == "YES":
                    # Back up before wiping
                    if os.path.exists(OUTPUT_FILE):
                        backup = OUTPUT_FILE + ".backup." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        shutil.copy2(OUTPUT_FILE, backup)
                        info(f"Old file backed up to {backup}")
                    annotations = {}
                    save_annotations(OUTPUT_FILE, annotations)
                    done = 0
                    ok("All annotations cleared. Starting fresh.")
                    break
                info("Cancelled.")
                continue

            err("Invalid option. Please choose one of the numbered choices.")

    # -------- 4. Open the OpenCV window --------
    WIN_NAME = "Module 1 Annotation - Integra"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, DISPLAY_WIDTH, MAX_CANVAS_HEIGHT)

    mouse_state = {"clicked": False, "x": -1, "y": -1}
    def _mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["clicked"] = True
            mouse_state["x"] = x
            mouse_state["y"] = y
    cv2.setMouseCallback(WIN_NAME, _mouse_cb)

    # -------- 5. Annotation loop --------
    idx             = 0
    current_cap     = None
    current_video   = None
    last_action     = "Welcome!"
    session_start   = time.time()
    session_scored  = 0
    session_skipped = 0

    print()
    hr()
    info("Annotating. Focus the video window and press keys to score.")
    info("Press Q at any time to save and quit. You can resume later.")
    hr()

    while 0 <= idx < total:
        seg = all_segments[idx]

        # Skip already-annotated segments when moving forward, unless in re-annotation mode or review mode
        if not review_mode and seg["segment_id"] in annotations and seg.get("_visited") is not True:
            if not reannotate_all and (reannotate_video_id is None or
                                       seg["video_id"] != reannotate_video_id):
                idx += 1
                continue

        # Open the right video file
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
        end_f   = grab_frame(current_cap,
                             max(seg["timestamp_end"] - 0.2, seg["timestamp_start"]))

        if review_mode:
            done = sum(1 for s in all_segments if annotations.get(s["segment_id"], {}).get("reviewed") is True)
        else:
            done = sum(1 for s in all_segments if s["segment_id"] in annotations)
        prev = annotations.get(seg["segment_id"], {}).get("raw_score", "-")

        # Compute ETA from this session's average
        eta_sec = None
        n_acts  = session_scored + session_skipped
        if n_acts >= 3:
            avg = (time.time() - session_start) / n_acts
            eta_sec = avg * max(0, total - done)

        # Render-and-input loop for this one segment
        active_thumb_idx = 1
        force_redraw     = True
        key              = -1

        while True:
            if force_redraw:
                main_f = [start_f, mid, end_f][active_thumb_idx]
                # Retrieve AI score and reasoning from annotations
                rec = annotations.get(seg["segment_id"], {})
                a_score = rec.get("ai_score")
                a_reason = rec.get("ai_reasoning")
                canvas, strip_y, strip_h, L_W, thumb_w = make_canvas(
                    main_f, [start_f, mid, end_f], seg, done, total, prev,
                    last_action, active_thumb_idx,
                    session_scored, session_skipped, eta_sec,
                    review_mode=review_mode, ai_score=a_score, ai_reasoning=a_reason)
                cv2.imshow(WIN_NAME, canvas)
                force_redraw = False

            key = cv2.waitKey(50) & 0xFF
            if key != 255:
                break

            if mouse_state["clicked"]:
                mx, my = mouse_state["x"], mouse_state["y"]
                mouse_state["clicked"] = False
                if strip_y <= my <= strip_y + strip_h:
                    right_panel_w = 140 if MAX_CANVAS_HEIGHT < 550 else 160
                    if L_W <= mx <= DISPLAY_WIDTH - right_panel_w:
                        rel_x = mx - L_W
                        if   rel_x < thumb_w:        active_thumb_idx = 0
                        elif rel_x < 2 * thumb_w:    active_thumb_idx = 1
                        else:                        active_thumb_idx = 2
                        force_redraw = True

        # ---- Decide what to do with the key press ----
        if key == ord("q"):
            break

        if key == ord("b"):
            j = idx - 1
            if j >= 0:
                all_segments[j]["_visited"] = True
                idx = j
            else:
                idx = 0
            last_action = "Moved Back"
            continue

        # "Keep previous score" shortcut (Space / Enter / N), only when valid
        keep_keys = (13, 10, 32, ord("n"), ord("N"))
        is_kept = False
        if key in keep_keys and prev != "-":
            if prev is None:
                # Previously skipped - treat keep as skip again
                key = ord("s")
            else:
                is_kept = True

        if key == ord("s"):
            if review_mode:
                annotations[seg["segment_id"]].update({
                    "raw_score":         None,
                    "normalized_score":  None,
                    "skipped":           True,
                    "reviewed":          True,
                    "final_score":       None,
                    "reviewed_at":       datetime.datetime.now().isoformat(timespec="seconds"),
                })
            else:
                annotations[seg["segment_id"]] = {
                    "video_id":          seg["video_id"],
                    "segment_index":     seg["segment_index"],
                    "segment_id":        seg["segment_id"],
                    "timestamp_start":   seg["timestamp_start"],
                    "timestamp_end":     seg["timestamp_end"],
                    "middle_frame_time": seg["middle_frame_time"],
                    "raw_score":         None,
                    "normalized_score":  None,
                    "skipped":           True,
                    "annotator":         ANNOTATOR,
                    "annotated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
                }
                fetch_ai_opinion_async(seg["segment_id"], start_f, mid, end_f, annotations, OUTPUT_FILE)

            save_annotations(OUTPUT_FILE, annotations)
            seg.pop("_visited", None)
            last_action     = "Skipped"
            session_skipped += 1
            idx += 1
            continue

        # Determine the score to save
        score = None
        if is_kept:
            score = prev
        elif ord("0") <= key <= ord("9"):
            score = key - ord("0")
        elif key == ord("t") or key == ord("T"):
            score = 10

        if score is not None:
            if review_mode:
                annotations[seg["segment_id"]].update({
                    "raw_score":         score,
                    "normalized_score":  normalize(score),
                    "skipped":           False,
                    "reviewed":          True,
                    "final_score":       score,
                    "reviewed_at":       datetime.datetime.now().isoformat(timespec="seconds"),
                })
            else:
                annotations[seg["segment_id"]] = {
                    "video_id":          seg["video_id"],
                    "segment_index":     seg["segment_index"],
                    "segment_id":        seg["segment_id"],
                    "timestamp_start":   seg["timestamp_start"],
                    "timestamp_end":     seg["timestamp_end"],
                    "middle_frame_time": seg["middle_frame_time"],
                    "raw_score":         score,
                    "normalized_score":  normalize(score),
                    "skipped":           False,
                    "annotator":         ANNOTATOR,
                    "annotated_at":      datetime.datetime.now().isoformat(timespec="seconds"),
                }
                fetch_ai_opinion_async(seg["segment_id"], start_f, mid, end_f, annotations, OUTPUT_FILE)

            save_annotations(OUTPUT_FILE, annotations)
            seg.pop("_visited", None)
            last_action     = f"Reviewed: {score}" if review_mode else (f"Kept: {score}" if is_kept else f"Saved: {score}")
            session_scored += 1
            idx += 1
        else:
            last_action = "Invalid key (try 0-9, T, S, B, Q)"

    # -------- 6. Clean up + session summary --------
    if current_cap is not None:
        current_cap.release()
    cv2.destroyAllWindows()
    save_annotations(OUTPUT_FILE, annotations)
    done = sum(1 for s in all_segments if s["segment_id"] in annotations)
    pct  = (done / total * 100.0) if total else 0.0
    elapsed = time.time() - session_start

    print()
    banner(["SESSION SUMMARY"])
    print()
    print(f"  Annotated this session : {C.GRN}{session_scored}{C.R} scored, "
          f"{C.YEL}{session_skipped}{C.R} skipped")
    print(f"  Time spent             : {_fmt_eta(elapsed)}")
    if session_scored + session_skipped > 0:
        avg = elapsed / (session_scored + session_skipped)
        print(f"  Avg. per segment       : {avg:.1f}s")
    print(f"  Total progress         : {C.B}{done}{C.R} / {total} segments ({pct:.1f}%)")
    print(f"  Saved to               : {C.B}{OUTPUT_FILE}{C.R}")
    print()
    if done < total:
        info(f"You have {total - done} segment(s) left. Run again to continue.")
    else:
        ok("All segments scored. Send your file to Rashmi for merging.")
    print()


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        merge_all()
    elif len(sys.argv) > 1 and sys.argv[1] == "--review-disagreements":
        try:
            run(review_mode=True)
        except KeyboardInterrupt:
            print()
            warn("Interrupted. Your last save is intact - run again to resume.")
    else:
        try:
            run(review_mode=False)
        except KeyboardInterrupt:
            print()
            warn("Interrupted. Your last save is intact - run again to resume.")