"""
streamlit_app.py

Dedicated UI for the Module 4 video-generation pipeline documented in
D:\\AI_Projects\\video_generation.md. Wraps the existing standalone scripts
as subprocesses (same commands, same venv python) so the cmd workflow keeps
working unchanged alongside this UI.

Usage:
    (from an activated venv, at the repo root)
    streamlit run streamlit_app.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
INPUT_DIR = REPO_ROOT / "input"
VIDEO_INPUT_DIR = REPO_ROOT / "video_input"
OUTPUT_DIR = REPO_ROOT / "output"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Mirrors src/module4_synthesis/align_sources.py's own candidate filenames.
SLIDE_CANDIDATES = ["lecture_slide.json", "lecture_021_module3_final_output.json"]
AUDIO_CANDIDATES = ["lecture_audio.json", "LecVideo_001_enriched.json"]

ALIGNED_SOURCES = INPUT_DIR / "aligned_sources.json"
VIDEO_JSON_INPUT = VIDEO_INPUT_DIR / "video_json_input.json"
VIDEO_JSON_CHECKED = VIDEO_INPUT_DIR / "video_json_input_checked.json"

st.set_page_config(page_title="Module 4 — Video Generator", layout="wide")


# ─── Small helpers ───────────────────────────────────────────────────────

def _has_candidate(directory: Path, candidates: list) -> bool:
    return any((directory / name).exists() for name in candidates)


def _fmt_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _file_info(path: Path) -> str:
    if not path.exists():
        return "not found"
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{_fmt_size(stat.st_size)}, modified {mtime}"


# ─── Environment status (read-only) ─────────────────────────────────────

def check_environment() -> dict:
    checks = {
        "venv python found": (REPO_ROOT / "venv" / "Scripts" / "python.exe").exists(),
        "ffmpeg on PATH": shutil.which("ffmpeg") is not None,
        "model checkpoint (models\\module4\\stage1)": (
            REPO_ROOT / "models" / "module4" / "stage1" / "model.safetensors"
        ).exists(),
    }

    env_path = REPO_ROOT / ".env"
    ollama_ok = False
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^OLLAMA_API_KEY=(.+)$", content, re.MULTILINE)
        if m:
            value = m.group(1).strip()
            ollama_ok = bool(value) and "your_ollama_cloud_api_key_here" not in value
    checks[".env has OLLAMA_API_KEY"] = ollama_ok

    if "torch_importable" not in st.session_state:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch, transformers"],
                cwd=str(REPO_ROOT), capture_output=True, timeout=60,
            )
            st.session_state["torch_importable"] = result.returncode == 0
        except Exception:
            st.session_state["torch_importable"] = False
    checks["torch / transformers importable"] = st.session_state["torch_importable"]

    return checks


with st.expander("Environment status", expanded=False):
    if st.button("Recheck", key="recheck_env"):
        st.session_state.pop("torch_importable", None)
        st.rerun()
    env_checks = check_environment()
    cols = st.columns(2)
    for i, (label, ok) in enumerate(env_checks.items()):
        with cols[i % 2]:
            st.write(("✅ " if ok else "❌ ") + label)
    if not all(env_checks.values()):
        st.caption(
            "Red items are covered by the Troubleshooting table in video_generation.md — "
            "fix them from cmd, this UI doesn't install/change your environment."
        )


# ─── Subprocess runner with live log + progress streaming ──────────────

def make_step2_parser():
    pattern = re.compile(r"\[(\d+)/(\d+)\]")

    def parser(line: str):
        m = pattern.search(line)
        if m and int(m.group(2)) > 0:
            return int(m.group(1)) / int(m.group(2))
        return None

    return parser


def make_step3_parser():
    pattern = re.compile(r"^\[(\d+)\]")
    total = None
    if VIDEO_JSON_INPUT.exists():
        try:
            data = json.loads(VIDEO_JSON_INPUT.read_text(encoding="utf-8"))
            total = len(data.get("fused_slides", [])) or None
        except Exception:
            total = None
    state = {"count": 0}

    def parser(line: str):
        if total and pattern.match(line):
            state["count"] += 1
            return state["count"] / total
        return None

    return parser


def make_step4_parser():
    pattern = re.compile(r"\]\s*(\d+)%")

    def parser(line: str):
        m = pattern.search(line)
        if m:
            return int(m.group(1)) / 100
        return None

    return parser


def run_step(cmd: list, progress_parser=None):
    """Runs cmd as a subprocess from REPO_ROOT, streaming stdout+stderr into
    live Streamlit placeholders. Returns (success, full_log_text)."""
    log_lines = []
    log_placeholder = st.empty()
    progress_line_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    log_placeholder.code("(waiting for output...)")
    status_placeholder.info("Running...")
    start = time.time()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # avoid child crashes writing block/emoji chars to a pipe

    try:
        process = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
        )
    except Exception as e:
        status_placeholder.error(f"Failed to start: {e}")
        return False, str(e)

    for raw in iter(process.stdout.readline, ""):
        line = raw.rstrip("\n")
        if line == "":
            continue
        pct = progress_parser(line) if progress_parser else None
        if pct is not None:
            progress_bar.progress(min(max(pct, 0.0), 1.0))
            progress_line_placeholder.text(line.strip())
        else:
            log_lines.append(line)
            log_placeholder.code("\n".join(log_lines[-400:]))

    process.stdout.close()
    returncode = process.wait()
    elapsed = time.time() - start
    full_log = "\n".join(log_lines)

    if returncode == 0:
        progress_bar.progress(1.0)
        status_placeholder.success(f"Done in {elapsed:.1f}s")
        return True, full_log

    status_placeholder.error(f"Failed — exit code {returncode} after {elapsed:.1f}s")
    st.error("Last output:")
    st.code("\n".join(log_lines[-30:]) or "(no output captured)")
    return False, full_log


def do_run(key: str, cmd: list, parser=None) -> bool:
    success, log = run_step(cmd, parser)
    st.session_state[f"{key}_status"] = "Success" if success else "Failed"
    st.session_state[f"{key}_log"] = log
    st.session_state[f"{key}_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return success


def step_caption(key: str, output_path: Path):
    status = st.session_state.get(f"{key}_status")
    if status:
        icon = "✅" if status == "Success" else "❌"
        st.caption(f"{icon} Last run: {status} at {st.session_state.get(f'{key}_time')}")
        with st.expander("Show last log", expanded=False):
            st.code(st.session_state.get(f"{key}_log", ""))
    st.caption(f"Output file: `{_rel(output_path)}` — {_file_info(output_path)}")


# ─── 1. Input files ──────────────────────────────────────────────────────

st.title("Module 4 — Video Generation Pipeline")
st.caption(f"Repo: {REPO_ROOT}")

st.header("1. Input files")
col1, col2, col3 = st.columns(3)
with col1:
    slide_upload = st.file_uploader("Slide/OCR JSON", type="json", key="slide_upload")
    st.caption(f"Current `input\\lecture_slide.json`: {_file_info(INPUT_DIR / 'lecture_slide.json')}")
with col2:
    audio_upload = st.file_uploader("Audio JSON", type="json", key="audio_upload")
    st.caption(f"Current `input\\lecture_audio.json`: {_file_info(INPUT_DIR / 'lecture_audio.json')}")
with col3:
    st.file_uploader("Importance JSON", type="json", key="importance_upload")
    st.caption(f"Current `input\\lecture_importance.json`: {_file_info(INPUT_DIR / 'lecture_importance.json')}")
    st.caption("pipeline — reserved")

if st.button(
    "Save uploaded files to input\\",
    disabled=not (slide_upload or audio_upload),
):
    errors = []
    if slide_upload is not None:
        raw = slide_upload.getvalue()
        try:
            json.loads(raw.decode("utf-8"))
        except Exception as e:
            errors.append(f"Slide JSON is not valid JSON: {e}")
        else:
            (INPUT_DIR / "lecture_slide.json").write_bytes(raw)
    if audio_upload is not None:
        raw = audio_upload.getvalue()
        try:
            json.loads(raw.decode("utf-8"))
        except Exception as e:
            errors.append(f"Audio JSON is not valid JSON: {e}")
        else:
            (INPUT_DIR / "lecture_audio.json").write_bytes(raw)
    if errors:
        for e in errors:
            st.error(e)
    else:
        st.success("Saved to input\\")
        st.rerun()

# ─── 2. Pipeline ─────────────────────────────────────────────────────────

st.header("2. Pipeline")
run_all = st.button("▶ Run Full Pipeline (Steps 1 → 4)", type="primary")
proceed = True  # gates automatic chaining when run_all is active

# --- Step 1: Align Sources ---
with st.expander("Step 1 — Align Sources", expanded=True):
    st.write(
        "Joins the slide/OCR JSON and audio JSON by timestamp window into "
        "`input\\aligned_sources.json`."
    )
    step1_prereq = _has_candidate(INPUT_DIR, SLIDE_CANDIDATES) and _has_candidate(INPUT_DIR, AUDIO_CANDIDATES)
    if not step1_prereq:
        st.warning("Save both input files above first (or place them manually in input\\).")
    run1 = st.button("Run Step 1", key="run1", disabled=not step1_prereq)
    if run_all and proceed and not step1_prereq:
        st.error("Skipping — Step 1 prerequisites missing.")
        proceed = False
    if run1 or (run_all and proceed and step1_prereq):
        ok = do_run(
            "step1",
            [sys.executable, str(REPO_ROOT / "src" / "module4_synthesis" / "align_sources.py")],
        )
        if run_all:
            proceed = ok
    step_caption("step1", ALIGNED_SOURCES)

# --- Step 2: Generate Video JSON (Step A) ---
with st.expander("Step 2 — Generate Video JSON (Step A, local model)", expanded=True):
    st.write(
        "Runs the fine-tuned local model (`models\\module4\\stage1\\`) over "
        "`aligned_sources.json` to produce `video_input\\video_json_input.json`. "
        "Takes a few minutes on CPU."
    )
    step2_prereq = ALIGNED_SOURCES.exists()
    if not step2_prereq:
        st.warning("Run Step 1 first — input\\aligned_sources.json not found.")
    run2 = st.button("Run Step 2", key="run2", disabled=not step2_prereq)
    if run_all and proceed and not step2_prereq:
        st.error("Skipping — Step 2 prerequisites missing.")
        proceed = False
    if run2 or (run_all and proceed and step2_prereq):
        ok = do_run(
            "step2",
            [sys.executable, str(SCRIPTS_DIR / "generate_video_json_input.py")],
            make_step2_parser(),
        )
        if run_all:
            proceed = ok
    step_caption("step2", VIDEO_JSON_INPUT)

# --- Step 3: Check Content Loss (Step B) ---
with st.expander("Step 3 — Check Content Loss (Step B, Ollama Cloud)", expanded=True):
    st.write(
        "Sends each slide's generated summary + its real source facts to "
        "Ollama Cloud to fix false attribution and missing facts. Writes "
        "`video_input\\video_json_input_checked.json`."
    )
    step3_prereq = VIDEO_JSON_INPUT.exists()
    if not step3_prereq:
        st.warning("Run Step 2 first — video_input\\video_json_input.json not found.")
    run3 = st.button("Run Step 3", key="run3", disabled=not step3_prereq)
    if run_all and proceed and not step3_prereq:
        st.error("Skipping — Step 3 prerequisites missing.")
        proceed = False
    if run3 or (run_all and proceed and step3_prereq):
        ok = do_run(
            "step3",
            [sys.executable, str(SCRIPTS_DIR / "check_content_loss.py")],
            make_step3_parser(),
        )
        if run_all:
            proceed = ok
    step_caption("step3", VIDEO_JSON_CHECKED)

# --- Step 4: Render Final Video (Step C) ---
with st.expander("Step 4 — Render Final Video (Step C)", expanded=True):
    st.write(
        "Renders slide images (Pillow), narration audio (edge-tts), and "
        "assembles the final MP4 (MoviePy/FFmpeg) into `output\\`. "
        "Several minutes for a full lecture."
    )
    step4_prereq = VIDEO_JSON_CHECKED.exists()
    if not step4_prereq:
        st.warning("Run Step 3 first — video_input\\video_json_input_checked.json not found.")
    run4 = st.button("Run Step 4", key="run4", disabled=not step4_prereq)
    if run_all and proceed and not step4_prereq:
        st.error("Skipping — Step 4 prerequisites missing.")
        proceed = False
    if run4 or (run_all and proceed and step4_prereq):
        output_name = f"lecture_summary_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        ok = do_run(
            "step4",
            [sys.executable, str(SCRIPTS_DIR / "generate_real_video.py"), output_name],
            make_step4_parser(),
        )
        if ok:
            st.session_state["last_output_file"] = output_name
        if run_all:
            proceed = ok
    last_output_name = st.session_state.get("last_output_file", "lecture_021_summary.mp4")
    step_caption("step4", OUTPUT_DIR / last_output_name)

if run_all:
    if proceed:
        st.success("Full pipeline completed successfully.")
    else:
        st.error("Pipeline stopped — see the failed/skipped step above.")

# ─── 3. Output ───────────────────────────────────────────────────────────

st.header("3. Output")
last_output = st.session_state.get("last_output_file")
candidate = (OUTPUT_DIR / last_output) if last_output else None
if candidate and candidate.exists():
    st.success(f"Latest render this session: {candidate.name} ({_file_info(candidate)})")
    st.video(str(candidate))
    with open(candidate, "rb") as f:
        st.download_button("Download MP4", data=f, file_name=candidate.name, mime="video/mp4")
else:
    st.caption("No video rendered yet in this session. Existing files in output\\:")
    existing = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if existing:
        for p in existing[:5]:
            st.write(f"- `{p.name}` — {_file_info(p)}")
    else:
        st.write("(none yet)")
