import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import numpy as np


def write_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return str(path)


def run_module1(video_path: str, output_dir: str) -> str:
    """
    Real Module 1 backend runner.

    Input:
        video_path: uploaded lecture video path from backend job folder
        output_dir: storage/jobs/{job_id}/outputs

    Output:
        storage/jobs/{job_id}/outputs/module1_output.json
        List of {segment_id, timestamp_start, timestamp_end, score_V}
        (schema validated in run_inference against src/utils/json_schema.py)
    """
    from src.module1_importance.feature_extractor import FrameFeatureExtractor
    from src.module1_importance.inference import run_inference

    project_root = Path(__file__).resolve().parents[3]

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    job_dir = output_dir.parent
    video_id = video_path.stem

    module1_temp_dir = job_dir / "temp" / "module1"
    module1_temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = module1_temp_dir / f"{video_id}_features.npy"
    timestamps_path = module1_temp_dir / f"{video_id}_timestamps.npy"
    final_module1_json = output_dir / "module1_output.json"

    # Falls back to random-weight dry-run inference (still schema-valid) until
    # the trained checkpoint exists at the project root — see train.py.
    model_path = project_root / "best_module1_model.pt"

    print("========== REAL MODULE 1 START ==========")
    print("Video path:", video_path)
    print("Output dir:", output_dir)
    print("Temp dir:", module1_temp_dir)
    print("Model checkpoint:", model_path,
          "(found)" if model_path.exists() else "(missing -> dry-run/random weights)")
    print("==========================================")

    # 1. Extract 1fps ResNet-50 features from the uploaded video
    extractor = FrameFeatureExtractor(device="cuda")
    features, timestamps = extractor.extract_from_video(str(video_path), fps=1)

    np.save(features_path, features)
    np.save(timestamps_path, np.array(timestamps))

    # 2. Run BiLSTM inference -> score_V per 10s segment
    run_inference(
        features_path=str(features_path),
        timestamps_path=str(timestamps_path),
        model_path=str(model_path),
        output_json_path=str(final_module1_json),
        device="cuda",
    )

    if not final_module1_json.exists():
        raise FileNotFoundError(f"Module 1 final output not found: {final_module1_json}")

    print("========== MODULE 1 COMPLETED ==========")
    print("Saved:", final_module1_json)
    print("=========================================")

    return str(final_module1_json)


def run_module2(video_path: str, output_dir: str) -> str:
    """
    Real Module 2 backend runner.

    Input:
        video_path: uploaded lecture video path from backend job folder
        output_dir: storage/jobs/{job_id}/outputs

    Output:
        storage/jobs/{job_id}/outputs/module2_output.json
        Flat list of {sentence, timestamp_start, timestamp_end, is_important, importance_ratio_T}
        (schema validated against src/utils/json_schema.py)

        storage/jobs/{job_id}/outputs/module2_enriched.json
        Topic/summary/key_points blocks grouped from high-importance segments
        (src/module2_summarization/gemini_enrich.py). Best-effort: if Gemini
        is unavailable (missing/expired key), this file is skipped rather than
        failing the whole module, matching Module 3's OCR-fallback pattern.
    """
    from src.utils.json_schema import validate_module2_output
    from src.module2_summarization import gemini_enrich

    project_root = Path(__file__).resolve().parents[3]

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    job_dir = output_dir.parent
    video_id = video_path.stem

    module2_temp_dir = job_dir / "temp" / "module2"
    module2_temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_json = module2_temp_dir / f"{video_id}_transcript.json"
    classified_json = module2_temp_dir / f"{video_id}_classified.json"
    final_module2_json = output_dir / "module2_output.json"

    bert_model_path = "src/module2_summarization/bert_model_v2"

    def run_command(command: list[str]) -> None:
        print("\nRunning command:")
        print(" ".join(command))
        subprocess.run(
            command,
            cwd=str(project_root),
            check=True
        )

    print("========== REAL MODULE 2 START ==========")
    print("Video path:", video_path)
    print("Output dir:", output_dir)
    print("Temp dir:", module2_temp_dir)
    print("BERT checkpoint:", bert_model_path)
    print("===========================================")

    # 1. Transcribe the lecture video with Whisper
    run_command([
        sys.executable,
        "src/module2_summarization/transcribe_video.py",
        "--video", str(video_path),
        "--output_json", str(raw_json)
    ])

    # 2. Classify sentence importance with BERT + rule-based feature boosting
    run_command([
        sys.executable,
        "src/module2_summarization/classify_with_features.py",
        "--input_json", str(raw_json),
        "--output_json", str(classified_json),
        "--model_path", bert_model_path
    ])

    # 3. Flatten segment-level detail into the frozen Module 2 schema
    with classified_json.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    flat_output = []
    for segment in segments:
        importance_ratio = segment["importance_ratio_T"]
        for sent in segment["sentences"]:
            flat_output.append({
                "sentence": sent["sentence"],
                "timestamp_start": sent["timestamp_start"],
                "timestamp_end": sent["timestamp_end"],
                "is_important": sent["is_important"],
                "importance_ratio_T": importance_ratio
            })

    errors = validate_module2_output(flat_output)
    if errors:
        raise ValueError(f"Module 2 output failed schema validation: {'; '.join(errors)}")

    write_json(final_module2_json, flat_output)

    # 4. Best-effort Gemini enrichment: group high-importance segments into
    # topic/summary blocks. Never fails the module - if Gemini is unavailable
    # (missing/expired key, rate limit, network), log it and move on, since
    # module2_output.json above already satisfies the schema Module 4 needs.
    enriched_json = output_dir / "module2_enriched.json"
    try:
        enriched_blocks = gemini_enrich.enrich_segments(segments)
        write_json(enriched_json, enriched_blocks)
        print(f"Gemini enrichment: {len(enriched_blocks)} topic blocks saved to {enriched_json}")
    except Exception as e:
        print(f"Gemini enrichment skipped (non-fatal): {e}")

    print("========== MODULE 2 COMPLETED ==========")
    print("Saved:", final_module2_json)
    print("=========================================")

    return str(final_module2_json)


def run_module3(video_path: str, output_dir: str) -> str:
    """
    Real Module 3 backend runner.

    Input:
        video_path: uploaded lecture video path from backend job folder
        output_dir: storage/jobs/{job_id}/outputs

    Output:
        storage/jobs/{job_id}/outputs/module3_output.json
    """

    project_root = Path(__file__).resolve().parents[3]

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    job_dir = output_dir.parent
    lecture_id = video_path.stem

    module3_temp_dir = job_dir / "temp" / "module3"
    frames_dir = module3_temp_dir / "slides" / lecture_id

    predictions_json = module3_temp_dir / f"{lecture_id}_predictions.json"
    anchors_json = module3_temp_dir / f"{lecture_id}_visual_anchors.json"
    ocr_json = module3_temp_dir / f"{lecture_id}_with_ocr.json"
    enhanced_json = module3_temp_dir / f"{lecture_id}_enhanced.json"
    filtered_json = module3_temp_dir / f"{lecture_id}_filtered_critical_important.json"

    final_module3_json = output_dir / "module3_output.json"

    module3_temp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    def run_command(command: list[str]) -> None:
        print("\nRunning command:")
        print(" ".join(command))
        subprocess.run(
            command,
            cwd=str(project_root),
            check=True
        )

    print("========== REAL MODULE 3 START ==========")
    print("Video path:", video_path)
    print("Output dir:", output_dir)
    print("Temp dir:", module3_temp_dir)
    print("Frames dir:", frames_dir)
    print("Final Module 3 JSON:", final_module3_json)
    print("=========================================")

    # 1. Extract slide frames from uploaded lecture video
    run_command([
        sys.executable,
        "src/module3_visual/slide_extractor.py",
        "--video", str(video_path),
        "--output", str(frames_dir),
        "--interval", "30"
    ])

    # 2. Run ViT inference on extracted frames
    run_command([
        sys.executable,
        "src/module3_visual/inference.py",
        "--frames_dir", str(frames_dir),
        "--output_json", str(predictions_json)
    ])

    # 3. Select visual anchor frames
    run_command([
        sys.executable,
        "src/module3_visual/select_visual_anchors.py",
        "--input_json", str(predictions_json),
        "--output_json", str(anchors_json),
        "--min_gap", "30",
        "--max_records", "80"
    ])

    # 4. Apply OCR to selected frames
    run_command([
        sys.executable,
        "src/module3_visual/ocr.py",
        "--input_json", str(anchors_json),
        "--output_json", str(ocr_json)
    ])

    # 5. Extract enhanced visual content information
    run_command([
        sys.executable,
        "src/module3_visual/enhanced_content_extractor.py",
        "--input_json", str(ocr_json),
        "--output_json", str(enhanced_json)
    ])

    # 6. Filter/relabel frames:
    # Critical  -> visual/diagram/mixed slides, sent to Gemini
    # Important -> useful text-only slides, OCR only
    with enhanced_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []

    critical_markers = {
        "diagram",
        "graph",
        "chart",
        "table",
        "flowchart",
        "architecture",
        "formula",
        "plot",
        "figure",
        "mixed"
    }

    for item in data:
        original_label = item.get("label")

        visual_type = str(item.get("visual_type") or "").lower()
        detected_visual_types = [
            str(x).lower()
            for x in item.get("detected_visual_types", [])
        ]
        visual_elements = item.get("visual_elements", [])
        element_types = []

        for element in visual_elements:
            if isinstance(element, dict):
                element_types.append(str(element.get("type", "")).lower())

        ocr_text = str(item.get("ocr_text") or "").strip()

        joined_visual_info = " ".join(
            [visual_type] + detected_visual_types + element_types
        )

        has_visual_content = any(
            marker in joined_visual_info
            for marker in critical_markers
        )

        is_text_only = (
            "text" in visual_type
            and not has_visual_content
            and len(ocr_text.split()) >= 3
        )

        if has_visual_content:
            item["original_label"] = original_label
            item["label"] = "Critical"
            item["label_score"] = 1.0
            item["routing_reason"] = "Visual structure detected, routed to Gemini."
            filtered.append(item)

        elif is_text_only:
            item["original_label"] = original_label
            item["label"] = "Important"
            item["label_score"] = 0.5
            item["routing_reason"] = "Useful text-only slide, routed to OCR-only analysis."
            filtered.append(item)

        else:
            # Skip weak, unknown, or empty frames
            continue

    with filtered_json.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)

    print("\n========== MODULE 3 FILTER SUMMARY ==========")
    print("Input records:", len(data))
    print("Filtered records:", len(filtered))
    print("Labels:", Counter(item.get("label") for item in filtered))
    print("Visual types:", Counter(item.get("visual_type") for item in filtered))
    print("Filtered JSON:", filtered_json)
    print("=============================================\n")

    # 7. Gemini for Critical frames, OCR-only for Important frames
    run_command([
        sys.executable,
        "src/module3_visual/enhanced_gemini_extractor.py",
        "--input_json", str(filtered_json),
        "--output_json", str(final_module3_json),
        "--sleep", "10"
    ])

    if not final_module3_json.exists():
        raise FileNotFoundError(f"Module 3 final output not found: {final_module3_json}")

    print("========== REAL MODULE 3 COMPLETED ==========")
    print("Saved:", final_module3_json)
    print("=============================================")

    return str(final_module3_json)

def run_module4(
    video_path: str,
    module1_json: str,
    module2_json: str,
    module3_json: str,
    output_dir: str
) -> tuple[str, str]:
    """
    Real Module 4 backend runner (Pipeline A: real-footage highlight reel).

    Fuses Module 1/2/3 outputs via ScoreFusion (S = w1*V + w2*T + w3*L),
    selects the top segments within the target duration cap, and cuts +
    concatenates those segments from the original uploaded video into a
    single condensed MP4.

    Output:
        storage/jobs/{job_id}/outputs/module4_final_output.json
        storage/jobs/{job_id}/outputs/summarized_video.mp4
    """
    from src.module4_synthesis.fusion import ScoreFusion
    from src.module4_synthesis.pipeline_a import HighlightVideoGenerator

    TARGET_SUMMARY_LENGTH = 600.0  # 10 minutes, matches .env.example's default

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_json_path = output_dir / "module4_final_output.json"
    final_video_path = output_dir / "summarized_video.mp4"

    print("========== REAL MODULE 4 START ==========")
    print("Video path:", video_path)
    print("Output dir:", output_dir)
    print("==========================================")

    with open(module1_json, "r", encoding="utf-8") as f:
        module1_data = json.load(f)
    with open(module2_json, "r", encoding="utf-8") as f:
        module2_data = json.load(f)
    with open(module3_json, "r", encoding="utf-8") as f:
        module3_data = json.load(f)

    fusion = ScoreFusion()
    fused_segments = fusion.fuse(module1_data, module2_data, module3_data)
    selected_segments = fusion.select_segments(fused_segments, max_duration=TARGET_SUMMARY_LENGTH)

    if not selected_segments and fused_segments:
        # Every segment fell under min_score_threshold (e.g. a quiet lecture
        # opening with no important speech or slides yet). Rather than fail
        # the whole job with no video at all, fall back to the highest-scoring
        # segments regardless of threshold, up to the duration cap.
        print("No segments met the fusion threshold - falling back to top-scoring segments regardless of threshold.")
        fallback = []
        total_duration = 0.0
        for seg in fused_segments:  # already sorted by fused_score, best first
            seg_duration = seg["timestamp_end"] - seg["timestamp_start"]
            if total_duration + seg_duration > TARGET_SUMMARY_LENGTH:
                continue
            fallback.append(seg)
            total_duration += seg_duration
            if total_duration >= min(30.0, TARGET_SUMMARY_LENGTH):
                break
        selected_segments = sorted(fallback, key=lambda s: s["timestamp_start"])

    if not selected_segments:
        raise ValueError(
            "Module 4 fusion selected zero segments - check Module 1/2/3 outputs; "
            "the input video may have no usable segments at all."
        )

    generator = HighlightVideoGenerator(max_duration=TARGET_SUMMARY_LENGTH)
    generator.generate(
        video_path=video_path,
        segments=selected_segments,
        output_path=str(final_video_path),
    )

    total_duration = sum(
        seg["timestamp_end"] - seg["timestamp_start"] for seg in selected_segments
    )

    final_data = {
        "module": "module4",
        "status": "completed",
        "description": "Real Module 4 fusion output (Pipeline A: real-footage highlight reel)",
        "fusion_weights": {"w1_visual": fusion.w1, "w2_text": fusion.w2, "w3_slide": fusion.w3},
        "total_segments_considered": len(fused_segments),
        "segments_selected": len(selected_segments),
        "selected_duration_seconds": round(total_duration, 2),
        "selected_segments": selected_segments,
        "final_video": str(final_video_path),
    }

    write_json(final_json_path, final_data)

    if not final_video_path.exists():
        raise FileNotFoundError(f"Module 4 final video not found: {final_video_path}")

    print("========== MODULE 4 COMPLETED ==========")
    print("Saved:", final_video_path)
    print("=========================================")

    return str(final_video_path), str(final_json_path)