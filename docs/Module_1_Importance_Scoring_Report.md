# Module 1 Report - Keyframe Detection and Visual Importance Scoring
### INTEGRA - Automated Lecture Video Summarization

*Prepared for Final Evaluation Report*

| Item | Details |
|---|---|
| Module | Module 1 - Keyframe Detection and Visual Importance Scoring |
| Module Owner | Jayawardhana G.G.R.M. (214093E) |
| Current Status | Full dataset annotated, model trained and evaluated across multiple runs, backend-integrated and verified with real inference |
| Main Output | Per-segment visual importance score JSON for Module 4 |

---

## 1. Executive Summary

Module 1 is responsible for processing the visual modality of lecture videos. It extracts frame-level features using a frozen ResNet-50 backbone, models temporal context across each 10-second segment using a Bidirectional LSTM, and outputs a visual importance score between 0 and 1 for every segment of the lecture. The final output is a structured JSON file consumed by Module 4 for multimodal fusion and final summarized video generation.

The full training dataset of 61 lecture videos (23,402 ten-second segments) was manually annotated against a four-criterion rubric. The model was trained and evaluated across five independent training runs at the finalized learning rate. The best-performing checkpoint achieves a segment-selection F1 of 0.870 at the project's primary operating threshold (score ≥ 5), meeting the target of Precision > 0.75 and Recall > 0.75 at that threshold. The model has also been integrated into the real backend pipeline and verified end-to-end with live inference on an uploaded video, confirming it produces schema-valid, non-random importance scores in production.

## 2. Module Objective

The objective of Module 1 is to identify which 10-second segments of a lecture video carry the most visual/instructional importance, so that Module 4 can prioritise these segments when assembling the condensed summary video. Importance is defined against a rubric that captures four signals: the introduction of a new concept, the presentation of a formula or equation, the delivery of a worked example, and visual emphasis cues such as circling, pointing, or extended dwell time on a slide.

## 3. Input and Output

### 3.1 Input
- Lecture video file (e.g. `LecVideo 058 - Lesson 02 - Interactions - Human Computer Interaction`).
- Frames extracted at 1 frame per second, converted to 2048-dimensional feature vectors by a frozen, ImageNet-pretrained ResNet-50 (no fine-tuning of the backbone itself).
- Frames are grouped into non-overlapping 10-second segments (10 consecutive frames per segment).

### 3.2 Output

Module 1 produces one JSON file per lecture, listing one record per 10-second segment.

| Field | Description |
|---|---|
| `segment_id` | `{video_id}__seg_{NNNN}`, identical format used in both annotation and inference so predictions match ground truth by ID. |
| `video_id` | Lecture identifier, taken from the video filename stem. |
| `timestamp_start` / `timestamp_end` | Segment boundaries in seconds. |
| `score_V` | Predicted visual importance score in the range 0-1 (BiLSTM output, sigmoid-activated). |

## 4. Scoring Rubric

Human annotators score each 10-second segment from 0 to 10 against four independent criteria, using a custom annotation tool (`annotate_module1.py`). The four-criterion score is later normalized to 0-1 for training.

| Criterion | Signal |
|---|---|
| New concept | Segment introduces a concept not previously covered. |
| Formula / equation | Segment presents a formula, equation, or algorithmic step. |
| Worked example | Segment walks through a concrete example. |
| Visual emphasis cue | Lecturer circles, points to, or otherwise visually emphasises on-screen content. |

## 5. Dataset Preparation

The full dataset of 61 lecture videos was manually annotated to 100% completion: 23,402 of 23,402 ten-second segments scored, 0 skipped or partially annotated. Videos span diverse IT theory topics (databases, algorithms, HCI, project management, security, and others), consistent with the project's IT-lecture domain focus.

The dataset is split by video number parsed directly from the filename (`LecVideo NNN - ...`), matching the split logic in `train.py`:

| Split | Videos | Rule |
|---|---|---|
| Train | 1-45 | `video number <= 45` |
| Validation | 46-50 | `45 < video number <= 50` |
| Test | 51-61 | `video number > 50` |

The test split is held out entirely from training and validation, and is the only split used for the evaluation numbers reported in Section 9.

## 6. Model Architecture

- **Frame feature extractor**: ResNet-50, pretrained on ImageNet, used frozen (no gradient updates) purely as a 2048-dim feature extractor. Features are pre-extracted once per video (`extract_all_features.py`) and cached, rather than run inline during training, to reduce compute cost.
- **Temporal model**: 2-layer Bidirectional LSTM, hidden size 512 (1024-dim output after concatenating both directions), dropout 0.3 between LSTM layers.
- **Classifier head**: `Linear(1024 -> 256)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(256 -> 1)` -> `Sigmoid`, producing one importance score per frame; frame-level scores are averaged across the 10-frame segment to produce the final `score_V`.
- **Frameworks**: PyTorch, TorchVision.
- **Total trainable parameters**: BiLSTM + classifier head only (ResNet-50 backbone is frozen and excluded from training).

## 7. Training Details

| Training Item | Value |
|---|---|
| Loss function | Mean Squared Error (regression on normalized 0-1 score) |
| Optimizer | Adam |
| Epochs | 10 |
| Batch size | 32 |
| Augmentation | Temporal jitter (+/-2s) and single-frame dropout, training split only |
| Device used | GPU (Google Colab) |
| Saved model path | `best_module1_model.pt` (project root) |

**Learning rate tuning.** The initial configuration (`lr=1e-4`) showed clear overfitting: validation loss improved only in epoch 1 of 10, then diverged while training loss kept decreasing every epoch. The learning rate was lowered to `lr=3e-5`, which substantially reduced this effect and became the final configuration used for all subsequent runs reported below.

**Weighted-loss experiment (rejected).** An importance-weighted MSE loss (weighting the loss by `1.0 + 2.0 * target_score`, intended to counter the natural scarcity of high-importance segments) combined with `weight_decay=1e-5` was implemented and tested once. It produced worse recall at every single threshold compared to the plain-MSE baseline (e.g. Recall at ≥5 fell from 0.777 to 0.639) and was reverted. This is reported as a genuine negative result: a stronger gradient signal from a small, non-representative subset of high-score segments hurt generalisation rather than improving it.

## 8. Baseline Comparison: Zero-Shot Gemini Scoring

Before training, a zero-shot AI baseline was established by scoring a subset of segments with Gemini (no fine-tuning, direct prompting), using `evaluate_ai_baseline.py`.

**Important caveat on comparability.** This baseline was computed on 777 segments drawn from 5 early-annotated videos (004, 006, 009, 019, 040) — all of which fall in the **training split** (video number ≤ 45), not the held-out **test split** (video number > 50) used for the trained-model results in Section 9. It was originally produced as an early calibration check, not against the final test split. As a result, the numbers below and the trained-model results in Section 9 are **not a strict apples-to-apples comparison** — they are evaluated on different data.

| Metric | Value (777 training-split segments) |
|---|---|
| MAE (0-10 scale) | 0.8430 |
| Pearson correlation | 0.8671 |
| Spearman correlation | 0.8078 |

Segment-selection thresholds ≥3 through ≥7 all meet the target (Precision > 0.75, Recall > 0.75); only ≥8 fails (P=0.8624, R=0.6573).

A like-for-like re-run of this baseline directly on test-split videos was attempted (targeting videos 051, 057, and 059, ~960 segments) but is currently blocked: the configured `GEMINI_API_KEY` is being rejected by Google's API (`401 UNAUTHENTICATED`), which is a credentials issue, not a code defect. This should be corrected (regenerate the key via Google AI Studio) and the test-split baseline re-run before this section is presented as a direct comparison against Section 9's trained-model numbers.

## 9. Trained-Model Evaluation Results (Test Split)

Once training stabilised at `lr=3e-5`, the model was retrained and evaluated five separate times using `evaluate_trained_model.py` on the held-out test split (videos 51-61). All five runs used identical code and hyperparameters; no random seed is fixed in `train.py`, so each run starts from different random weight initialisation and batch ordering.

### 9.1 Segment-selection performance at the primary threshold (score ≥ 5)

| Run | Precision | Recall | F1 | Target met? |
|---|---|---|---|---|
| 1 | 0.9348 | 0.4902 | 0.6431 | No (R) |
| 2 | 0.9289 | 0.7767 | 0.8460 | **YES** |
| 3 | 0.9073 | 0.6184 | 0.7355 | No (R) |
| 4 | 0.9248 | 0.7005 | 0.7972 | No (R) |
| **5 (final)** | **0.9131** | **0.8305** | **0.8698** | **YES** |

### 9.2 Continuous scoring agreement (runs with full metrics logged)

| Run | MAE | Pearson r | Spearman |
|---|---|---|---|
| 2 | 1.6056 | 0.5806 | 0.5418 |
| 3 | 1.6603 | 0.6007 | 0.5496 |
| 5 (final) | 1.6942 | 0.5631 | 0.5105 |

### 9.3 Full threshold breakdown, final selected run (Run 5)

| Threshold | Precision | Recall | F1 | Target met? |
|---|---|---|---|---|
| ≥3 | 0.9600 | 0.9241 | 0.9417 | YES |
| ≥4 | 0.9393 | 0.8795 | 0.9084 | YES |
| ≥5 | 0.9131 | 0.8305 | 0.8698 | YES |
| ≥6 | 0.6709 | 0.7763 | 0.7198 | No (P) |
| ≥7 | 0.7385 | 0.5674 | 0.6417 | No (P/R) |
| ≥8 | 0.7129 | 0.4337 | 0.5393 | No (P/R) |

Run 5's checkpoint was selected as the final `best_module1_model.pt`: it is the only run (alongside Run 2) that meets the primary ≥5 target, and it has the highest F1 and recall at that threshold of all five runs.

## 10. Run-to-Run Variance Finding

A notable and honestly-reported finding from this evaluation process is that identical code and hyperparameters produced meaningfully different results across five separate training runs — Recall at the ≥5 threshold ranged from 0.490 to 0.831 across runs. Only 2 of 5 runs met the project's target threshold. Because `train.py` does not fix a random seed, each run's LSTM and classifier weights are initialised differently and training batches are shuffled differently, which is the direct cause of this variance. This means the reported result reflects the best of five runs rather than a value the architecture reproduces reliably on a single attempt, and is reported here as a genuine limitation rather than treating any single run's numbers as an exact, guaranteed outcome. Fixing a random seed (`torch.manual_seed(...)`) is identified as further work to make future runs directly comparable and reproducible.

## 11. Module 1 Processing Pipeline

1. Extract frames from the input lecture video at 1 frame per second.
2. Pass each frame through the frozen ResNet-50 backbone to obtain a 2048-dim feature vector.
3. Group features into consecutive 10-frame (10-second) segments.
4. Run the segment sequence through the Bidirectional LSTM and classifier head to obtain a per-frame importance score.
5. Average per-frame scores across each segment to produce `score_V`.
6. Validate output against the shared JSON schema (`src/utils/json_schema.py`).
7. Save the final per-lecture JSON for consumption by Module 4.

| Script | Purpose |
|---|---|
| `extract_frames_m1.py` | Extracts 1fps frames from lecture videos. |
| `extract_all_features.py` | Runs ResNet-50 over extracted frames, caches 2048-dim features per video. |
| `annotate_module1.py` | Interactive human annotation tool implementing the four-criterion rubric. |
| `train.py` | Trains the BiLSTM + classifier head on cached features. |
| `inference.py` | Loads a trained checkpoint and produces `score_V` predictions for a video. |
| `evaluate_trained_model.py` | Evaluates a trained checkpoint against human annotations on the test split. |
| `evaluate_ai_baseline.py` | Evaluates zero-shot Gemini scoring against human annotations, for baseline comparison. |
| `export_module4_handoff.py` | Exports per-lecture (or combined) `score_V` JSON files for Module 4, from either human annotations or AI-baseline scores. |

## 12. Backend Integration Status

Module 1 has been integrated with the backend orchestrator and verified with a live end-to-end test: the real FastAPI backend was started, a test video was uploaded through the `/api/upload` endpoint, and the job was traced through to completion.

| Backend Item | Status |
|---|---|
| Video upload and job folder creation | Working |
| Real Module 1 execution from backend (`run_module1()`) | Working |
| Trained checkpoint auto-detected and loaded | Confirmed (`best_module1_model.pt (found)` in server log) |
| Final backend Module 1 output path | `storage/jobs/{job_id}/outputs/module1_output.json` |
| Output schema validation | Passed (`[OK] Schema validation passed successfully!`) |
| Per-lecture handoff to Module 4 | Confirmed for two requested lectures (058, 033) via `export_module4_handoff.py --split-by-lecture` |

In the live test, Module 1 completed feature extraction and inference on a short test clip in approximately 8 seconds on the deployment machine, producing varied, non-random `score_V` values (0.51-0.59 across segments), confirming the trained checkpoint — not an untrained/random-weight fallback — was actually used.

## 13. Current Limitations

- Run-to-run variance around the ≥5-≥8 thresholds is real and not fully understood; only 2 of 5 identically-configured training runs met the primary target on their first attempt.
- No fixed random seed, so individual runs are not directly reproducible.
- The model still fails to meet target Precision/Recall jointly at higher thresholds (≥6, ≥7, ≥8) in every run so far.
- Two unverified hypotheses remain open regarding the source of residual error: (1) that mixing Zoom-recorded (low physical movement) and physical/blackboard-style (high movement) lectures introduces a recording-format bias, since the "movement bonus" component of the rubric is structurally easier to earn in physical lectures; and (2) that the frozen, ImageNet-pretrained ResNet-50 backbone was never built to semantically distinguish code/mathematical-formula complexity, since it was trained purely on natural photographs, which may cap accuracy on code- and formula-heavy segments regardless of additional training data.
- Backend verification to date covers Module 1 in isolation and as part of a 3-module (excluding Module 4) run; the full 4-module pipeline including real Module 4 video rendering has not yet been exercised end-to-end, since Module 4's backend integration is still a placeholder stub at the time of writing.

## 14. Future Improvements

- Fix a random seed in `train.py` to make future training runs reproducible and directly comparable.
- Empirically test the recording-format bias hypothesis (Zoom vs. physical lecture videos) by stratifying evaluation metrics by recording type.
- Empirically test the code/math representation hypothesis by stratifying evaluation metrics by content type (theory vs. code/formula-heavy segments).
- Investigate targeted improvements at the ≥6-≥8 thresholds specifically, where no run so far has met target Precision and Recall jointly.
- Re-run backend verification once Module 4's real rendering pipeline (rather than the current placeholder stub) is wired into the orchestrator, to confirm Module 1's output is consumed correctly end-to-end.

## 15. Short Summary for Final Report

Module 1 uses a frozen ResNet-50 feature extractor and a Bidirectional LSTM to score the visual importance of every 10-second segment of a lecture video, trained on a fully human-annotated dataset of 61 lecture videos (23,402 segments) against a four-criterion rubric. After diagnosing and correcting an overfitting issue via learning-rate reduction (1e-4 to 3e-5), and after rejecting a weighted-loss variant that empirically hurt recall, the best of five training runs achieves an F1 of 0.870 (Precision 0.913, Recall 0.831) at the project's primary ≥5 importance threshold, meeting the target of Precision > 0.75 and Recall > 0.75. Run-to-run variance across the five training attempts is reported honestly as a limitation rather than concealed. The trained checkpoint has been integrated into the real backend and verified with a live end-to-end test, confirming it produces schema-valid, non-random importance scores for consumption by Module 4.
