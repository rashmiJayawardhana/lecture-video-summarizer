# Calibration Report — LecVideo 045

## Summary

| Metric | Count |
|--------|------:|
| Total segments | 289 |
| ✅ Exact agreement (diff = 0) | 180 (62%) |
| ✅ Acceptable (diff = 1) | 36 (12%) |
| ⚠️ Needs review (diff = 2) | 18 (6%) |
| ❌ Critical divergence (diff ≥ 3) | **55 (19%)** |

Overall: **74% of segments are aligned** (diff ≤ 1). **26% need discussion**.

---

## Pattern Observations

### 🔴 Fathhy scored the intro as 0 (segments 0–18)
Fathhy scored segments 0–18 as `0` while everyone else gave `1`. Per the guide, **Score 0 = nothing at all on screen** (black frame, logo), while **Score 1 = lecturer visible, slide empty/decorative**. If the lecturer is on camera in those segments, the correct score is **1**, not 0. Fathhy should re-check these.

### 🔴 Rashmi over-scores content slides (+2 to +5 above others)
Rashmi's scores are consistently the highest — often 7–9 where others score 3–6. Likely cause: **not muting audio**. The guide says to mute the audio and score only what is visually on screen. Rashmi appears to be letting the lecturer's speech inflate the visual content score. This needs recalibration.

### 🔴 Ravindu under-scores content slides (segments 163–189 area)
Ravindu scores segments like 163–171 as `1` or `3` where others give `5–8`. These appear to be the WBS/Project Charter slides which are genuine content slides (base 5–8). Ravindu may be scoring the lecturer-only moments from the filmstrip start instead of the main slide content.

---

## Critical Divergences (diff ≥ 3) — Full List

| Seg | Fathhy | Lathisana | Rashmi | Ravindu | Diff | Notes |
|----:|-------:|----------:|-------:|--------:|-----:|-------|
| 3 | 0 | 1 | 2 | 3 | 3 | Intro/title area. Correct answer: **1** (lecturer visible, minimal content) |
| 26 | 2 | 1 | 3 | 3 | 2 | Section transition. Correct: **2–3** |
| 44 | 7 | 7 | 9 | 1 | 8 | Core diagram. Ravindu far too low. Correct: **8–9** |
| 45 | 7 | 1 | 9 | 9 | 8 | Same core diagram. Lathisana far too low. Correct: **8–9** |
| 50 | 8 | 8 | 9 | 5 | 4 | Full content slide. Ravindu too low. Correct: **8–9** |
| 51 | 7 | 7 | 9 | 5 | 4 | Full content slide. Ravindu too low. Correct: **7–9** |
| 64 | 5 | 5 | 9 | 3 | 6 | Diagram slide. Rashmi over-scored, Ravindu under-scored. Correct: **5–6** |
| 68–71 | 6 | 6 | 7 | 3 | 4 | Project Charter cells. Ravindu far too low. Correct: **6–7** |
| 72–74 | 6 | 3–6 | 6 | 2–3 | 3–5 | Partially filled template. Correct: **5–6** |
| 102–103 | varies | varies | varies | varies | 3+ | Check individually |
| 109–110 | varies | varies | varies | varies | 3+ | Check individually |
| 132–136 | varies | varies | varies | varies | 3+ | Complex diagram segments — Rashmi likely over-scoring |
| 152–155 | 1–4 | 2–6 | 8 | 3–4 | 4–7 | WBS diagrams. Rashmi over-scores. Correct: **7–8** per guide |
| 163–171 | 5 | 4–5 | 8 | 1–3 | 4–7 | WBS Dictionary. Ravindu under-scores. Correct: **5–7** |
| 173 | 0 | 2 | 8 | 1 | 8 | **Biggest divergence.** Correct answer needs visual check |
| 185–189 | 1–7 | 1–7 | 6–7 | 1–9 | 3–8 | Ravindu/Rashmi both off. Needs group review |
| 205–208 | 1–4 | 1–5 | 7–9 | 1–9 | 5–8 | Rashmi over-scores. Correct: **3–6 range** |
| 226–229 | 1–3 | 1–3 | 6–7 | 3–4 | 3–5 | Rashmi over-scores |
| 257–259 | 3 | 3 | 7–8 | 1–4 | 4–5 | Rashmi over-scores |
| 263–264 | 1–5 | 3–4 | 8 | 1–3 | 4–7 | Rashmi over-scores |

---

## ⚠️ Needs Review (diff = 2) — Summary

Segments: `26, 32, 56, 57, 159, 174, 175, 212, 217, 221, 223, 245, 249, 256, 275, 287`

These are borderline cases — mostly section transition slides or partially visible diagrams. Acceptable to decide by majority vote.

---

## Per-Annotator Feedback

### Fathhy
- **Issue**: Scored segments 0–18 as `0` — likely used score 0 for "quiet/filler opening" but the rule is `0 = literally nothing on screen`. If the lecturer is visible, it's at minimum a `1`.
- **Action**: Re-check and re-score segments 0–18. Likely all should be `1`.

### Lathisana
- **Issue**: Segment 45 scored `1` (a major divergence — others scored 7–9). This is a core diagram segment.
- **Action**: Re-check segment 45 and surrounding segments 44–51.

### Rashmi
- **Issue**: Consistently the highest scorer. Likely including audio content in visual scores. Segments 64, 152–155, 163–171, 173, 205–208 appear inflated by 2–5 points.
- **Action**: **Mute the video** and re-score all segments marked DIVERGE. The audio must not influence the score.

### Ravindu
- **Issue**: Segments 44, 45, 50, 51, 68–74, 163–171 significantly under-scored. May be scoring the lecturer-only opening frame rather than the main slide content shown in the middle/end of the segment.
- **Action**: Re-check these segments using all three filmstrip thumbnails (start, middle, end), not just the first frame.

---

## Recommended Next Steps

1. **Share this report** with the team in your group chat.
2. **Rashmi** re-annotates all `DIVERGE` segments with audio muted.
3. **Ravindu** re-checks the WBS/Project Charter segments (44–74, 163–189) using the filmstrip thumbnails.
4. **Fathhy** re-checks segments 0–18 (change all `0` to `1` if lecturer is visible).
5. **Lathisana** re-checks segment 45.
6. Hold a quick 15-minute group call to agree on the **canonical scores for the top 10 DIVERGE segments** (44, 45, 50, 51, 64, 68–74, 152–155, 173).
7. Once agreed, Rashmi runs `--merge` again and uploads the updated `module1_annotations.json` to Google Drive.
