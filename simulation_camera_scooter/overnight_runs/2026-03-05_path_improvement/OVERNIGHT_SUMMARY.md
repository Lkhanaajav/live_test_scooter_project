# Overnight Summary (2026-03-05)

## Mission
Improve drivable path generation and selection from the **current imperfect segmentation output** (no retraining), with visual evidence.

## Baseline (E0)
- Video: `test_video_mar3_1_h264.mp4`
- Run: `baseline/logs/run_20260305_173724.csv`
- Baseline output video: `baseline/baseline_mar3_overlay.mp4`
- Key result:
  - `has_path = 0.0%`
  - `has_model_path = 0.0%`
  - `num_paths_mean = 0.0`
- Visual baseline finding:
  - BEV mask often collapses to a tiny near-bottom fragment; graph search produces no valid candidate path.
  - Camera overlay has no drivable path most of the time.

## Changes Implemented
Primary code edits were in `realtime_nav_core.py`:
1. Added fallback path recovery in `BEVPathExtractor` when graph-based extraction fails:
   - row-wise fallback centerline extraction from BEV mask
   - forward extension to minimum usable span
   - short-term previous-path hold for continuity
2. Added lightweight sparse-mask tolerance in preprocessing:
   - relaxed minimum-width enforcement when occupancy is very low
3. Added safety/robustness on fallback outputs:
   - ego-path memory state (`prev_best_path_m`, `no_path_counter`)
   - fallback lateral clamp (`fallback_output_lateral_clip_m`) to prevent persistent one-sided steering lock

## Experiments
- E1 (fallback + hold): major path recovery, but persistent one-sided steering bias (~+22 deg) remained.
- E2 (aggressive recenter): rejected (introduced hard left lock near -30 deg).
- E3 (ego-anchor only): removed E2 failure but still showed large one-sided bias.
- E4 (fallback lateral clamp): best tradeoff tonight.
- E5 (linear fallback fitting): tested after E4 and rolled back (worse heading behavior than E4).

## Best Result (E4)
- Run: `exp4_lateral_clip/logs/run_20260305_180941.csv`
- Video used for visual check: `exp4_lateral_clip/exp4_lateral_clip_mar3_overlay.mp4`
- Visual sample frames: `exp4_lateral_clip/frames/`
- Quantitative comparison on same video:
  - Baseline: `has_path=0.0%`, mean path length `0.0 px`
  - E4: `has_path=100%`, mean path length `73.81 px`
  - E4 mean absolute heading on path frames: `6.62 deg` (vs E1 `22.10 deg`)
  - E4 heading jump p95: `0.575 deg/frame`

## Validation Answers
- Is final chosen path better than baseline?
  - **Yes.** Baseline had no valid path; E4 produces continuous path across the run.
- Is it more stable?
  - **Yes.** No frame-to-frame path disappearance collapse; moderate heading transitions.
- Is it more centered?
  - **Partially.** Better command behavior than E1/E3, but overlay remains left-biased in many frames.
- Is it less sensitive to segmentation holes/noise?
  - **Yes.** Fallback + hold logic prevents all-zero path collapse from sparse/noisy BEV.
- Which exact change helped the most?
  - **Fallback path recovery + previous-path hold (E1 foundation)** was the largest gain.
- Which attempted ideas failed?
  - **E2 aggressive low-confidence recentering** caused steering lock and was rejected.
- What still breaks under bad video?
  - With heavily biased BEV fragments, path overlay can still drift to one side due calibration/projection mismatch.
- What should be improved next when better videos are available?
  - Recalibrate BEV and retune fallback confidence gating with physically aligned videos.

## Current Best Configuration
In `realtime_nav_core.py`:
- `PathExtractorConfig` includes fallback controls (enabled by default).
- `BEVPathExtractor.process()` now:
  - tries graph path first,
  - falls back to mask-derived centerline,
  - then holds previous valid path briefly if needed.
- Fallback lateral output is bounded with `fallback_output_lateral_clip_m=0.25`.

## Re-run Instructions
1. Best current behavior (selected):
   - `python live_heading_demo.py --video test_video_mar3_1_h264.mp4 --save --headless --log --log-dir overnight_runs/2026-03-05_path_improvement/exp4_lateral_clip/logs --no-detection`
2. Baseline reference:
   - `python live_heading_demo.py --video test_video_mar3_1_h264.mp4 --save --headless --log --log-dir overnight_runs/2026-03-05_path_improvement/baseline/logs --no-detection`
3. Tests:
   - `python -m pytest tests -q`

## Artifacts
- Session log: `SESSION_LOG.md`
- Visual findings: `VISUAL_FINDINGS.md`
- Experiment log: `EXPERIMENT_LOG.md`
- Code change log: `CHANGELOG_OVERNIGHT.md`
