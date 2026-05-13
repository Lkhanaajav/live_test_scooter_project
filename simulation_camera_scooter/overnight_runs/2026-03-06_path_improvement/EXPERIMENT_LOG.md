# EXPERIMENT LOG

## E0 - Baseline (current code at start, predictor ON)
- Change: none.
- Command:
  - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/baseline/logs --no-detection`
- Artifacts:
  - Video: `baseline/baseline_june_overlay.mp4`
  - Log: `baseline/logs/run_20260306_011609.csv`
  - Frames: `baseline/frames/`, `baseline/frames_bev/`
- Expected effect: reference for before/after comparisons.
- Actual observed effect:
  - `has_path=100%`
  - `mean |heading|=0.824 deg`, `p95 |heading|=2.693 deg`
  - During collapse window (~frames 1520-1600): steering latched near `-10.62 deg`; `|heading|>8 deg` for 38 frames.
- Decision: kept as baseline.

## E0b - Diagnostic baseline (`--no-predict`)
- Change: disable predictor only.
- Command:
  - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/baseline_no_predict/logs --no-detection --no-predict`
- Artifacts:
  - Video: `baseline_no_predict/baseline_no_predict_june_overlay.mp4`
  - Log: `baseline_no_predict/logs/run_20260306_012206.csv`
- Expected effect: isolate predictor impact.
- Actual observed effect:
  - Severe persistent bias lock: `mean |heading|=19.98 deg`, `p95 |heading|=28.13 deg`.
- Decision: diagnostic reference only.

## E1 - Low-evidence robustness + scoring/continuity improvements (SELECTED)
- Change set:
  - `bev_predictor.py`: occupancy-aware skip gating, empty-prediction path invalidation, stale path-model clearing.
  - `realtime_nav_core.py`: low-evidence detection, aggressive hold decay for near-empty BEV, center/continuity candidate scoring, bounded discontinuity hold/reacquire.
  - `live_heading_demo.py` + `data_logger.py` + `visualization.py`: path source + BEV occupancy diagnostics logged/overlaid.
  - `tests/test_bev_predictor.py`: updated/added tests for occupancy guard behavior.
- Commands:
  - Test: `py -m pytest tests -q`
  - Validation run: `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/exp1_lowconf_guard/logs --no-detection`
- Artifacts:
  - Video: `exp1_lowconf_guard/exp1_june_overlay.mp4`
  - Log: `exp1_lowconf_guard/logs/run_20260306_013930.csv`
  - Frames: `exp1_lowconf_guard/frames/`
- Expected effect:
  - reduce stale steering lock during BEV collapse,
  - improve center/continuity preference,
  - maintain overall stability.
- Actual observed effect:
  - `mean |heading|`: `0.824 -> 0.690 deg`
  - `p95 |heading|`: `2.693 -> 2.407 deg`
  - `p95 heading jump`: `0.546 -> 0.538 deg/frame`
  - collapse-window `|heading|>8` frames: `38 -> 2`
  - collapse-window mean `|heading|`: `5.85 -> 2.58 deg`
- Decision: selected best.

## E1b - Post-change diagnostic (`--no-predict`)
- Change: none beyond E1; validation mode only.
- Command:
  - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/exp1_no_predict_check/logs --no-detection --no-predict`
- Artifacts:
  - Video: `exp1_no_predict_check/exp1_no_predict_june_overlay.mp4`
  - Log: `exp1_no_predict_check/logs/run_20260306_014316.csv`
- Actual observed effect:
  - `mean |heading|`: `19.98 -> 0.776 deg`
  - `p95 |heading|`: `28.13 -> 2.646 deg`
  - confirms discontinuity lock issue was largely removed.
- Decision: supports keeping E1 changes.

## E2 - Low-confidence fallback tuning pass (REJECTED)
- Change:
  - tuned `fallback_low_conf_prev_blend`, `fallback_recenter_gain`, `fallback_low_conf_max_abs_lateral_m`.
- Command:
  - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/exp2_lowconf_tune/logs --no-detection`
- Artifacts:
  - Video: `exp2_lowconf_tune/exp2_june_overlay.mp4`
  - Log: `exp2_lowconf_tune/logs/run_20260306_015033.csv`
- Actual observed effect:
  - Regression vs E1: `mean |heading|` and `p95 |heading|` increased; new spikes near `-8.9 deg` around frame ~1730.
- Decision: rejected and reverted.

## E3: Skeleton-Geodesic Fallback Candidate + Selection Override (2026-03-06 afternoon)

### E3.1 Skeleton geodesic fallback candidate
- **Change**: Added `_fallback_skeleton_geodesic()` in `realtime_nav_core.py`:
  - Builds shortest-path tree over skeleton pixels from ego-near start pixel.
  - Picks far endpoint by forward-progress minus center penalty.
  - Converts to metric polyline and ego-anchors.
- **Why**: Dijkstra was active but often only produced one short branch candidate due graph fragmentation; needed a pure-skeleton path alternative.
- **Expected**: Better branch-end reach and fewer early false turns.
- **Observed**: Candidate reach improved in isolated diagnostics (e.g., frame 350 fallback skeleton path span near full forward extent), but full-run selection still frequently chose graph path at failure windows.
- **Result**: **Kept** as a fallback/candidate source.

### E3.2 Skeleton selected before centerline fallback
- **Change**: In `process()`, added skeleton fallback stage before centerline fallback when graph path fit fails.
- **Why**: Prefer geometry from skeleton over mask-only centerline when graph is weak.
- **Observed**: On targeted frame checks, frame 350 switched to `fallback_skeleton`; however full long-run still had major graph-selected spikes at frames 350/355/360.
- **Result**: **Kept**, but not sufficient alone.

### E3.3 Targeted graph-vs-skeleton override
- **Change**: Added override rule to force skeleton candidate when selected graph candidate is short-progress and laterally biased near ego while skeleton near-ego lateral is centered.
- **Why**: Specifically target false early branch turns from near-ego triangle artifacts.
- **Observed**: In short sequential diagnostics, override can trigger; in full run (`verify_no_predict_skel_override`) major spikes at 350/355/360 remained, indicating condition still not firing in those states.
- **Result**: **Kept (inconclusive)**.

### E3.4 Skeleton hold strategy (rejected)
- **Change**: Added temporary multi-frame hold when `fallback_skeleton` selected.
- **Why**: Prevent immediate snap-back from skeleton to short graph branch.
- **Observed**: Caused broad steering bias drift and much worse overall stability:
  - mean|heading| ~2.532, p95 ~6.61, max ~21.58.
- **Result**: **Rejected** and reverted.

### E3.5 Comparative runs
- `verify_no_predict_skel_fallback/logs/run_20260306_130808.csv`
  - mean|heading| 1.010, p95 2.997, max 26.862
  - sources: graph 1861, fallback_skeleton 109, fallback_centerline 48, fallback_hold 39
- `verify_no_predict_skel_override/logs/run_20260306_132732.csv`
  - mean|heading| 1.011, p95 2.961, max 26.862
  - sources: graph 1856, fallback_skeleton 117, fallback_centerline 45, fallback_hold 39
- Baseline comparator for low-spike behavior:
  - `verify_no_predict_fbcand_cond/logs/run_20260306_125128.csv`
  - mean|heading| 1.337, p95 2.271, max 7.420 (but over-relied on fallback_centerline)

### E3.6 Near-ego branch-angle/lateral score penalty (rejected)
- **Change**: Added near-ego short-candidate heading/lateral penalty terms in `_score_candidates`.
- **Why**: Attempted to suppress tiny-triangle false turn capture near start.
- **Observed**: Helped frame 230 transiently but did not fix 350/355/360 and worsened late spikes (e.g., frame 1530).
- **Result**: **Rejected** and reverted from code.
