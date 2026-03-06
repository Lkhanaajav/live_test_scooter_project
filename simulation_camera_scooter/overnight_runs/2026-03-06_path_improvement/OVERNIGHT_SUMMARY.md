# OVERNIGHT SUMMARY

## Mission
Improve drivable path generation/selection from current imperfect segmentation output (no retraining), with evidence from baseline-vs-after runs and visual inspection.

## Baseline Problems
- Baseline (predict ON) was mostly stable overall, but had a critical failure mode:
  - in collapse windows (~frames 1520-1600), BEV occupancy dropped sharply,
  - steering stayed latched around `-10.6 deg` for too long.
- Diagnostic no-predict run exposed severe stale-lock behavior (`mean |heading| ~19.98 deg`).

## Best Improvements Found (E1 - Selected)
1. Predictor occupancy guard + stale path invalidation.
2. Low-evidence aggressive fallback-hold decay.
3. Center/continuity candidate scoring in path selection.
4. Bounded discontinuity-hold reacquire in pure pursuit.
5. Added `path_source` + BEV occupancy diagnostics to logs/HUD for transparent debugging.

## Validation Results
Using the same video (`test_video_june_03_3.mp4`):

### Baseline (predict ON)
- Log: `baseline/logs/run_20260306_011609.csv`
- `mean |heading| = 0.824 deg`
- `p95 |heading| = 2.693 deg`
- collapse window (1520-1600):
  - max `|heading| = 10.62 deg`
  - `|heading| > 8` on 38 frames

### E1 (predict ON, selected)
- Log: `exp1_lowconf_guard/logs/run_20260306_013930.csv`
- `mean |heading| = 0.690 deg`
- `p95 |heading| = 2.407 deg`
- collapse window (1520-1600):
  - max `|heading| = 8.10 deg`
  - `|heading| > 8` on 2 frames
- Path-source distribution:
  - `predict: 1457`, `graph: 523`, `fallback_centerline: 55`, `fallback_hold: 22`

### Diagnostic no-predict comparison
- Before (baseline_no_predict):
  - `mean |heading| = 19.98 deg`, `p95 |heading| = 28.13 deg`
- After E1 (exp1_no_predict_check):
  - `mean |heading| = 0.776 deg`, `p95 |heading| = 2.646 deg`

## What Worked Best
- Occupancy-aware stale-path suppression + aggressive low-evidence hold decay.
- This combination materially reduced prolonged wrong steering when BEV evidence collapsed.

## What Did Not Work
- E2 low-confidence recenter parameter tuning introduced new spikes (up to ~8.9 deg around frame ~1730) and worsened p95 metrics.
- E2 was rejected and reverted.

## Current Best Configuration
- Code state corresponds to E1-selected changes (post-E2 revert).
- Core files: `bev_predictor.py`, `realtime_nav_core.py`, `live_heading_demo.py`, `visualization.py`, `data_logger.py`.

## Rerun Instructions
1. Selected best run (predict ON):
   - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/exp1_lowconf_guard/logs --no-detection`
2. Baseline reference:
   - `python live_heading_demo.py --video test_video_june_03_3.mp4 --save --headless --log --log-dir overnight_runs/2026-03-06_path_improvement/baseline/logs --no-detection`
3. Tests:
   - `py -m pytest tests -q`

## BEV Assessment
- Is the BEV transform working reasonably?
  - Yes on this video; mean survival in baseline was ~63% and usable in most frames.
- Main BEV failure modes:
  - intermittent occupancy collapse and fragmentation.
- How much do BEV failures hurt path planning?
  - Significantly; they caused stale steering lock in baseline.
- What BEV improvements were attempted?
  - occupancy-gated predictor skip logic,
  - low-evidence fallback hold decay,
  - source/occupancy diagnostics.
- Which BEV fix helped most?
  - occupancy-gated predictor + low-evidence hold decay.
- Is final BEV more stable/usable than baseline?
  - Practically yes: better recovery and less prolonged bias in collapse windows.

## Paper/Image Comparison
- Which paper/report images were inspected?
  - `bev_mask_raw.png`, `bev_clean.png`, `bev_skeleton.png`, `skeleton_paths_overlay.png`, `planned_vs_skeleton_overlay.png`, `cam_paths_0001.png`.
- What looked better there?
  - cleaner contiguous BEV masks, dominant center trunk, branch selection without edge drift.
- What concrete clues were learned?
  - prioritize center continuity and avoid trusting stale paths under weak evidence.
- What changes were attempted from those clues?
  - center/continuity scoring, low-evidence mode behavior, explicit source/confidence diagnostics.
- Did they help?
  - Yes, quantitatively and visually (E1 selected).

## Similar Work Review Summary
- Outside ideas that influenced implementation:
  - robust path extraction from noisy segmentation (distance-transform/branch filtering concepts),
  - pure pursuit reacquisition behavior,
  - practical OpenCV morphology/DT/CC guidance.
- Useful adopted/adapted ideas:
  - occupancy as confidence gate,
  - temporal persistence with bounded decay/reacquire,
  - center/continuity-aware path selection.
- Too heavy for tonight:
  - full dense inverse-distance global search each frame.
- Most impactful research-inspired change:
  - confidence-aware fallback/persistence control during BEV collapse.

## Afternoon Continuation (Deep Dijkstra / Skeleton Investigation)

### What was verified
- Dijkstra is running (including skip-frame path now in prior fix), but in hard windows the selected start component can still expose only one practical graph candidate.
- Deep frame diagnostics showed mismatch between visible skeleton extent and selected graph candidate length at branch-entry failure frames.

### New changes tested
1. Pure skeleton geodesic fallback path (`fallback_skeleton`) added and integrated into planner fallback chain.
2. Targeted graph-vs-skeleton override rule for short, laterally-biased near-ego graph paths.
3. Temporary skeleton-hold logic (reverted; too much global bias).

### Validation runs added
- `verify_no_predict_skel_fallback` (`run_20260306_130808.csv`)
- `verify_no_predict_skel_hold` (`run_20260306_131916.csv`) [rejected strategy]
- `verify_no_predict_skel_override` (`run_20260306_132732.csv`)

### Metric snapshot
- `verify_no_predict_skel_override`:
  - mean|heading|: 1.011 deg
  - p95|heading|: 2.961 deg
  - max|heading|: 26.862 deg
  - path_source: graph 1856 / fallback_skeleton 117 / fallback_centerline 45 / fallback_hold 39
- Comparator (`verify_no_predict_fbcand_cond`):
  - mean|heading|: 1.337 deg
  - p95|heading|: 2.271 deg
  - max|heading|: 7.420 deg
  - but heavily fallback-dominated (fallback_centerline 1916, graph 102)

### Current interpretation
- If priority is strict suppression of spikes tonight, `verify_no_predict_fbcand_cond` still gives best max-heading behavior but relies heavily on fallback centerline.
- If priority is keeping graph/Dijkstra active, `verify_no_predict_skel_override` preserves graph usage but still has unresolved spike windows (notably around frames 350/355/360 and 1460).

### Best-known unresolved root issue
- Branch-entry false turns are caused by under-constrained candidate selection near ego branch artifacts: graph path can still beat skeleton fallback in full stateful run at critical frames.
