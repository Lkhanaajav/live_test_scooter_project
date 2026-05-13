# Experiment Log

## E0 - Baseline capture
- Change: none.
- Why: establish true starting behavior.
- Command:
  - `python live_heading_demo.py --video test_video_mar3_1_h264.mp4 --save --headless --log --log-dir overnight_runs/2026-03-05_path_improvement/baseline/logs --no-detection`
- Expected effect: reference for before/after.
- Actual observed effect:
  - `has_path=0.0%`
  - no candidate paths across all frames.
  - BEV mask often too sparse near ego only.
- Decision: kept as baseline reference.

## E1 - Fallback centerline + path hold
- Change:
  - Added fallback path generation from sparse BEV mask rows.
  - Added short-term hold of previous valid path.
  - Relaxed sparse-mask min-width preprocessing.
- Why: baseline complete path starvation.
- Expected effect: recover continuous drivable path under sparse/noisy masks.
- Actual observed effect:
  - `has_path=99.74%`
  - path is continuous, but steering bias remained high (one-sided tendency; mean |heading| ~22.1 deg).
- Decision: concept kept, needed refinement.

## E2 - Aggressive confidence recentering
- Change:
  - Strong low-confidence recentering and continuity gating.
- Why: reduce E1 one-sided bias.
- Expected effect: centered path under weak BEV.
- Actual observed effect:
  - severe regression: persistent steering saturation near `-30 deg`.
- Decision: rejected.

## E3 - E2 rollback + ego anchor
- Change:
  - Rolled back aggressive E2 behavior.
  - Anchored fallback path near ego center.
- Why: preserve E1 recovery while removing E2 regression.
- Expected effect: remove side lock, keep path availability high.
- Actual observed effect:
  - path availability remained high, but one-sided bias still large (mean |heading| ~22.6 deg).
- Decision: partial; not final.

## E4 - Fallback lateral clip (selected)
- Change:
  - Added `fallback_output_lateral_clip_m` and applied clip in fallback/hold paths.
- Why: prevent persistent large steering offsets from weak BEV fragments.
- Expected effect: keep 100% path availability with safer, more centered steering behavior.
- Actual observed effect:
  - `has_path=100%`
  - mean |heading| reduced to `6.62 deg`
  - heading jump p95 `0.575 deg/frame`
  - visually stable path with no prolonged +/-30 deg lock.
- Decision: selected best overnight configuration.

## E5 - Linear fallback model (rolled back)
- Change:
  - Replaced fallback/hold path fitting with a linear model for low-confidence cases.
- Why: test whether simpler fitting reduces curvature artifacts on weak masks.
- Expected effect: lower heading oscillation and fewer curve-induced offsets.
- Actual observed effect:
  - No availability gain over E4 (`has_path` already 100%).
  - Mean absolute heading increased versus E4 on the March clip.
  - Visual steering quality degraded relative to E4.
- Decision: rejected and reverted; final code keeps E4 behavior.

## Cross-experiment metrics (same video)
- baseline: has_path `0.0%`, mean |heading| N/A
- E1: has_path `99.74%`, mean |heading| `22.10 deg`
- E2: has_path `100%`, mean |heading| `29.96 deg` (rejected)
- E3: has_path `100%`, mean |heading| `22.57 deg`
- E4: has_path `100%`, mean |heading| `6.62 deg` (selected)

## E6 - Replace bottom rectangle with thin ego bridge
- Change:
  - Replaced synthetic bottom rectangle anchor with a thin line bridge from ego point to nearest nearby BEV mask pixel.
  - Moved anchor stage to run after clean_sidewalk_mask and before ego_connected_mask.
- Why:
  - Rectangle anchor was creating fake wings / artificial start curvature.
- Expected effect:
  - Preserve connectivity without injecting wide artificial geometry near ego.
- Actual observed effect (June run, 438 frames):
  - mean |heading| reduced from 1.022 -> 0.696 deg (same-length comparison).
  - p95 |heading| reduced from 4.063 -> 3.011 deg.
  - stab_corr_deg stats unchanged (suggesting improvement came from mask topology, not stabilizer change).
- Decision:
  - kept.
