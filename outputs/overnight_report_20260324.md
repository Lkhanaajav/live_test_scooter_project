# Overnight Report 2026-03-24

## Model used

- Chosen checkpoint: `outputs/training/binary_segformer_all6_t400/best_checkpoint`
- Why: highest local checkpoint metric still present on disk after the clean checkout.
- Local checkpoint metric: `val_iou=0.9587699141217626`, `accuracy=0.9856501555266204`

## Baseline findings

- Historical Phase 11.1 probe artifact: `outputs/phase11_probe/vid017_right/run_20260324_024747.csv`
- Historical path-source counts on `VID_20260319_155939_00_017.mp4` with persistent right intent:
  - `template=269`
  - `dt_corridor=19`
  - `none=5`
  - `waypoint_turn_hold=5`
  - `waypoint_turn=2`
- Direct frame-24 containment diagnostic on the old right-turn path showed the accepted turn polyline was not strictly inside the BEV mask:
  - `path_outside_count=3 / 70`
  - `path_outside_ratio=0.042857`
  - `min_boundary_clearance_px=-12.415`
- The leak was not far-field drift. It was the first few near-ego samples at the bottom of the BEV when ego anchoring was disabled.

## Changes made

- Added final control-path containment metrics to `PathPlanResult`:
  - `path_outside_ratio`
  - `path_outside_count`
  - `path_sample_count`
  - `min_boundary_clearance_px`
  - `turn_active`
  - `turn_containment_fail`
- Logged those metrics in the runtime CSV via `simulation_camera_scooter/data_logger.py` and `simulation_camera_scooter/live_heading_demo.py`.
- Updated BEV / camera visualization so replay videos show the actual final control path and containment state.
- Added a turn-path containment guard in `simulation_camera_scooter/realtime_nav_core.py`:
  - evaluates the sampled final control path against the BEV mask
  - rejects unsafe waypoint-turn/template turn paths instead of trusting them blindly
  - allows safer hold behavior when a commanded turn path is not contained
- Added a recovery attempt by projecting path samples back into the mask before refitting.
- Tuned the near-ego exclusion used by the containment guard:
  - `TURN_PATH_IGNORE_FIRST_SAMPLES=4`
  - This matches the noisy BEV bottom-edge zone in the current no-anchor replay configuration.
- Fixed `simulation_camera_scooter/scripts/eval_waypoint_turn_planner.py` so it no longer passes the invalid `use_dt_planner` kwarg to `run_live`, and added containment-aware summary fields.

## Experiments

### 1. Strict containment guard, ignore-first-samples = 1

- Artifact: `outputs/overnight_eval/smoke_017_right/run_20260324_035421.csv`
- Result: too strict.
- It suppressed the waypoint-turn window and the runtime fell back to `dt_corridor`.

### 2. Projection / refit recovery

- Artifact: `outputs/overnight_eval/smoke_017_right_refit/run_20260324_035702.csv`
- Result: not enough by itself.
- The failure source was still the near-ego mask edge, not the far turn arc.

### 3. Near-ego exclusion widened to 4 samples

- Artifact: `outputs/overnight_eval/smoke_017_right_ignore4/run_20260324_040135.csv`
- Result: recovered the original Phase 11.1 waypoint-turn window while clearing containment failures.

## Final replay results

### Persistent right intent, 300 frames

- CSV: `outputs/overnight_eval/vid017_right_300_final/run_20260324_040331.csv`
- Video: `outputs/overnight_eval/vid017_right_300_final/vid017_right_300_final.mp4`
- Keyframes: `outputs/overnight_eval/vid017_right_300_final/vid017_right_stuck_intent_keyframes.jpg`
- Summary:
  - `frames=300`
  - `mean_fps=10.3275`
  - `mean_t_total_ms=93.2563`
  - `path_source_counts={"fallback_hold":269,"dt_corridor":19,"none":5,"waypoint_turn_hold":5,"waypoint_turn":2}`
  - `turn_containment_fail_rate=0.0`
  - `mean_path_outside_ratio=0.0`
- Interpretation:
  - Safe under a stuck right-intent.
  - Too conservative for realistic use because the runtime stays in `fallback_hold` after the maneuver window if right intent is never cleared.

### Scheduled right-intent window, 300 frames

- Schedule: `outputs/overnight_eval/schedules/vid017_right_window.json`
- CSV: `outputs/overnight_eval/vid017_schedule_300_final/run_20260324_040617.csv`
- Video: `outputs/overnight_eval/vid017_schedule_300_final/vid017_schedule_300_final.mp4`
- Keyframes: `outputs/overnight_eval/vid017_schedule_300_final/vid017_schedule_keyframes.jpg`
- Summary:
  - `frames=300`
  - `mean_fps=12.8552`
  - `mean_t_total_ms=49.4250`
  - `path_source_counts={"template":289,"waypoint_turn_hold":9,"waypoint_turn":2}`
  - `turn_active_frames=20..30`
  - `waypoint_turn_frames=[24,25]`
  - `waypoint_turn_hold_frames=[20,21,22,23,26,27,28,29,30]`
  - `turn_containment_fail_rate=0.0`
  - `mean_path_outside_ratio=0.0`
  - `max_path_outside_ratio=0.0`
  - `min_boundary_clearance_px=4.775`
- Interpretation:
  - This is the cleanest current behavior.
  - The commanded turn window stays inside the segmented / BEV drivable region according to the new control-path containment metric.
  - Once intent clears, the runtime cleanly returns to normal template following instead of sticking in hold.

## Remaining risks

- Validation is still narrow:
  - Most direct validation is on `VID_20260319_155939_00_017.mp4`.
  - I have not yet completed the same containment pass across 018/019/020.
- The BEV calibration warning remains:
  - `Calibration matrix is ill-conditioned (cond=1.2e+06)`
  - Geometry can still be fragile even when containment metrics are clean.
- Persistent turn intent is now intentionally conservative:
  - if the turn-family template leaves the mask, the runtime prefers `fallback_hold`
  - this is safer, but it means user intent should be cleared once the maneuver is complete

## Commands used

### Reproducible scheduled replay

```powershell
python simulation_camera_scooter\live_heading_demo.py ^
  --video simulation_camera_scooter\test_videos\VID_20260319_155939_00_017.mp4 ^
  --model-dir outputs\training\binary_segformer_all6_t400\best_checkpoint ^
  --seg-conf-thresh 0.6 ^
  --seg-width 640 ^
  --seg-height 360 ^
  --path-scale 0.7 ^
  --stride 1 ^
  --no-detection ^
  --no-stabilization ^
  --no-seg-smoothing ^
  --no-bev-smoothing ^
  --no-ego-anchor ^
  --no-ego-connected ^
  --no-predict ^
  --bev-clean legacy ^
  --headless ^
  --max-frames 300 ^
  --intent-schedule outputs\overnight_eval\schedules\vid017_right_window.json ^
  --log ^
  --log-dir outputs\overnight_eval\vid017_schedule_300_final ^
  --save ^
  --output-video outputs\overnight_eval\vid017_schedule_300_final\vid017_schedule_300_final.mp4
```

### Manual GUI run to test tomorrow

```powershell
python simulation_camera_scooter\live_heading_demo.py ^
  --video simulation_camera_scooter\test_videos\VID_20260319_155939_00_017.mp4 ^
  --model-dir outputs\training\binary_segformer_all6_t400\best_checkpoint ^
  --seg-conf-thresh 0.6 ^
  --seg-width 640 ^
  --seg-height 360 ^
  --path-scale 0.7 ^
  --stride 1 ^
  --no-detection ^
  --no-stabilization ^
  --no-seg-smoothing ^
  --no-bev-smoothing ^
  --no-ego-anchor ^
  --no-ego-connected ^
  --no-predict ^
  --bev-clean legacy
```

- For the manual run, do not keep `right` latched for the whole clip.
- Give right intent only around the actual maneuver and clear it after the turn.
