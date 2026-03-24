# Must Read: Phase 11.1 Turn Containment

Read this before changing the turn planner or testing a new checkpoint.

## Current validated model

- Runtime default now selects the best local checkpoint on disk by recorded validation IoU from `training_summary.json`.
- In this checkout, that resolves to `outputs/training/binary_segformer_all6_t400/best_checkpoint`.
- Recorded local metric for that checkpoint: `val_iou=0.9587699141217626`, `accuracy=0.9856501555266204`.
- If `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint` comes back, it should win automatically because its handoff metric was `val_iou=0.9602`.

## What changed

- Predictive BEV reuse is blocked during active turn intent and turn-lock behavior.
- The runtime now measures the final committed control path against the actual BEV drivable mask.
- New metrics are logged and visualized:
  - `path_outside_ratio`
  - `path_outside_count`
  - `path_sample_count`
  - `min_boundary_clearance_px`
  - `turn_active`
  - `turn_containment_fail`
- Unsafe commanded-turn paths are no longer trusted just because the planner liked them.
- Recovery order for unsafe turn paths:
  1. check the smoothed/final control path
  2. retry with the raw fitted path
  3. try projection back into the mask and refit
  4. if containment still fails, fall back to hold + slowdown
- `TURN_PATH_IGNORE_FIRST_SAMPLES=4` is intentional for the current replay setup because the first few near-ego samples are noisy when `--no-ego-anchor --no-ego-connected` are used.

## What confidence does

- `approval_confidence` is planner confidence in `[0, 1]`. It is not model accuracy and it is not the same thing as segmentation IoU.
- For waypoint turns, confidence is a weighted blend of support score, containment, near containment, and corridor confidence.
- For templates, confidence is a weighted blend of total template score and corridor confidence, with penalties when approval fails and a reuse floor when reuse is allowed.
- Runtime uses planner confidence to:
  - decide whether a result is low-confidence
  - compute `suggested_slowdown`
  - drive confidence-adaptive path smoothing
- Containment guard is stricter than planner confidence. A path can still be rejected if the final control path leaves the BEV mask.
- Segmentation frame-reuse has its own confidence logic. That path is already disabled during active turns.

## Validated behavior on VID_017

- Main validated clip: `simulation_camera_scooter/test_videos/VID_20260319_155939_00_017.mp4`
- Tracked replay schedule: `simulation_camera_scooter/intent_schedules/vid017_right_window.json`
- Best replay result so far uses a scheduled right-intent window from frames `20` through `30`.
- Result summary for the 300-frame scheduled replay:
  - `path_source_counts={"template":289,"waypoint_turn_hold":9,"waypoint_turn":2}`
  - `waypoint_turn_frames=[24,25]`
  - `waypoint_turn_hold_frames=[20,21,22,23,26,27,28,29,30]`
  - `turn_containment_fail_rate=0.0`
  - `max_path_outside_ratio=0.0`
  - `min_boundary_clearance_px=4.775`
  - `mean_fps=12.8552`
- Important behavior note:
  - if `right` stays latched for the whole clip, the runtime now prefers safe `fallback_hold` behavior once the maneuver window is over because the right-turn template tail does not stay contained
  - that is intentional safety behavior, not a regression

## First test to run

Scheduled replay:

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
  --intent-schedule simulation_camera_scooter\intent_schedules\vid017_right_window.json
```

Manual GUI test:

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

- For the manual test, give right intent only during the actual maneuver and clear it after the turn.

## Remaining risks

- Validation is still concentrated on `VID_017`. `VID_018`, `VID_019`, and `VID_020` still need the same containment pass.
- The current calibration still warns that the homography is ill-conditioned. Geometry can remain fragile even when containment metrics look clean.
- This work makes commanded turns safer, but it is intentionally conservative under stale or stuck intent.

## Local-only evidence

- Detailed CSVs, replay videos, and keyframe strips were saved under `outputs/overnight_eval/`.
- The scratch report from the overnight pass is `outputs/overnight_report_20260324.md`.
- Those `outputs/` artifacts are local evidence and are intentionally not committed.
