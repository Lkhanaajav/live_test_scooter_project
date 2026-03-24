# Runtime Runbook

This file is the single handoff for the current BEV + segmentation runtime.
If another AI or engineer picks this up later, start here.

## Current Runtime Goal

- Default no-intent behavior: follow the drivable corridor smoothly.
- Intent behavior: accept `left` / `right` turn intent from manual keys now, later GPS.
- Keep recent research/planner work intact and reversible.

## Current Important Files

- Runtime entry: [live_heading_demo.py](simulation_camera_scooter/live_heading_demo.py)
- Planner core: [realtime_nav_core.py](simulation_camera_scooter/realtime_nav_core.py)
- BEV calibration loader: [bev_calibration.py](simulation_camera_scooter/bev_calibration.py)
- Calibration preview tool: [calibrate_bev_examples.py](simulation_camera_scooter/scripts/calibrate_bev_examples.py)
- Shared constants: [config.py](simulation_camera_scooter/config.py)
- HUD drawing: [visualization.py](simulation_camera_scooter/visualization.py)

## Current Model

Use the latest and best segmentation checkpoint (val IoU 0.9602, trained 2026-03-19):

- `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`

This model was fine-tuned from `old400_plus_img_1931` on 1419 frames spanning 9 videos including the newest VID_017–020 1080p campus clips. It supersedes both `oneformer_teacher` (IoU 0.9437) and `all6_t400` (IoU 0.9588).

## Current BEV Geometry

Current committed runtime geometry in [config.py](simulation_camera_scooter/config.py):

- `BEV_SIZE = (360, 660)`
- `NAV_BEV_LATERAL_M = 6.0`
- `NAV_BEV_FORWARD_M = 11.0`

This is a taller BEV than earlier experiments.

## Current Calibration

Current committed calibration metadata in [bev_calibration_meta.json](simulation_camera_scooter/bev_calibration_meta.json):

- source resolution: `1920x1080`
- BEV resolution: `360x660`
- source frame id: `900`

The committed calibration is meant for the newer `1920x1080` test videos, not the earlier `IMG_1931` iPhone clip.

## How To Recalibrate

Use the frame-based preview tool, not the old blind click flow.

Run:

```powershell
python simulation_camera_scooter\scripts\calibrate_bev_examples.py --examples-dir outputs\evaluation\new_videos_manual_bev
```

Controls:

- `n` next frame
- `p` previous frame
- `r` reset points
- `s` save calibration
- `q` quit

What it does:

- left side: segmented camera frame
- right side: live warped BEV preview

Save only when the preview BEV looks correct for the active `BEV_SIZE` in [config.py](simulation_camera_scooter/config.py).

## Recommended Debug Run

This is the current honest debug command. It avoids predictor reuse and skip-frame reuse:

```powershell
python simulation_camera_scooter\live_heading_demo.py ^
  --video simulation_camera_scooter\test_videos\VID_20260319_155939_00_017.mp4 ^
  --model-dir outputs\training\binary_segformer_old400_img1931_vid017_020\best_checkpoint ^
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

Notes:

- `--no-predict` alone is not enough if `stride > 1`. Use `--stride 1` for a truly no-reuse run.
- This run is intentionally slow. It is for correctness/debug, not demo smoothness.

## Controls

The live window currently uses arrow keys:

- `Up` = straight
- `Left` = left intent
- `Right` = right intent
- `Down` = clear intent
- `q` or `Esc` = quit

The overlay also shows the active intent.

## Current Planner Behavior

### No Intent

No-intent mode should generally look best.

- preferred path source: `dt_corridor`
- reason: it uses distance-transform safe-corridor extraction from the BEV mask and usually gives the cleanest centerline

### With Intent

Intent mode is experimental and under active iteration.

Current design goal:

- no intent: use normal corridor following
- left/right intent: override toward a turn target suitable for future GPS turn commands

Current manual-intent path priority:

1. `manual_endpoint`
2. `manual_dt_corridor`
3. `manual_locked`
4. non-skeleton fallbacks only if needed

Important:

- manual intent should not rely on skeleton as the primary source
- if it visually falls back into ugly skeleton-like turning, treat that as a bug/regression

## What `dt_corridor` Means

`dt_corridor` in [safe_corridor.py](simulation_camera_scooter/safe_corridor.py):

- computes distance transform on the BEV drivable mask
- treats high-clearance pixels as low cost
- runs forward Dijkstra from ego
- smooths that corridor centerline

This is why straight/no-intent mode often looks much better than branch/skeleton-based turning.

## Current Known Issues

- Manual turn behavior is still not fully satisfactory.
- Endpoint-based turn override is still being tuned.
- The runtime is slow in full debug mode because segmentation still dominates runtime.
- Lowering segmentation resolution changes visible mask quality, so do not do that casually.
- `stride > 1` can still make the overlay look like it is reusing old results even when predictor is off.

## What Not To Do

- Do not assume a new BEV size can reuse an old calibration.
- Do not use skeleton as the preferred manual turn planner.
- Do not change segmentation resolution just to get FPS unless you accept mask quality changes.
- Do not assume `--no-predict` means zero reuse if `stride` is greater than `1`.

## Good Next Steps

If continuing work later, the safest order is:

1. Validate calibration on the current BEV size.
2. Validate no-intent `dt_corridor` behavior first.
3. Tune intent-turn override separately from straight mode.
4. Only after behavior is correct, work on speed/demo mode.

## Git Scope For This Work

If pushing runtime changes, include these tracked files:

- [bev_calibration.py](simulation_camera_scooter/bev_calibration.py)
- [bev_calibration.npy](simulation_camera_scooter/bev_calibration.npy)
- [bev_calibration_meta.json](simulation_camera_scooter/bev_calibration_meta.json)
- [config.py](simulation_camera_scooter/config.py)
- [live_heading_demo.py](simulation_camera_scooter/live_heading_demo.py)
- [realtime_nav_core.py](simulation_camera_scooter/realtime_nav_core.py)
- [calibrate_bev_examples.py](simulation_camera_scooter/scripts/calibrate_bev_examples.py)
- [visualization.py](simulation_camera_scooter/visualization.py)
- [RUNTIME_RUNBOOK.md](simulation_camera_scooter/RUNTIME_RUNBOOK.md)

Do not push local-only artifacts by accident:

- `outputs/evaluation/`
- `simulation_camera_scooter/test_videos/` local added clips unless intentionally committing them
- `segmentation_preview.py` unless you explicitly want that tool in repo
- `bev_H.npy`
- `bev_Hinv.npy`
