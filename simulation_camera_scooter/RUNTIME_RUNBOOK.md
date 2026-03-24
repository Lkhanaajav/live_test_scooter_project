# Runtime Runbook

This file records the current accepted runtime configuration and the exact latest full run artifact.

## Current Accepted Run

Date:
- `2026-03-24`

Video:
- `simulation_camera_scooter/test_videos/VID_20260319_155939_00_017.mp4`

Segmentation model:
- `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint`

Exact command:

```powershell
python simulation_camera_scooter\live_heading_demo.py ^
  --video simulation_camera_scooter\test_videos\VID_20260319_155939_00_017.mp4 ^
  --model-dir outputs\training\binary_segformer_oneformer_teacher\best_checkpoint ^
  --seg-conf-thresh 0.6 ^
  --seg-width 640 ^
  --seg-height 360 ^
  --path-scale 0.7 ^
  --stride 4 ^
  --no-detection ^
  --no-predict ^
  --save ^
  --headless ^
  --output-video outputs\evaluation\vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun.mp4 ^
  --log ^
  --log-dir outputs\evaluation\vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun_logs
```

## Exact Artifact

Video:
- [vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun.mp4](/C:/Users/miji0000/Desktop/thesis_prep/outputs/evaluation/vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun.mp4)

Log:
- [run_20260324_163125.csv](/C:/Users/miji0000/Desktop/thesis_prep/outputs/evaluation/vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun_logs/run_20260324_163125.csv)

Metadata:
- [run_20260324_163125_meta.json](/C:/Users/miji0000/Desktop/thesis_prep/outputs/evaluation/vid_20260319_155939_00_017_teacher640_stride4_headless_recal_nopredict_overlay2_full_rerun_logs/run_20260324_163125_meta.json)

## Run Summary

- `1800` frames processed
- mean loop FPS: `9.968`
- mean fresh segmentation Hz: `2.509`
- mean total pipeline time: `69.51 ms`
- mean segmentation IoU: `0.9536`
- `has_path`: `100%`
- `path_source`: `template` on all `1800` frames
- mean best path length: `480 px`

Interpretation:
- This run is `stride 4` and `--no-predict`, so predictor reuse is off.
- Fresh segmentation is still lower than loop FPS because stride skipping is still active.
- The rendered path now shows a transparent suggested path with the controller direction drawn strongly on top.

## Current Calibration

Active calibration files:
- [bev_calibration.npy](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration.npy)
- [bev_calibration_meta.json](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration_meta.json)

Current calibration facts:
- source resolution: `1920x1080`
- source frame id: `900`
- BEV size: `360x660`
- current homography condition number: about `5.99e5`

Active source points:
- bottom-left: `(277, 880)`
- bottom-right: `(1776, 880)`
- top-right: `(1171, 460)`
- top-left: `(745, 460)`

Calibration tooling:
- runtime loader: [bev_calibration.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration.py)
- preview tool: [calibrate_bev_examples.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/scripts/calibrate_bev_examples.py)

Important:
- the runtime loader now accepts both `source_frame_width/source_frame_height` and the older `source_width/source_height` metadata keys
- the preview tool now writes the runtime key names the loader expects

## Current Behavior

Controls:
- `Up` or `s` = straight
- `Left` or `l` = left
- `Right` or `r` = right
- `Down` or `c` = clear
- `q` or `Esc` = quit

Overlay behavior:
- full planned path is drawn as a transparent suggestion
- the controller lookahead ray and target are drawn on top as the strong active command

Planner behavior:
- active runtime family: `dt_ridge`
- template planner: enabled
- no predictive reuse in this accepted run
- turn-intent frames still force fresh compute rather than holding reused results

## Current Important Files

- runtime entry: [live_heading_demo.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/live_heading_demo.py)
- planner core: [realtime_nav_core.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/realtime_nav_core.py)
- waypoint-turn planner: [waypoint_turn_planner.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/waypoint_turn_planner.py)
- HUD drawing: [visualization.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/visualization.py)
- calibration loader: [bev_calibration.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration.py)
- calibration preview tool: [calibrate_bev_examples.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/scripts/calibrate_bev_examples.py)
- logger: [data_logger.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/data_logger.py)

## Git Scope

Tracked files that belong with this runtime state:
- [simulation_camera_scooter/bev_calibration.npy](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration.npy)
- [simulation_camera_scooter/bev_calibration.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration.py)
- [simulation_camera_scooter/bev_calibration_meta.json](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/bev_calibration_meta.json)
- [simulation_camera_scooter/data_logger.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/data_logger.py)
- [simulation_camera_scooter/live_heading_demo.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/live_heading_demo.py)
- [simulation_camera_scooter/realtime_nav_core.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/realtime_nav_core.py)
- [simulation_camera_scooter/scripts/calibrate_bev_examples.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/scripts/calibrate_bev_examples.py)
- [simulation_camera_scooter/tests/test_realtime_nav_core.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/tests/test_realtime_nav_core.py)
- [simulation_camera_scooter/tests/test_waypoint_turn_planner.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/tests/test_waypoint_turn_planner.py)
- [simulation_camera_scooter/visualization.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/visualization.py)
- [simulation_camera_scooter/waypoint_turn_planner.py](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/waypoint_turn_planner.py)
- [simulation_camera_scooter/RUNTIME_RUNBOOK.md](/C:/Users/miji0000/Desktop/thesis_prep/simulation_camera_scooter/RUNTIME_RUNBOOK.md)

Local-only artifacts that should stay untracked:
- `outputs/evaluation/`
- `simulation_camera_scooter/test_videos/`
- `bev_H.npy`
- `bev_Hinv.npy`
- calibration backup files unless intentionally preserving them in git
