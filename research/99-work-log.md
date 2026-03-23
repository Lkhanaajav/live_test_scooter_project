# Work Log

## 2026-03-19

### Repo mapping
- Inspected the runtime entrypoints and planning stack:
  - `simulation_camera_scooter/live_heading_demo.py`
  - `simulation_camera_scooter/fast_road_detector.py`
  - `simulation_camera_scooter/realtime_nav_core.py`
  - `simulation_camera_scooter/dt_path_planner.py`
  - `simulation_camera_scooter/masks.py`
  - `simulation_camera_scooter/object_detector.py`
- Confirmed the current baseline is:
  - binary SegFormer-style segmentation
  - homography BEV warp
  - BEV cleanup
  - DT-based BEV planning

### Existing artifacts reviewed
- Runtime profiling:
  - `outputs/profiling/runtime_512/summary.md`
  - `outputs/profiling/runtime_512/full/run_20260319_024759.csv`
- Segmentation replay summaries:
  - `outputs/evaluation/binary_model_replay_full/summary.md`
  - `EVALUATION_REPORT.md`
- Speed studies:
  - `outputs/comparisons/resolution_benchmark_img1931_1024w30/summary.md`
  - `outputs/comparisons/onnx_vs_pytorch_img1931_1024w30_256x144/summary.md`
  - `INFERENCE_SPEEDUP_RESEARCH.md`

### Main initial findings
- CPU detection was the biggest steady-state FPS penalty.
- Disabling the predictor was catastrophic for runtime.
- One profiled BEV run mostly returned `path_source = none`, so the old average path timing was misleading.
- Existing replay evidence already suggested the better binary checkpoint was a real segmentation improvement.

### New implementation work
- Added `simulation_camera_scooter/image_path_planner.py`
  - `CameraMidpointPlanner`
  - `CameraDtPlanner`
- Added `simulation_camera_scooter/tests/test_image_path_planner.py`
- Added `simulation_camera_scooter/scripts/eval_hand_annotated_pipeline.py`

### Verification
- Ran:
```powershell
pytest simulation_camera_scooter/tests/test_image_path_planner.py `
  simulation_camera_scooter/tests/test_dt_path_planner.py `
  simulation_camera_scooter/tests/test_boundary_inference.py -q
```
- Result:
  - `16 passed`

### Research evaluation
- Ran:
```powershell
python simulation_camera_scooter/scripts/eval_hand_annotated_pipeline.py --per-video-limit 8
```
- Produced:
  - segmentation summary tables
  - planner summary tables
  - planner comparison images

### New artifacts generated or copied into `research/`
- Tables:
  - `research/artifacts/tables/segmentation_hand_annotations_summary.csv`
  - `research/artifacts/tables/segmentation_hand_annotations_by_video.csv`
  - `research/artifacts/tables/planner_hand_annotations_summary.csv`
  - `research/artifacts/tables/planner_hand_annotations_by_video.csv`
  - `research/artifacts/tables/fps_offenders_summary.csv`
- Images:
  - `research/artifacts/images/planner_bev_vs_img_dt_contact_sheet.png`
  - `research/artifacts/images/planner_compare_IMG_1922_frame_001124.png`
  - `research/artifacts/images/planner_compare_IMG_1924_frame_001428.png`
  - `research/artifacts/images/planner_compare_IMG_1924_frame_003378.png`
  - `research/artifacts/images/segmentation_compare_img_1878.jpg`
  - `research/artifacts/images/segmentation_compare_img_1922.jpg`
- Videos:
  - `research/artifacts/videos/segmentation_compare_img_1878_side_by_side.mp4`
  - `research/artifacts/videos/planner_compare_slideshow.mp4`

### Literature and code review performed
- Segmentation:
  - SegFormer paper + code
  - OneFormer paper + code
  - PIDNet paper + code
- BEV:
  - Lift-Splat-Shoot paper + code
  - BEVFormer paper + code
  - GitNet paper
  - FocusBEV paper
  - SkyEye repo
  - MonoScene paper + code
- Pathing:
  - CenterLineDet paper / project
  - Nav2 Smac Planner docs / code
  - scikit-image morphology docs

### Final conclusions reached
- Current shipped segmentation checkpoint is not good enough.
- The better binary SegFormer checkpoint is the correct immediate replacement.
- Monocular BEV is too fragile to remain the sole planning domain.
- Image-space midpoint planning should replace BEV DT as the default planner.
- Image-space DT should remain as the robust fallback.
- CPU detection, predictor disablement, high seg resolution, and always-on BEV planning were the biggest avoidable FPS drains.

### New data refresh with four new test videos
- Identified the four new videos under `simulation_camera_scooter/test_videos/`:
  - `VID_20260319_155939_00_017.mp4`
  - `VID_20260319_160039_00_018.mp4`
  - `VID_20260319_160139_00_019.mp4`
  - `VID_20260319_160240_00_020.mp4`
- Extracted `180` frames per video at `1280x720` into:
  - `outputs/datasets/frames_vid017_020_t180`
- Generated binary teacher pseudo-labels for all `720` extracted frames into:
  - `outputs/pseudo_labels/pseudo_vid017_020_binary`

### Mixed teacher-student training pass
- Combined the old strong mixed dataset with the new four-video pseudo labels:
  - `699` older image/mask pairs
  - `720` new pseudo-labeled pairs
  - `1419` total pairs
- Built combined training roots:
  - `outputs/datasets/frames_old400_img1931_vid017_020`
  - `outputs/pseudo_labels/pseudo_old400_img1931_vid017_020_binary/masks`
- Trained a new checkpoint initialized from:
  - `outputs/training/binary_segformer_old400_plus_img_1931_t300/best_checkpoint`
- New training output:
  - `outputs/training/binary_segformer_old400_img1931_vid017_020`
- Best validation IoU:
  - `0.9602279425304405`
- Tuned threshold:
  - `0.60`

### Runtime flag sweep with the new model
- Benchmarked on:
  - `simulation_camera_scooter/test_videos/VID_20260319_160139_00_019.mp4`
- Common settings:
  - new checkpoint `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
  - `--seg-conf-thresh 0.60`
  - `--seg-width 640 --seg-height 360`
  - `--max-frames 300`
  - `--no-detection`
  - `--log`
- Cases run:
  - `headless_nosave`
  - `headless_save`
  - `gui_nosave`
  - `gui_save`
- Wrote summaries to:
  - `outputs/profiling/new_model_flag_sweep/summary.csv`
  - `outputs/profiling/new_model_flag_sweep/summary.json`
  - `outputs/profiling/new_model_flag_sweep/summary.md`
  - `research/artifacts/tables/new_model_flag_sweep_summary.csv`
  - `research/artifacts/tables/new_model_flag_sweep_summary.json`
  - `research/artifacts/tables/new_model_flag_sweep_summary.md`

### Runtime conclusion from the new model sweep
- `--headless` changed throughput by only a few milliseconds.
- `--save` changed throughput by only a few milliseconds.
- Mean segmentation cost was about `32-33 ms/frame`.
- Mean BEV cost was about `19 ms/frame`.
- Mean pathing cost was about `515-520 ms/frame`.
- The main live bottleneck is still the BEV DT planner, not display or video writing overhead.

### Real-time deployment pass on `VID_20260319_155939_00_017.mp4`
- Restored the `1024x576` BEV calibration metadata so runtime uses the intended `1024x576 -> 1920x1080` rescaling path.
- Fixed the fast-planner bridge in `simulation_camera_scooter/path_planners/adapter.py` so reversed or slightly negative-forward centerlines can still be fit into a valid cubic path.
- Fixed the potential-field integrator to keep forward progress and stay within the BEV bounds.
- Added explicit live runtime planner selection in `simulation_camera_scooter/live_heading_demo.py` with:
  - `--planner-family dt_ridge`
  - `--planner-family vectorized_dt`
  - `--planner-family weighted_centroid`
  - `--planner-family potential_field`
  - `--planner-family skeleton_hybrid`
- Kept `--legacy-planner` as a backwards-compatible alias for `--planner-family potential_field`.
- Added adapter tests in `simulation_camera_scooter/tests/test_path_planner_adapter.py`.
- Verified tests:
  - `pytest simulation_camera_scooter/tests/test_path_planner_adapter.py simulation_camera_scooter/tests/test_path_planners.py -q`
  - result: `41 passed`
- Added intent-aware far-field bias to `simulation_camera_scooter/path_planners/weighted_centroid.py` so manual turn schedules can influence the real-time planner.
- Verified live intent scheduling on a short `VID_...017` weighted-centroid run with a synthetic right-turn window:
  - `outputs/profiling/vid017_wc_schedule_test/run_20260320_010315.csv`
  - schedule event was loaded and applied
  - turn-lock engaged and released during the scheduled window

### Real-time result that hit the target
- Produced a saved demo run at:
  - `outputs/runs/vid017_weighted_centroid_rt/VID_20260319_155939_00_017_weighted_centroid_rt.mp4`
- Configuration:
  - new student checkpoint
  - `--planner-family weighted_centroid`
  - detection disabled
  - `640x360` segmentation
  - headless + save
- Summary:
  - `300` frames
  - `has_path = 100%`
  - mean FPS after startup = `6.90`
  - 5th percentile FPS after startup = `6.42`
  - mean total time = `105.64 ms/frame`
  - mean segmentation time = `34.92 ms/frame`
  - mean BEV time = `22.18 ms/frame`
  - mean path time = `2.61 ms/frame`
- Added run summary:
  - `outputs/runs/vid017_weighted_centroid_rt/summary.md`
- Added an intent schedule template for future manual turn windows:
  - `outputs/runs/vid017_weighted_centroid_rt/intent_schedule_template.json`

### Four-video planner overlay batch with the new model
- Patched `simulation_camera_scooter/scripts/eval_path_planners.py` so it now saves:
  - per-planner camera-overlay videos with the planner path warped back onto the original camera view
  - per-planner BEV overlay videos
  - the existing side-by-side comparison panel
- Added script flags:
  - `--model-dir`
  - `--seg-conf-thresh`
- Verified the patched evaluator on a short smoke test:
  - `outputs/smoke_eval_path_planners/planner_comparison/`
- Ran the full four-video batch with:
  - checkpoint `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
  - threshold `0.60`
  - `frame-step = 3`
  - all four March 19 videos: `017`, `018`, `019`, `020`
- Main output directory:
  - `outputs/path_planner_eval_new_model_all4_step3/planner_comparison/`
- Aggregate summary:
  - `dt_ridge_baseline`: `1288.2 ms`, heading std `0.93 deg`, path jump `0.17 deg`
  - `vectorized_dt`: `82.0 ms`, heading std `29.26 deg`, path jump `3.71 deg`
  - `weighted_centroid`: `2.7 ms`, heading std `14.09 deg`, path jump `2.42 deg`
  - `potential_field`: `7.0 ms`, heading std `40.03 deg`, path jump `5.77 deg`
  - `skeleton_hybrid`: `11.6 ms`, heading std `34.66 deg`, path jump `11.29 deg`
- Saved artifacts:
  - `44` MP4 files
  - total size about `3.48 GB`
- Practical result:
  - `dt_ridge_baseline` remains the quality oracle but is too slow for live use
  - `weighted_centroid` remains the most deployable fast planner on these four videos

### Full-rate rerender of the best two planners on `VID_...017`
- Extended `simulation_camera_scooter/scripts/eval_path_planners.py` with `--planners` so targeted rerenders can be run without paying for every planner each time.
- Verified the filtered path on a short smoke run:
  - `outputs/smoke_eval_path_planners_best2_vid017_fullrate/planner_comparison/`
- Ran the full `1800`-frame `VID_20260319_155939_00_017.mp4` clip at full frame rate with:
  - checkpoint `outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
  - planners `dt_ridge_baseline` and `weighted_centroid`
  - threshold `0.60`
  - `frame-step = 1`
- Main output directory:
  - `outputs/path_planner_eval_new_model_vid017_best2_fullrate/planner_comparison/VID_20260319_155939_00_017/`
- Saved videos:
  - `comparison.mp4`
  - `camera_overlays/dt_ridge_baseline.mp4`
  - `camera_overlays/weighted_centroid.mp4`
  - `bev_overlays/dt_ridge_baseline.mp4`
  - `bev_overlays/weighted_centroid.mp4`
- Final metrics on the full clip:
  - `dt_ridge_baseline`: heading std `0.33 deg`, path jump `0.03 deg`, lateral std `1.320 m`, `1079.5 ms/frame`
  - `weighted_centroid`: heading std `11.96 deg`, path jump `0.84 deg`, lateral std `0.921 m`, `2.6 ms/frame`
