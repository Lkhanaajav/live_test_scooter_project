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
