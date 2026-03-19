# Current Pipeline Summary

## Scope
This document maps the current end-to-end path pipeline in the repo before any new recommendation is applied. The baseline runtime entrypoint is `simulation_camera_scooter/live_heading_demo.py`.

## Baseline Runtime Flow
1. **Frame input**
   - Camera or video frames are opened in `live_heading_demo.py`.
   - Optional stabilization runs before perception.

2. **Segmentation**
   - `FastRoadDetector` in `simulation_camera_scooter/fast_road_detector.py` loads the current model from `simulation_camera_scooter/models/my-segformer-road`.
   - The detector is effectively used as a **binary drivable-mask model**: probability of class `road_id=1` is thresholded into a 0/255 mask.
   - `live_heading_demo.py` currently passes `seg_conf_thresh=0.5`, even though `FastRoadDetector.Config` defaults to `0.6`.
   - The raw mask is post-processed by:
     - `suppress_grass_in_mask(...)`
     - `TemporalMaskSmoother`

3. **Object detection**
   - Optional YOLO nano detection runs through `simulation_camera_scooter/object_detector.py`.
   - Detections are projected into BEV for obstacle blocking and slowdown logic.

4. **BEV projection**
   - `load_bev_params(...)` loads the homography.
   - `cv2.warpPerspective(...)` maps the camera mask into a `600 x 600` BEV canvas.
   - Current metric span is `10 m` forward by `10 m` lateral, so the nominal scale is about `60 px/m`.

5. **BEV cleanup**
   - `simulation_camera_scooter/masks.py` applies:
     - `clean_bev_mask_enhanced(...)` when `MORPH_ENHANCED = True`
     - or `clean_sidewalk_mask(...)` in the legacy path
   - The cleaned mask is then ego-anchored and reduced to the ego-connected component.

6. **Path extraction**
   - `BEVPathExtractor` in `simulation_camera_scooter/realtime_nav_core.py` is the main planner wrapper.
   - In the current config, `DT_PLANNER_ENABLED = True`, so `BEVPathExtractor` instantiates `DtPathPlanner` from `simulation_camera_scooter/dt_path_planner.py`.
   - If DT is disabled, the older template / graph / corridor fallback chain is used.

7. **Control**
   - The selected path is converted into heading and speed commands by the pure-pursuit controller and scooter command layer.

8. **Predictive frame reuse**
   - `BEVPredictiveTracker` is used to skip full recomputation on selected frames.
   - On skip frames the repo can reuse or predict BEV/path state instead of rerunning segmentation + BEV + planning.

## Current Config Snapshot
The current codebase defaults relevant to this thesis pass are:

| Setting | Current value |
|---|---:|
| Segmentation model | `simulation_camera_scooter/models/my-segformer-road` |
| Runtime seg threshold in `live_heading_demo.py` | `0.5` |
| BEV size | `600 x 600` |
| Forward BEV span | `10.0 m` |
| Lateral BEV span | `10.0 m` |
| Enhanced BEV morphology | `True` |
| DT planner enabled | `True` |
| Planner mode | `dijkstra` |
| Predictor enabled by default | `True` |

## Measured Baseline Runtime
From `outputs/profiling/runtime_512/summary.md`:

| Case | FPS | Total ms | Seg ms | Det ms | BEV ms | Path ms |
|---|---:|---:|---:|---:|---:|---:|
| `full_cpu_detection` | 12.214 | 82.319 | 18.82 | 39.032 | 14.631 | 3.708 |
| `full_gpu_detection` | 20.123 | 51.581 | 17.02 | 12.216 | 13.438 | 3.152 |
| `no_detection` | 25.272 | 40.045 | 20.25 | 0.102 | 10.989 | 3.825 |
| `no_detection_no_predict` | 8.938 | 113.027 | 75.261 | 0.356 | 29.504 | 3.012 |

Two important points:

1. **CPU object detection is the biggest steady-state latency source** in the measured baseline.
2. The reported `~3 ms` pathfinding average is misleading when BEV occupancy collapses; on a failing run the planner is mostly returning `"none"` rather than solving a valid path.

## What The Current System Is Good At
- It already has useful instrumentation and replay infrastructure.
- It preserves the old baseline clearly enough to compare against.
- It already contains temporal reuse logic, which is critical for runtime.

## What The Current System Assumes
- The ground near the scooter is approximately planar enough for a static homography.
- The segmentation mask is stable enough that a binary BEV warp still preserves path topology.
- A BEV centerline is the most convenient planning space.

Those assumptions are exactly what the later experiments stress-tested.
