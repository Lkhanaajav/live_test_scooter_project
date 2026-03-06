# CHANGELOG OVERNIGHT

## Edited Files

### `simulation_camera_scooter/bev_predictor.py`
- Added occupancy-state tracking (`last_occ_ratio`) and minimum occupancy thresholds.
- Updated `should_skip()` to block skip when predicted BEV occupancy is too low.
- Updated `on_compute_frame()` to:
  - store occupancy ratio,
  - reduce confidence when BEV occupancy is weak,
  - clear stale path state when no valid path exists.
- Updated `on_skip_frame()` to:
  - detect sparse/empty predicted masks,
  - invalidate predicted path and reduce confidence,
  - force return to compute-frame behavior.

### `simulation_camera_scooter/realtime_nav_core.py`
- `PathExtractorConfig`:
  - added low-evidence thresholds,
  - added candidate center/continuity scoring weights.
- `PurePursuitConfig`:
  - added `discont_max_hold_frames`.
- `PathPlanResult`:
  - added `path_source` and `mask_occ_ratio` diagnostics.
- `BEVPathExtractor`:
  - tracks mask occupancy,
  - computes low-evidence mode in `process()`,
  - applies aggressive decay in `_hold_previous_path(aggressive=True)`,
  - adds center + temporal continuity terms in `_score_candidates()`.
- `AdaptivePurePursuitController`:
  - added bounded discontinuity reject counter to avoid indefinite stale-path lock.

### `simulation_camera_scooter/live_heading_demo.py`
- Skip-frame `PathPlanResult` now records `path_source` and occupancy ratio.
- Logger includes `path_source` and `bev_mask_occ_ratio`.
- BEV HUD call now passes source/occupancy diagnostics.

### `simulation_camera_scooter/data_logger.py`
- Added CSV fields: `planner_mode`, `path_source`, `bev_mask_occ_ratio`.

### `simulation_camera_scooter/visualization.py`
- `draw_bev_hud()` now optionally displays path source and BEV occupancy ratio.

### `simulation_camera_scooter/tests/test_bev_predictor.py`
- Updated skip-related tests for occupancy-guard behavior.
- Added tests for low-occupancy skip blocking and predicted-path invalidation on sparse masks.

## Reverted/Rejected Changes
- Reverted E2 low-confidence fallback parameter tuning in `PathExtractorConfig` after regression (`exp2_lowconf_tune`).

## 2026-03-06 Afternoon Update

### `simulation_camera_scooter/realtime_nav_core.py`
- Added skeleton-based fallback path generator:
  - `_fallback_skeleton_geodesic(skel_bin)`
  - geodesic path from ego-near skeleton seed to forward endpoint, converted to metric path.
- Added configuration knobs:
  - `skeleton_fallback_enabled`
  - `skeleton_fallback_min_forward_m`
  - `skeleton_fallback_center_penalty`
  - `skeleton_override_graph_short_progress_m`
  - `skeleton_override_graph_near_abs_lat_m`
  - `skeleton_override_skel_near_abs_lat_m`
- Added candidate/fallback integration:
  - skeleton fallback candidate appended when graph candidates are under-constrained.
  - fallback order updated to prefer skeleton fallback before centerline fallback.
- Added targeted selection override:
  - when selected graph is short and laterally biased near ego while skeleton candidate is centered, force skeleton candidate.
- Added `path_source="fallback_skeleton"` reporting.
- Tried and reverted temporary `skeleton_hold_frames` logic (rejected due global bias drift).
