# Changelog: Research Implementation
## Autonomous Scooter Project — Three Navigation Improvements

**Date:** 2026-03-17

---

## Files Modified

### `simulation_camera_scooter/config.py`
**Type:** Modified — added new config entries

New constants added at end of file:
- `MORPH_ENHANCED = True` — enable enhanced mask pipeline
- `MORPH_HOLE_FILL_MAX_M2 = 5.0` — max hole area for flood-fill (m²)
- `MORPH_GAUSS_SIGMA_PX = 1.2` — Gaussian sigma for boundary smoothing
- `MORPH_GAUSS_THRESH = 0.35` — re-binarize threshold after Gaussian
- `MORPH_EGO_BAND_FRAC = 0.20` — bottom fraction for ego-clearance scoring
- `DT_CORRIDOR_ENABLED = True` — enable DT safe corridor
- `DT_CORRIDOR_COST_EXPONENT = 1.5` — cost = 1/(dt+eps)^exponent
- `DT_CORRIDOR_LATERAL_DRIFT_PX = 30` — lateral freedom per Dijkstra step
- `DT_CORRIDOR_SG_WINDOW = 9` — Savitzky-Golay window for centerline smoothing
- `DT_CORRIDOR_CONFIDENCE_NORM_M = 1.5` — clearance giving confidence=1.0
- `PATH_SMOOTH_ENABLED = True` — enable cubic coefficient EMA
- `PATH_SMOOTH_MIN_ALPHA = 0.35` — minimum EMA alpha (high smoothing)
- `PATH_SMOOTH_MAX_ALPHA = 0.85` — maximum EMA alpha (fast tracking)
- `PATH_SMOOTH_RESET_THRESH = 2.0` — coefficient jump threshold for reset
- `HEADING_SMOOTH_ENABLED = True` — enable heading circular EMA
- `HEADING_SMOOTH_ALPHA = 0.50` — heading EMA alpha
- `HEADING_SMOOTH_RESET_DEG = 45.0` — heading delta triggering filter reset

**Backward impact:** None — all existing imports unaffected. New constants only.

---

### `simulation_camera_scooter/masks.py`
**Type:** Modified — added three new functions + enhanced pipeline function

**Added imports:** `MORPH_ENHANCED`, `MORPH_HOLE_FILL_MAX_M2`, `MORPH_GAUSS_SIGMA_PX`, `MORPH_GAUSS_THRESH`, `MORPH_EGO_BAND_FRAC` from config

**New private functions:**
- `_flood_fill_holes(mask_255, max_hole_area_px)` — fills enclosed holes via corner flood-fill (arXiv:2505.12206)
- `_select_component_by_ego_clearance(mask_255, ego_band_frac)` — picks navigable component by DT at ego position

**New public function:**
- `clean_bev_mask_enhanced(mask, m_per_px, min_width_m=0.50, enhanced=True)` — full enhanced pipeline

**Unchanged:** `split_masks`, `suppress_grass_in_mask`, `clean_sidewalk_mask`, `select_main_component`, `anchor_ego_to_mask`, `ego_connected_mask` — all signatures and behavior preserved exactly.

---

### `simulation_camera_scooter/template_path_planner.py`
**Type:** Modified — added DT corridor integration wrapper

**Added imports:** `Optional` from typing; lazy import of `DtCorridorResult` from `safe_corridor`

**New public dataclass:**
- `CorridorWithDt` — bundles a `Corridor` with optional `DtCorridorResult` field

**Unchanged:** All existing functions, dataclasses, and signatures. The `Corridor` dataclass remains `frozen=True` and unmodified. The `CorridorWithDt` is additive and purely optional for callers.

---

### `simulation_camera_scooter/realtime_nav_core.py`
**Type:** Modified — integrated PathTemporalSmoother

**Added imports:** `PATH_SMOOTH_ENABLED` from config; lazy import of `PathTemporalSmoother`

**Modified:** `BEVPathExtractor.__init__` — added `self._path_smoother` instance (None if disabled or unavailable)

**New private method:**
- `BEVPathExtractor._apply_path_smoothing(path_model, confidence, path_source)` — applies EMA to CubicPathModel coefficients and rebuilds model; returns original model unchanged if smoother disabled or curvature guard fires

**Modified locations:**
- Template path approval block: calls `_apply_path_smoothing` after `_fit_regularized_cubic`
- Graph/fallback path block: calls `_apply_path_smoothing` at `has_path` commit point

**Unchanged:** All existing function signatures, `PathPlanResult` fields, `AdaptivePurePursuitController`, `ControlOutput`, `CubicPathModel`. The smoother is purely additive and the `path_model.coeff` field in returned results will contain smoothed coefficients when enabled.

---

### `simulation_camera_scooter/heading.py`
**Type:** Modified — added temporal heading filter

**Added imports:** `HEADING_SMOOTH_ENABLED` from config; lazy import of `HeadingTemporalFilter`

**New module-level state:** `_heading_filter` (None until first call)

**New private function:** `_get_heading_filter()` — lazy init

**New public functions:**
- `compute_heading_smooth(path_pts, confidence=1.0)` — calls `compute_heading()` then applies HeadingTemporalFilter; falls back to raw if filter unavailable
- `reset_heading_filter()` — resets module-level filter state

**Unchanged:** `compute_heading`, `heading_to_command`, `compute_speed`, `apply_planner_speed_limit` — all signatures and behavior preserved exactly.

---

## Files Created

### `simulation_camera_scooter/safe_corridor.py` (NEW)
Research impl: Distance Transform Safe Corridor (Idea 2)

Contents:
- `DtCorridorResult` dataclass — result container with centerline_px, width_m_per_point, confidence, dt_map, centerline_m
- `DtSafeCorridor` class — EDT + Dijkstra corridor extractor with configurable cost exponent, lateral drift, and Savitzky-Golay smoothing
- `get_default_dt_corridor()` — module-level lazy-init convenience function
- Graceful fallback when scipy unavailable (`_HAS_SCIPY` guard)

Dependencies: `scipy.ndimage.distance_transform_edt`, `scipy.signal.savgol_filter` (optional), `numpy`, `heapq`, `config`

---

### `simulation_camera_scooter/path_smoother.py` (NEW)
Research impl: Temporal Path Smoothing (Idea 3)

Contents:
- `PathTemporalSmoother` class — EMA on 4D cubic coefficients with adaptive alpha and reset logic
- `HeadingTemporalFilter` class — circular EMA on heading angle with ±180° wrap handling
- `_path_source_category(source)` — maps path_source strings to reset-detection categories

Dependencies: `numpy`, `config` (for default parameter values)

---

### `simulation_camera_scooter/scripts/eval_research_improvements.py` (NEW)
Evaluation script comparing baseline vs enhanced conditions

Contents:
- `FrameMetrics` class — per-frame metric collector with `.summary()` method
- `_passthrough_bev()` — stub BEV transform for testing without full pipeline
- `run_condition()` — runs pipeline on video for one set of improvements
- `write_report()` — writes EVALUATION_REPORT.md with comparison table
- `main()` — CLI entry point with `--video` and `--max-frames` arguments

---

## Files NOT Modified

- `simulation_camera_scooter/skeleton.py` — no changes needed
- `simulation_camera_scooter/tests/` — no test files modified
- All other pipeline files — unchanged

---

## Dependency Requirements

| New Dependency | Required By | Availability |
|---------------|-------------|-------------|
| `scipy.ndimage` | `safe_corridor.py` | Optional (graceful fallback) |
| `scipy.signal` | `safe_corridor.py` | Optional (graceful fallback) |
| `numpy` | All new files | Already required |
| `cv2` (OpenCV) | `masks.py` (existing) | Already required |

To install scipy if not present:
```bash
pip install scipy
```

All improvements degrade gracefully when optional dependencies are missing.

---

## Configuration Quick Reference

```python
# In config.py — set these to control improvements

# Idea 1: Enhanced morphological mask
MORPH_ENHANCED = True   # set False to use original clean_sidewalk_mask()

# Idea 2: DT safe corridor
DT_CORRIDOR_ENABLED = True   # set False to skip DtSafeCorridor

# Idea 3: Temporal path smoothing
PATH_SMOOTH_ENABLED = True      # set False to skip coefficient EMA
HEADING_SMOOTH_ENABLED = True   # set False to skip heading filter
```
