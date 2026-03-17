# Implementation Plan: Research Pipeline Improvements
## Autonomous Scooter Project — Phase Research Branch

**Date:** 2026-03-17
**Author:** Research Engineer

---

## Overview

Three improvements will be implemented as additive, backward-compatible changes:

| # | Name | Files Changed | Config Flag |
|---|------|--------------|-------------|
| 1 | Enhanced Morphological BEV Mask | `masks.py`, `config.py` | `MORPH_ENHANCED` |
| 2 | DT Safe Corridor | `safe_corridor.py` (new), `template_path_planner.py` | `DT_CORRIDOR_ENABLED` |
| 3 | Temporal Path Smoothing | `path_smoother.py` (new), `realtime_nav_core.py`, `heading.py` | `PATH_SMOOTH_ENABLED`, `HEADING_SMOOTH_ENABLED` |

---

## Idea 1: Enhanced Morphological BEV Mask Pipeline

### Problem
`clean_sidewalk_mask()` in `masks.py`:
- Uses distance transform with a fixed threshold (DT_CORE_THRESH=2.0) that cuts too aggressively on thin paths
- `select_main_component()` uses largest-area heuristic, not navigability-aware scoring
- No hole-filling beyond standard morphological close
- Jagged contours from quantization not smoothed

### Solution: `clean_bev_mask_enhanced()`

**New function signature:**
```python
def clean_bev_mask_enhanced(
    mask: np.ndarray,
    m_per_px: float,
    min_width_m: float = 0.50,
    enhanced: bool = True,
) -> np.ndarray
```

**Algorithm steps:**
```
Step 1: Standard morphological close (k=7) + open (k=5)
        [existing — keep as-is]

Step 2: [NEW, enhanced only] Flood-fill hole filling
        - Invert mask
        - Flood-fill from all 4 corners → marks background
        - Invert result → enclosed holes
        - Fill only holes < max_hole_area (avoid filling open space at path end)
        max_hole_area = 5.0 / (m_per_px^2) pixels

Step 3: [NEW, enhanced only] Gaussian blur before re-binarize
        - Apply GaussianBlur(sigma=1.2) to float mask [0,1]
        - Re-binarize at 0.35 threshold
        - Net effect: rounds jagged corners, fills 1-pixel gaps
        - sigma is in pixels; at BEV scale (~0.017m/px) this = ~2cm smoothing

Step 4: [NEW, enhanced only] DT-based component selection
        - Compute EDT on smoothed mask
        - For each component, score = max DT in bottom 20% rows
          (measures clearance near ego position)
        - Select component with highest ego-clearance score
        - Fallback to largest-area if scores tie

Step 5: Standard DT core extraction (existing clean_sidewalk_mask logic)
        [only if enhanced=False — enhanced path skips this, uses lighter erosion]
```

**Config additions (config.py):**
```python
MORPH_ENHANCED = True
MORPH_HOLE_FILL_MAX_M2 = 5.0      # max hole area to fill (square meters)
MORPH_GAUSS_SIGMA_PX = 1.2        # Gaussian sigma before re-binarize
MORPH_GAUSS_THRESH = 0.35         # threshold after Gaussian blur
MORPH_EGO_BAND_FRAC = 0.20        # bottom fraction of BEV for ego-clearance scoring
```

**Backward compatibility:**
- `clean_sidewalk_mask()` unchanged
- `select_main_component()` unchanged
- `clean_bev_mask_enhanced(..., enhanced=False)` falls back to `clean_sidewalk_mask()` behavior
- New function is purely additive

**Integration points:**
- Called from whichever pipeline entry calls `clean_sidewalk_mask` (currently `masks.py` standalone)
- The enhanced function is available for callers; existing callers continue to work unchanged

---

## Idea 2: Distance Transform Safe Corridor

### Problem
`corridor_from_mask()` in `template_path_planner.py`:
- Row-wise scan fails at bifurcations (picks wrong branch arbitrarily)
- Noisy centerline — each row independently scored, no spatial continuity
- No global optimality — cannot "look ahead" to find the highest-clearance path through a turn
- Width estimate from left/right boundaries is unreliable when mask has holes

### Solution: `DtSafeCorridor` class in `safe_corridor.py`

**Algorithm:**
```
1. EDT computation
   dt = scipy.ndimage.distance_transform_edt(bev_mask > 0)
   - dt[i,j] = distance to nearest background pixel in pixels

2. Cost grid
   cost = 1.0 / (dt + 0.5) ^ 1.5
   - High DT (far from wall) → low cost → preferred path
   - Exponent 1.5 (configurable via DT_CORRIDOR_COST_EXPONENT)
   - eps=0.5 prevents division-by-zero at mask boundary

3. Dijkstra from ego
   Start: (row=H-1, col=W//2) — bottom center of BEV
   Movement: forward only (row decreases toward top of BEV)
   Lateral drift: ±30px per row step (allows following turns)
   Termination: reaches row 0 or runs out of valid pixels

4. Backtrack optimal path
   Trace parent pointers from furthest reached row back to start
   Result: (N,2) array of (row, col) pixel coordinates

5. Smoothing
   Apply scipy.signal.savgol_filter(window=9, polyorder=2) to col coordinates
   - Removes quantization jitter from grid-constrained Dijkstra path

6. Convert to metric
   centerline_m: forward_m = (H-1-row) / (H-1) * bev_forward_m
                 lateral_m = (col / (W-1) - 0.5) * bev_lateral_m

7. Confidence
   mean_dt_px = mean(dt[centerline_rows, centerline_cols])
   mean_dt_m = mean_dt_px * m_per_px
   valid_fraction = len(centerline) / H
   confidence = clip(mean_dt_m / 1.5, 0, 1) * valid_fraction
```

**Class interface:**
```python
@dataclass
class DtCorridorResult:
    centerline_px: np.ndarray      # (N,2) row,col
    width_m_per_point: np.ndarray  # (N,) clearance in meters
    confidence: float              # 0–1
    dt_map: np.ndarray             # full EDT map (for debug)
    centerline_m: np.ndarray       # (N,2) forward_m, lateral_m

class DtSafeCorridor:
    def __init__(self, bev_forward_m=10.0, bev_lateral_m=10.0,
                 cost_exponent=1.5, lateral_drift_px=30,
                 sg_window=9, sg_poly=2):
        ...

    def extract(self, bev_mask_uint8: np.ndarray,
                prev_centerline_px=None) -> DtCorridorResult:
        ...
```

**Integration in template_path_planner.py:**
- `CorridorResult` dataclass gets new optional field: `dt_corridor: Optional[DtCorridorResult] = None`
- `corridor_from_mask()` signature unchanged
- When `DT_CORRIDOR_ENABLED=True`, callers can pass the DtCorridorResult alongside the Corridor for enhanced scoring
- The DT centerline can supplement or replace the row-wise centerline in template scoring

**Config additions:**
```python
DT_CORRIDOR_ENABLED = True
DT_CORRIDOR_COST_EXPONENT = 1.5
DT_CORRIDOR_LATERAL_DRIFT_PX = 30
DT_CORRIDOR_SG_WINDOW = 9
DT_CORRIDOR_CONFIDENCE_NORM_M = 1.5   # clearance that gives confidence=1.0
```

---

## Idea 3: Temporal Path Smoothing

### Problem
- Each frame produces a fresh cubic fit from raw path points
- Coefficient jitter between frames → heading jitter at controller input
- `compute_heading()` in `heading.py` takes raw path points → no temporal filtering
- When segmentation is stable but corridor boundary flickers, coefficients flip unnecessarily

### Solution: Two new classes in `path_smoother.py`

#### PathTemporalSmoother

```python
class PathTemporalSmoother:
    def smooth(self, coeffs: np.ndarray, confidence: float, path_source: str) -> np.ndarray
    def reset(self)
```

**Algorithm:**
```
alpha = clip(confidence * 1.3, PATH_SMOOTH_MIN_ALPHA, PATH_SMOOTH_MAX_ALPHA)

# Reset conditions (use new_coeffs directly):
- path_source changed (graph ↔ template)
- |new_coeffs - prev_smoothed|.max() > PATH_SMOOTH_RESET_THRESH
- First call after reset

# Normal operation:
smoothed = alpha * new_coeffs + (1 - alpha) * prev_smoothed

# Confidence interpretation:
- high confidence (near 1.0) → alpha near 0.85 → fast tracking
- low confidence (near 0.3) → alpha near 0.35 → heavy smoothing / inertia
```

#### HeadingTemporalFilter

```python
class HeadingTemporalFilter:
    def update(self, heading_deg: float, confidence: float) -> float
    def reset(self)
```

**Algorithm:**
```
alpha_h = HEADING_SMOOTH_ALPHA = 0.50

# Circular EMA (handles ±180° wraparound):
delta = heading_deg - prev_heading
# Wrap delta to [-180, 180]:
delta = (delta + 180) % 360 - 180
# Reset condition:
if |delta| > 45° → reset, use heading_deg directly

smoothed = prev_heading + alpha_h * delta

# Exposure: update() returns smoothed_heading
```

**Integration in heading.py:**
- Add optional module-level `_heading_filter = HeadingTemporalFilter()` instance
- Add `compute_heading_smooth(path_pts, confidence=1.0)` function that calls `compute_heading()` then applies filter
- Original `compute_heading()` unchanged

**Integration in realtime_nav_core.py:**
- Add `_path_smoother: PathTemporalSmoother` instance to `BEVPathExtractor.__init__`
- After `_fit_regularized_cubic` returns a model, apply smoother to `path_model.coeff`
- Rebuild CubicPathModel with smoothed coefficients
- Only when `PATH_SMOOTH_ENABLED=True`

**Config additions:**
```python
PATH_SMOOTH_ENABLED = True
PATH_SMOOTH_MIN_ALPHA = 0.35
PATH_SMOOTH_MAX_ALPHA = 0.85
PATH_SMOOTH_RESET_THRESH = 2.0
HEADING_SMOOTH_ENABLED = True
HEADING_SMOOTH_ALPHA = 0.50
HEADING_SMOOTH_RESET_DEG = 45.0
```

---

## Implementation Order and Dependencies

```
config.py   (add all new config entries — no dependencies)
    ↓
masks.py    (Idea 1 — depends on config only)
    ↓
safe_corridor.py   (Idea 2 — new file, depends on scipy)
    ↓
template_path_planner.py  (Idea 2 integration — minimal changes)
    ↓
path_smoother.py   (Idea 3 — new file, no external deps beyond numpy)
    ↓
heading.py         (Idea 3 integration — minimal changes)
    ↓
realtime_nav_core.py  (Idea 3 integration — minimal changes)
    ↓
scripts/eval_research_improvements.py  (evaluation)
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Gaussian smoothing oversmooths thin corridors | `enhanced=False` fallback; sigma=1.2px is conservative |
| Dijkstra too slow on large BEV (600×500) | Only traverse road pixels; prune by EDT>0; target <10ms |
| Path smoother causes lag at turns | Reset on large coefficient jump (>2.0) and source change |
| Heading filter wraps incorrectly | Circular delta approach handles ±180° correctly |
| scipy not available | Import guard + fallback in safe_corridor.py |
| New coefficients break CubicPathModel | Evaluate curvature after smoothing; clamp to kappa_max |

---

## Testing Strategy

1. **Unit test:** `clean_bev_mask_enhanced` on synthetic masks (rectangle with holes, narrow corridor)
2. **Unit test:** `DtSafeCorridor.extract` on straight and curved synthetic BEV masks
3. **Unit test:** `PathTemporalSmoother` — verify reset on source change, EMA convergence
4. **Unit test:** `HeadingTemporalFilter` — verify wrap-around handling at ±180°
5. **Integration test:** Run eval_research_improvements.py on test video, compare metrics
6. **Regression test:** `pytest tests/ -x -q` — existing tests must pass unchanged
