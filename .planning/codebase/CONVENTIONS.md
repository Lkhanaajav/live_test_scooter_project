# Code Conventions

This document captures the coding conventions, patterns, and practices used across the scooter navigation codebase.

## File Structure & Organization

- **Main package**: `simulation_camera_scooter/` (31 Python modules at root level)
- **Tests**: `simulation_camera_scooter/tests/` (12 test files using pytest)
- **Single source of truth**: `config.py` — all shared constants and configuration
- **Modular design**: high cohesion, low coupling; typical module size 200–800 lines
- **Consistent naming**: kebab-case for filenames, snake_case for functions/variables

### Module Categories

- **Core pipeline**: `realtime_nav_core.py`, `template_path_planner.py`, `skeleton.py`
- **Perception**: `fast_road_detector.py`, `boundary_model.py`, `bev_obstacle.py`, `object_detector.py`
- **Navigation**: `gps_navigator.py`, `intent_picker.py`, `heading.py`
- **Utilities**: `data_logger.py`, `stabilization.py`, `visualization.py`, `bev_predictor.py`

---

## Python Coding Style

### PEP 8 Compliance

All code follows **PEP 8** with these specific applications:

- **Line length**: target ~100 characters (practical soft limit; longer acceptable for clarity)
- **Blank lines**: 2 between module-level functions/classes, 1 between methods
- **Indentation**: 4 spaces (no tabs)
- **Imports**: grouped and ordered via isort convention
  - Standard library
  - Third-party packages (`numpy`, `cv2`, `torch`, etc.)
  - Local imports (from `config`, sibling modules)

### Import Organization

**Example** (`realtime_nav_core.py`):
```python
from dataclasses import dataclass, field, replace
import heapq
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    BEV_OBSTACLE_PENALTY_WEIGHT,
    PATH_SMOOTH_ENABLED,
    ...
)
from template_path_planner import (
    CorridorConfig,
    TemplateApprovalResult,
    ...
)
```

**Key patterns**:
- Import all config constants at module top
- Lazy imports for optional/research features (try/except around imports)
- Import modules (`import config as cfg_module`), not scattered constants

### Type Hints

**Mandatory** on all public function signatures:

```python
def compute_heading(path_pts: Sequence[Tuple[float, float]]) -> float:
    """Compute heading angle from a BEV path."""
    ...

def process(self, mask: np.ndarray) -> PathPlanResult:
    """Extract path from BEV mask."""
    ...
```

**Types used**:
- `float`, `int`, `str`, `bool` for primitives
- `np.ndarray` for arrays (with optional dtype hint in docstring)
- `Optional[T]` for nullable types
- `Sequence[T]`, `List[T]`, `Tuple[T, ...]` from `typing`
- Custom dataclass types (`PathPlanResult`, `ControlOutput`)

**Where NOT required**:
- Loop variables in tight math-heavy sections
- Return values of private functions (optional for clarity)

### Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| **Constants** | `UPPER_SNAKE_CASE` | `ROAD_ID`, `SPEED_MAX`, `BEV_FORWARD_M`, `HEADING_STRAIGHT_THRESH` |
| **Variables** | `lower_snake_case` | `best_path_m`, `control_path_px`, `mean_curvature_m_inv` |
| **Functions** | `lower_snake_case` | `compute_heading()`, `prune_small_branches()`, `project_foot_to_bev()` |
| **Classes** | `PascalCase` | `DataLogger`, `PathExtractorConfig`, `CubicPathModel`, `BEVPathExtractor` |
| **Private methods** | `_lower_snake_case` | `_commit_selected_path()`, `_obstacle_penalty()`, `_safe_norm()` |
| **Module files** | `lower_snake_case.py` | `realtime_nav_core.py`, `bev_obstacle.py`, `template_path_planner.py` |

**Metric units in variable names** (strongly encouraged):
- `_m` = meters (distance, position)
- `_px` = pixels (image/mask coordinates)
- `_deg` = degrees (angles)
- `_rad` = radians (angles)
- `_mps` = meters per second (velocity)
- `_s` = seconds (time)
- `_hz` = hertz (frequency)
- `_m_inv` = meters^-1 (curvature: 1/radius)

**Example**:
```python
best_path_m: np.ndarray      # [N,2] in meters (forward, lateral)
control_path_px: np.ndarray  # [N,2] in pixels (x, y)
lookahead_m: float           # 1.5 meters
steer_deg: float             # ±30 degrees
dt_s: float                  # 0.125 seconds
```

---

## Data Structures & Immutability

### Dataclasses for Configuration & Results

All configuration and result types use **immutable dataclasses**:

```python
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class PathExtractorConfig:
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0
    work_size: Tuple[int, int] = (220, 220)
    min_sidewalk_width_m: float = 0.50
    branch_min_len_m: float = 0.30
    # ... ~50 more config fields
```

**Pattern benefits**:
- Readable initialization: `cfg = PathExtractorConfig(bev_forward_m=12.0)`
- Dataclass generates `__init__`, `__repr__`, `__eq__` automatically
- Type-checked at module load time
- Avoids scattered magic numbers

**Special case — mutable default fields**:
```python
@dataclass
class PathExtractorConfig:
    template_planner_cfg: TemplatePlannerConfig = field(default_factory=TemplatePlannerConfig)
```

### Regular Classes for Stateful Objects

Classes with internal state (caches, EMA filters, counters) use regular `class` syntax:

```python
class BEVPathExtractor:
    """Medial-axis path extraction with caching and branch tracking."""

    def __init__(self, cfg: Optional[PathExtractorConfig] = None):
        self.cfg = cfg or PathExtractorConfig()
        self.prev_first_edge_sig: Optional[Tuple[int, int, int, int]] = None
        self.prev_best_path_m: Optional[np.ndarray] = None
        self.branch_hold_counter: int = 0
        self.obstacle_zones: list = []
```

**Instance variables initialized in `__init__`**: avoid lazy initialization unless performance-critical.

### Immutability in Return Values

Functions return **new** objects, never mutate inputs:

```python
# CORRECT: create new object
def resample_polyline(pts_m: np.ndarray, ds_m: float) -> np.ndarray:
    if pts_m is None or len(pts_m) < 2:
        return np.zeros((0, 2), dtype=np.float32)  # new array
    seg = np.diff(pts_m, axis=0)  # numpy creates a copy
    ...
    return np.stack([x, y], axis=1)  # new array, original unchanged

# AVOID: modifying in-place unless explicitly documented
def process_mask(mask: np.ndarray) -> None:
    mask[:trim_px, :] = 0  # WRONG: mutates caller's array
```

---

## Documentation & Docstrings

### Docstring Style: Google Format

All public functions and classes use **Google-style docstrings**:

```python
def compute_heading(path_pts: Sequence[Tuple[float, float]]) -> float:
    """
    Compute heading angle from a BEV path.

    Returns angle in degrees: 0 = straight ahead, negative = left, positive = right.

    Args:
        path_pts: Sequence of (x, y) coordinates in BEV frame.

    Returns:
        Heading angle in degrees.
    """
    ...

class DataLogger:
    """
    Logs every frame's data to a timestamped CSV file for post-hoc analysis.

    Each row captures: timing, heading, speed, detections, GPS, path info.
    """

    FIELDNAMES = [...]

    def __init__(self, log_dir: str = "logs"):
        """Initialize logger and create CSV file.

        Args:
            log_dir: Output directory for logs (created if missing).
        """
        ...
```

### Module Docstrings

Every `.py` file starts with a module docstring:

```python
"""
heading.py
==========
Heading computation, command classification, and speed profiling.

Research impl: Temporal heading filter (Idea 3).
  compute_heading_smooth() wraps compute_heading() with a circular EMA filter
  to prevent single-frame heading spikes from flipping command classification.
  Papers: Regulated Pure Pursuit (arXiv:2305.20026)
"""
```

**Structure**:
- Filename
- = underline
- One-line purpose
- Blank line
- Detailed description
- (Optional) research/paper citations
- (Optional) key functions listed

### Docstring Content

**For complex algorithms**, include:
- Mathematical relationships (e.g., cubic path equation, arc geometry)
- Parameter constraints (e.g., "radius must be > 0.5m")
- Return semantics (e.g., "points in metric frame (x=forward, y=lateral)")
- References to papers or research

**Example**:
```python
def compute_speed(
    heading_deg: float,
    min_obstacle_dist: Optional[float] = None,
    has_path: bool = True
) -> float:
    """
    Compute target speed based on heading angle and obstacle proximity.

    Speed profile:
      - Heading < STRAIGHT_THRESH: SPEED_MAX
      - Heading in [STRAIGHT, TURN]: interpolate between MAX and TURN
      - Heading >= TURN: SPEED_SHARP_TURN
      - Obstacle < STOP_M: SPEED_STOP
      - Obstacle < CLOSE_M: SPEED_OBSTACLE_NEAR

    Args:
        heading_deg: Path heading angle in degrees (0 = straight, ±90 = sharp turn).
        min_obstacle_dist: Distance to nearest obstacle in meters, or None if clear.
        has_path: Whether a valid path was found.

    Returns:
        Target speed in m/s.
    """
```

---

## Error Handling

### Explicit Error Handling

**All errors are handled explicitly** at system boundaries:

```python
def _checkpoint_val_iou(model_dir: str) -> float:
    """Return recorded validation IoU for a checkpoint, or -1 when unavailable."""
    summary_path = os.path.join(model_dir, "training_summary.json")
    if not os.path.isfile(summary_path):
        return -1.0  # explicit failure signal
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return -1.0  # catch all JSON errors
    best_metrics = data.get("best_metrics") or {}
    metrics = data.get("metrics") or {}
    value = best_metrics.get("best_val_iou", metrics.get("iou"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0  # type conversion failure
```

**Pattern**: Use sentinel values (`-1.0`, `None`, empty collections) or boolean flags to signal errors, not silent failures.

### Logging, Not Printing

Use print-to-console sparingly; reserve it for user-facing messages:

```python
# Logger initialization (GOOD: user wants to know)
print(f"[Logger] Logging to {self.csv_path}")

# Math debugging (AVOID in production code unless flag-gated)
print(f"DEBUG: skeleton_px shape = {skel.shape}")
```

### Lazy Import for Research Features

Optional/research implementations use lazy imports with feature flags:

```python
try:
    from path_smoother import PathTemporalSmoother as _PathTemporalSmoother
    _HAS_PATH_SMOOTHER = True
except ImportError:
    _HAS_PATH_SMOOTHER = False
    _PathTemporalSmoother = None  # type: ignore[assignment,misc]

# Later, in function:
def compute_heading_smooth(path_pts, confidence: float = 1.0) -> float:
    """Heading with optional temporal smoothing."""
    raw = compute_heading(path_pts)
    filt = _get_heading_filter()
    if filt is None:
        return raw  # fallback to raw heading
    return filt.update(raw, confidence)
```

**Benefits**:
- Backward compatible: code works even if new module missing
- Feature toggle: disable via config flag (`HEADING_SMOOTH_ENABLED`)
- Safe imports: no circular dependencies

---

## Logging Conventions

### Print-Based Logging

Simple print statements with `[Tag]` prefix for clarity:

```python
print(f"[Logger] Logging to {self.csv_path}")
print(f"[Logger] Metadata saved to {self.meta_path}")
print(f"[Logger] Closed. {self._row_count} rows written to {self.csv_path}")
```

### CSV/Structured Logging

For telemetry, use `DataLogger` class (in `data_logger.py`):

```python
logger = DataLogger(log_dir="logs")
logger.log(
    frame_id=frame_num,
    heading_raw_deg=raw_heading,
    heading_smoothed_deg=smooth_heading,
    command=cmd,
    speed_raw_mps=speed,
    has_path=has_valid_path,
    ...
)
logger.save_metadata(model_name="binary_segformer", run_type="live_test")
logger.close()
```

**Fieldnames**: pre-defined as class constant (60+ fields for comprehensive telemetry).

---

## Common Patterns

### Utility Helper Functions

Small, reusable utility functions are defined at module scope:

```python
def _odd(v: int) -> int:
    """Convert to odd integer (for kernel sizes)."""
    v = max(1, int(v))
    return v if (v % 2 == 1) else v + 1

def _safe_norm(v: np.ndarray) -> float:
    """Compute norm with 1e-9 epsilon to avoid division-by-zero."""
    return float(np.linalg.norm(v) + 1e-9)

def _clip(v: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return float(max(lo, min(hi, v)))
```

**Convention**: prefix with `_` for private utilities.

### Math-Heavy Section Optimization

Vectorized numpy operations for performance:

```python
def _polyline_curvature_mean(pts_m: np.ndarray) -> float:
    """Mean absolute curvature from discrete 3-point estimate."""
    if pts_m is None or len(pts_m) < 3:
        return 0.0
    p0 = pts_m[:-2]
    p1 = pts_m[1:-1]
    p2 = pts_m[2:]
    a = p1 - p0
    b = p2 - p1
    c = p2 - p0
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1) + 1e-6
    kappa = 2.0 * cross / denom
    return float(np.mean(np.abs(kappa)))
```

**Pattern**: avoid Python loops; use numpy broadcasting for 2+ array operations.

### Configuration-First Design

**Never hardcode** values in functions. Always extract to `config.py`:

```python
# config.py
HEADING_STRAIGHT_THRESH = 5.0   # degrees
HEADING_TURN_THRESH = 20.0      # degrees
SPEED_MAX = 1.5                 # m/s
SPEED_TURN = 0.8                # m/s
SPEED_SHARP_TURN = 0.4          # m/s
SPEED_OBSTACLE_NEAR = 0.2       # m/s
SPEED_STOP = 0.0                # m/s

# heading.py
from config import (
    HEADING_STRAIGHT_THRESH,
    HEADING_TURN_THRESH,
    SPEED_MAX,
    SPEED_TURN,
    SPEED_SHARP_TURN,
    SPEED_OBSTACLE_NEAR,
    SPEED_STOP,
)

def compute_speed(heading_deg: float, ...) -> float:
    if abs(heading_deg) < HEADING_STRAIGHT_THRESH:
        return SPEED_MAX
    if abs(heading_deg) >= HEADING_TURN_THRESH:
        return SPEED_SHARP_TURN
    ...
```

### Result Container Pattern

Use dataclasses to return multiple values with semantic labels:

```python
@dataclass
class PathPlanResult:
    has_path: bool
    path_model: Optional[CubicPathModel]
    best_path_m: np.ndarray
    best_path_px: np.ndarray
    control_path_px: np.ndarray
    candidate_paths_m: List[np.ndarray]
    candidate_paths_px: List[np.ndarray]
    skeleton_px: np.ndarray
    graph_nodes: int
    graph_edges: int
    t_skeleton_ms: float
    t_path_ms: float
    path_source: str = "none"
    # ... and 15 more detailed fields

# Usage:
result = extractor.process(bev_mask)
if result.has_path:
    print(f"Found {result.graph_nodes} nodes")
    print(f"Best path: {result.best_path_m.shape[0]} points")
```

---

## Real-Time Constraints

### Target Performance

- **Frequency**: ≥10 Hz on Raspberry Pi 4 (100 ms per frame)
- **Blocking**: avoid any blocking calls in main pipeline loop
- **Memory**: pre-allocate arrays where possible; avoid repeated malloc

### Profiling Patterns

Methods often measure execution time:

```python
def process(self, mask: np.ndarray) -> PathPlanResult:
    t0_skel = time.time()
    skel = extract_skeleton(mask)
    t_skeleton_ms = 1000 * (time.time() - t0_skel)

    t0_path = time.time()
    result = self._find_best_path(skel, mask)
    t_path_ms = 1000 * (time.time() - t0_path)

    return PathPlanResult(
        ...,
        t_skeleton_ms=t_skeleton_ms,
        t_path_ms=t_path_ms,
    )
```

---

## BEV Coordinate System

**Critical convention** for all path/mask processing:

- **Origin**: (0, 0) at bottom-left of BEV image
- **X-axis**: forward direction (but stored as pixel row in images)
- **Y-axis**: lateral (stored as pixel column in images)
- **Forward**: decreasing row index (top-down in visualization)

**Metric conversion** (from pixels to real-world):

```python
# BEV image shape: (H=220, W=220) pixels
# Metric frame: X (forward) = [0, 11] m, Y (lateral) = [-6, +6] m

forward_m = (220 - row) / 220 * 11.0  # top of image = forward
lateral_m = (col - 110) / 220 * 6.0   # center = 0, right = +, left = -
```

**In code** (`realtime_nav_core.py`):
```python
x_m = 1.0  # 1 meter forward
y_m = 0.5  # 0.5 meters to the right (lateral)
```

---

## Summary Checklist

Before committing code:

- [ ] All public functions have type hints and Google-style docstrings
- [ ] Constants are in `config.py`, not hardcoded
- [ ] Dataclasses use immutable defaults; regular classes handle state
- [ ] No mutations of input arrays; all returns are new objects
- [ ] Error handling is explicit (try/except, sentinel values)
- [ ] Private functions prefixed with `_`
- [ ] Metric units in variable names (`_m`, `_px`, `_deg`, `_s`)
- [ ] BEV coordinates respected (origin bottom-left, forward = decreasing y)
- [ ] No print debugging (use logging if needed)
- [ ] Real-time constraints checked (timing measurements in results)
