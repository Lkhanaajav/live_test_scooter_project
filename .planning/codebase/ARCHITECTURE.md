# Architecture — Autonomous Scooter Navigation

## Overview

The scooter navigation system is a **real-time perception-to-control pipeline** designed for monocular-camera autonomous driving on a Raspberry Pi-class embedded computer. The entire system is implemented in Python with a focus on low-latency execution (target ≥10 Hz) and safety-critical reliability.

**Core principle**: Single source of truth for all configuration constants (`config.py`), immutable data structures, and explicit error handling at system boundaries.

---

## Pipeline Architecture

### High-Level Data Flow

```
Camera Frame
    ↓
[Segmentation] ← SegFormer (binary road/sidewalk)
    ↓
[BEV Transform] ← Homography matrix + temporal smoothing
    ↓
[Path Planning] ← Template arc planner + waypoint turn planner
    ↓
[Obstacle Detection] ← YOLOv8n + BEV projection
    ↓
[Intent Conditioning] ← GPS waypoint following (straight/left/right)
    ↓
[Control Output] ← Pure pursuit + speed profile + safety gates
    ↓
Speed (m/s) + Heading (degrees)
```

### Core Pipeline Stages

#### 1. **Segmentation** (`fast_road_detector.py`, `config.py`)

- **Model**: SegFormer (binary classifier or multi-class)
- **Input**: Camera frame resized to 640×360 or 512×288 (configurable)
- **Output**: Semantic mask (uint8) with road (ID=1) and sidewalk (ID=2) labels
- **Optimization**: GPU inference with optional ONNX acceleration; frame skipping for low-power mode
- **Temporal smoothing**: EMA blending with mask stability gates (IoU threshold: `SEG_IOU_FAIL=0.22`, `SEG_IOU_WARN=0.35`)
- **Constants source**: `config.py` lines 13–91 (model selection, resolution, stability gates)

#### 2. **BEV (Bird's Eye View) Transform** (`bev_calibration.py`, `realtime_nav_core.py`)

- **Technique**: Perspective homography (4-point correspondence)
- **Calibration**: Stored in `bev_calibration.npy` and `bev_calibration_meta.json`
- **Output**: BEV mask (360×660 pixels, bottom-left origin)
- **Coordinate system**:
  - (0, 0) = bottom-left (ego position)
  - Forward direction = upward (decreasing y)
  - Lateral range ±6m (configurable: `NAV_BEV_LATERAL_M`)
  - Forward range 11m (configurable: `NAV_BEV_FORWARD_M`)
- **Enhancement options**:
  - Predictive frame reuse (`bev_predictor.py`, flag: `PREDICT_ENABLED`)
  - Enhanced morphological cleaning (`masks.py`, flag: `MORPH_ENHANCED`)
  - Distance-transform safe corridor extraction (`safe_corridor.py`, flag: `DT_CORRIDOR_ENABLED`)

#### 3. **Path Planning** (Multiple strategies, Phase 11+)

**Primary: Template Arc Planner** (`template_path_planner.py`)
- Ego-anchored template arcs (8 pre-computed candidates) approved against corridor geometry
- Returns cubic path coefficients: `(a0, a1, a2, a3)`
- Habitat: Medial-axis skeleton extraction (older) superseded by template approach
- Design: Lightweight, explainable rule-based alternatives to deep learning

**Specialized: GPS-Intent Waypoint Turn Planner** (`waypoint_turn_planner.py`, Phase 11.1)
- Builds turn paths when GPS intent is "left" or "right"
- Scans a decision band (2–7m forward) for commanded-side corridor support
- Generates cubic Hermite paths (entry → apex → exit)
- Gates on: (a) target support threshold, (b) path containment ratio
- Output: `WaypointTurnResult` with path, confidence, slowdown recommendation
- Enabled via: `WAYPOINT_TURN_ENABLED=True` in config.py

**Fallback methods** (research improvements):
- DT Ridge Planner (`dt_path_planner.py` if available): Dijkstra on distance-transform cost grid
- Skeleton Hybrid: Coarse medial-axis + DT refinement (`skeleton.py`)

#### 4. **Obstacle Detection** (`object_detector.py`, `bev_obstacle.py`)

- **Model**: YOLOv8n (3.2 MB nano; 80 COCO classes, filtered to person/bike/car/etc.)
- **Detection classes** (config.py line 168–169): person, bicycle, car, motorcycle, bus, truck, cat, dog
- **BEV projection**:
  - YOLO foot-point (bbox bottom-center) → homography to BEV pixel space
  - Pixel → metric conversion (forward_m, lateral_m)
  - EMA grid accumulation (`BEV_OBSTACLE_ALPHA=0.50`)
- **Safety gates**:
  - `OBSTACLE_CLOSE_M=3.0`: trigger speed reduction
  - `OBSTACLE_STOP_M=1.0`: trigger full stop
  - `BEV_HARD_BLOCK_DIST_M=1.2`: hard mask blocking (closer obstacles)
- **Integration**: Obstacle density maps into path scoring via `BEV_OBSTACLE_PENALTY_WEIGHT=3.0`

#### 5. **Intent & GPS Navigation** (`gps_navigator.py`, `intent_picker.py`)

- **GPS**: WGS84 (lat/lon) → metric waypoint followers
- **Intent types**: STRAIGHT (heading <12°), LEFT/RIGHT (12–40°), SHARP (>40°)
- **Heading estimation** (`heading.py`): from medial-axis skeleton derivative or path curvature
- **Temporal filtering** (`path_smoother.py`): EMA smoothing on heading angle with reset hysteresis
- **Speed profile** (config.py lines 159–164):
  - `SPEED_MAX=1.5`: clear straight path
  - `SPEED_TURN=0.8`: gentle turns
  - `SPEED_SHARP_TURN=0.4`: sharp turns
  - `SPEED_OBSTACLE_NEAR=0.3`: obstacle within 3m
  - `SPEED_STOP=0.0`: full stop

#### 6. **Control Output** (`realtime_nav_core.py`)

- **Controller**: Adaptive pure pursuit with heading and speed regulation
- **Outputs**: `(speed_mps, steer_angle_deg)`
- **Execution frequency**: ≥10 Hz (real-time constraint)
- **Safety gates**:
  - Segmentation instability → `SPEED_SEG_UNSTABLE=0.20`
  - Missing path → hold previous command or fallback planner
  - Obstacle proximity → speed reduction or stop

---

## Module Boundaries & Responsibilities

### Core Orchestrator

**`realtime_nav_core.py`** (128 KB, 3600+ lines)
- Single entry point for the navigation pipeline
- Manages frame processing loop and timing
- Coordinates: segmentation → BEV → obstacle → path planning → control
- Implements safety gates, temporal smoothing, and fallback chains
- Exports: `BEVPathExtractor`, `BEVPathResult`, `FrameBuffer`

### Segmentation & BEV

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `config.py` | All shared constants (330+ lines) | `MODEL_DIR`, speed/heading/obstacle thresholds |
| `fast_road_detector.py` | SegFormer inference + logging | `FastRoadDetector`, `load_model()` |
| `bev_calibration.py` | H matrix persistence & visualization | `BEVCalibrator`, `load_calibration()` |
| `bev_predictor.py` | Frame reuse prediction via motion | `BEVPredictor`, `predict_next_bev()` |
| `masks.py` | Morphological cleaning + DT processing | `clean_bev_mask()`, `clean_bev_mask_enhanced()` |
| `stabilization.py` | Camera shake compensation | `TemporalMaskSmoother` |

### Path Planning

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `template_path_planner.py` | Corridor + template approval | `Corridor`, `TemplateApprovalResult`, `approve_template_bank()` |
| `waypoint_turn_planner.py` | GPS-intent turn planning | `WaypointTurnPlanner`, `plan_waypoint_turn()` |
| `skeleton.py` | Medial-axis extraction (legacy) | `extract_skeleton()`, `prune_branches()` |
| `safe_corridor.py` | DT-based corridor (research) | `DtSafeCorridor`, `get_default_dt_corridor()` |
| `dt_path_planner.py` | DT Ridge path (research) | `DtPathPlanner`, `DtPlannerConfig` |

### Obstacle & Intent

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `object_detector.py` | YOLOv8 inference wrapper | `ObjectDetector`, `filter_detections()` |
| `bev_obstacle.py` | BEV projection + grid EMA | `ObstacleEMAGrid`, `project_foot_to_bev()` |
| `gps_navigator.py` | Waypoint following | `GPSNavigator`, `compute_gps_intent()` |
| `intent_picker.py` | Intent type classification | `pick_intent()` |
| `heading.py` | Heading from skeleton/path | `compute_heading()` |
| `path_smoother.py` | Temporal EMA on path/heading | `PathTemporalSmoother`, `HeadingTemporalFilter` |

### Visualization & Logging

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `visualization.py` | HUD overlays (camera + BEV) | `draw_heading_hud()`, `draw_bev_hud()` |
| `data_logger.py` | Session telemetry (CSV + MP4) | `SessionLogger`, `log_frame()` |
| `analyze_log.py` | Post-processing analysis | `FrameAnalyzer` |

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `live_heading_demo.py` | Standalone demo (segmentation → BEV → heading) |
| `simple_road_pipeline.py` | Simplified binary road pipeline (research variant) |
| `camera_waypoint_pipeline.py` | Image-space path planner (legacy) |
| `boundary_model.py`, `boundary_inference.py` | Road boundary segmentation (experimental) |
| `scooter_commander.py` | Hardware interface (if present) |

---

## Entry Points

### Primary

- **`realtime_nav_core.py`**: Main pipeline orchestrator
  - Import: `from realtime_nav_core import BEVPathExtractor, FrameBuffer`
  - Initialize: `extractor = BEVPathExtractor(config_dict)`
  - Call: `result = extractor.process_frame(frame_bgr, H, detections)`

### Demos & Scripts

- **`live_heading_demo.py`** (standalone, 63 KB): Full pipeline with GUI visualization
- **`simple_road_pipeline.py`**: Simplified path pipeline (binary mask only)
- **Scripts in `scripts/`**: Evaluation, training, analysis, replay (see STRUCTURE.md)

### Testing

- **`tests/conftest.py`**: pytest fixtures (camera frames, calibration, detections)
- **Test files** (e.g., `test_realtime_nav_core.py`): Unit + integration tests

---

## Key Abstractions & Interfaces

### Data Classes (Immutable, Frozen)

```python
# From template_path_planner.py
@dataclass(frozen=True)
class Corridor:
    left_poly_m: np.ndarray       # (N, 2) meter coords
    right_poly_m: np.ndarray
    centerline_m: np.ndarray
    width_m_per_point: np.ndarray # scalar per row

@dataclass(frozen=True)
class TemplateApprovalResult:
    path_m: np.ndarray            # (n_pts, 2) cubic path
    success: bool
    reason: str
    confidence: float             # [0, 1]
    speed_slowdown: float         # [0, 1]

# From waypoint_turn_planner.py
@dataclass(frozen=True)
class WaypointTurnResult:
    path_m: Optional[np.ndarray]
    confidence: float             # [0, 1]
    hold: bool                    # if True, skip turn
    slowdown: float               # [0, 1] speed reduction
    side: Optional[str]           # "left"/"right"
```

### Configuration Classes

```python
@dataclass(frozen=True)
class TemplatePlannerConfig:
    bev_forward_m: float
    bev_lateral_m: float
    bev_height_px: int
    bev_width_px: int
    # ... 20+ tunable parameters

@dataclass(frozen=True)
class WaypointTurnPlannerConfig:
    decision_band_min_m: float
    acquire_support_min: float
    # ... 12+ thresholds from config.py
```

### Safety Interface

```python
# From realtime_nav_core.py
class BEVPathResult:
    path_m: np.ndarray            # output path (meters)
    confidence: float             # [0, 1]
    method: str                   # "template", "waypoint", "dt", "skeleton"
    fallback_reason: Optional[str]
    speed_slowdown: float         # [0, 1]
```

---

## Safety Architecture

### Safety Gates (Non-negotiable)

#### Segmentation Stability
- **Trigger**: Frame-to-frame IoU < `SEG_IOU_FAIL` (0.22)
- **Action**: Speed cap at `SPEED_SEG_UNSTABLE` (0.20 m/s)
- **Hold count**: Must exceed `SEG_FAIL_HOLD_FRAMES` (6 consecutive)
- **Location**: `realtime_nav_core.py` lines ~800–850

#### Obstacle Proximity
- **Close** (3m): Reduce speed to `SPEED_OBSTACLE_NEAR` (0.30 m/s)
- **Stop** (1m): Full stop (`SPEED_STOP` = 0.0)
- **Hard block** (1.2m): Remove from BEV mask entirely
- **Location**: `realtime_nav_core.py` obstacle gate, `bev_obstacle.py`

#### Path Confidence
- **Template path** confidence < 0.4 → fallback chain
- **Waypoint turn** confidence < 0.35 (`WAYPOINT_LOW_CONFIDENCE_THRESHOLD`) → hold recommendation
- **Missing path**: Hold previous command or use simple-road fallback
- **Location**: `realtime_nav_core.py`, `template_path_planner.py`

#### Path Containment (Turns)
- **Constraint**: ≥60% of path inside corridor (waypoint), ≥60% template (template planner)
- **Near-field stricter**: ≥70% in first 1.2m
- **Rasterization spill**: ≤2 pixels outside corridor allowed
- **Location**: `waypoint_turn_planner.py`, `realtime_nav_core.py`

#### Heading Validity
- **Reset condition**: Heading jump > 45° in one frame → filter reset
- **Output clipping**: [-90°, +90°] (lateral only)
- **Location**: `path_smoother.py`, `heading.py`

### Fallback Chain

```
Template Path Planning
    ↓ (if failed or low confidence)
Waypoint Turn Planning
    ↓ (if not applicable or failed)
DT Ridge Planner (if enabled)
    ↓ (if failed)
Skeleton Hybrid (if enabled)
    ↓ (if all failed)
Hold previous command
```

### Error Handling Patterns

**Rule 1**: Explicit try-catch at module boundaries (imports, inference calls)
- Example: `_HAS_DT_CORRIDOR`, `_HAS_WAYPOINT_TURN` flags (realtime_nav_core.py lines 44–79)

**Rule 2**: Fail-safe defaults (never None, always valid fallback)
- Example: Missing obstacle → treat as clear
- Example: Missing path → hold previous or stop

**Rule 3**: Detailed logging (frame-by-frame telemetry)
- Location: `data_logger.py` (CSV + MP4 with HUD overlay)

---

## Real-Time Constraints

### Target Execution Frequency
- **Primary**: ≥10 Hz on Raspberry Pi 4
- **Low-power**: ~8 Hz with frame skipping

### Latency Budget (per frame)
```
Segmentation inference:    ~60 ms (SegFormer on GPU)
BEV transform:            ~5 ms
Obstacle detection:       ~40 ms (YOLOv8n on GPU)
Path planning:            ~10 ms (template) / ~20 ms (waypoint)
Control output:           ~5 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                    ~120 ms (8–10 Hz)
```

### Optimization Strategies

1. **Frame skipping** on straight road (`PREDICT_ENABLED`)
   - Reuse BEV for up to 3 consecutive frames
   - Condition: low curvature, high confidence

2. **Low-power mode** (`LOW_POWER_*` constants)
   - Input resolution: 512×288 (vs. 640×360)
   - Stride: process every 2nd frame
   - DT corridor disabled

3. **GPU acceleration**
   - Segmentation + YOLOv8n on CUDA (if available)
   - Optional ONNX runtime for mobile deployment

---

## Design Principles

1. **Config.py is the single source of truth**
   - Never hardcode thresholds in modules
   - All tunable constants centralized
   - Enable fast experimentation without code changes

2. **Immutability by default**
   - Use `@dataclass(frozen=True)` for all outputs
   - Return new objects, never modify inputs
   - Prevents hidden side effects in time-critical code

3. **Explicit error handling**
   - No silent failures
   - Log context and recommendations
   - Safety gates are non-negotiable

4. **Safety-first over performance**
   - Prefer conservative thresholds
   - Always preserve stopping capability
   - Extensive telemetry for debugging

5. **Testability**
   - Unit test each stage independently
   - Mock hardware (camera, motor)
   - Test coverage ≥80% (pytest)
