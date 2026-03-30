# CLAUDE.md — Autonomous Scooter Navigation Project

This file provides context to Claude Code when working in this repository.

## Project Overview

Monocular-camera autonomous driving system for a scooter. Runs on a Raspberry Pi / onboard computer with a single forward-facing camera. No LiDAR. No stereo.

**Core pipeline** (all Python, all real-time):
1. **Segmentation** — SegFormer (`models/my-segformer-road`) classifies road vs sidewalk
2. **BEV transform** — Homography projects front-view mask to bird's-eye view
3. **Path planning** — Medial-axis skeleton + template path planner (Phase 11)
4. **Obstacle detection** — YOLOv8n (`yolov8n.pt`) for dynamic objects
5. **GPS navigation** — Intent conditioning from GPS waypoints
6. **Speed/heading control** — Rule-based pure pursuit + safety gates

## Directory Structure

```
simulation_camera_scooter/   # Main Python package
  config.py                  # ALL shared constants (speed, thresholds, BEV params)
  realtime_nav_core.py        # Top-level pipeline orchestrator
  template_path_planner.py    # Phase 11: template arc planner
  bev_predictor.py            # Predictive BEV frame reuse
  bev_obstacle.py             # Obstacle projection in BEV
  gps_navigator.py            # GPS waypoint following
  intent_picker.py            # GPS-conditioned intent (straight/left/right)
  skeleton.py                 # Medial-axis path extraction
  heading.py                  # Heading estimation from path
  object_detector.py          # YOLOv8 wrapper
  fast_road_detector.py       # Low-latency road segmentation
  boundary_model.py           # Road boundary model
  visualization.py            # BEV + overlay rendering
  data_logger.py              # Session logging
  stabilization.py            # Camera shake compensation
  tests/                      # pytest test suite
  models/                     # Model checkpoints
  annotation_frames/          # Fine-tuning frame extracts

thesis/                      # LaTeX paper (do not modify unless asked)
```

## Key Design Rules

- **config.py is the single source of truth** — never hardcode values in modules
- **Safety gates are non-negotiable** — `SEG_IOU_FAIL`, `SPEED_SEG_UNSTABLE`, `OBSTACLE_STOP_M` etc. must not be relaxed without explicit user request
- **Real-time constraint** — target ≥ 10 Hz on Raspberry Pi 4; avoid anything that blocks the main loop
- **No deep learning in the planner** — template path planner is deliberately rule-based for explainability and safety
- **BEV coordinate system** — (0,0) is bottom-left of BEV image; forward is up (decreasing y)

## Current Status (Phase 11 complete)

- Template arc path planner with GPS intent conditioning
- 8-meter planning horizon
- Annotation frame extraction tools for fine-tuning

## Workflow Commands

| Command | Use |
|---------|-----|
| `/python-review` | Review Python code quality, type hints, security |
| `/code-review` | General code review before committing |
| `/plan` | Plan a new feature or phase (waits for confirm) |
| `/build-fix` | Fix import errors / mypy failures |
| `/tdd` | Test-driven development workflow |
| `/learn` | Extract reusable patterns from the session |
| `/verify` | Run verification loop on completed work |
| `/refactor-clean` | Clean dead code / dead imports |

## Testing

```bash
cd simulation_camera_scooter
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing
```

## Python Environment

- Python 3.10+
- Key deps: `torch`, `transformers` (SegFormer), `ultralytics` (YOLOv8), `opencv-python`, `numpy`, `scipy`
- Install: `pip install -r requirements.txt` (if present) or install manually

## Agents Available

- `python-reviewer` — PEP 8, type hints, security, Pythonic idioms
- `planner` — feature/phase planning with structured plan format
- `security-reviewer` — safety-critical code review
- `code-reviewer` — general quality review
- `build-error-resolver` — fix mypy/import errors fast
- `refactor-cleaner` — remove dead code

## Critical Files

- `config.py` — all tunable constants
- `realtime_nav_core.py` — pipeline entry point
- `template_path_planner.py` — current active planner (Phase 11)
- `tests/` — regression tests, run before every commit

<!-- GSD:project-start source:PROJECT.md -->
## Project

**Master's Thesis Rewrite**

A full rewrite of the Master's thesis for the University of Oklahoma ECE department on monocular-camera-based autonomous sidewalk navigation. The existing draft (1,323 lines of LaTeX) has solid experimental data and figures but needs restructuring, professional prose, coherent narrative, and a proper evaluation framework (vs. baseline, not model-vs-model). The goal is a publication-quality 60-80 page thesis ready for committee review.

**Core Value:** A professional, cohesive thesis that tells a clear scientific story: simple image-space geometry outperforms complex BEV pipelines for monocular sidewalk navigation on embedded platforms — supported by systematic evaluation against proper baselines.

### Constraints

- **Timeline:** ~1 week — full rewrite but no new experiments
- **Format:** University of Oklahoma thesis format (already in template)
- **Length:** 60-80 pages double-spaced
- **Data:** Must use existing experimental results — no new runs
- **Figures:** Reuse existing figures — no new data visualizations
- **Tool:** LaTeX only (main.tex + references.bib)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Overview
## Core Language & Runtime
- **Python 3.10+** — Primary development language
- **Target Platforms:**
- **Real-time Constraint:** ≥10 Hz on-device inference and path planning (non-blocking main loop)
## Deep Learning Frameworks
### PyTorch & Transformers
- **torch** (PyTorch 2.0+) — Core deep learning runtime
- **transformers** (Hugging Face) — SegFormer semantic segmentation
### Ultralytics YOLOv8
- **ultralytics** — YOLOv8-nano object detection (3.2 MB, lightweight)
## Computer Vision & Image Processing
- **opencv-python (cv2)** — Core image I/O and transformations
- **numpy** — Numerical arrays and linear algebra
## Scientific Computing
- **scipy** — Advanced numerical operations
- **networkx** — Graph algorithms for path planning
- **PIL (Pillow)** — Image file I/O
## Data & Logging
- **csv** (stdlib) — Per-frame session logging
- **json** (stdlib) — Model metadata and configuration
- **pandas** — Data frame operations (optional, used in analysis scripts)
## System Monitoring & Hardware
- **psutil** — CPU/memory profiling
- **platform** (stdlib) — OS/architecture detection
- **threading** (stdlib) — Background I/O threads
## Serial Communication
- **pyserial** — Hardware interfaces (optional, graceful degradation if absent)
## Testing & Development
- **pytest** — Test framework
- **black** — Code formatting (optional, not enforced in CI)
- **mypy** — Static type checking (optional)
## Configuration Management
### Single Source of Truth
- **Segmentation:** model dir, input resolution, IOU thresholds
- **BEV transforms:** homography points, ego position, coordinate scaling
- **Path planning:** skeleton thresholds, template configs, approval gates
- **Obstacle detection:** YOLO confidence, BEV projection radii, stop distances
- **Speed profiles:** max speed, turn speeds, obstacle slowdown
- **GPS:** waypoint radius, steering gains
- **Safety gates:** instability thresholds, hold-frame counts
- **Research improvements:** flags for enhanced morphology, DT corridors, temporal smoothing
### Calibration Files
- `bev_calibration.npy` — 4-point homography matrix (binary numpy array)
- `bev_calibration_meta.json` — Optional ego-position fraction (dict)
- Auto-loaded on startup; gracefully uses defaults if absent
### Model Directories
## Build & Installation
### Prerequisites
# Core dependencies
# Optional (for GPS/serial)
# Development only
### Environment Setup
### Performance Optimization Notes
- **Low-Power Profile** (`config.py` lines 240–242):
- **BEV Predictive Reuse** (`bev_predictor.py`):
- **Temporal Smoothing**:
## Architecture Highlights
| Component | Tech | Purpose |
|-----------|------|---------|
| **Segmentation** | SegFormer (transformers) | Road mask classification |
| **Obstacle Detection** | YOLOv8n (ultralytics) | Dynamic object tracking |
| **BEV Transform** | OpenCV (homography) | Front-view → bird's-eye projection |
| **Path Extraction** | NumPy + SciPy (Distance Transform, Dijkstra) | Skeleton-based corridor planning |
| **Path Planning** | Rule-based (scipy.interpolate) | Template arc fitting + GPS intent |
| **Control Law** | Pure pursuit (config-driven) | Steering angle computation |
| **Logging** | CSV + JSON (stdlib) | Session data & post-hoc analysis |
| **Serial I/O** | pyserial (optional) | GPS + motor control |
## Performance Targets
- **Frame Rate:** ≥10 Hz sustained on RPi 4 (target timing budget ~100 ms/frame)
- **Segmentation Latency:** <50 ms/frame (SegFormer B0 or nano variants)
- **YOLO Latency:** <30 ms/frame (YOLOv8n)
- **Path Planning:** <20 ms/frame (skeleton extraction + template fitting)
- **Memory:** <2 GB total (including model weights in RAM)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

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
## Python Coding Style
### PEP 8 Compliance
- **Line length**: target ~100 characters (practical soft limit; longer acceptable for clarity)
- **Blank lines**: 2 between module-level functions/classes, 1 between methods
- **Indentation**: 4 spaces (no tabs)
- **Imports**: grouped and ordered via isort convention
### Import Organization
- Import all config constants at module top
- Lazy imports for optional/research features (try/except around imports)
- Import modules (`import config as cfg_module`), not scattered constants
### Type Hints
- `float`, `int`, `str`, `bool` for primitives
- `np.ndarray` for arrays (with optional dtype hint in docstring)
- `Optional[T]` for nullable types
- `Sequence[T]`, `List[T]`, `Tuple[T, ...]` from `typing`
- Custom dataclass types (`PathPlanResult`, `ControlOutput`)
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
- `_m` = meters (distance, position)
- `_px` = pixels (image/mask coordinates)
- `_deg` = degrees (angles)
- `_rad` = radians (angles)
- `_mps` = meters per second (velocity)
- `_s` = seconds (time)
- `_hz` = hertz (frequency)
- `_m_inv` = meters^-1 (curvature: 1/radius)
## Data Structures & Immutability
### Dataclasses for Configuration & Results
- Readable initialization: `cfg = PathExtractorConfig(bev_forward_m=12.0)`
- Dataclass generates `__init__`, `__repr__`, `__eq__` automatically
- Type-checked at module load time
- Avoids scattered magic numbers
### Regular Classes for Stateful Objects
### Immutability in Return Values
## Documentation & Docstrings
### Docstring Style: Google Format
### Module Docstrings
- Filename
- = underline
- One-line purpose
- Blank line
- Detailed description
- (Optional) research/paper citations
- (Optional) key functions listed
### Docstring Content
- Mathematical relationships (e.g., cubic path equation, arc geometry)
- Parameter constraints (e.g., "radius must be > 0.5m")
- Return semantics (e.g., "points in metric frame (x=forward, y=lateral)")
- References to papers or research
## Error Handling
### Explicit Error Handling
### Logging, Not Printing
### Lazy Import for Research Features
- Backward compatible: code works even if new module missing
- Feature toggle: disable via config flag (`HEADING_SMOOTH_ENABLED`)
- Safe imports: no circular dependencies
## Logging Conventions
### Print-Based Logging
### CSV/Structured Logging
## Common Patterns
### Utility Helper Functions
### Math-Heavy Section Optimization
### Configuration-First Design
### Result Container Pattern
## Real-Time Constraints
### Target Performance
- **Frequency**: ≥10 Hz on Raspberry Pi 4 (100 ms per frame)
- **Blocking**: avoid any blocking calls in main pipeline loop
- **Memory**: pre-allocate arrays where possible; avoid repeated malloc
### Profiling Patterns
## BEV Coordinate System
- **Origin**: (0, 0) at bottom-left of BEV image
- **X-axis**: forward direction (but stored as pixel row in images)
- **Y-axis**: lateral (stored as pixel column in images)
- **Forward**: decreasing row index (top-down in visualization)
## Summary Checklist
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
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Overview
## Pipeline Architecture
### High-Level Data Flow
```
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
- **Enhancement options**:
#### 3. **Path Planning** (Multiple strategies, Phase 11+)
- Ego-anchored template arcs (8 pre-computed candidates) approved against corridor geometry
- Returns cubic path coefficients: `(a0, a1, a2, a3)`
- Habitat: Medial-axis skeleton extraction (older) superseded by template approach
- Design: Lightweight, explainable rule-based alternatives to deep learning
- Builds turn paths when GPS intent is "left" or "right"
- Scans a decision band (2–7m forward) for commanded-side corridor support
- Generates cubic Hermite paths (entry → apex → exit)
- Gates on: (a) target support threshold, (b) path containment ratio
- Output: `WaypointTurnResult` with path, confidence, slowdown recommendation
- Enabled via: `WAYPOINT_TURN_ENABLED=True` in config.py
- DT Ridge Planner (`dt_path_planner.py` if available): Dijkstra on distance-transform cost grid
- Skeleton Hybrid: Coarse medial-axis + DT refinement (`skeleton.py`)
#### 4. **Obstacle Detection** (`object_detector.py`, `bev_obstacle.py`)
- **Model**: YOLOv8n (3.2 MB nano; 80 COCO classes, filtered to person/bike/car/etc.)
- **Detection classes** (config.py line 168–169): person, bicycle, car, motorcycle, bus, truck, cat, dog
- **BEV projection**:
- **Safety gates**:
- **Integration**: Obstacle density maps into path scoring via `BEV_OBSTACLE_PENALTY_WEIGHT=3.0`
#### 5. **Intent & GPS Navigation** (`gps_navigator.py`, `intent_picker.py`)
- **GPS**: WGS84 (lat/lon) → metric waypoint followers
- **Intent types**: STRAIGHT (heading <12°), LEFT/RIGHT (12–40°), SHARP (>40°)
- **Heading estimation** (`heading.py`): from medial-axis skeleton derivative or path curvature
- **Temporal filtering** (`path_smoother.py`): EMA smoothing on heading angle with reset hysteresis
- **Speed profile** (config.py lines 159–164):
#### 6. **Control Output** (`realtime_nav_core.py`)
- **Controller**: Adaptive pure pursuit with heading and speed regulation
- **Outputs**: `(speed_mps, steer_angle_deg)`
- **Execution frequency**: ≥10 Hz (real-time constraint)
- **Safety gates**:
## Module Boundaries & Responsibilities
### Core Orchestrator
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
## Entry Points
### Primary
- **`realtime_nav_core.py`**: Main pipeline orchestrator
### Demos & Scripts
- **`live_heading_demo.py`** (standalone, 63 KB): Full pipeline with GUI visualization
- **`simple_road_pipeline.py`**: Simplified path pipeline (binary mask only)
- **Scripts in `scripts/`**: Evaluation, training, analysis, replay (see STRUCTURE.md)
### Testing
- **`tests/conftest.py`**: pytest fixtures (camera frames, calibration, detections)
- **Test files** (e.g., `test_realtime_nav_core.py`): Unit + integration tests
## Key Abstractions & Interfaces
### Data Classes (Immutable, Frozen)
```python
```
### Configuration Classes
```python
```
### Safety Interface
```python
```
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
```
### Error Handling Patterns
- Example: `_HAS_DT_CORRIDOR`, `_HAS_WAYPOINT_TURN` flags (realtime_nav_core.py lines 44–79)
- Example: Missing obstacle → treat as clear
- Example: Missing path → hold previous or stop
- Location: `data_logger.py` (CSV + MP4 with HUD overlay)
## Real-Time Constraints
### Target Execution Frequency
- **Primary**: ≥10 Hz on Raspberry Pi 4
- **Low-power**: ~8 Hz with frame skipping
### Latency Budget (per frame)
```
```
### Optimization Strategies
## Design Principles
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
