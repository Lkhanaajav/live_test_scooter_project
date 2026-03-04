# Architecture

**Analysis Date:** 2026-03-04

## Pattern Overview

**Overall:** Modular sensor-to-control pipeline with three functional layers: perception, path planning, and control.

**Key Characteristics:**
- Real-time multi-module pipeline (8 Hz target on embedded boards)
- Stateful controllers with temporal smoothing and hysteresis
- Model-based inference (SegFormer for segmentation, YOLO for detection)
- Coordinate frame transformations (image → BEV → metric → control)
- Graceful degradation with fallback behaviors when path detection fails

## Layers

**Perception Layer:**
- Purpose: Semantic understanding of environment (road/sidewalk detection, obstacle detection)
- Location: `fast_road_detector.py`, YOLOv8 integration in `live_heading_demo.py`
- Contains: Deep learning inference (SegFormer segmentation), object detection, frame timing/memory profiling
- Depends on: OpenCV, PyTorch, Hugging Face transformers, YOLO model files
- Used by: BEV transformation layer

**Geometric Processing Layer:**
- Purpose: Convert perception outputs to navigable path representation
- Location: `realtime_nav_core.py` (BEVPathExtractor class)
- Contains: BEV coordinate transforms, morphological preprocessing, medial-axis skeleton extraction, graph construction, path search with branch pruning
- Depends on: NumPy, OpenCV (thinning, distance transform, connected components)
- Used by: Control layer, visualization

**Control Layer:**
- Purpose: Generate steering and speed commands from path representation
- Location: `realtime_nav_core.py` (AdaptivePurePursuitController class)
- Contains: Pure pursuit controller with adaptive lookahead, steering rate limiting, fallback decay, discontinuity detection
- Depends on: Path model (CubicPathModel), kinematics parameters
- Used by: Command output (serial, visualization, logging)

**Integration Layer:**
- Purpose: Orchestrate end-to-end pipeline with auxiliary features (GPS, logging, camera input)
- Location: `live_heading_demo.py`, `camera_waypoint_pipeline.py`
- Contains: Main event loop, camera capture, serial communication (GPS/scooter), CSV logging, visualization, frame stabilization, temporal smoothing
- Depends on: All other layers, serial/threading libraries
- Used by: Direct execution

**Analysis/Reporting Layer:**
- Purpose: Post-hoc evaluation and thesis figure generation
- Location: `analyze_log.py`
- Contains: CSV log parsing, timing statistics, visualization generation (matplotlib)
- Depends on: Pandas, NumPy, Matplotlib
- Used by: Research reporting

## Data Flow

**Real-time Navigation Loop:**

1. **Capture**: Read frame from camera (USB/iPhone Continuity) or video file
2. **Detect Road**: SegFormer inference on frame → semantic map (road/sidewalk/background)
3. **Detect Obstacles**: YOLO inference → bounding boxes + distances (if enabled)
4. **BEV Transform**: Perspective warp semantic map to bird's-eye-view coordinate frame
5. **Extract Skeleton**: Morphological preprocessing → distance transform → medial-axis thinning → remove small branches
6. **Build Graph**: Extract connectivity from skeleton pixels → nodes + edges with geometric properties
7. **Search Paths**: Depth-limited best-first search from ego position → scored candidates
8. **Fit Spline**: Cubic polynomial y(x) fit with curvature regularization (if valid path exists)
9. **Control**: Pure pursuit with adaptive lookahead → steering angle + speed command
10. **Fuse GPS** (optional): NMEA waypoint tracking adjusts heading reference
11. **Smooth/Log**: Exponential moving average on steering, CSV row per frame
12. **Output**: Send to scooter serial port, display visualization, update state machines

**State Management:**
- `BEVPathExtractor`: Maintains previous edge signature, end heading, branch hold counter for hysteresis
- `AdaptivePurePursuitController`: Stores previous steering angle, path model, failure counter for fallback decay
- `DataLogger`: Accumulates frame data, flushes to CSV on file rotation
- `live_heading_demo.py`: Maintains camera calibration (homography), frame stabilization trajectory, mask temporal buffers

## Key Abstractions

**CubicPathModel:**
- Purpose: Differentiable path representation for control law evaluation
- Examples: `realtime_nav_core.py` lines 196-248
- Pattern: Immutable data container with query methods (y_of_x, heading_of_x, curvature_of_x, sample_xy)
- Enables efficient lookahead point computation without re-evaluating skeleton

**AxisEdge:**
- Purpose: Skeleton graph edge with metric and topological properties
- Examples: `realtime_nav_core.py` lines 145-155
- Pattern: Dataclass aggregating pixel path, metric coordinates, curvature, signature (for hysteresis)
- Used for branch pruning, candidate path assembly, stability tracking

**PathCandidate:**
- Purpose: Scored navigation alternative for best-path selection
- Examples: `realtime_nav_core.py` lines 158-166
- Pattern: Dataclass with cost function combining progress/curvature/heading continuity
- Enables branch switching with configurable hysteresis margin

**ControlOutput:**
- Purpose: Unified representation of steering + speed command with diagnostics
- Examples: `realtime_nav_core.py` lines 186-194
- Pattern: Dataclass containing steering angle (deg), lookahead distance, curvature command, heading, target point, path validity flag
- Passed to serial output, visualization, logging

**PathExtractorConfig / PurePursuitConfig:**
- Purpose: Hyperparameter bundles for tuning perception-to-control pipeline
- Examples: `realtime_nav_core.py` lines 91-142
- Pattern: Immutable dataclasses with metric-aware thresholds (meters, degrees, frames)
- Enables A/B testing different configurations without code modification

## Entry Points

**live_heading_demo.py:**
- Location: `live_heading_demo.py` main block (line ~800+)
- Triggers: Direct execution or as module import
- Responsibilities: Real-time camera loop, GPS serial interface, obstacle detection, scooter serial output, frame visualization, CSV logging
- Supports: Multiple input sources (USB camera, iPhone Continuity, video file), optional GPS/serial scooter control, calibration mode

**camera_waypoint_pipeline.py:**
- Location: `camera_waypoint_pipeline.py` main block
- Triggers: Alternative entry point for waypoint-following variant
- Responsibilities: Similar to live_heading_demo but adds GPS waypoint navigation (bearing correction), simplified obstacle handling
- Supports: Video input, GPS CSV route file, BEV visualization

**fast_road_detector.py (standalone):**
- Location: `fast_road_detector.py` lines 100+ (FastRoadDetector class)
- Triggers: Can be invoked standalone for road detection benchmarking
- Responsibilities: SegFormer model loading, inference, performance profiling (FPS, memory)
- Supports: Batch processing, FPS measurement, system info collection

**analyze_log.py:**
- Location: `analyze_log.py` main block
- Triggers: Post-processing invocation with CSV log paths
- Responsibilities: Statistical analysis, figure generation, ablation comparison
- Supports: Glob pattern matching (multiple runs), LaTeX table output, matplotlib visualization

## Error Handling

**Strategy:** Graceful degradation with fallback behaviors rather than exception propagation.

**Patterns:**

- **Path Extraction Failure**: If skeleton is empty or all paths too short, returns empty PathPlanResult with has_path=False. Controller falls back to previous steering angle with exponential decay (fallback_decay=0.85).

- **Discontinuous Path Update**: If new path deviates >0.45 m laterally or >25 degrees in heading from previous, reject update for current cycle. Maintain previous path model to prevent jitter.

- **Segmentation Instability**: Temporal IoU checking (SEG_IOU_FAIL=0.22, SEG_IOU_WARN=0.35) caps speed if inconsistency detected. After 6 consecutive unstable frames, apply SPEED_SEG_UNSTABLE=0.20 m/s limit.

- **Obstacle Proximity**: If any detection within OBSTACLE_STOP_M=1.0 m, set speed to 0. If within OBSTACLE_CLOSE_M=3.0 m, reduce to SPEED_OBSTACLE_NEAR=0.3 m/s.

- **GPS Failure**: Optional serial input. If no valid fix or timeout, skip GPS fusion step. GPS_STEER_GAIN=0.35 modulates correction to prevent over-dependence.

- **Camera Frame Issues**: Stabilization module (STAB_MAX_CORRECTION_PX=50, STAB_MAX_CORRECTION_DEG=3.0) detects optical flow shake, applies trajectory smoothing (STAB_SMOOTHING_RADIUS=20 frames).

## Cross-Cutting Concerns

**Logging:**
- Per-frame CSV via DataLogger in `live_heading_demo.py`. Fieldnames include timing (8 modules), heading, speed, detections, GPS, path geometry. Flushed every N frames or on SIGINT.
- Format: Timestamp-indexed, easily post-processable with pandas/numpy.
- Usage: `analyze_log.py` generates thesis figures and ablation comparisons.

**Validation:**
- Coordinate transforms (metric_from_pixel, pixel_from_metric) validate bounds clipping to prevent out-of-bounds array access.
- Polyline functions (_polyline_length_m, _polyline_curvature_mean) handle edge cases (None input, < 2 points).
- Cubic fit regularization (_fit_regularized_cubic) enforces non-decreasing x and curvature bounds.

**Authentication:**
- GPS NMEA parsing validates fix quality (gps_fix_quality ≥ 1 = valid).
- Serial port communication (scooter control) expects checksummed output packets.

**Performance:**
- Target 8 Hz (125 ms per cycle) on Rock 5B (small ARM board).
- Profiling: SegFormer inference ~80-120 ms, skeleton extraction ~10-20 ms, control ~2-5 ms.
- Memory: Efficient numpy operations, no dynamic allocation per frame (pre-allocated buffers for BEV).

**Configurability:**
- All thresholds exposed in module-level constants (HEADING_STRAIGHT_THRESH, SPEED_MAX, etc.) or dataclass configs.
- No hardcoded magic numbers in control law.

---

*Architecture analysis: 2026-03-04*
