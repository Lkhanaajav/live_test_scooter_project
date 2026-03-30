# Technology Stack

## Overview

The autonomous scooter navigation system is a **Python-based real-time perception and control pipeline** designed to run on edge compute (Raspberry Pi / Rock 5B class). The system uses monocular camera vision only—no LiDAR or stereo—and emphasizes lightweight, rule-based algorithms for explainability and deterministic timing.

## Core Language & Runtime

- **Python 3.10+** — Primary development language
- **Target Platforms:**
  - Raspberry Pi 4B+ (4GB+ RAM recommended)
  - Rock 5B or similar ARM64 SBCs
  - Ubuntu 22.04 LTS (both development and target)
- **Real-time Constraint:** ≥10 Hz on-device inference and path planning (non-blocking main loop)

## Deep Learning Frameworks

### PyTorch & Transformers

- **torch** (PyTorch 2.0+) — Core deep learning runtime
  - CUDA/MPS acceleration where available; CPU fallback for embedded targets
  - Device detection in `object_detector.py` prioritizes CUDA → MPS → CPU
- **transformers** (Hugging Face) — SegFormer semantic segmentation
  - Model: **SegformerForSemanticSegmentation** + **SegformerImageProcessor**
  - Used for road vs. sidewalk binary segmentation
  - Model files: `/simulation_camera_scooter/models/my-segformer-road`
  - Fine-tuning checkpoints: `/outputs/training/binary_segformer_*`

### Ultralytics YOLOv8

- **ultralytics** — YOLOv8-nano object detection (3.2 MB, lightweight)
  - Model: `yolov8n.pt` (downloaded auto-on-first-run from HuggingFace/ultralytics)
  - Detects: person, bicycle, car, motorcycle, bus, truck, cat, dog
  - Confidence threshold: 0.35 (tunable in `config.py`)
  - Runs inference on BGR frames; class filtering done in-code

## Computer Vision & Image Processing

- **opencv-python (cv2)** — Core image I/O and transformations
  - Camera frame capture
  - BEV homography transforms (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`)
  - Segmentation mask cleaning (morphological ops, edge detection)
  - Visualization (polylines, circles, text overlays, transparency)
  - Video codec (MP4 write with H.264)

- **numpy** — Numerical arrays and linear algebra
  - Matrix operations for BEV calibration (`bev_calibration.py`)
  - Distance transforms (scipy.ndimage.distance_transform_edt)
  - Coordinate transformations (homography, pixel↔metric conversions)

## Scientific Computing

- **scipy** — Advanced numerical operations
  - `scipy.ndimage.distance_transform_edt` — Distance transform for path skeleton
  - `scipy.interpolate.splprep` — Spline fitting for smooth path coefficients
  - `scipy.signal.savgol_filter` — Savitzky-Golay smoothing (temporal + spatial)

- **networkx** — Graph algorithms for path planning
  - Used in skeleton-based and Dijkstra-based corridor extraction
  - Graph construction from BEV mask topology

- **PIL (Pillow)** — Image file I/O
  - Dataset annotation frame loading (`boundary_dataset.py`)
  - PNG/JPG reading for test data

## Data & Logging

- **csv** (stdlib) — Per-frame session logging
  - Timestamped CSV rows with timing, heading, speed, GPS, detections, path info
  - Log files: `logs/run_YYYYMMDD_HHMMSS.csv` and `_meta.json`
  - 40+ fields per frame for post-hoc analysis and thesis experiments

- **json** (stdlib) — Model metadata and configuration
  - BEV calibration metadata: `bev_calibration_meta.json`
  - Training summaries: `outputs/training/*/training_summary.json`
  - GPS waypoint files (implicit CSV format, not JSON)

- **pandas** — Data frame operations (optional, used in analysis scripts)
  - Post-processing logged CSV data
  - Aggregating performance metrics across runs

## System Monitoring & Hardware

- **psutil** — CPU/memory profiling
  - Used in `fast_road_detector.py` to track real-time performance
  - Memory usage tracking (GPU allocated/reserved, CPU resident)

- **platform** (stdlib) — OS/architecture detection
  - Runtime environment discovery in performance logging

- **threading** (stdlib) — Background I/O threads
  - GPS serial reader (`gps_navigator.py`)
  - Non-blocking architecture for real-time loop

## Serial Communication

- **pyserial** — Hardware interfaces (optional, graceful degradation if absent)
  - GPS receiver (NMEA sentence parsing): `gps_navigator.py`
  - Scooter motor/steering commands: `scooter_commander.py`
  - Protocol: simple text lines (`CMD,<steer_deg>,<speed_mps>\n`)
  - Gracefully disables features if library not installed (print warning + continue)

## Testing & Development

- **pytest** — Test framework
  - Test directory: `/simulation_camera_scooter/tests/`
  - Markers: `@pytest.mark.unit`, `@pytest.mark.integration`
  - Shared fixtures in `conftest.py`
  - Target coverage: ≥80% (run with `--cov=.`)
  - Current test suite: 13 test modules, 104 passing tests

- **black** — Code formatting (optional, not enforced in CI)
  - PEP 8 compliance
  - Configuration in `.flake8` or `pyproject.toml` (if present)

- **mypy** — Static type checking (optional)
  - Type annotations on all function signatures (enforced via code review)
  - Python 3.10+ syntax (dataclass, type hints)

## Configuration Management

### Single Source of Truth

All tunable constants are in `/simulation_camera_scooter/config.py`:

- **Segmentation:** model dir, input resolution, IOU thresholds
- **BEV transforms:** homography points, ego position, coordinate scaling
- **Path planning:** skeleton thresholds, template configs, approval gates
- **Obstacle detection:** YOLO confidence, BEV projection radii, stop distances
- **Speed profiles:** max speed, turn speeds, obstacle slowdown
- **GPS:** waypoint radius, steering gains
- **Safety gates:** instability thresholds, hold-frame counts
- **Research improvements:** flags for enhanced morphology, DT corridors, temporal smoothing

Module imports use: `from config import CONSTANT_NAME`

No hardcoded values in other modules (enforced via code review).

### Calibration Files

- `bev_calibration.npy` — 4-point homography matrix (binary numpy array)
- `bev_calibration_meta.json` — Optional ego-position fraction (dict)
- Auto-loaded on startup; gracefully uses defaults if absent

### Model Directories

**Priority-based model selection** (`config.py` lines 23–85):
1. `outputs/training/binary_segformer_oneformer_teacher/best_checkpoint` (highest priority)
2. `outputs/training/binary_segformer_old400_plus_img_1931_t300/best_checkpoint`
3. `outputs/training/binary_segformer_all6_t400/best_checkpoint`
4. `simulation_camera_scooter/models/my-segformer-road` (fallback)

Selection is based on recorded `training_summary.json::best_metrics::best_val_iou`. Highest IoU wins.

YOLOv8 model: auto-downloaded from Hugging Face on first load (cached locally).

## Build & Installation

### Prerequisites

```bash
# Core dependencies
pip install torch transformers ultralytics opencv-python numpy scipy networkx pandas pillow psutil

# Optional (for GPS/serial)
pip install pyserial

# Development only
pip install pytest black mypy
```

### Environment Setup

1. Clone repository to `C:\Users\lhana\OneDrive\Desktop\scootedr\live_test_scooter_project`
2. Ensure `/simulation_camera_scooter/models/` and `/outputs/training/` are populated with trained checkpoints
3. Optional: place GPS waypoint CSV and BEV calibration NPY files in project root
4. Run tests: `cd simulation_camera_scooter && pytest tests/ -v`

### Performance Optimization Notes

- **Low-Power Profile** (`config.py` lines 240–242):
  - `LOW_POWER_STRIDE=2`: process every 2nd frame
  - `LOW_POWER_DETECTION_STRIDE=2`: run YOLO every 2nd inference frame
  - `LOW_POWER_PATH_SCALE=0.65`: scale BEV processing grid

- **BEV Predictive Reuse** (`bev_predictor.py`):
  - Skip segmentation on straight roads (up to 3 frames)
  - Blend predicted vs. computed BEV masks (75% weight on compute frames)
  - Confidence floor: stop skipping if IoU drops below 0.50

- **Temporal Smoothing**:
  - EMA on segmentation masks (α=0.55–0.65 tunable)
  - Cubic path coefficient smoothing (α=0.35–0.85 adaptive to confidence)
  - Circular EMA on heading angle (±180° wrap-aware)

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
