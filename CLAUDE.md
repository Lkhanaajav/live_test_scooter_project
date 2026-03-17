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
