# Autonomous Scooter Navigation System
### Undergraduate Thesis — Lkhanaajav Mijiddorj | University of Oklahoma

> Real-time autonomous navigation for electric scooters using semantic segmentation, Bird's Eye View path planning, and a custom fine-tuned SegFormer model — running at 8 Hz on embedded hardware.

---

## What This Project Does

This system lets an electric scooter navigate sidewalks autonomously using only a single forward-facing camera. No LiDAR, no GPS lock required for local navigation. It detects drivable road surface in real time, transforms the camera view into a top-down (BEV) representation, plans a safe path through the corridor, and sends steering/speed commands to the scooter.

The full pipeline runs at **8 Hz on a Rock 5B embedded board** — fast enough for real-world navigation.

---

## Technical Stack

| Layer | Approach |
|---|---|
| Road Segmentation | Fine-tuned SegFormer-B0 (HuggingFace Transformers) |
| Model Training | Teacher-student: OneFormer (Swin-L) → SegFormer-B0 |
| View Transform | Bird's Eye View (BEV) via homography calibration |
| Path Planning | Template bank + pure pursuit controller |
| Safe Corridor | Distance Transform (EDT) based clearance computation |
| Obstacle Detection | Object detection overlay on BEV mask |
| Hardware Target | Rock 5B (ARM64), 8 Hz real-time |
| Language | Python 3.10+ |
| Key Libraries | PyTorch, HuggingFace Transformers, OpenCV, NumPy |

---

## Architecture

```
Camera Frame (1280x720 @ 30fps)
        │
        ▼
┌─────────────────────────┐
│  FastRoadDetector        │  ← SegFormer-B0 fine-tuned on sidewalk data
│  (fast_road_detector.py) │     binary: drivable / non-drivable
└────────────┬────────────┘
             │ segmentation mask
             ▼
┌─────────────────────────┐
│  BEV Transform           │  ← perspective homography (4-point calibration)
│  (bev_calibration.py)    │     camera view → top-down 600x500px map
└────────────┬────────────┘
             │ BEV road mask
             ▼
┌─────────────────────────┐
│  Mask Cleanup            │  ← morphological ops + flood-fill hole removal
│  (masks.py)              │     Gaussian boundary smoothing, ego-clearance selection
└────────────┬────────────┘
             │ clean BEV mask
             ▼
┌─────────────────────────┐
│  Safe Corridor + Planner │  ← EDT clearance map, template path bank
│  (realtime_nav_core.py)  │     adaptive pure pursuit with T/Y branch handling
└────────────┬────────────┘
             │ heading + speed
             ▼
┌─────────────────────────┐
│  Scooter Commander       │  ← motor/steering commands over serial
│  (scooter_commander.py)  │
└─────────────────────────┘
```

---

## Model Training

The road segmentation model was trained using a **teacher-student approach** to overcome limited labeled data:

1. **Teacher:** `OneFormer (Swin-L, Cityscapes)` — strong pretrained segmentation model used to generate pseudo-labels on unlabeled sidewalk footage
2. **Student:** `SegFormer-B0` — lightweight model fine-tuned on pseudo-labels, optimized for 8 Hz inference on embedded hardware
3. **Dataset:** 400 frames extracted from 4 sidewalk videos, 320 train / 80 validation
4. **Label collapse:** `road + sidewalk → drivable`, everything else `→ non-drivable`

The student model achieves comparable accuracy to the teacher at a fraction of the compute cost.

---

## Key Research Improvements

Three research-backed improvements were implemented over the baseline:

**1. Enhanced Morphological BEV Mask Pipeline** (`masks.py`)
- Flood-fill hole filling for enclosed regions the standard morphological close misses
- Gaussian boundary smoothing (σ=1.2px) to reduce segmentation quantization artifacts
- Distance transform ego-clearance component selection — picks the corridor closest to the scooter, not just the largest area

**2. Distance Transform Safe Corridor** (`safe_corridor.py`)
- Replaces row-wise scan that failed at path bifurcations (T/Y intersections)
- EDT-based corridor gives consistent clearance margin across the full path
- Referenced: Regulated Pure Pursuit (arXiv:2305.20026)

**3. Temporal Path Smoother** (`path_smoother.py`)
- Reduces steering oscillation from frame-to-frame path jitter
- Weighted temporal blending of waypoint sequences
- Referenced: Trajectory Prediction Survey (arXiv:2503.03262)

---

## Repository Structure

```
simulation_camera_scooter/
├── fast_road_detector.py      # SegFormer inference + temporal smoothing
├── bev_calibration.py         # Interactive 4-point BEV calibration tool
├── realtime_nav_core.py       # BEV path extraction + pure pursuit controller
├── camera_waypoint_pipeline.py # Full pipeline from camera frame to waypoint
├── masks.py                   # BEV mask cleanup (morphological + flood-fill)
├── safe_corridor.py           # EDT-based safe corridor computation
├── path_smoother.py           # Temporal path smoother
├── scooter_commander.py       # Hardware serial interface
├── gps_navigator.py           # GPS-based waypoint navigation
├── object_detector.py         # Obstacle detection overlay
├── config.py                  # Centralized configuration
├── models/                    # Fine-tuned SegFormer checkpoints
└── scripts/                   # Evaluation + benchmarking scripts
research/                      # Literature review and architecture decisions
docs/thesis_improvement_plan/  # Detailed improvement planning documents
```

---

## How to Run

### Requirements

```bash
pip install torch torchvision transformers opencv-python numpy networkx
```

### BEV Calibration (one-time setup)

```bash
cd simulation_camera_scooter
python bev_calibration.py --video your_video.MOV
# Click 4 sidewalk corners when prompted, press S to save
```

### Live Camera Navigation

```bash
python camera_waypoint_pipeline.py --camera 0
```

### Evaluate on Video

```bash
python fast_road_detector.py --video test_video.MOV --output result/output.mp4
```

### Run Benchmarks

```bash
python scripts/benchmark_seg_stability.py
python scripts/eval_research_improvements.py
```

---

## AI-Assisted Development

This project used a multi-agent Claude Code workflow for development — specialized agents handled planning, code review, debugging, and verification in parallel. The `.claude/agents/` directory contains the agent definitions:

| Agent | Role |
|---|---|
| `gsd-planner` | Breaks features into actionable implementation phases |
| `gsd-executor` | Implements planned tasks and writes tests |
| `gsd-debugger` | Diagnoses errors from hardware test logs |
| `gsd-verifier` | Validates output against research requirements |
| `code-reviewer` | Reviews diffs for correctness and style |
| `build-error-resolver` | Fixes dependency and import errors |

---

## Results

- **Inference speed:** 8 Hz sustained on Rock 5B (ARM64, no GPU)
- **BEV mask quality:** Flood-fill + EDT corridor eliminated corridor dropouts at T/Y intersections
- **Path smoothness:** Temporal smoother reduced heading oscillation by ~40% (measured over 5 test runs)
- **Model size:** SegFormer-B0 student model — 3.7M parameters, ~15MB on disk

---

## Background

This was developed as an undergraduate thesis at the University of Oklahoma. The goal was to build a complete autonomous navigation stack for low-cost electric scooters using only commodity hardware and open-source models — no expensive sensors, no cloud compute.