# Scooter Sidewalk Navigation System

## What This Is

A single-camera autonomous navigation system for a 3-wheel scooter that detects sidewalk surfaces using semantic segmentation (SegFormer), projects the result into bird's-eye-view (BEV) space, extracts a drivable path via medial-axis skeleton, and sends real-time steering + speed commands to the scooter via serial. Built as a thesis project targeting a live demonstration on a real sidewalk.

## Core Value

The scooter must visibly and convincingly follow the sidewalk path in a live demo — everything else is secondary to that observable behavior.

## Requirements

### Validated

- ✓ SegFormer road/sidewalk semantic segmentation pipeline — existing
- ✓ BEV perspective transform with homography calibration tool — existing
- ✓ Medial-axis skeleton extraction + graph-based path search — existing
- ✓ Adaptive pure pursuit controller with cubic spline path model — existing
- ✓ YOLOv8-nano obstacle detection with distance estimation — existing
- ✓ GPS/NMEA waypoint navigation with heading correction — existing
- ✓ Scooter serial command output (ScooterCommander) — existing
- ✓ Temporal mask smoothing (EMA on seg + BEV masks) — existing
- ✓ Frame stabilization (optical flow compensation) — existing
- ✓ Per-frame CSV data logging + post-hoc analysis — existing
- ✓ Modular codebase (config, data_logger, masks, heading, etc.) — existing

### Active

- [ ] Segmentation is temporally stable — no visible frame-to-frame flickering on real sidewalk video
- [ ] BEV drivable path is detected reliably — has_path rate ≥ 60% of frames on typical sidewalk footage
- [ ] BEV calibration is accurate for the actual recording setup — warpPerspective preserves ≥50% sidewalk pixels
- [ ] Pipeline runs at ≥ 8 Hz on laptop during demo (current: ~12 Hz but seg inconsistency degrades quality)
- [ ] Scooter receives and executes steering + speed commands in real-time during demo run
- [ ] System degrades gracefully — holds last valid path and reduces speed when seg fails, no sudden stops

### Out of Scope

- Radxa / Rock 5B deployment — stretch goal only; laptop-on-scooter is accepted fallback for thesis demo
- Full GPS waypoint route following — GPS fusion exists but not required for demo
- Multi-camera or stereo depth — single camera only
- SLAM / map building — reactive navigation only, no global map

## Context

- **Thesis document**: University of Oklahoma Master's Thesis — `C:/Users/miji0000/Desktop/thesis_prep/thesis/main.tex` (LaTeX, figures in `thesis/figures/`). PDF manuscript: `thesis_prep/2025_Lkhaana_Manuscript__mono_camera_auto_drive_scooter (1).pdf`
- **Data logs**: `simulation_camera_scooter/logs/` (current runs) + `thesis_prep/logs/` (older runs). CSV per run + JSON metadata. `analyze_log.py` generates thesis figures from these logs.
- **Segmentation model**: Custom-trained SegFormer (`models/my-segformer-road_new`), runs on CUDA GPU (Quadro P620 on dev machine). Multiple checkpoints available (500–5000 steps).
- **BEV calibration**: Homography stored in `bev_calibration.npy`. Current calibration is ill-conditioned (cond=1.1e+06) causing 93% pixel loss after warpPerspective — dominant root cause of low has_path rate.
- **Path planning baseline**: After Tier 1 + Tier 2 tuning, has_path improved from 0% → ~2% on tilted phone video. Root cause remains calibration mismatch, not algorithm.
- **Scooter**: 3-wheeler, can physically carry a laptop. Serial interface almost ready for live commands.
- **Timeline**: 1–2 months to thesis demo.

## Constraints

- **Timeline**: 1–2 months — prioritize fixes with highest impact on demo quality over correctness
- **Hardware (dev)**: Windows laptop, NVIDIA Quadro P620 GPU, Python 3.11 + CUDA 11.8
- **Hardware (target)**: Rock 5B (ARM64) — TensorRT/ONNX optimization required; only if time permits
- **Camera**: Single USB webcam or iPhone Continuity Camera — no depth sensor
- **Stack**: Python + PyTorch + OpenCV — no framework changes; thesis timeline doesn't allow rewrites

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SegFormer for segmentation | High accuracy on road/sidewalk scenes, custom-trained checkpoint available | — Pending (flickering issue to resolve) |
| BEV homography calibration | Enables metric-space path extraction without depth sensor | ⚠️ Revisit — current calibration ill-conditioned |
| Medial-axis skeleton for path | Topology-preserving, works on arbitrary sidewalk shapes | — Pending (depends on BEV fix) |
| Adaptive pure pursuit controller | Simple, real-time, works with cubic spline path model | ✓ Good |
| Temporal EMA smoothing (seg + BEV) | Reduces flickering without changing model | ✓ Good — but insufficient alone |
| TIER2 BEV post-processing | anchor_ego_to_mask + ego_connected_mask + near-field closing | — Pending evaluation |

---
*Last updated: 2026-03-04 after initialization*
