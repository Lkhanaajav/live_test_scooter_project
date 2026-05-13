# Technology Stack

**Analysis Date:** 2026-03-04

## Languages

**Primary:**
- Python 3.11.9 - Core application, all modules

## Runtime

**Environment:**
- CPython 3.11.9

**Package Manager:**
- pip 24.x
- Lockfile: Not detected (no requirements.txt, pyproject.toml, or Pipfile present)

## Frameworks

**Core ML/Vision:**
- PyTorch 2.6.0+cu118 - Deep learning backend for SegFormer and auxiliary models
- torchvision 0.21.0+cu118 - Vision utilities
- transformers 4.52.2 - Hugging Face model framework (loads SegFormer)
- Ultralytics 8.3.96 - YOLOv8 object detection framework

**Segmentation:**
- SegFormer (from transformers) - Road/sidewalk semantic segmentation model
- segmentation_models_pytorch 0.5.0 - Segmentation model utilities

**Image Processing:**
- OpenCV 4.12.0.88 (opencv-python & opencv-contrib-python) - Video capture, frame processing, image transformations, drawing

**Data Processing:**
- numpy 2.1.1 - Numerical operations, array manipulation
- pandas 2.2.2 - Logging data (CSV writing)
- scipy 1.13.1 - Scientific computing, signal processing
- NetworkX 3.4.2 - Graph operations for path/skeleton analysis

**System/Utilities:**
- psutil 7.0.0 - System monitoring (CPU, memory, GPU usage tracking)

**Testing/Profiling:**
- torch-tb-profiler 0.4.3 - PyTorch profiling
- torchviz 0.0.3 - Tensor visualization
- ultralytics-thop 2.0.14 - FLOPs/throughput profiling

## Key Dependencies

**Critical:**
- PyTorch + CUDA 11.8 - Neural network inference (SegFormer, YOLOv8)
- transformers 4.52.2 - Model loading and inference (SegFormer semantic segmentation)
- OpenCV 4.12.0.88 - Camera capture and real-time image processing
- Ultralytics 8.3.96 - YOLOv8-nano object detection

**Infrastructure:**
- pyserial 3.5 - GPS and scooter serial communication
- numpy 2.1.1 - Matrix operations for BEV transformation and path calculations
- NetworkX 3.4.2 - Skeleton extraction and path graph analysis

## Configuration

**Environment:**
- No .env file used; configuration via command-line arguments
- Model paths: hardcoded to `models/my-segformer-road_new` and `models/my-segformer-road`
- Log output: `logs/` directory for CSV and JSON metadata

**Build:**
- No build configuration files detected
- Runs as pure Python scripts (no compilation)

**CLI Arguments:**
- `--camera` - Select camera device (0=default, 1=iPhone Continuity)
- `--video` - Input video file instead of live camera
- `--calibrate` - BEV calibration mode
- `--gps-device` - Serial port for GPS (e.g., COM3)
- `--gps-waypoints` - CSV file with waypoints
- `--serial-port` - Serial port for scooter commands (e.g., COM4)
- Additional: segmentation resolution, detection confidence, speed limits

## Platform Requirements

**Development:**
- Windows/macOS/Linux with Python 3.11+
- NVIDIA GPU with CUDA 11.8 support (can fall back to CPU, slower)
- Webcam or video input source
- Optional: Serial devices for GPS and scooter control

**Production:**
- Target deployment: Rock 5B (ARM64 board) for real-time 8 Hz operation
- Fallback: MacBook, Windows with GPU or CPU inference
- Camera: iPhone Continuity Camera (macOS) or standard USB webcam
- GPS module: NMEA-compatible serial device
- Scooter control: Serial-connected motor controller

## Key Models

**Road Segmentation:**
- SegFormer (custom trained) - Located in `models/my-segformer-road_new/`
- Checkpoints: `models/checkpoint-500` through `models/checkpoint-5000` available
- Purpose: Pixel-level road/sidewalk classification

**Object Detection:**
- YOLOv8-nano (`yolov8n.pt`) - 3.2 MB lightweight model
- Purpose: Pedestrian, bicycle, car, motorcycle detection
- Config in `live_heading_demo.py` line 85

## Inference Flow

1. **Video Input** (OpenCV) → Live camera or video file
2. **SegFormer** (PyTorch/transformers) → Road mask (640×360 or 512×288 resolution)
3. **BEV Transformation** (OpenCV/numpy) → Bird's-eye view (600×500px)
4. **Skeleton + Path** (NetworkX, cv2) → Extract walkable path
5. **YOLOv8** (Ultralytics) → Detect obstacles in frame
6. **Pure Pursuit Controller** (realtime_nav_core.py) → Calculate steering angle
7. **GPS/Waypoint Navigation** (pyserial, custom) → Target heading correction
8. **Scooter Command** (pyserial) → Send steering + speed over serial

---

*Stack analysis: 2026-03-04*
