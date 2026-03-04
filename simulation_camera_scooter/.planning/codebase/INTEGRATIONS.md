# External Integrations

**Analysis Date:** 2026-03-04

## APIs & External Services

**Model Hosting:**
- Hugging Face Model Hub (implicit) - SegFormer weights loaded via transformers library
  - Access method: AutoImageProcessor and SegformerForSemanticSegmentation via transformers
  - No explicit API key required (public models)

## Data Storage

**Databases:**
- None detected

**File Storage:**
- Local filesystem only
  - Model checkpoints: `models/my-segformer-road_new/` and `models/my-segformer-road/` directories
  - Trained checkpoints: `models/checkpoint-500/` through `models/checkpoint-5000/`
  - Inference results: `camera_results/` directory
  - Logs: `logs/` directory for CSV and JSON metadata

**Caching:**
- None detected (models cached in local `models/` directories)

## Authentication & Identity

**Auth Provider:**
- None - System is entirely offline/local

**Implementation:**
- No authentication required
- All operations self-contained on single machine or robot platform

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Datadog, etc.)

**Logs:**
- Custom CSV logging to `logs/run_[timestamp].csv`
  - Fieldnames: frame numbers, timestamps, FPS, heading, speed, steering angle, GPS data, obstacle detections
  - Location: `live_heading_demo.py` lines 140-178 (`RunLogger` class)
- JSON metadata: `logs/run_[timestamp]_meta.json`
  - Records system info (CPU, GPU, RAM), model versions, configuration used
- Console output: Print statements for status messages

## Serial Devices & Connections

**GPS (NMEA over Serial):**
- Protocol: NMEA 0183 (standard GPS)
- Library: pyserial 3.5
- Serial parameters: Configurable baud rate (default 9600)
- CLI flag: `--gps-device COM3` (Windows) or `/dev/ttyUSB0` (Linux)
- Data parsed: GGA (fix quality, lat/lon) and RMC (heading, speed)
- Waypoint navigation: Loads CSV file with lat/lon/name format
- Implementation: `live_heading_demo.py` lines 301-507 (`GPSNavigator` class)
- Threading: Background thread for serial reads to avoid blocking main loop

**Scooter Control (Serial):**
- Protocol: Custom binary/text format over serial
- Library: pyserial 3.5
- Serial parameters: Configurable baud rate (default 115200 for scooter)
- CLI flag: `--serial-port COM4` (Windows) or `/dev/ttyUSB1` (Linux)
- Command format: Steering angle (degrees) + speed (m/s) values
- Implementation: `live_heading_demo.py` lines 513-590 (`ScooterController` class)
- Error handling: Fails silently if pyserial not installed or port unavailable

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Hardware Interfaces

**Camera Input:**
- OpenCV cv2.VideoCapture()
- Supported sources: USB webcams, iPhone Continuity Camera (macOS), video files
- Selection: `--camera 0` or `--camera 1` (for iPhone via Continuity)
- Resolution configuration: SEG_INPUT_RES (640×360) or LOW_POWER_SEG_INPUT_RES (512×288)

**GPU/Accelerator:**
- NVIDIA CUDA 11.8 (optional)
- PyTorch detects and uses GPU if available, falls back to CPU
- Check: `torch.cuda.is_available()` in `fast_road_detector.py` line 60

**Performance Monitoring Hardware:**
- CPU/Memory monitoring via psutil
- GPU memory tracking via torch.cuda (if CUDA available)

## Data Flow & Processing Pipeline

**Main Processing Loop** (live_heading_demo.py):
1. Video frame capture (OpenCV)
2. SegFormer inference (PyTorch) → road mask
3. BEV transformation (OpenCV perspective transform)
4. Skeleton extraction (NetworkX graph)
5. YOLOv8 inference (Ultralytics) → obstacle detection
6. Pure Pursuit steering calculation (realtime_nav_core.py)
7. GPS heading fusion (GPSNavigator thread)
8. Scooter command generation (ScooterController)
9. Logging (CSV/JSON to logs/)

**Configuration Files Parsed:**
- GPS waypoints: CSV format (lat,lon,name) - sample at `sample_route.csv`
- BEV calibration points: Hardcoded numpy arrays or loaded from .npy file (`../bev_calibration.npy`)

## External Model Dependencies

**SegFormer:**
- Source: Hugging Face transformers library
- Model type: Vision transformer for semantic segmentation
- Resolution: 640×360 input (default)
- Classes: Road (ID=1), Sidewalk (ID=2)
- Loading: `AutoImageProcessor` and `SegformerForSemanticSegmentation` in `fast_road_detector.py`

**YOLOv8-nano:**
- Source: Ultralytics official (`yolov8n.pt`)
- Weights downloaded/cached by ultralytics on first run
- Classes: Person, bicycle, car, motorcycle, bus, truck, cat, dog (COCO subset)
- Confidence threshold: `YOLO_CONF_THRESH = 0.35` (line 84 in live_heading_demo.py)

## Environment Variables

**Not used** - All configuration via CLI arguments or hardcoded constants

## Dependencies with Network Access

**Initial Model Download:**
- transformers library downloads SegFormer from Hugging Face (first run)
- ultralytics library downloads yolov8n.pt from Ultralytics CDN (first run)
- Subsequent runs: Models cached locally in `models/` and `.cache/` directories

---

*Integration audit: 2026-03-04*
