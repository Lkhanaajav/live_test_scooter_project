# External Integrations

## Deep Learning Model Integrations

### SegFormer Semantic Segmentation

**Purpose:** Binary road (sidewalk/drivable) segmentation from monocular camera frames.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/fast_road_detector.py` (primary inference engine)
- **Model Class:** `SegformerForSemanticSegmentation` (from `transformers`)
- **Image Processor:** `SegformerImageProcessor` (automatic normalization + resizing)

**Checkpoint Loading:**
```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_DIR)
processor = SegformerImageProcessor.from_pretrained(MODEL_DIR)
```

**Model Directories (priority-ranked):**
1. `/outputs/training/binary_segformer_oneformer_teacher/best_checkpoint` — Val IoU: 0.9437 (best)
2. `/outputs/training/binary_segformer_old400_plus_img_1931_t300/best_checkpoint`
3. `/outputs/training/binary_segformer_old400_img1931_vid017_020/best_checkpoint`
4. `/outputs/training/binary_segformer_all6_t400/best_checkpoint`
5. `/simulation_camera_scooter/models/my-segformer-road` — Fallback default

**Auto-Selection Logic:**
- Scans priority list for existence
- Reads `training_summary.json::best_metrics::best_val_iou` from each checkpoint
- Loads checkpoint with highest recorded IoU
- Falls back to `/my-segformer-road` if no training summaries found

**Input/Output:**
- Input resolution: 640×360 (default), configurable to 512×288 (low-power mode)
- Output: Logits for 2 classes (road=1, sidewalk=2)
- Confidence threshold: 0.60 (tunable in `config.py`)
- Post-processing: argmax → binary mask (road vs. sidewalk)

**Configuration Constants** (`config.py`):
- `MODEL_DIR` — Auto-resolved checkpoint path
- `SEG_INPUT_RES=(640, 360)` — Standard resolution
- `LOW_POWER_SEG_INPUT_RES=(512, 288)` — Lightweight variant
- `ROAD_ID=1, SIDEWALK_ID=2` — Class IDs

**Performance:**
- Latency: ~40–50 ms/frame on RTX 3080 (development), ~200–300 ms on RPi 4
- Uses GPU acceleration (CUDA/MPS) when available; CPU fallback

---

### YOLOv8-nano Object Detection

**Purpose:** Real-time detection of dynamic obstacles (person, bicycle, car, etc.).

**Integration Points:**
- **Module:** `/simulation_camera_scooter/object_detector.py`
- **Model Class:** `YOLO` (from `ultralytics`)
- **Auto-Download:** Model cached to `~/.cache/ultralytics/` on first load

**Model Loading:**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.to(device)  # device auto-detected (CUDA > MPS > CPU)
```

**Model Details:**
- **Name:** `yolov8n.pt` (nano variant, 3.2 MB)
- **Classes Tracked:** {0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 15: cat, 16: dog}
- **Confidence Threshold:** 0.35 (tunable in `config.py`)

**Output:**
Each detection dict contains:
- `bbox: (x1, y1, x2, y2)` — Bounding box in pixel coordinates
- `class_id, class_name` — Object type
- `conf` — Confidence score
- `center: (cx, cy)` — Box centroid
- `height_px` — Used for monocular distance estimation

**Distance Estimation:**
- Module: `/simulation_camera_scooter/object_detector.py::estimate_obstacle_distance()`
- Method: Monocular distance from foot-of-bounding-box position
- Output: Approximate distance in meters
- Formula: `distance ~ k / (normalized_y - offset)` with pinhole model assumptions

**Configuration Constants** (`config.py`):
- `OBSTACLE_CLASSES` — Dictionary of class IDs to names
- `YOLO_CONF_THRESH=0.35` — Confidence cutoff
- `YOLO_MODEL_NAME="yolov8n.pt"` — Model variant
- `OBSTACLE_CLOSE_M=3.0` — Distance triggering slowdown
- `OBSTACLE_STOP_M=1.0` — Distance triggering stop
- `BEV_OBSTACLE_RADIUS_PX` — BEV footprint radius per class (15–60 px)

**Performance:**
- Latency: ~20–30 ms/frame on RTX 3080, ~100–150 ms on RPi 4
- GPU acceleration where available; CPU fallback graceful

**BEV Integration:**
- **Module:** `/simulation_camera_scooter/bev_obstacle.py`
- Detections projected to bird's-eye view via homography
- EMA-blended obstacle grid (α=0.50, configurable)
- Hard-block obstacles closer than 1.2 m (safety gate)
- Penalty weight in path scoring: 3.0× per unit obstacle density

---

## Hardware Integrations

### Camera Interface

**Purpose:** Real-time video frame capture.

**Integration Points:**
- **Module:** Implied in `live_heading_demo.py`, `fast_road_detector.py`, etc.
- **Library:** OpenCV (`cv2.VideoCapture`)

**Usage Pattern:**
```python
import cv2
cap = cv2.VideoCapture(0)  # Default webcam (dev only)
# OR
cap = cv2.VideoCapture("video_file.mp4")  # File input (testing)
ret, frame = cap.read()  # Returns BGR numpy array (H, W, 3)
```

**Frame Properties:**
- **Format:** BGR (OpenCV native)
- **Typical Resolution:** 1920×1080 (input), resized to 640×360 for segmentation
- **Frame Rate:** 30 FPS (video mode) or unbuffered (live capture)

**Configuration Assumptions:**
- Forward-facing monocular camera on scooter
- Approximate field-of-view: 55° vertical (used in distance estimation)
- Camera height: ~0.8 m above ground (tunable)

**BEV Calibration:**
- Homography matrix stored in `bev_calibration.npy`
- 4-point source (image) + 4-point destination (BEV) corners
- Ego position fraction optionally stored in `bev_calibration_meta.json`
- Loaded dynamically; defaults provided in `config.py`

---

### GPS Receiver (Optional)

**Purpose:** Waypoint navigation and GPS-conditioned turn intent.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/gps_navigator.py`
- **Hardware:** NMEA-capable GPS receiver (u-blox, SiRF, etc.)
- **Serial Protocol:** Configurable baud rate (default 9600)

**GPS Navigator Class:**
```python
from gps_navigator import GPSNavigator
nav = GPSNavigator(
    serial_device="/dev/ttyUSB0",  # Linux/RPi
    baud=9600,
    waypoints_file="waypoints.csv"
)
# Background thread reads NMEA: GGA, RMC sentences
lat, lon, heading = nav.lat, nav.lon, nav.heading_gps
```

**Waypoint Format** (CSV):
```
latitude,longitude[,optional_name]
37.12345,-122.45678,waypoint_1
37.12350,-122.45650,waypoint_2
```

**State Tracked:**
- `lat, lon` — Current position (degrees)
- `speed_mps` — GPS speed over ground
- `heading_gps` — Compass heading from GPS RMC (degrees from north)
- `fix_quality` — NMEA GGA fix quality (0=invalid, 1=GPS, 2=DGPS)
- `hdop` — Horizontal dilution of precision
- `current_wp_idx` — Index of current target waypoint
- `wp_reached_radius_m` — Proximity threshold for waypoint completion

**Integration with Path Planner:**
- **Module:** `/simulation_camera_scooter/intent_picker.py`
- Maps GPS-to-next-waypoint vector to turn intent (STRAIGHT, LEFT, RIGHT, SHARP)
- Intent conditioning applied in template path approval (`template_path_planner.py`)

**Safety Notes:**
- GPS is **not** used for immediate collision avoidance (vision-primary)
- GPS updates at ~1 Hz (slow compared to vision @ 10 Hz)
- Graceful degradation: system operates vision-only if GPS unavailable

**Configuration Constants** (`config.py`):
- `EARTH_RADIUS_M=6_371_000` — WGS84 radius for bearing calculations
- `GPS_STEER_GAIN=0.35` — Proportional steering gain to GPS error
- `GPS_STEER_BIAS_MAX_DEG=12.0` — Max steering bias from GPS

---

### Scooter Motor & Steering Control (Optional)

**Purpose:** Send steering angle and speed commands to onboard scooter hardware.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/scooter_commander.py`
- **Hardware:** Scooter motor controller + servo steering
- **Serial Protocol:** Text-based command protocol

**ScooterCommander Class:**
```python
from scooter_commander import ScooterCommander
cmd = ScooterCommander(port="/dev/ttyUSB1", baud=115200)
cmd.send_command(steer_deg=-12.5, speed_mps=1.2)  # Steer left, full speed
cmd.stop()  # Emergency halt
```

**Command Protocol:**
```
Format: CMD,<steer_deg>,<speed_mps>\n
Example: CMD,-12.5,1.2\n     (steer left 12.5°, speed 1.2 m/s)
Example: CMD,0.0,0.0\n       (stop)
```

**State:**
- `steer_deg` — Steering angle (negative=left, positive=right), typically ±45°
- `speed_mps` — Forward speed (0–2 m/s typical)

**Speed Limits** (applied by planner):
- `SPEED_MAX=1.5` — Full speed on straight, clear path
- `SPEED_TURN=0.8` — Reduced speed during gentle turns
- `SPEED_SHARP_TURN=0.4` — Sharp turns (>40° heading)
- `SPEED_OBSTACLE_NEAR=0.3` — Obstacle within 3 m
- `SPEED_STOP=0.0` — Emergency halt / segmentation failure

**Control Flow:**
1. Path planner → curvature κ (m⁻¹)
2. Pure pursuit → target heading angle
3. GPS intent → bias to next waypoint
4. Safety gates → speed reduction
5. `send_command(steer, speed)` → scooter hardware

**Graceful Degradation:**
- System continues planning/inference if serial unavailable
- Commands printed to console (debug mode)
- No hardware communication required for algorithm testing

---

## Data Logging & Storage

### Per-Frame CSV Logging

**Purpose:** Detailed frame-by-frame telemetry for thesis analysis and debugging.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/data_logger.py`
- **Output:** `logs/run_YYYYMMDD_HHMMSS.csv` + `_meta.json`

**Logged Fields** (40+ per frame):
- **Timing:** `t_segmentation`, `t_detection`, `t_bev`, `t_skeleton`, `t_pathfinding`, `t_gps_fusion`, `t_command`, `t_total_pipeline`, `fps`, `compute_hz`
- **Heading & Control:** `heading_raw_deg`, `heading_smoothed_deg`, `command`, `gps_intent_family`, `speed_raw_mps`, `speed_smoothed_mps`
- **Segmentation:** `seg_iou`, `seg_unstable_frames`, `stability_mode`
- **Path Info:** `has_candidate_path`, `has_model_path`, `approval_confidence`, `planner_family`, `path_source`
- **Obstacles:** `num_detections`, `min_obstacle_dist_m`, `detection_classes`, `detection_distances`
- **GPS:** `gps_lat`, `gps_lon`, `gps_fix_quality`, `gps_wp_dist_m`
- **Mask Stats:** `sidewalk_mask_pixels`, `bev_mask_pixels`, `skeleton_pixels`

**Usage Pattern:**
```python
from data_logger import DataLogger
logger = DataLogger(log_dir="logs")
logger.log(frame_id=0, timestamp=time.time(), heading_raw_deg=5.2, ...)
```

**File Format:**
- CSV with header row (field names)
- Per-frame rows with comma-separated values
- Meta JSON file contains run metadata (model dir, config constants, etc.)

**Analysis:**
- Scripts in `/scripts/`: `eval_simple_road.py`, `analyze_log.py` use these CSVs
- Pandas-based post-processing for aggregating metrics

---

### Annotation Frame Extraction

**Purpose:** Extract and cache fine-tuning data for model retraining.

**Integration Points:**
- **Module:** `live_heading_demo.py` (frame extraction logic)
- **Storage:** `/simulation_camera_scooter/annotation_frames/`

**Extraction Triggers:**
- Periodic frame sampling during live runs
- Bounding-box confidence → frame inclusion scoring
- Manual frame selection via UI (in advanced workflows)

**Output Format:**
- Original camera frame: `.jpg` (full resolution)
- Segmentation mask: `.png` (binary or class-indexed)
- Metadata: `.json` (frame ID, timestamp, GPS, heading, segmentation IoU)

**Use Case:**
Fine-tune SegFormer on campus-specific sidewalk data to improve domain adaptation.

---

## External Datasets & Benchmarks

### Cityscapes Evaluation

**Purpose:** Validate road segmentation performance on public benchmark.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/eval_cityscapes.py`
- **Dataset:** Cityscapes drivable-area segmentation (public download)
- **Metric:** mIoU (mean Intersection-over-Union) over test set

**Workflow:**
1. Download Cityscapes `leftImg8bit_test_val.zip` + `gtFine_test_val.zip`
2. Run evaluation script: `python eval_cityscapes.py --model_dir <checkpoint> --data_dir <cityscapes>`
3. Output: mIoU score, per-class metrics, confusion matrix

**Recent Results** (from thesis deliverables):
- SegFormer B0 on Cityscapes drivable-area: mIoU ≈ 0.82–0.85

---

### RUGD (Road Understudied Geographical Domains) Evaluation

**Purpose:** Test generalization on rural/unpaved road segmentation.

**Integration Points:**
- **Module:** `/simulation_camera_scooter/eval_rugd.py`
- **Dataset:** RUGD unpaved road segmentation benchmark
- **Metric:** mIoU

**Use Case:**
Evaluate sidewalk robustness on domain shifts (campus → new environments).

---

## Internal Research Implementations

### Enhanced Morphological Mask Pipeline (Idea 1)

**Module:** `/simulation_camera_scooter/masks.py`
- **Function:** `clean_bev_mask_enhanced(mask, ego_x_px, ego_y_px, ...)`
- **Operations:**
  - Flood-fill holes up to 5 m² (configured in `MORPH_HOLE_FILL_MAX_M2`)
  - Gaussian boundary smoothing (σ=1.2 px)
  - Distance-transform ego-clearance scoring
  - Re-binarization at threshold 0.35
- **Enable Flag:** `MORPH_ENHANCED=True` in config
- **Backward Compatible:** Exact original behavior when disabled

---

### Distance-Transform Safe Corridor (Idea 2)

**Module:** `/simulation_camera_scooter/safe_corridor.py` (new)
- **Class:** `DtSafeCorridor`
- **Method:** `extract(bev_mask, forward_m, lateral_m, ...)`
- **Algorithm:**
  - Euclidean distance transform on mask
  - Dijkstra path-finding on cost grid: `1/(dt+ε)^1.5`
  - Savitzky-Golay smoothing of centerline (window=9)
- **Output:** `DtCorridorResult` with centerline pixels, width per point, confidence, DT map
- **Enable Flag:** `DT_CORRIDOR_ENABLED=True`
- **Cost Exponent:** `DT_CORRIDOR_COST_EXPONENT=1.5` (tunable)

---

### Temporal Path Smoothing (Idea 3)

**Module:** `/simulation_camera_scooter/path_smoother.py` (new)
- **Class:** `PathTemporalSmoother`
  - EMA on cubic path coefficients (α=0.35–0.85, adaptive to confidence)
  - Reset on topology change (coeff jump > 2.0)
- **Class:** `HeadingTemporalFilter`
  - Circular EMA on heading angle (handles ±180° wrap)
  - Reset on >45° heading jump
- **Enable Flags:**
  - `PATH_SMOOTH_ENABLED=True`
  - `HEADING_SMOOTH_ENABLED=True`
- **Parameters:**
  - `PATH_SMOOTH_MIN_ALPHA=0.35, MAX_ALPHA=0.85` — Confidence-adaptive smoothing
  - `HEADING_SMOOTH_ALPHA=0.50` — Heading filter rate
  - `HEADING_SMOOTH_RESET_DEG=45.0` — Jump threshold

---

## Testing & Validation Integrations

### Pytest Test Suite

**Location:** `/simulation_camera_scooter/tests/`
**Test Modules:**
- `test_realtime_nav_core.py` — Path extraction & control law
- `test_template_path_planner.py` — Template arc fitting & approval
- `test_waypoint_turn_planner.py` — GPS-intent turn selection
- `test_bev_obstacle.py` — Obstacle projection
- `test_bev_predictor.py` — Predictive BEV reuse
- `test_heading.py` — Heading computation
- `test_boundary_model.py`, `test_boundary_dataset.py` — Road boundary inference
- `test_temporal_smoother.py` — EMA filtering

**Fixtures** (`conftest.py`):
- `straight_bev_mask`, `turn_bev_mask` — Synthetic test corridors
- `sample_detections` — Mock YOLO output
- Shared initialization (pin ego position to 0.5 for test reproducibility)

**Run Command:**
```bash
cd simulation_camera_scooter
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Coverage Target:** ≥80% (currently 104 tests, all passing)

---

## External File Formats & Locations

| Type | Location | Format | Purpose |
|------|----------|--------|---------|
| **Segmentation Model** | `/simulation_camera_scooter/models/my-segformer-road` | HuggingFace transformer checkpoint | Default road segmentation model |
| **Fine-tuned Models** | `/outputs/training/binary_segformer_*` | Same (auto-selected by best IoU) | Campus-specific segmentation |
| **YOLO Model** | `yolov8n.pt` (auto-cached) | ONNX + PyTorch weights | Object detection |
| **BEV Calibration** | `bev_calibration.npy` | NumPy binary array (4×2 float32) | Homography matrix |
| **Calibration Metadata** | `bev_calibration_meta.json` | JSON dict | Ego position fraction |
| **GPS Waypoints** | User-provided CSV | CSV: lat,lon[,name] | Navigation targets |
| **Session Logs** | `logs/run_YYYYMMDD_HHMMSS.csv` | CSV with 40+ fields | Per-frame telemetry |
| **Annotation Frames** | `annotation_frames/` | JPG + PNG + JSON | Fine-tuning dataset |
| **Thesis Results** | `EVALUATION_REPORT.md`, `TRAINING_LOG.md` | Markdown | Research deliverables |

---

## Graceful Degradation

The system is designed to operate with missing components:

1. **No GPU** → Falls back to CPU inference (slower but functional)
2. **No pyserial** → GPS and scooter control disabled; system continues planning
3. **No GPS receiver** → Navigation defaults to vision-based straight paths
4. **No calibration file** → Uses default homography (generic camera model)
5. **No YOLO model** → Obstacle detection disabled; path planning continues
6. **Model checkpoint missing** → Uses oldest available checkpoint or fallback model

All degradation is logged to stdout (print statements); no silent failures.

---

## Security & Safety Notes

- **No Network Communication** — All inference runs locally (no cloud API calls)
- **Model Weights** — Stored locally; no automatic downloads during inference
- **YOLO Auto-Download** — Only on first instantiation; thereafter uses cache
- **Serial Protocols** — Text-based (human-readable for debugging); no authentication required
- **Configuration** — All constants in Python (no external config servers)
- **Data Logging** — Stored locally in CSV format (no external analytics)
