# Testing Patterns

**Analysis Date:** 2026-03-04

## Test Framework

**Runner:**
- Not detected - no pytest, unittest, or vitest configuration found
- No `pytest.ini`, `tox.ini`, `setup.cfg`, or test runner config files present

**Assertion Library:**
- Not used - no formal testing framework integrated

**Run Commands:**
```bash
# No automated test suite detected
# Testing is manual/scripted through argument flags
```

## Test File Organization

**Location:**
- No dedicated test files (`*_test.py`, `*_spec.py`, `test_*.py`) found in codebase
- Testing done via script execution with different argument combinations

**Naming:**
- Not applicable - no test files present

**Structure:**
- N/A - no test suite structure

## Test Structure

**Manual Testing via Script Arguments:**

Codebase uses CLI arguments for testing different configurations rather than automated tests.

**From `fast_road_detector.py`:**
```python
# Lines 459-482: Argument-driven testing
parser = argparse.ArgumentParser(description='Road Detection with GPU/CPU toggle')
parser.add_argument('--use-gpu', action='store_true', help='Use GPU for processing')
parser.add_argument('--use-cpu', action='store_true', help='Force CPU usage')
parser.add_argument('--video', type=str, default="test_video_june_03_1.MOV", help='Input video path')
parser.add_argument('--output', type=str, default="result/fast_overlay.mp4", help='Output video path')
parser.add_argument('--save-metrics', action='store_true', help='Save detailed performance metrics')
```

**From `camera_waypoint_pipeline.py`:**
```python
# Lines 365-376: Multiple configuration testing
parser.add_argument("--camera-id", type=int, default=DEFAULT_CAMERA_ID, help="Camera device ID")
parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
parser.add_argument("--save-video", action="store_true", help="Save output video")
parser.add_argument("--resize-w", type=int, default=None, help="Resize width before inference")
parser.add_argument("--resize-h", type=int, default=None, help="Resize height before inference")
parser.add_argument("--no-window", action="store_true", help="Disable preview window")
parser.add_argument("--gps-test", action="store_true", help="Only read GPS and print lat/lon")
```

## Mocking

**Framework:**
- No mocking framework detected
- No unittest.mock or similar imports

**Patterns:**
- Graceful fallback for missing dependencies instead of mocks:

From `camera_waypoint_pipeline.py` lines 121-126:
```python
def gps_test(device, baud=9600):
    try:
        import serial
    except Exception as exc:
        print("❌ pyserial not installed. Run: pip install pyserial")
        raise exc
```

From `live_heading_demo.py` lines 232-242 (TinyObjectDetector):
```python
def _load(self, model_name):
    try:
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        # ...
    except ImportError:
        print("[ObjDet] WARNING: ultralytics not installed...")
        self.model = None  # Graceful degradation
```

From `analyze_log.py` lines 29-38:
```python
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed...")
```

**What to Mock:**
- None explicitly mocked - use graceful degradation with optional dependencies

**What NOT to Mock:**
- Core CV2/NumPy operations are real
- Model inference runs on actual data (real video files or camera streams)

## Fixtures and Factories

**Test Data:**
- Video files used as fixtures: `test_video_june_03_3.MOV`, `test_video_mar3.MOV`, `test_video_mar3_1.MOV`
- CSV logs created during runs: `logs/run_YYYYMMDD_HHMMSS.csv` with metadata `_meta.json`
- Hardcoded calibration points for BEV transform: `DEFAULT_SRC_POINTS`, `DEFAULT_DST_POINTS` in `live_heading_demo.py`

**Location:**
- Test videos in project root: `/test_video_*.MOV`, `/test_video_*.mp4`
- CSV logs in `logs/` directory (auto-created during runs)
- Calibration data: `bev_calibration.npy` (loaded/saved dynamically)

**Factory Functions:**

From `fast_road_detector.py` lines 453-457:
```python
def toggle_device(detector: FastRoadDetector, use_gpu: bool) -> FastRoadDetector:
    """Quickly toggle between CPU and GPU usage."""
    config = detector.config
    config.use_gpu = use_gpu
    return FastRoadDetector(config)
```

From `camera_waypoint_pipeline.py` lines 218-220:
```python
def initialize_model():
    cfg = Config(model_dir="models/my-segformer-road_new", conf_thresh=0.5, road_id=ROAD_ID)
    return FastRoadDetector(cfg)
```

## Coverage

**Requirements:**
- Not enforced - no coverage tool configured

**View Coverage:**
- Not applicable

## Test Types

**Unit Tests:**
- Not present
- Testing relies on component-level argument configurations

**Integration Tests:**
- Implicit via end-to-end script execution
- `fast_road_detector.py`: Processes full video, logs metrics
- `live_heading_demo.py`: Real-time integration of segmentation + navigation + GPS + detection
- `camera_waypoint_pipeline.py`: Combines BEV transform + path planning + visualization

**E2E Tests:**
- Manual execution with test videos
- Real-time camera testing with `--camera` argument
- GPS integration testing with `--gps-device` argument

**Example from `live_heading_demo.py`:**
Entire script is an integration test runner with modes:
```bash
python live_heading_demo.py                           # real-time camera
python live_heading_demo.py --calibrate               # calibration mode
python live_heading_demo.py --video test.mp4          # video playback test
python live_heading_demo.py --gps-device COM3         # GPS integration
python live_heading_demo.py --serial-port COM4        # scooter control test
```

## Common Patterns

**Async Testing:**
- Threading used for real-time operations
- Example from `live_heading_demo.py`: GPS reader runs in background thread
- No async/await patterns detected

**Error Testing:**
- Try-except blocks used to validate error conditions
- Graceful fallback testing (GPU unavailable → CPU)
- Model loading tests both local and HuggingFace paths

From `fast_road_detector.py` lines 155-192:
```python
def _load_model(self):
    try:
        # ... load from local path
        if os.path.isdir(model_dir):
            # local loading path
        else:
            # fallback to HuggingFace
            self.processor = AutoImageProcessor.from_pretrained(model_dir)
    except Exception as e:
        self.logger.error(f"Error loading model: {e}")
        raise
```

**Performance Testing:**
- Metrics collection via `DataLogger` in `live_heading_demo.py`
- Timing breakdown captured per module: segmentation, detection, BEV, skeleton, pathfinding, GPS fusion, command
- CSV logging with per-frame analysis

From `fast_road_detector.py` lines 402-415:
```python
# Update performance metrics
self.performance_metrics.frame_count = frame_idx
self.performance_metrics.fps = processed_idx / self.performance_metrics.pure_processing_time
self._save_metrics()

def _save_metrics(self):
    """Save performance metrics to a JSON file."""
    metrics = {
        "system_info": asdict(self.system_info),
        "performance_metrics": asdict(self.performance_metrics),
        "config": asdict(self.config)
    }
```

**Validation Testing:**
- Argument validation via argparse
- Shape validation in frame processing: `if model_out.shape != infer_frame.shape[:2]`
- Bounds checking: `np.clip()`, `_clip()` functions used extensively

From `camera_waypoint_pipeline.py` lines 277-283:
```python
model_out, _ = model.process_frame(infer_frame)
if model_out.shape != infer_frame.shape[:2]:
    model_out = cv2.resize(model_out, (infer_frame.shape[1], infer_frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
```

## Data Logging for Analysis

**CSV Logging Pattern** (from `analyze_log.py`):
Scripts log frame-by-frame data for post-hoc analysis:

```python
# From live_heading_demo.py DataLogger class (lines 139-210)
FIELDNAMES = [
    "frame_id", "timestamp", "wall_clock",
    # Timing (ms)
    "t_segmentation", "t_detection", "t_bev", "t_skeleton",
    "t_pathfinding", "t_gps_fusion", "t_command", "t_total_pipeline",
    "fps",
    # Heading & control
    "heading_raw_deg", "heading_smoothed_deg", "command",
    "speed_raw_mps", "speed_smoothed_mps",
    # Path info
    "has_path", "num_paths", "best_path_length_px",
    # Object detection
    "num_detections", "min_obstacle_dist_m",
]
```

**Analysis Tools** (from `analyze_log.py`):
```bash
python analyze_log.py logs/run_*.csv                        # load and summarize
python analyze_log.py logs/run_*.csv --output figures/      # generate plots
```

Generated reports include:
- Per-module timing breakdown
- Heading stability plots
- FPS stability analysis
- Command distribution pie charts
- Obstacle detection histograms
- Ablation comparison (multiple runs)

---

*Testing analysis: 2026-03-04*
