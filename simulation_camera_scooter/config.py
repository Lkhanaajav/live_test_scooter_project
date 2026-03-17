"""
config.py
=========
Shared constants for the scooter navigation pipeline.
All module-level constants extracted from live_heading_demo.py.
"""

import json
import os
import numpy as np

# =============================================================================
# Segmentation
# =============================================================================
ROAD_ID = 1
SIDEWALK_ID = 2
# Best checkpoint from benchmark (Plan 01-01): 99.3% frames stable — was: my-segformer-road_new
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "my-segformer-road",  # updated from benchmark Plan 01-01
)
SEG_INPUT_RES = (640, 360)
LOW_POWER_SEG_INPUT_RES = (512, 288)

# =============================================================================
# BEV (Bird's Eye View)
# =============================================================================
# Default BEV points (scooter camera) -- override with calibration
DEFAULT_SRC_POINTS = np.array(
    [[6.0, 1072.0], [1907.0, 1072.0], [1017.0, 457.0], [834.0, 469.0]],
    dtype=np.float32,
)

DEFAULT_DST_POINTS = np.array(
    [
        [100, 576],  # bottom-left
        [500, 576],  # bottom-right
        [400, 120],  # top-right
        [200, 120],  # top-left
    ],
    dtype=np.float32,
)

BEV_SIZE = (600, 600)
TRIM_BOTTOM = 0           # was 20 — tier1 tuning: keep near-field BEV pixels
CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bev_calibration.npy",
)
CALIBRATION_META_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bev_calibration_meta.json",
)


def _load_bev_calibration_meta() -> dict:
    """Load optional BEV runtime metadata saved alongside the 4-point calibration."""
    if not os.path.exists(CALIBRATION_META_FILE):
        return {}
    try:
        with open(CALIBRATION_META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_BEV_META = _load_bev_calibration_meta()
BEV_EGO_X_FRAC = float(np.clip(float(_BEV_META.get("ego_x_frac", 0.5)), 0.05, 0.95))


def bev_ego_x_px(width: int) -> float:
    """Return the calibrated BEV ego x-position in pixels for the given width."""
    return float(np.clip(BEV_EGO_X_FRAC, 0.05, 0.95) * max(1.0, float(width - 1)))

# =============================================================================
# Skeleton / path tuning
# =============================================================================
DT_CORE_THRESH = 2.0      # was 6.0 — tier1 tuning: allow thinner paths through distance transform
PRUNE_BRANCH_LEN = 12
BOTTOM_BAND_PX = 30

# =============================================================================
# Heading thresholds (degrees from vertical/forward)
# =============================================================================
HEADING_STRAIGHT_THRESH = 12.0   # < 12 deg = STRAIGHT
HEADING_TURN_THRESH = 40.0       # 12-40 deg = LEFT/RIGHT, >40 = SHARP

# =============================================================================
# Speed profile (m/s) based on heading + obstacle proximity
# =============================================================================
SPEED_MAX = 1.5            # full speed on straight, clear path
SPEED_TURN = 0.8           # reduced speed during turns
SPEED_SHARP_TURN = 0.4     # sharp turns
SPEED_OBSTACLE_NEAR = 0.3  # obstacle within close range
SPEED_STOP = 0.0           # full stop

# =============================================================================
# Obstacle detection
# =============================================================================
OBSTACLE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                    5: "bus", 7: "truck", 15: "cat", 16: "dog"}
OBSTACLE_CLOSE_M = 3.0     # meters -- trigger slowdown
OBSTACLE_STOP_M = 1.0      # meters -- trigger stop
YOLO_CONF_THRESH = 0.35    # detection confidence threshold

# BEV obstacle projection constants (Phase 03.1)
BEV_OBSTACLE_ALPHA = 0.50           # EMA decay rate for obstacle grid (weight for incoming frame)
BEV_OBSTACLE_RADIUS_PX = {          # BEV footprint radius per YOLO class_name
    "person": 15,
    "bicycle": 18,
    "car": 40,
    "motorcycle": 20,
    "bus": 60,
    "truck": 55,
    "cat": 10,
    "dog": 10,
    "default": 15,
}
BEV_OBSTACLE_PENALTY_WEIGHT = 3.0   # cost adder per unit obstacle density in _score_candidates()
BEV_HARD_BLOCK_DIST_M = 1.2         # obstacles closer than this get hard-blocked in BEV mask
YOLO_MODEL_NAME = "yolov8n.pt"  # 3.2 MB nano model

# =============================================================================
# GPS
# =============================================================================
EARTH_RADIUS_M = 6_371_000.0
GPS_STEER_GAIN = 0.35
GPS_STEER_BIAS_MAX_DEG = 12.0

# =============================================================================
# Colors (BGR)
# =============================================================================
COLOR_STRAIGHT = (0, 255, 0)     # green
COLOR_LEFT = (255, 165, 0)       # orange
COLOR_RIGHT = (0, 165, 255)      # blue
COLOR_SHARP = (0, 0, 255)        # red
COLOR_STOP = (0, 0, 200)         # dark red
COLOR_OBJ_BOX = (0, 255, 255)    # cyan -- object detection box
COLOR_OBJ_WARN = (0, 0, 255)     # red -- close obstacle

PATH_COLORS = [
    (0, 255, 255), (255, 255, 0), (255, 0, 255),
    (0, 165, 255), (0, 255, 128), (128, 0, 255),
]

# =============================================================================
# Frame stabilization (camera shake compensation)
# =============================================================================
STABILIZATION_ENABLED = True
STAB_SMOOTHING_RADIUS = 20       # frames for trajectory smoothing window
STAB_MAX_CORRECTION_PX = 50      # max pixel shift correction per frame
STAB_MAX_CORRECTION_DEG = 3.0    # max rotation correction (degrees)

# =============================================================================
# Temporal mask smoothing (reduces segmentation flickering)
# =============================================================================
MASK_SMOOTH_ALPHA = 0.65         # EMA weight for current frame (0-1) — tuned Plan 01-02 sweep — was: 0.45
MASK_SMOOTH_CONSISTENCY_THRESH = 0.20  # IoU below this = likely shake artifact — tuned Plan 01-02 sweep — was: 0.30
BEV_SMOOTH_ALPHA = 0.55          # BEV mask temporal smoothing weight

# =============================================================================
# Segmentation stability safety gate
# =============================================================================
SEG_IOU_FAIL = 0.22              # severe instability threshold
SEG_IOU_WARN = 0.35              # mild instability threshold
SEG_FAIL_HOLD_FRAMES = 6         # consecutive unstable frames to trigger limit
SPEED_SEG_UNSTABLE = 0.20        # speed cap (m/s) when segmentation unstable

# =============================================================================
# Low-power profile (for small onboard computers)
# =============================================================================
LOW_POWER_STRIDE = 2
LOW_POWER_DETECTION_STRIDE = 2
LOW_POWER_PATH_SCALE = 0.65

# =============================================================================
# Real-time BEV navigation module (medial-axis + adaptive pure pursuit)
# =============================================================================
NAV_BEV_FORWARD_M = 10.0
NAV_BEV_LATERAL_M = 10.0
NAV_WORK_GRID_BASE = 220

# BEV motion compensation: forward pixel displacement per meter of travel
# BEV_SIZE[1]=600 px covers NAV_BEV_FORWARD_M=10 m → 60 px/m
BEV_PIXELS_PER_METER_FORWARD = BEV_SIZE[1] / NAV_BEV_FORWARD_M
# BEV_SIZE[0]=600 px covers NAV_BEV_LATERAL_M=10 m → 60 px/m
BEV_PIXELS_PER_METER_LATERAL = BEV_SIZE[0] / NAV_BEV_LATERAL_M

# =============================================================================
# BEV Predictive Frame Reuse
# =============================================================================
PREDICT_ENABLED = True
PREDICT_MAX_SKIP_STRAIGHT = 3    # max consecutive skips on straight road
PREDICT_MAX_SKIP_TURN = 1        # max consecutive skips on gentle turn
PREDICT_CURVATURE_STRAIGHT = 0.10  # kappa threshold for "straight" (m^-1)
PREDICT_CURVATURE_SHARP = 0.30     # kappa threshold for "sharp turn" (m^-1)
PREDICT_BLEND_ALPHA = 0.75        # real vs predicted blend weight on compute frames
PREDICT_CONFIDENCE_FLOOR = 0.50   # below this IoU confidence, stop skipping

# =============================================================================
# Research impl: Enhanced Morphological BEV Mask Pipeline (Idea 1)
# Papers: Road Seg ADAS (arXiv:2505.12206), Skelite (arXiv:2503.07369)
# =============================================================================
MORPH_ENHANCED = True                  # enable enhanced mask cleaning pipeline
MORPH_HOLE_FILL_MAX_M2 = 5.0          # max enclosed hole area to flood-fill (m^2)
MORPH_GAUSS_SIGMA_PX = 1.2            # Gaussian sigma (px) applied before re-binarize
MORPH_GAUSS_THRESH = 0.35             # threshold for re-binarize after Gaussian
MORPH_EGO_BAND_FRAC = 0.20            # bottom fraction of BEV used for ego-clearance scoring

# =============================================================================
# Research impl: Distance Transform Safe Corridor (Idea 2)
# Papers: Dual-BEV Nav (arXiv:2501.18351), ESDF corridor planning
# =============================================================================
DT_CORRIDOR_ENABLED = True            # enable DT-based safe corridor extraction
DT_CORRIDOR_COST_EXPONENT = 1.5       # cost = 1/(dt+eps)^exponent  (higher = stronger clearance preference)
DT_CORRIDOR_LATERAL_DRIFT_PX = 30     # max lateral step per row in Dijkstra search
DT_CORRIDOR_SG_WINDOW = 9             # Savitzky-Golay smoothing window (must be odd)
DT_CORRIDOR_CONFIDENCE_NORM_M = 1.5   # clearance (m) that maps to confidence=1.0

# =============================================================================
# Research impl: Temporal Path Smoother (Idea 3)
# Papers: Trajectory Prediction Survey (arXiv:2503.03262), Regulated PP (arXiv:2305.20026)
# =============================================================================
PATH_SMOOTH_ENABLED = True            # enable EMA smoothing on cubic path coefficients
PATH_SMOOTH_MIN_ALPHA = 0.35          # minimum EMA alpha (heavy smoothing, low confidence)
PATH_SMOOTH_MAX_ALPHA = 0.85          # maximum EMA alpha (fast tracking, high confidence)
PATH_SMOOTH_RESET_THRESH = 2.0        # max coefficient jump before reset (per coefficient)
HEADING_SMOOTH_ENABLED = True         # enable circular EMA on heading angle
HEADING_SMOOTH_ALPHA = 0.50           # EMA alpha for heading filter
HEADING_SMOOTH_RESET_DEG = 45.0       # heading delta (deg) that triggers filter reset
