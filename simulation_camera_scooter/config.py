"""
config.py
=========
Shared constants for the scooter navigation pipeline.
All module-level constants extracted from live_heading_demo.py.
"""

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
        [100, 480],  # bottom-left
        [500, 480],  # bottom-right
        [400, 100],  # top-right
        [200, 100],  # top-left
    ],
    dtype=np.float32,
)

BEV_SIZE = (600, 500)
TRIM_BOTTOM = 0           # was 20 — tier1 tuning: keep near-field BEV pixels
CALIBRATION_FILE = "bev_calibration.npy"

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
