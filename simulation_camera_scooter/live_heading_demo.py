#!/usr/bin/env python3
"""
live_heading_demo.py
====================
Live camera -> SegFormer -> BEV -> Skeleton -> Path -> Heading + Speed command.
+ YOLOv8-nano object detection (pedestrians, bicycles, cars)
+ GPS waypoint navigation (NMEA over serial)
+ Precise steering degree + speed output for scooter serial control

Run on MacBook with iPhone Continuity Camera (or any webcam).

Usage:
    python live_heading_demo.py                            # default camera 0
    python live_heading_demo.py --camera 1                 # iPhone camera
    python live_heading_demo.py --calibrate                # BEV calibration mode
    python live_heading_demo.py --video test.mp4           # test with video
    python live_heading_demo.py --gps-device COM3          # with GPS
    python live_heading_demo.py --gps-waypoints route.csv  # follow GPS route
    python live_heading_demo.py --serial-port COM4         # send commands to scooter
"""

import os, sys, cv2, math, time, argparse, threading, csv, json
from datetime import datetime
import numpy as np
import networkx as nx
from collections import deque

# ---- Detector (sidewalk segmentation) ----
from fast_road_detector import FastRoadDetector, Config

# =============================================================================
# Constants
# =============================================================================
ROAD_ID = 1
SIDEWALK_ID = 2
MODEL_DIR = "models/my-segformer-road_new"
SEG_INPUT_RES = (640, 360)

# Default BEV points (scooter camera) -- override with calibration
DEFAULT_SRC_POINTS = np.array([
    [0.0,   717.0],
    [1278.0, 717.0],
    [860.0,  337.0],
    [573.0,  329.0]
], dtype=np.float32)

DEFAULT_DST_POINTS = np.array([
    [100, 480],  # bottom-left
    [500, 480],  # bottom-right
    [400, 100],  # top-right
    [200, 100]   # top-left
], dtype=np.float32)

BEV_SIZE = (600, 500)
TRIM_BOTTOM = 20

# Skeleton / path tuning
DT_CORE_THRESH = 6.0
PRUNE_BRANCH_LEN = 12
BOTTOM_BAND_PX = 30

# Heading thresholds (degrees from vertical/forward)
HEADING_STRAIGHT_THRESH = 12.0   # < 12 deg = STRAIGHT
HEADING_TURN_THRESH = 40.0       # 12-40 deg = LEFT/RIGHT, >40 = SHARP

# Speed profile (m/s) based on heading + obstacle proximity
SPEED_MAX = 1.5            # full speed on straight, clear path
SPEED_TURN = 0.8           # reduced speed during turns
SPEED_SHARP_TURN = 0.4     # sharp turns
SPEED_OBSTACLE_NEAR = 0.3  # obstacle within close range
SPEED_STOP = 0.0           # full stop

# Obstacle detection
OBSTACLE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                    5: "bus", 7: "truck", 15: "cat", 16: "dog"}
OBSTACLE_CLOSE_M = 3.0     # meters -- trigger slowdown
OBSTACLE_STOP_M = 1.0      # meters -- trigger stop
YOLO_CONF_THRESH = 0.35    # detection confidence threshold
YOLO_MODEL_NAME = "yolov8n.pt"  # 3.2 MB nano model

# GPS
EARTH_RADIUS_M = 6_371_000.0

# Colors
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

CALIBRATION_FILE = "bev_calibration.npy"


# =============================================================================
# Data Logger -- per-frame CSV for thesis experiments
# =============================================================================
class DataLogger:
    """
    Logs every frame's data to a timestamped CSV file for post-hoc analysis.
    Each row captures: timing, heading, speed, detections, GPS, path info.
    """

    FIELDNAMES = [
        # Identity
        "frame_id", "timestamp", "wall_clock",
        # Timing (ms)
        "t_segmentation", "t_detection", "t_bev", "t_skeleton",
        "t_pathfinding", "t_gps_fusion", "t_command", "t_total_pipeline",
        "fps",
        # Heading & control
        "heading_raw_deg", "heading_smoothed_deg", "command",
        "speed_raw_mps", "speed_smoothed_mps", "serial_cmd",
        # Path info
        "has_path", "num_paths", "best_path_length_px",
        "num_graph_nodes", "num_graph_edges",
        # Object detection
        "num_detections", "min_obstacle_dist_m",
        "detection_classes", "detection_distances",
        # GPS
        "gps_lat", "gps_lon", "gps_fix_quality",
        "gps_wp_name", "gps_wp_dist_m", "gps_correction_deg",
        # Mask stats
        "sidewalk_mask_pixels", "bev_mask_pixels", "skeleton_pixels",
    ]

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"run_{ts}.csv")
        self.meta_path = os.path.join(log_dir, f"run_{ts}_meta.json")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._start_time = time.time()
        self._row_count = 0
        print(f"[Logger] Logging to {self.csv_path}")

    def log(self, **kwargs):
        """Write one row. Missing fields default to empty string."""
        kwargs.setdefault("timestamp", time.time() - self._start_time)
        kwargs.setdefault("wall_clock", datetime.now().isoformat())
        row = {k: kwargs.get(k, "") for k in self.FIELDNAMES}
        self._writer.writerow(row)
        self._row_count += 1
        # Flush every 50 rows for safety
        if self._row_count % 50 == 0:
            self._file.flush()

    def save_metadata(self, **kwargs):
        """Save run configuration as JSON alongside the CSV."""
        meta = {
            "csv_file": self.csv_path,
            "start_time": datetime.now().isoformat(),
            "total_frames": self._row_count,
            **kwargs,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[Logger] Metadata saved to {self.meta_path}")

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"[Logger] Closed. {self._row_count} rows written to {self.csv_path}")


# =============================================================================
# Object Detection -- YOLOv8-nano (ultralytics, ~3.2 MB)
# =============================================================================
class TinyObjectDetector:
    """
    Wraps ultralytics YOLOv8-nano for lightweight obstacle detection.
    Model is auto-downloaded from HuggingFace/ultralytics on first run.
    Only keeps relevant classes (person, bicycle, car, etc.).
    """

    def __init__(self, model_name=YOLO_MODEL_NAME, conf=YOLO_CONF_THRESH,
                 classes=None, device="cpu"):
        self.conf = conf
        self.classes = classes or list(OBSTACLE_CLASSES.keys())
        self.device = device
        self.model = None
        self._load(model_name)

    def _load(self, model_name):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            # Force CPU (tiny model, no GPU needed)
            self.model.to(self.device)
            print(f"[ObjDet] YOLOv8-nano loaded ({model_name}, device={self.device})")
        except ImportError:
            print("[ObjDet] WARNING: ultralytics not installed. "
                  "Run: pip install ultralytics")
            print("[ObjDet] Object detection DISABLED.")
            self.model = None

    def detect(self, frame_bgr):
        """
        Run detection on a BGR frame.
        Returns list of dicts: {bbox: (x1,y1,x2,y2), class_id, class_name, conf, center}
        """
        if self.model is None:
            return []

        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "class_name": OBSTACLE_CLASSES.get(cls_id, f"cls_{cls_id}"),
                    "conf": conf,
                    "center": (cx, cy),
                    "height_px": y2 - y1,  # used for rough distance estimate
                })
        return detections


def estimate_obstacle_distance(det, frame_h, camera_fov_v_deg=55.0,
                               camera_height_m=0.8):
    """
    Rough monocular distance estimate using bounding-box bottom position.
    Uses the pinhole model: objects at the bottom of the frame are closer.
    Returns estimated distance in meters (very approximate).
    """
    _, y1, _, y2 = det["bbox"]
    # Use bottom of bounding box (foot position)
    foot_y = y2
    # Normalized position: 0 = top, 1 = bottom
    norm_y = foot_y / frame_h
    if norm_y < 0.3:
        return 15.0  # far away
    # Simple inverse model: distance ~ k / (norm_y - offset)
    distance = camera_height_m / max(0.01, math.tan(
        math.radians(camera_fov_v_deg * (norm_y - 0.5))))
    return max(0.5, min(20.0, abs(distance)))


# =============================================================================
# GPS Navigation
# =============================================================================
class GPSNavigator:
    """
    Reads NMEA sentences from a serial GPS and steers toward waypoints.
    Waypoints loaded from CSV: lat,lon[,name]
    """

    def __init__(self, serial_device=None, baud=9600, waypoints_file=None):
        self.lat = None
        self.lon = None
        self.speed_mps = 0.0
        self.heading_gps = 0.0  # degrees from north (from GPS RMC)
        self.fix_quality = 0
        self.last_update = 0.0
        self.lock = threading.Lock()
        self._running = False
        self._thread = None

        # Waypoints
        self.waypoints = []  # list of (lat, lon, name)
        self.current_wp_idx = 0
        self.wp_reached_radius_m = 5.0

        if waypoints_file:
            self._load_waypoints(waypoints_file)

        if serial_device:
            self._start_serial(serial_device, baud)

    def _load_waypoints(self, path):
        """Load waypoints from CSV: lat,lon[,name]"""
        self.waypoints = []
        try:
            with open(path, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    lat = float(parts[0])
                    lon = float(parts[1])
                    name = parts[2].strip() if len(parts) > 2 else f"WP{i}"
                    self.waypoints.append((lat, lon, name))
            print(f"[GPS] Loaded {len(self.waypoints)} waypoints from {path}")
        except Exception as e:
            print(f"[GPS] ERROR loading waypoints: {e}")

    def _start_serial(self, device, baud):
        """Start background thread reading NMEA from serial."""
        try:
            import serial as pyserial
            self._ser = pyserial.Serial(device, baud, timeout=1)
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[GPS] Reading from {device} @ {baud} baud")
        except ImportError:
            print("[GPS] WARNING: pyserial not installed. Run: pip install pyserial")
            print("[GPS] GPS DISABLED.")
        except Exception as e:
            print(f"[GPS] ERROR opening {device}: {e}")

    def _read_loop(self):
        """Background thread: read NMEA sentences."""
        while self._running:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
                if not line.startswith("$"):
                    continue
                self._parse_nmea(line)
            except Exception:
                time.sleep(0.1)

    def _parse_nmea(self, sentence):
        """Parse GGA and RMC sentences."""
        parts = sentence.split(",")
        with self.lock:
            if parts[0].endswith("GGA") and len(parts) >= 10:
                lat = self._nmea_to_decimal(parts[2], parts[3])
                lon = self._nmea_to_decimal(parts[4], parts[5])
                if lat is not None and lon is not None:
                    self.lat = lat
                    self.lon = lon
                    self.last_update = time.time()
                try:
                    self.fix_quality = int(parts[6])
                except (ValueError, IndexError):
                    pass

            elif parts[0].endswith("RMC") and len(parts) >= 8:
                lat = self._nmea_to_decimal(parts[3], parts[4])
                lon = self._nmea_to_decimal(parts[5], parts[6])
                if lat is not None and lon is not None:
                    self.lat = lat
                    self.lon = lon
                    self.last_update = time.time()
                try:
                    self.speed_mps = float(parts[7]) * 0.514444  # knots to m/s
                except (ValueError, IndexError):
                    pass
                try:
                    self.heading_gps = float(parts[8])
                except (ValueError, IndexError):
                    pass

    @staticmethod
    def _nmea_to_decimal(raw, hemi):
        if not raw:
            return None
        try:
            raw = float(raw)
        except ValueError:
            return None
        deg = int(raw // 100)
        minutes = raw - deg * 100
        val = deg + minutes / 60.0
        if hemi in ("S", "W"):
            val = -val
        return val

    def get_position(self):
        """Returns (lat, lon) or (None, None)."""
        with self.lock:
            return self.lat, self.lon

    def get_bearing_to_waypoint(self):
        """
        Returns (bearing_deg, distance_m, wp_name) to the current waypoint.
        bearing_deg: 0=North, 90=East, etc.
        Returns (None, None, None) if no GPS fix or no waypoints.
        """
        with self.lock:
            lat, lon = self.lat, self.lon

        if lat is None or lon is None:
            return None, None, None
        if self.current_wp_idx >= len(self.waypoints):
            return None, None, "ARRIVED"

        wp_lat, wp_lon, wp_name = self.waypoints[self.current_wp_idx]
        bearing = self._bearing(lat, lon, wp_lat, wp_lon)
        dist = self._haversine(lat, lon, wp_lat, wp_lon)

        # Auto-advance to next waypoint if close enough
        if dist < self.wp_reached_radius_m:
            print(f"[GPS] Reached waypoint: {wp_name} ({dist:.1f}m)")
            self.current_wp_idx += 1
            if self.current_wp_idx < len(self.waypoints):
                next_wp = self.waypoints[self.current_wp_idx]
                print(f"[GPS] Next waypoint: {next_wp[2]}")
            else:
                print("[GPS] All waypoints reached!")

        return bearing, dist, wp_name

    def get_gps_heading_correction(self):
        """
        Returns a correction angle (degrees) to steer toward the next waypoint.
        Positive = turn right, negative = turn left.
        Returns (0.0, None, None) if no GPS data.
        Returns (0.0, None, "ARRIVED") if all waypoints reached.
        """
        bearing, dist, wp_name = self.get_bearing_to_waypoint()
        if bearing is None:
            # Preserve wp_name so "ARRIVED" state is communicated
            return 0.0, dist, wp_name

        with self.lock:
            current_heading = self.heading_gps

        # Difference between desired bearing and current heading
        diff = bearing - current_heading
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff, dist, wp_name

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Distance in meters between two GPS coordinates."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        """Bearing in degrees from (lat1,lon1) to (lat2,lon2). 0=N, 90=E."""
        dlon = math.radians(lon2 - lon1)
        lat1r, lat2r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r) -
             math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        # Close the serial port if open
        if hasattr(self, "_ser") and self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass


# =============================================================================
# Scooter Serial Commander
# =============================================================================
class ScooterCommander:
    """
    Sends steering angle (degrees) and speed (m/s) to the scooter
    over a serial connection.

    Protocol: each command is a line:
        CMD,<steer_deg>,<speed_mps>\n
    Example:
        CMD,-12.5,1.2\n    means steer 12.5 deg left at 1.2 m/s
        CMD,0.0,0.0\n      means stop
    """

    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.port = port
        if port:
            try:
                import serial as pyserial
                self.ser = pyserial.Serial(port, baud, timeout=0.1)
                print(f"[Scooter] Serial connected: {port} @ {baud}")
            except ImportError:
                print("[Scooter] WARNING: pyserial not installed.")
            except Exception as e:
                print(f"[Scooter] ERROR: {e}")

    def send_command(self, steer_deg, speed_mps):
        """Send steering + speed command. Returns the command string."""
        cmd = f"CMD,{steer_deg:.1f},{speed_mps:.2f}\n"
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
            except Exception as e:
                print(f"[Scooter] Write error: {e}")
        return cmd.strip()

    def stop(self):
        """Emergency stop."""
        self.send_command(0.0, 0.0)
        if self.ser:
            self.ser.close()


# =============================================================================
# BEV Calibration Tool
# =============================================================================
def run_calibration(camera_id=0, video_path=None):
    """Interactive 4-point BEV calibration. Click 4 sidewalk corners."""
    print("\n=== BEV CALIBRATION MODE ===")
    print("Click 4 points on the sidewalk in order:")
    print("  1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left")
    print("Press 'r' to reset, 's' to save, 'q' to quit.\n")

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame.")
        cap.release()
        return

    points = []
    labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"  Point {len(points)}: ({x}, {y}) - {labels[len(points)-1]}")

    cv2.namedWindow("BEV Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("BEV Calibration", on_click)

    while True:
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(display, f"{i+1}: {labels[i]}", (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
        info = f"Points: {len(points)}/4 | 'r'=reset 's'=save 'q'=quit"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("BEV Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
            points.clear()
            print("  Points reset.")
        elif key == ord('s') and len(points) == 4:
            src = np.array(points, dtype=np.float32)
            np.save(CALIBRATION_FILE, src)
            print(f"\n  Saved calibration to {CALIBRATION_FILE}")
            print(f"  Source points: {src.tolist()}")
            break
        elif key == ord('q') or key == 27:
            print("  Calibration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# Load BEV calibration
# =============================================================================
def load_bev_params():
    """Load calibration or use defaults. Returns (H, Hinv, src_points)."""
    if os.path.exists(CALIBRATION_FILE):
        src = np.load(CALIBRATION_FILE).astype(np.float32)
        print(f"Loaded BEV calibration from {CALIBRATION_FILE}")
    else:
        src = DEFAULT_SRC_POINTS
        print("Using default BEV calibration (scooter camera). Run --calibrate for iPhone.")
    dst = DEFAULT_DST_POINTS
    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = np.linalg.inv(H)
    return H, Hinv, src


# =============================================================================
# BEV Mask Cleaning
# =============================================================================
def clean_sidewalk_mask(bev_mask_255, dt_thresh=DT_CORE_THRESH):
    mask = bev_mask_255.copy().astype(np.uint8)
    k7 = np.ones((7, 7), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    core = (dist > float(dt_thresh)).astype(np.uint8) * 255
    core = cv2.dilate(core, k5, iterations=1)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k5, iterations=1)
    return core


def select_main_component(mask_255, bottom_band_px=45, center_weight=0.35):
    bin_ = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if num <= 1:
        return mask_255
    H, W = mask_255.shape
    bottom_band_px = int(max(1, min(H, bottom_band_px)))
    bottom_slice = labels[max(0, H - bottom_band_px):, :]
    center_x = W / 2.0
    best_label, best_score = None, -1.0
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        touches_bottom = bool(np.any(bottom_slice == idx))
        cx, _ = centroids[idx]
        center_bonus = 1.0 - min(1.0, abs(cx - center_x) / (center_x + 1e-6))
        score = area
        if touches_bottom:
            score += area * 0.5
        score += area * float(center_weight) * max(0.0, center_bonus)
        if score > best_score:
            best_score = score
            best_label = idx
    if best_label is None:
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_255, dtype=np.uint8)
    out[labels == best_label] = 255
    return out


# =============================================================================
# Skeletonization (Guo-Hall via OpenCV)
# =============================================================================
def skeletonize_guohall(mask_255):
    try:
        from cv2.ximgproc import thinning, THINNING_GUOHALL
        bin_ = ((mask_255 > 0).astype(np.uint8)) * 255
        skel = thinning(bin_, THINNING_GUOHALL)
        # thinning() may return 0/1 or 0/255 depending on OpenCV version
        # normalize to 0/255 safely (avoids 255*255 overflow)
        return ((skel > 0).astype(np.uint8)) * 255
    except ImportError:
        img = (mask_255 > 0).astype(np.uint8)
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel * 255


def extract_skeleton(bev_binary, trim_px=5):
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(bev_binary, cv2.MORPH_CLOSE, kernel)
    clean = cv2.medianBlur(clean, 5)
    _, binary = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize_guohall(binary)
    if trim_px > 0:
        skel[:trim_px, :] = 0
        skel[-trim_px:, :] = 0
        skel[:, :trim_px] = 0
        skel[:, -trim_px:] = 0
    return skel


def prune_small_branches(skel, min_len=PRUNE_BRANCH_LEN):
    s = skel.copy()
    for _ in range(min_len):
        nb = cv2.filter2D((s > 0).astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        endpoints = ((s > 0) & (nb == 2))
        s[endpoints] = 0
    return s


def prune_graph_branches(G, min_branch_length=40):
    G = G.copy()
    endpoints = [n for n in list(G.nodes) if G.degree[n] == 1]
    to_remove = set()
    for ep in endpoints:
        path = [ep]
        current, prev = ep, None
        total_len = 0.0
        while True:
            nbrs = [n for n in G.neighbors(current) if n != prev]
            if not nbrs:
                break
            nxt = nbrs[0]
            total_len += G[current][nxt]["weight"]
            path.append(nxt)
            if G.degree[nxt] != 2:
                break
            prev, current = current, nxt
        if total_len < min_branch_length:
            to_remove.update(path)
    G.remove_nodes_from(to_remove)
    return G


# =============================================================================
# Heading computation
# =============================================================================
def compute_heading(path_pts):
    """
    Compute heading angle from a BEV path.
    Returns angle in degrees: 0 = straight ahead, negative = left, positive = right.
    """
    if len(path_pts) < 2:
        return 0.0
    start = np.array(path_pts[0], dtype=float)
    idx = min(len(path_pts) - 1, max(1, len(path_pts) * 2 // 5))
    end = np.array(path_pts[idx], dtype=float)
    dx = end[0] - start[0]
    dy = start[1] - end[1]
    if dy <= 0:
        return 0.0
    angle_rad = math.atan2(dx, dy)
    return math.degrees(angle_rad)


def heading_to_command(angle_deg):
    """Convert heading angle to command string and color."""
    abs_angle = abs(angle_deg)
    if abs_angle < HEADING_STRAIGHT_THRESH:
        return "STRAIGHT", COLOR_STRAIGHT
    elif abs_angle < HEADING_TURN_THRESH:
        if angle_deg < 0:
            return "LEFT", COLOR_LEFT
        else:
            return "RIGHT", COLOR_RIGHT
    else:
        if angle_deg < 0:
            return "SHARP LEFT", COLOR_SHARP
        else:
            return "SHARP RIGHT", COLOR_SHARP


def compute_speed(angle_deg, min_obstacle_dist, has_path):
    """
    Compute target speed based on heading angle and nearest obstacle distance.
    Returns speed in m/s.
    """
    if not has_path:
        return SPEED_STOP

    # Obstacle override
    if min_obstacle_dist is not None:
        if min_obstacle_dist < OBSTACLE_STOP_M:
            return SPEED_STOP
        elif min_obstacle_dist < OBSTACLE_CLOSE_M:
            return SPEED_OBSTACLE_NEAR

    # Speed from heading
    abs_angle = abs(angle_deg)
    if abs_angle < HEADING_STRAIGHT_THRESH:
        return SPEED_MAX
    elif abs_angle < HEADING_TURN_THRESH:
        # Linear interpolation between SPEED_TURN and SPEED_MAX
        t = (abs_angle - HEADING_STRAIGHT_THRESH) / (HEADING_TURN_THRESH - HEADING_STRAIGHT_THRESH)
        return SPEED_MAX - t * (SPEED_MAX - SPEED_TURN)
    else:
        return SPEED_SHARP_TURN


# =============================================================================
# HUD drawing
# =============================================================================
def draw_heading_hud(img, command, angle_deg, speed_mps, color, fps, pipeline_ms,
                     detections=None, gps_info=None):
    """Draw heading, speed, obstacles, and GPS on the camera view."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Large command text at top center ----
    text = command
    (tw, th), baseline = cv2.getTextSize(text, font, 1.8, 4)
    tx = (w - tw) // 2
    ty = 60
    cv2.rectangle(img, (tx - 15, ty - th - 15), (tx + tw + 15, ty + baseline + 15),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, 1.8, color, 4, cv2.LINE_AA)

    # ---- Angle + speed below command ----
    cmd_line = f"Steer: {angle_deg:+.1f} deg | Speed: {speed_mps:.2f} m/s"
    (cw, ch), _ = cv2.getTextSize(cmd_line, font, 0.7, 2)
    cx = (w - cw) // 2
    cy = ty + 40
    cv2.rectangle(img, (cx - 8, cy - ch - 5), (cx + cw + 8, cy + 8), (0, 0, 0), -1)
    cv2.putText(img, cmd_line, (cx, cy), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- Serial command format ----
    serial_cmd = f"CMD,{angle_deg:+.1f},{speed_mps:.2f}"
    (sw, sh), _ = cv2.getTextSize(serial_cmd, font, 0.55, 1)
    sx = (w - sw) // 2
    sy = cy + 30
    cv2.putText(img, serial_cmd, (sx, sy), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # ---- Direction arrow ----
    arrow_cx = w // 2
    arrow_cy = h // 2
    arrow_len = 120
    arrow_angle_rad = math.radians(-angle_deg)
    arrow_ex = int(arrow_cx + arrow_len * math.sin(arrow_angle_rad))
    arrow_ey = int(arrow_cy - arrow_len * math.cos(arrow_angle_rad))
    cv2.arrowedLine(img, (arrow_cx, arrow_cy + 30), (arrow_ex, arrow_ey - 30),
                    color, 6, cv2.LINE_AA, tipLength=0.3)

    # ---- Object detection boxes ----
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            dist = det.get("distance_m", None)
            label = f"{det['class_name']} {det['conf']:.0%}"
            if dist is not None:
                label += f" ~{dist:.1f}m"
            box_color = COLOR_OBJ_WARN if (dist and dist < OBSTACLE_CLOSE_M) else COLOR_OBJ_BOX
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(img, label, (x1, y1 - 8), font, 0.5, box_color, 1, cv2.LINE_AA)

    # ---- GPS info (top-right) ----
    if gps_info:
        gps_lines = gps_info if isinstance(gps_info, list) else [gps_info]
        for i, line in enumerate(gps_lines):
            cv2.putText(img, line, (w - 350, 30 + i * 22),
                        font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # ---- FPS / latency at bottom left ----
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Pipeline: {pipeline_ms:.0f} ms",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (10, h - 20 - i * 25),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def draw_bev_hud(bev_rgb, paths, best_idx, skel, bev_sidewalk, command, angle_deg,
                 speed_mps=0.0):
    """Draw BEV visualization with paths, heading, and speed."""
    h_bev, w_bev = bev_sidewalk.shape
    vis = np.full((h_bev, w_bev, 3), (240, 240, 240), dtype=np.uint8)
    vis[bev_sidewalk > 0] = (220, 235, 245)

    skel_thick = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    vis[skel_thick > 0] = (180, 180, 180)

    for idx, (path_pts, plen) in enumerate(paths):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        thickness = 6 if idx == best_idx else 2
        path_np = np.int32(path_pts).reshape(-1, 1, 2)
        cv2.polylines(vis, [path_np], False, color, thickness, cv2.LINE_AA)

    if paths and best_idx >= 0:
        best_path = paths[best_idx][0]
        path_np = np.int32(best_path).reshape(-1, 1, 2)
        cv2.polylines(vis, [path_np], False, (0, 200, 0), 8, cv2.LINE_AA)

    # Command + speed text
    cv2.putText(vis, f"{command} ({angle_deg:+.1f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, f"Speed: {speed_mps:.2f} m/s", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 150), 1)

    return vis


# =============================================================================
# Mask splitting
# =============================================================================
def split_masks(model_output):
    m = model_output.astype(np.uint8)
    if set(np.unique(m).tolist()) <= {0, 255}:
        sidewalk = (m > 0).astype(np.uint8) * 255
        road = np.zeros_like(sidewalk)
    else:
        sidewalk = (m == SIDEWALK_ID).astype(np.uint8) * 255
        road = (m == ROAD_ID).astype(np.uint8) * 255
    return sidewalk, road


# =============================================================================
# Main live loop
# =============================================================================
def run_live(camera_id=0, video_path=None, save_video=False, stride=1,
             enable_detection=True, gps_device=None, gps_waypoints=None,
             serial_port=None, enable_logging=False, log_dir="logs"):
    print("\n=== LIVE HEADING + OBJECT DETECTION + GPS DEMO ===")

    # Initialize data logger
    logger = None
    if enable_logging:
        logger = DataLogger(log_dir=log_dir)

    # Load BEV calibration
    H_mat, Hinv, src_pts = load_bev_params()

    # Initialize segmentation model
    print("Loading SegFormer model...")
    cfg = Config(
        model_dir=MODEL_DIR,
        conf_thresh=0.5,
        road_id=ROAD_ID,
        inference_resize=SEG_INPUT_RES,
    )
    seg_model = FastRoadDetector(cfg)
    print("SegFormer ready.")

    # Initialize object detector (YOLOv8-nano)
    obj_detector = None
    if enable_detection:
        print("Loading YOLOv8-nano object detector...")
        obj_detector = TinyObjectDetector(
            model_name=YOLO_MODEL_NAME,
            conf=YOLO_CONF_THRESH,
        )

    # Initialize GPS navigator
    gps_nav = None
    if gps_device or gps_waypoints:
        gps_nav = GPSNavigator(
            serial_device=gps_device,
            waypoints_file=gps_waypoints,
        )

    # Initialize scooter serial commander
    scooter = ScooterCommander(port=serial_port)

    # Open camera or video
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"Playing video: {video_path}")
    else:
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Camera {camera_id} opened.")

    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    # Smoothing buffers
    heading_buffer = deque(maxlen=5)
    speed_buffer = deque(maxlen=5)
    last_mask = None
    frame_id = 0
    fps_counter = deque(maxlen=30)
    vw = None

    print("Running... Press 'q' or ESC to quit.\n")
    print(f"{'Frame':>6} | {'Command':>12} | {'Steer':>8} | {'Speed':>7} | "
          f"{'Obst':>5} | {'ms':>5} | {'FPS':>5}")
    print("-" * 72)

    try:
        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                if video_path:
                    break
                print("Frame lost, retrying...")
                continue

            frame_h, frame_w = frame.shape[:2]
            run_net = (frame_id % stride == 0)

            # --- 1) Segmentation ---
            t_seg_start = time.time()
            if run_net or last_mask is None:
                seg, _ = seg_model.process_frame(frame)
                if seg.shape != frame.shape[:2]:
                    seg = cv2.resize(seg, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
                last_mask = seg
            seg = last_mask
            sidewalk_mask, road_mask = split_masks(seg)
            t_seg = (time.time() - t_seg_start) * 1000

            # --- 2) Object detection ---
            t_det_start = time.time()
            detections = []
            min_obstacle_dist = None
            if obj_detector and run_net:
                detections = obj_detector.detect(frame)
                for det in detections:
                    det["distance_m"] = estimate_obstacle_distance(det, frame_h)
                if detections:
                    min_obstacle_dist = min(d["distance_m"] for d in detections)
            t_det = (time.time() - t_det_start) * 1000

            # --- 3) BEV projection + cleaning ---
            t_bev_start = time.time()
            bev_sidewalk = cv2.warpPerspective(sidewalk_mask, H_mat, BEV_SIZE)
            if TRIM_BOTTOM > 0:
                bev_sidewalk = bev_sidewalk[:-TRIM_BOTTOM, :]
            bev_sidewalk = clean_sidewalk_mask(bev_sidewalk, DT_CORE_THRESH)
            bev_sidewalk = select_main_component(bev_sidewalk)
            t_bev = (time.time() - t_bev_start) * 1000

            # --- 4) Skeleton + graph ---
            t_skel_start = time.time()
            skel = extract_skeleton(bev_sidewalk, trim_px=5)
            skel = prune_small_branches(skel, PRUNE_BRANCH_LEN)

            Hm, Wm = skel.shape
            G = nx.Graph()
            for y in range(Hm):
                xs = np.where(skel[y] > 0)[0]
                for x in xs:
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            ny_, nx_ = y + dy, x + dx
                            if 0 <= ny_ < Hm and 0 <= nx_ < Wm and skel[ny_, nx_] > 0:
                                G.add_edge((x, y), (nx_, ny_), weight=math.hypot(dx, dy))

            G = prune_graph_branches(G, min_branch_length=PRUNE_BRANCH_LEN * 3)
            t_skel = (time.time() - t_skel_start) * 1000

            # --- 5) Path finding ---
            t_path_start = time.time()
            paths = []
            best_idx = -1
            heading_raw = 0.0
            command = "STOP"
            cmd_color = COLOR_STOP
            has_path = False
            best_path_len = 0.0

            if G.number_of_nodes() > 1:
                endpoints = [n for n in G.nodes if G.degree[n] == 1]
                center_x = Wm // 2

                band_nodes = [n for n in G.nodes if n[1] >= Hm - BOTTOM_BAND_PX]
                band_eps = [n for n in endpoints if n[1] >= Hm - BOTTOM_BAND_PX]

                def start_key(p):
                    return (p[1], -abs(p[0] - center_x))

                if band_eps:
                    start = max(band_eps, key=start_key)
                elif band_nodes:
                    start = max(band_nodes, key=start_key)
                elif endpoints:
                    start = max(endpoints, key=lambda p: p[1])
                else:
                    start = max(G.nodes, key=lambda p: p[1])

                for end in endpoints:
                    if end == start:
                        continue
                    try:
                        path = nx.dijkstra_path(G, start, end, weight="weight")
                        plen = nx.path_weight(G, path, weight="weight")
                        paths.append((path, plen))
                    except nx.NetworkXNoPath:
                        continue

                paths.sort(key=lambda x: x[1], reverse=True)

                if paths:
                    best_score = float('inf')
                    for idx, (path_pts, plen) in enumerate(paths):
                        if len(path_pts) < 2:
                            continue
                        h_angle = abs(compute_heading(path_pts))
                        end_x = path_pts[-1][0]
                        lat_shift = abs(end_x - center_x) / (Wm / 2.0)
                        score = 0.6 * h_angle + 0.4 * lat_shift * 45.0
                        if score < best_score:
                            best_score = score
                            best_idx = idx

                    if best_idx >= 0:
                        heading_raw = compute_heading(paths[best_idx][0])
                        best_path_len = paths[best_idx][1]
                        has_path = True
            t_path = (time.time() - t_path_start) * 1000

            # --- 6) GPS correction ---
            t_gps_start = time.time()
            gps_info = None
            gps_lat, gps_lon, gps_fix = None, None, 0
            gps_wp_name, gps_wp_dist, gps_correction = None, None, 0.0
            heading_angle = heading_raw

            if gps_nav:
                gps_correction, gps_wp_dist, gps_wp_name = gps_nav.get_gps_heading_correction()
                gps_lat, gps_lon = gps_nav.get_position()
                with gps_nav.lock:
                    gps_fix = gps_nav.fix_quality
                gps_info = []
                if gps_lat is not None:
                    gps_info.append(f"GPS: {gps_lat:.6f}, {gps_lon:.6f}")
                if gps_wp_name:
                    gps_info.append(f"WP: {gps_wp_name} ({gps_wp_dist:.0f}m)" if gps_wp_dist else f"WP: {gps_wp_name}")
                    gps_info.append(f"GPS correction: {gps_correction:+.1f} deg")
                    heading_angle = 0.7 * heading_angle + 0.3 * gps_correction
            t_gps = (time.time() - t_gps_start) * 1000

            # --- 7) Smooth heading + compute speed ---
            t_cmd_start = time.time()
            heading_buffer.append(heading_angle)
            heading_smoothed = float(np.mean(heading_buffer))
            command, cmd_color = heading_to_command(heading_smoothed)

            speed_raw = compute_speed(heading_smoothed, min_obstacle_dist, has_path)
            speed_buffer.append(speed_raw)
            speed_smoothed = float(np.mean(speed_buffer))

            serial_str = scooter.send_command(heading_smoothed, speed_smoothed)
            t_cmd = (time.time() - t_cmd_start) * 1000

            # --- Total pipeline time ---
            t_total = (time.time() - t0) * 1000
            fps_counter.append(time.time())
            if len(fps_counter) > 1:
                fps = (len(fps_counter) - 1) / (fps_counter[-1] - fps_counter[0])
            else:
                fps = 0.0

            # --- 8) Log frame data ---
            if logger:
                det_classes = ";".join(d["class_name"] for d in detections) if detections else ""
                det_dists = ";".join(f'{d["distance_m"]:.2f}' for d in detections) if detections else ""
                logger.log(
                    frame_id=frame_id,
                    # Timing
                    t_segmentation=round(t_seg, 2),
                    t_detection=round(t_det, 2),
                    t_bev=round(t_bev, 2),
                    t_skeleton=round(t_skel, 2),
                    t_pathfinding=round(t_path, 2),
                    t_gps_fusion=round(t_gps, 2),
                    t_command=round(t_cmd, 2),
                    t_total_pipeline=round(t_total, 2),
                    fps=round(fps, 2),
                    # Heading & control
                    heading_raw_deg=round(heading_raw, 3),
                    heading_smoothed_deg=round(heading_smoothed, 3),
                    command=command,
                    speed_raw_mps=round(speed_raw, 3),
                    speed_smoothed_mps=round(speed_smoothed, 3),
                    serial_cmd=serial_str,
                    # Path info
                    has_path=int(has_path),
                    num_paths=len(paths),
                    best_path_length_px=round(best_path_len, 1),
                    num_graph_nodes=G.number_of_nodes(),
                    num_graph_edges=G.number_of_edges(),
                    # Detections
                    num_detections=len(detections),
                    min_obstacle_dist_m=round(min_obstacle_dist, 2) if min_obstacle_dist else "",
                    detection_classes=det_classes,
                    detection_distances=det_dists,
                    # GPS
                    gps_lat=gps_lat if gps_lat else "",
                    gps_lon=gps_lon if gps_lon else "",
                    gps_fix_quality=gps_fix,
                    gps_wp_name=gps_wp_name or "",
                    gps_wp_dist_m=round(gps_wp_dist, 1) if gps_wp_dist else "",
                    gps_correction_deg=round(gps_correction, 2) if gps_correction else "",
                    # Mask stats
                    sidewalk_mask_pixels=int(np.count_nonzero(sidewalk_mask)),
                    bev_mask_pixels=int(np.count_nonzero(bev_sidewalk)),
                    skeleton_pixels=int(np.count_nonzero(skel)),
                )

            # --- 9) Visualization ---
            cam_vis = frame.copy()
            seg_overlay = np.zeros_like(cam_vis)
            seg_overlay[sidewalk_mask > 0] = (0, 200, 0)
            seg_overlay[road_mask > 0] = (255, 120, 0)
            cam_vis = cv2.addWeighted(seg_overlay, 0.35, cam_vis, 1.0, 0)

            if paths and best_idx >= 0:
                best_path = paths[best_idx][0]
                pts = np.array(best_path, dtype=np.float32).reshape(-1, 1, 2)
                cam_pts = cv2.perspectiveTransform(pts, Hinv).reshape(-1, 2)
                cam_pts_int = np.int32(cam_pts).reshape(-1, 1, 2)
                cv2.polylines(cam_vis, [cam_pts_int], False, cmd_color, 8, cv2.LINE_AA)

            cam_vis = draw_heading_hud(cam_vis, command, heading_smoothed, speed_smoothed,
                                       cmd_color, fps, t_total,
                                       detections=detections, gps_info=gps_info)

            bev_vis = draw_bev_hud(None, paths, best_idx, skel, bev_sidewalk,
                                   command, heading_smoothed, speed_smoothed)

            display_h = 480
            cam_display = cv2.resize(cam_vis,
                (int(display_h * cam_vis.shape[1] / cam_vis.shape[0]), display_h))
            bev_display = cv2.resize(bev_vis,
                (int(display_h * bev_vis.shape[1] / bev_vis.shape[0]), display_h))

            combined = np.hstack((cam_display, bev_display))
            cv2.imshow("Live Heading + Detection + GPS", combined)

            if save_video and vw is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                vw = cv2.VideoWriter("heading_demo_output.mp4", fourcc, 15,
                                     (combined.shape[1], combined.shape[0]))
            if vw:
                vw.write(combined)

            # Console output (every 10 frames)
            if frame_id % 10 == 0:
                obs_str = f"{min_obstacle_dist:.1f}m" if min_obstacle_dist else "none"
                print(f"{frame_id:6d} | {command:>12s} | {heading_smoothed:+7.1f}d | "
                      f"{speed_smoothed:5.2f}ms | {obs_str:>5s} | {t_total:5.0f} | {fps:5.1f}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Save logger metadata and close
        if logger:
            logger.save_metadata(
                camera_id=camera_id, video_path=video_path,
                stride=stride, detection_enabled=enable_detection,
                gps_device=gps_device, gps_waypoints=gps_waypoints,
                serial_port=serial_port, total_frames=frame_id,
                model_dir=MODEL_DIR, seg_resolution=SEG_INPUT_RES,
                yolo_model=YOLO_MODEL_NAME if enable_detection else "disabled",
            )
            logger.close()
        # Cleanup
        scooter.stop()
        if gps_nav:
            gps_nav.stop()
        cap.release()
        if vw:
            vw.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Processed {frame_id} frames.")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live heading + object detection + GPS navigation demo")

    # Camera / video
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (0=default, 1=iPhone usually)")
    parser.add_argument("--video", type=str, default=None,
                        help="Video file path (instead of live camera)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run BEV calibration mode")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (1=all)")
    parser.add_argument("--save", action="store_true",
                        help="Save output video")

    # Object detection
    parser.add_argument("--no-detection", action="store_true",
                        help="Disable YOLOv8-nano object detection")

    # GPS
    parser.add_argument("--gps-device", type=str, default=None,
                        help="GPS serial device (e.g., COM3 or /dev/ttyUSB0)")
    parser.add_argument("--gps-baud", type=int, default=9600,
                        help="GPS baud rate (default: 9600)")
    parser.add_argument("--gps-waypoints", type=str, default=None,
                        help="CSV file with GPS waypoints: lat,lon[,name]")

    # Scooter serial
    parser.add_argument("--serial-port", type=str, default=None,
                        help="Serial port for scooter commands (e.g., COM4)")
    parser.add_argument("--serial-baud", type=int, default=115200,
                        help="Scooter serial baud rate (default: 115200)")

    # Data logging
    parser.add_argument("--log", action="store_true",
                        help="Enable per-frame CSV data logging for experiments")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for log files (default: logs/)")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration(camera_id=args.camera, video_path=args.video)
    else:
        run_live(
            camera_id=args.camera,
            video_path=args.video,
            save_video=args.save,
            stride=args.stride,
            enable_detection=not args.no_detection,
            gps_device=args.gps_device,
            gps_waypoints=args.gps_waypoints,
            serial_port=args.serial_port,
            enable_logging=args.log,
            log_dir=args.log_dir,
        )
