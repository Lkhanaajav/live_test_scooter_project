"""
object_detector.py
==================
YOLOv8-nano obstacle detection + monocular distance estimation.
"""

import math

from config import OBSTACLE_CLASSES, YOLO_CONF_THRESH, YOLO_MODEL_NAME, OBSTACLE_CLOSE_M


def _default_device():
    """Prefer accelerator backends when available instead of forcing CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class TinyObjectDetector:
    """
    Wraps ultralytics YOLOv8-nano for lightweight obstacle detection.
    Model is auto-downloaded from HuggingFace/ultralytics on first run.
    Only keeps relevant classes (person, bicycle, car, etc.).
    """

    def __init__(self, model_name=YOLO_MODEL_NAME, conf=YOLO_CONF_THRESH,
                 classes=None, device=None):
        self.conf = conf
        self.classes = classes or list(OBSTACLE_CLASSES.keys())
        self.device = device or _default_device()
        self.model = None
        self._load(model_name)

    def _load(self, model_name):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
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
