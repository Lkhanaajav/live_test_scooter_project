"""
data_logger.py
==============
Per-frame CSV logger for thesis experiments.
"""

import csv
import json
import os
import time
from datetime import datetime


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
        "gps_intent_family", "planner_intent_family", "turn_lock_family",
        "speed_raw_mps", "speed_smoothed_mps", "serial_cmd",
        "pp_lookahead_m", "pp_kappa_cmd_m_inv", "pp_target_x_m", "pp_target_y_m",
        "pp_valid_path",
        "seg_iou", "seg_unstable_frames", "stability_mode",
        "stab_corr_px", "stab_corr_deg",
        # Path info
        "has_candidate_path", "has_model_path", "has_control_path", "has_path", "num_paths", "best_path_length_px",
        "num_graph_nodes", "num_graph_edges", "planner_mode", "path_source", "bev_mask_occ_ratio",
        "approval_confidence", "approval_margin", "planner_low_confidence", "planner_slowdown",
        "selected_template_id", "selected_template_family",
        "corridor_confidence", "corridor_valid_ratio", "corridor_forward_span_m", "corridor_width_cv",
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

