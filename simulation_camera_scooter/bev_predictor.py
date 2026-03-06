"""
bev_predictor.py
================
BEV Predictive Frame Reuse — skip expensive segmentation on frames where
the BEV scene can be accurately predicted from ego-motion.

On straight roads (low curvature), multiple consecutive frames are predicted
via affine warping of the previous BEV mask + path.  On turns or when
prediction quality drops, the full pipeline runs.

Design: "human attention" analogy — coast on predictions when straight,
full attention at intersections/turns.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    BEV_SIZE,
    NAV_BEV_FORWARD_M,
    NAV_BEV_LATERAL_M,
    PREDICT_ENABLED,
    PREDICT_MAX_SKIP_STRAIGHT,
    PREDICT_MAX_SKIP_TURN,
    PREDICT_CURVATURE_STRAIGHT,
    PREDICT_CURVATURE_SHARP,
    PREDICT_BLEND_ALPHA,
    PREDICT_CONFIDENCE_FLOOR,
    BEV_PIXELS_PER_METER_FORWARD,
    BEV_PIXELS_PER_METER_LATERAL,
)


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two float masks in [0, 1]."""
    ba = a > 0.5
    bb = b > 0.5
    inter = float(np.logical_and(ba, bb).sum())
    union = float(np.logical_or(ba, bb).sum())
    return inter / max(union, 1.0)


class BEVPredictiveTracker:
    """Predicts BEV mask and path on frames where segmentation is skipped."""

    def __init__(self):
        self.last_bev_mask: Optional[np.ndarray] = None      # float32, 0-1
        self.last_path_points_m: Optional[np.ndarray] = None  # (N, 2) metric
        self.last_path_model = None                           # CubicPathModel
        self.last_curvature: float = 0.0
        self.cumulative_skip_count: int = 0
        self.prediction_confidence: float = 1.0
        self.last_occ_ratio: float = 0.0
        self._min_occ_ratio: float = 0.012
        self._min_occ_pixels: int = 450

        # BEV geometry
        self._bev_h, self._bev_w = BEV_SIZE[1], BEV_SIZE[0]
        self._px_per_m_fwd = BEV_PIXELS_PER_METER_FORWARD
        self._px_per_m_lat = BEV_PIXELS_PER_METER_LATERAL

    # ------------------------------------------------------------------
    # Ego displacement
    # ------------------------------------------------------------------
    def compute_ego_displacement(
        self, speed_mps: float, dt: float, path_model=None, steer_deg: float = 0.0
    ) -> Tuple[float, float, float]:
        """Compute ego displacement in BEV metric space (dx_fwd, dy_lat, dtheta).

        Primary: path_model.query_s(ds) for displacement along the path curve.
        Fallback: dead-reckoning from speed + steer.
        """
        ds = speed_mps * dt
        if ds < 1e-6:
            return 0.0, 0.0, 0.0

        if path_model is not None and path_model.length_m > ds:
            x, y, _kappa = path_model.query_s(ds)
            dtheta = path_model.heading_of_x(x)
            return float(x), float(y), float(dtheta)

        # Dead-reckoning fallback (bicycle model approximation)
        steer_rad = math.radians(steer_deg)
        wheelbase = 0.72  # PurePursuitConfig default
        if abs(steer_rad) < 1e-6:
            return ds, 0.0, 0.0
        dtheta = ds * math.tan(steer_rad) / wheelbase
        dx = ds * math.cos(dtheta / 2.0)
        dy = ds * math.sin(dtheta / 2.0)
        return float(dx), float(dy), float(dtheta)

    # ------------------------------------------------------------------
    # BEV mask warping
    # ------------------------------------------------------------------
    def warp_bev_mask(
        self, bev_mask_f: np.ndarray, dx_m: float, dy_m: float, dtheta_rad: float
    ) -> np.ndarray:
        """Affine warp of BEV mask by ego displacement.

        In BEV, ego is at bottom-center.  Forward motion shifts mask DOWN,
        lateral motion shifts LEFT/RIGHT, rotation rotates around ego point.
        New pixels from top/sides fill with 0 (unknown).
        """
        h, w = bev_mask_f.shape[:2]
        ego_x = w / 2.0
        ego_y = float(h - 1)

        # Rotation around ego point
        R = cv2.getRotationMatrix2D((ego_x, ego_y), math.degrees(dtheta_rad), 1.0)
        # Translation: forward (dx) → shift down; lateral (dy) → shift right
        R[0, 2] += dy_m * self._px_per_m_lat
        R[1, 2] += dx_m * self._px_per_m_fwd
        return cv2.warpAffine(
            bev_mask_f, R, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    # ------------------------------------------------------------------
    # Path warping
    # ------------------------------------------------------------------
    def warp_path(
        self, path_points_m: np.ndarray, dx_m: float, dy_m: float, dtheta_rad: float
    ) -> np.ndarray:
        """Transform path points by ego displacement.

        Path is in ego-centric coords (col0=forward, col1=lateral).
        When ego moves forward by dx, world shifts back.
        """
        if path_points_m is None or len(path_points_m) < 2:
            return np.zeros((0, 2), dtype=np.float32)

        shifted = path_points_m.astype(np.float64) - np.array([dx_m, dy_m])
        cos_t = math.cos(-dtheta_rad)
        sin_t = math.sin(-dtheta_rad)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = (shifted @ R.T).astype(np.float32)
        # Trim points behind ego (x < 0)
        valid = rotated[:, 0] >= 0.0
        return rotated[valid]

    # ------------------------------------------------------------------
    # Adaptive skip decision
    # ------------------------------------------------------------------
    def should_skip(self, speed: float = 0.0, dt: float = 0.0) -> bool:
        """Decide whether to skip segmentation on this frame.

        Skip more on straight roads, less on turns:
        - |kappa| < 0.10 (straight): up to PREDICT_MAX_SKIP_STRAIGHT skips
        - |kappa| < 0.30 (gentle turn): up to PREDICT_MAX_SKIP_TURN skip
        - |kappa| >= 0.30 (sharp turn): never skip

        Override: don't skip if confidence is low or no stored state.
        """
        if not PREDICT_ENABLED:
            return False
        if self.last_bev_mask is None:
            return False
        if self.last_occ_ratio < self._min_occ_ratio:
            return False
        if self.prediction_confidence < PREDICT_CONFIDENCE_FLOOR:
            return False

        kappa = abs(self.last_curvature)

        # Near-stopped: barely moving → safe to coast
        if speed < 0.2:
            max_skip = PREDICT_MAX_SKIP_STRAIGHT
        elif kappa < PREDICT_CURVATURE_STRAIGHT:
            max_skip = PREDICT_MAX_SKIP_STRAIGHT
        elif kappa < PREDICT_CURVATURE_SHARP:
            max_skip = PREDICT_MAX_SKIP_TURN
        else:
            max_skip = 0

        return self.cumulative_skip_count < max_skip

    # ------------------------------------------------------------------
    # Compute frame handler (segmentation ran)
    # ------------------------------------------------------------------
    def on_compute_frame(
        self,
        real_bev_mask: np.ndarray,
        path_result,
        speed: float,
        dt: float,
        steer_deg: float,
    ) -> np.ndarray:
        """Called when segmentation runs.  Blend predicted + real, update state.

        Returns blended BEV mask (uint8, 0/255).
        """
        real_f = real_bev_mask.astype(np.float32) / 255.0 if real_bev_mask.dtype == np.uint8 else real_bev_mask.astype(np.float32)

        if self.last_bev_mask is not None and self.cumulative_skip_count > 0:
            # Predict where old mask should be now
            total_dt = dt * (self.cumulative_skip_count + 1)
            dx, dy, dtheta = self.compute_ego_displacement(
                speed, total_dt, self.last_path_model, steer_deg
            )
            predicted = self.warp_bev_mask(self.last_bev_mask, dx, dy, dtheta)

            # Evaluate prediction quality
            iou = _compute_iou(predicted, real_f)
            self.prediction_confidence = 0.7 * self.prediction_confidence + 0.3 * iou

            # Blend: mostly real, prediction fills gaps
            alpha = PREDICT_BLEND_ALPHA
            blended = alpha * real_f + (1.0 - alpha) * predicted
        else:
            blended = real_f
            # First frame or no skips: confidence stays high
            self.prediction_confidence = min(1.0, 0.9 * self.prediction_confidence + 0.1)

        # Store state
        self.last_bev_mask = blended.copy()
        self.last_occ_ratio = float(np.count_nonzero(blended > 0.45)) / max(1.0, float(blended.size))
        if self.last_occ_ratio < self._min_occ_ratio:
            self.prediction_confidence = min(self.prediction_confidence, 0.30)
        if path_result is not None and path_result.has_path:
            self.last_path_points_m = (
                path_result.best_path_m.copy() if path_result.best_path_m is not None else None
            )
            self.last_path_model = path_result.path_model
            self.last_curvature = (
                path_result.path_model.curvature_of_x(1.0)
                if path_result.path_model is not None and path_result.path_model.x_max_m >= 1.0
                else 0.0
            )
        else:
            self.last_path_points_m = None
            self.last_path_model = None
            self.last_curvature = 0.0

        self.cumulative_skip_count = 0

        # Return uint8 binary mask
        return (blended > 0.45).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Skip frame handler (no segmentation)
    # ------------------------------------------------------------------
    def on_skip_frame(
        self, speed: float, dt: float, steer_deg: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Called when segmentation is skipped.  Return predicted mask + path.

        Returns:
            predicted_mask_255: uint8 BEV mask (0/255)
            predicted_path_m:   (N, 2) path in metric coords, or None
        """
        dx, dy, dtheta = self.compute_ego_displacement(
            speed, dt, self.last_path_model, steer_deg
        )

        predicted_mask = self.warp_bev_mask(self.last_bev_mask, dx, dy, dtheta)
        predicted_path = self.warp_path(self.last_path_points_m, dx, dy, dtheta)

        # Propagate prediction forward (stored state advances)
        self.last_bev_mask = predicted_mask
        mask_255 = (predicted_mask > 0.45).astype(np.uint8) * 255
        occ_pixels = int(np.count_nonzero(mask_255))
        occ_ratio = float(occ_pixels) / max(1.0, float(mask_255.size))
        self.last_occ_ratio = occ_ratio
        if occ_pixels < self._min_occ_pixels or occ_ratio < self._min_occ_ratio:
            # Predicted BEV is too sparse/empty; do not trust predicted path.
            self.prediction_confidence = min(self.prediction_confidence, 0.25)
            self.last_path_model = None
            self.cumulative_skip_count = max(self.cumulative_skip_count, PREDICT_MAX_SKIP_STRAIGHT)
            return mask_255, None

        self.last_path_points_m = predicted_path if len(predicted_path) >= 2 else self.last_path_points_m
        self.cumulative_skip_count += 1
        return mask_255, predicted_path if len(predicted_path) >= 2 else None
