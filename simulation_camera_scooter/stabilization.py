"""
stabilization.py
================
Frame stabilization and temporal mask smoothing.
"""

import math
from collections import deque

import cv2
import numpy as np

from config import (
    STAB_SMOOTHING_RADIUS,
    STAB_MAX_CORRECTION_PX,
    STAB_MAX_CORRECTION_DEG,
    MASK_SMOOTH_ALPHA,
    MASK_SMOOTH_CONSISTENCY_THRESH,
)


class FrameStabilizer:
    """
    Lightweight video stabilization using optical-flow trajectory smoothing.

    How it works:
      1. Detect feature points in the previous grayscale frame.
      2. Track them into the current frame with Lucas-Kanade optical flow.
      3. Estimate a rigid transform (translation + rotation) between frames.
      4. Accumulate transforms into a "trajectory" (cumulative position).
      5. Smooth the trajectory with a moving-average window.
      6. Apply the difference (smoothed - actual) as a corrective warp.

    This removes high-frequency camera shake while preserving the low-frequency
    intentional motion (scooter turns, etc.).  Essential for reliable
    segmentation and BEV projection from a vibrating scooter handlebar.
    """

    def __init__(self,
                 smoothing_radius=STAB_SMOOTHING_RADIUS,
                 max_shift_px=STAB_MAX_CORRECTION_PX,
                 max_angle_deg=STAB_MAX_CORRECTION_DEG):
        self.smoothing_radius = smoothing_radius
        self.max_shift = float(max_shift_px)
        self.max_angle = math.radians(max_angle_deg)
        self.prev_gray = None
        # Cumulative trajectory
        self.cum_dx = 0.0
        self.cum_dy = 0.0
        self.cum_da = 0.0
        # Trajectory history for moving-average smoothing
        buf_len = smoothing_radius * 2 + 1
        self.traj_x = deque(maxlen=buf_len)
        self.traj_y = deque(maxlen=buf_len)
        self.traj_a = deque(maxlen=buf_len)
        # OpenCV feature-detection / optical-flow params
        self._feat_params = dict(
            maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3
        )
        self._lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.last_correction_px = 0.0
        self.last_correction_deg = 0.0
        self.last_raw_dx = 0.0
        self.last_raw_dy = 0.0

    def stabilize(self, frame):
        """Return a shake-compensated copy of *frame* (BGR, uint8)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.traj_x.append(0.0)
            self.traj_y.append(0.0)
            self.traj_a.append(0.0)
            self.last_correction_px = 0.0
            self.last_correction_deg = 0.0
            self.last_raw_dx = 0.0
            self.last_raw_dy = 0.0
            return frame

        # --- detect & track features ---
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self._feat_params)
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray
            self.last_correction_px = 0.0
            self.last_correction_deg = 0.0
            self.last_raw_dx = 0.0
            self.last_raw_dy = 0.0
            return frame

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self._lk_params
        )
        good = (status.ravel() == 1)
        if good.sum() < 6:
            self.prev_gray = gray
            self.last_correction_px = 0.0
            self.last_correction_deg = 0.0
            self.last_raw_dx = 0.0
            self.last_raw_dy = 0.0
            return frame

        # --- estimate rigid motion (translation + rotation) ---
        m, _ = cv2.estimateAffinePartial2D(prev_pts[good], curr_pts[good])
        if m is None:
            self.prev_gray = gray
            self.last_correction_px = 0.0
            self.last_correction_deg = 0.0
            self.last_raw_dx = 0.0
            self.last_raw_dy = 0.0
            return frame

        dx = m[0, 2]
        dy = m[1, 2]
        # Expose raw frame-to-frame translation for downstream motion compensation
        self.last_raw_dx = dx
        self.last_raw_dy = dy
        da = math.atan2(m[1, 0], m[0, 0])

        # --- accumulate trajectory ---
        self.cum_dx += dx
        self.cum_dy += dy
        self.cum_da += da
        self.traj_x.append(self.cum_dx)
        self.traj_y.append(self.cum_dy)
        self.traj_a.append(self.cum_da)

        # --- smooth trajectory (moving average) ---
        smooth_x = float(np.mean(self.traj_x))
        smooth_y = float(np.mean(self.traj_y))
        smooth_a = float(np.mean(self.traj_a))

        # correction = smoothed - actual  (clamped for safety)
        corr_dx = np.clip(smooth_x - self.cum_dx, -self.max_shift, self.max_shift)
        corr_dy = np.clip(smooth_y - self.cum_dy, -self.max_shift, self.max_shift)
        corr_da = np.clip(smooth_a - self.cum_da, -self.max_angle, self.max_angle)
        self.last_correction_px = float(math.hypot(corr_dx, corr_dy))
        self.last_correction_deg = float(math.degrees(corr_da))

        # --- build corrective affine warp (rotate around frame center) ---
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        cos_a = math.cos(corr_da)
        sin_a = math.sin(corr_da)
        M = np.array([
            [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy + corr_dx],
            [sin_a,  cos_a, -sin_a * cx + (1 - cos_a) * cy + corr_dy]
        ], dtype=np.float64)

        stabilized = cv2.warpAffine(frame, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)
        self.prev_gray = gray
        return stabilized


class TemporalMaskSmoother:
    """
    Exponential moving average (EMA) smoother for binary segmentation masks.

    When camera shake causes a sudden segmentation change (low IoU with the
    running average), the smoother trusts the historical average more,
    preventing transient mis-classifications from propagating to BEV / path.
    """

    def __init__(self, alpha=MASK_SMOOTH_ALPHA,
                 consistency_thresh=MASK_SMOOTH_CONSISTENCY_THRESH):
        """
        Args:
            alpha: EMA weight for current frame (higher = more responsive).
            consistency_thresh: IoU below this triggers conservative blending.
        """
        self.alpha = alpha
        self.consistency_thresh = consistency_thresh
        self.running_avg = None

    def _warp_mask(self, mask_f, dy):
        """Shift mask downward by dy pixels to compensate for forward ego-motion.

        When the scooter moves forward, ground-plane features move downward in
        the image.  Warping the previous overlay by the same translation aligns
        it with the current frame before blending, so the smoothed result stays
        spatially coherent rather than ghosting behind the scooter's motion.
        """
        if abs(dy) < 0.5:
            return mask_f
        shift = int(round(dy))
        rows, cols = mask_f.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, shift]])
        return cv2.warpAffine(
            mask_f, M, (cols, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def smooth(self, mask_255, flow_dy=0.0, return_iou=False):
        """
        Accept a uint8 binary mask (0 / 255) and return a temporally
        smoothed binary mask (0 / 255).

        Args:
            mask_255:  Current segmentation mask (uint8, 0/255).
            flow_dy:   Downward pixel translation between the previous and
                       current frame (from FrameStabilizer.last_raw_dy).
                       Positive = scooter moved forward.  0 = no compensation.
            return_iou: If True, also return the consistency IoU score.
        """
        mask_f = mask_255.astype(np.float32) / 255.0

        if self.running_avg is None:
            self.running_avg = mask_f.copy()
            return (mask_255, 1.0) if return_iou else mask_255

        # Warp the previous overlay to where it should appear in this frame
        warped_avg = self._warp_mask(self.running_avg, flow_dy)

        # Consistency check: IoU between current mask and motion-compensated history
        iou = self._iou(mask_f, warped_avg)

        if iou < self.consistency_thresh:
            # Very different from history -- likely a shake artifact
            effective_alpha = self.alpha * 0.25
        elif iou < 0.5:
            # Moderate change -- partially trust history
            effective_alpha = self.alpha * 0.5
        else:
            effective_alpha = self.alpha

        # Blend current mask with motion-compensated overlay; store result as
        # next frame's "previous" so the overlay propagates, not the raw mask.
        self.running_avg = (effective_alpha * mask_f
                            + (1.0 - effective_alpha) * warped_avg)

        # Threshold back to binary
        smoothed = (self.running_avg > 0.45).astype(np.uint8) * 255
        return (smoothed, iou) if return_iou else smoothed

    @staticmethod
    def _iou(a, b):
        """IoU between two float masks in [0, 1]."""
        ba = a > 0.5
        bb = b > 0.5
        inter = float(np.logical_and(ba, bb).sum())
        union = float(np.logical_or(ba, bb).sum())
        return inter / max(union, 1.0)
