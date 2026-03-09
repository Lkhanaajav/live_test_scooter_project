"""
bev_obstacle.py
===============
Standalone BEV (Bird's Eye View) obstacle projection and temporal smoothing module.

Provides:
  - project_foot_to_bev: project YOLO detection foot-point from camera image space
    to BEV pixel coordinates using a homography matrix H.
  - detection_to_metric: convert a detection to (forward_m, lateral_m) metric
    coordinates in the BEV reference frame.
  - ObstacleEMAGrid: temporal EMA smoothing grid that accumulates obstacle
    footprints over time, decaying toward zero when no detections arrive.

No dependency on realtime_nav_core.py or live_heading_demo.py — this module
is independently importable and fully unit-tested in tests/test_bev_obstacle.py.

BEV coordinate system (from RESEARCH.md):
  - BEV image shape: numpy (500, 600) — HxW, row=y, col=x
  - BEV origin: bottom-center = ego position
  - forward_m: 0 at bottom (row=499), +NAV_BEV_FORWARD_M at top (row=0)
  - lateral_m: -5 at left (col=0), 0 at center (col=300), +5 at right (col=599)
"""

import numpy as np
import cv2

from config import (
    NAV_BEV_FORWARD_M,
    NAV_BEV_LATERAL_M,
    BEV_OBSTACLE_ALPHA,
    BEV_OBSTACLE_RADIUS_PX,
)


def project_foot_to_bev(
    det: dict,
    H: np.ndarray,
    bev_h: int = 500,
    bev_w: int = 600,
) -> tuple:
    """Project a detection's foot-point (bbox bottom-center) into BEV pixel space.

    Parameters
    ----------
    det : dict
        Detection dict with key "bbox": (x1, y1, x2, y2) in camera image pixels.
    H : np.ndarray
        3x3 homography matrix (image -> BEV pixel space), dtype float64.
    bev_h : int
        BEV image height in pixels (rows). Default 500.
    bev_w : int
        BEV image width in pixels (cols). Default 600.

    Returns
    -------
    (bev_x_px, bev_y_px) : tuple[float, float]
        BEV pixel coordinates of the foot-point, clamped to [0, bev_w-1] x [0, bev_h-1].
        Never raises on out-of-bounds input.
    """
    x1, y1, x2, y2 = det["bbox"]
    foot = np.array([[[(x1 + x2) / 2.0, float(y2)]]], dtype=np.float64)
    bev_pt = cv2.perspectiveTransform(foot, H)[0, 0]
    bx = float(np.clip(bev_pt[0], 0.0, bev_w - 1))
    by = float(np.clip(bev_pt[1], 0.0, bev_h - 1))
    return bx, by


def detection_to_metric(
    det: dict,
    H: np.ndarray,
    bev_mask_shape: tuple,
) -> tuple:
    """Convert a YOLO detection to metric BEV coordinates (forward_m, lateral_m).

    Uses the same pixel-to-metric formula as realtime_nav_core._metric_from_pixel:
      forward_m = (bev_h - 1 - by) / (bev_h - 1) * NAV_BEV_FORWARD_M
      lateral_m = (bx / (bev_w - 1) - 0.5) * NAV_BEV_LATERAL_M

    Forward is clamped to >= 0.3 m to filter out obstacles at/behind ego.

    Parameters
    ----------
    det : dict
        Detection dict with key "bbox": (x1, y1, x2, y2).
    H : np.ndarray
        3x3 homography matrix (image -> BEV pixel space), dtype float64.
    bev_mask_shape : tuple
        Shape of the BEV mask as (height, width) or (height, width, channels).

    Returns
    -------
    (forward_m, lateral_m) : tuple[float, float]
        Metric position in BEV frame. forward_m >= 0.3, lateral_m is signed.
    """
    bev_h, bev_w = bev_mask_shape[:2]
    bx, by = project_foot_to_bev(det, H, bev_h=bev_h, bev_w=bev_w)
    forward_m = max(0.3, (bev_h - 1 - by) / max(1.0, float(bev_h - 1)) * NAV_BEV_FORWARD_M)
    lateral_m = (bx / max(1.0, float(bev_w - 1)) - 0.5) * NAV_BEV_LATERAL_M
    return float(forward_m), float(lateral_m)


class ObstacleEMAGrid:
    """Temporal EMA (Exponential Moving Average) obstacle density grid in BEV space.

    Accumulates obstacle footprints across frames: present detections raise
    density, absent frames decay it toward zero at rate (1 - alpha) per frame.

    Usage
    -----
    grid = ObstacleEMAGrid(bev_h=500, bev_w=600)
    density = grid.update([(bx1, by1), (bx2, by2)])  # nonzero near obstacles
    density = grid.update([])                          # decays toward zero
    grid.reset()                                       # zero all cells
    """

    def __init__(
        self,
        bev_h: int = 500,
        bev_w: int = 600,
        alpha: float = BEV_OBSTACLE_ALPHA,
        footprint_radius_px: int = BEV_OBSTACLE_RADIUS_PX["default"],
    ) -> None:
        """
        Parameters
        ----------
        bev_h : int
            BEV grid height (rows). Default 500.
        bev_w : int
            BEV grid width (cols). Default 600.
        alpha : float
            EMA weight for the incoming frame (0 = no update, 1 = replace).
            Default BEV_OBSTACLE_ALPHA (0.50).
        footprint_radius_px : int
            Radius in BEV pixels for the circular footprint drawn at each foot-point.
            Default BEV_OBSTACLE_RADIUS_PX["default"] (15).
        """
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.alpha = float(alpha)
        self.radius = int(footprint_radius_px)
        self.grid = np.zeros((bev_h, bev_w), dtype=np.float32)

    def update(self, foot_bev_pts: list) -> np.ndarray:
        """Update the EMA grid with a new set of detection foot-points.

        Parameters
        ----------
        foot_bev_pts : list[tuple[float, float]]
            List of (bev_x_px, bev_y_px) foot-point coordinates for this frame.
            Pass an empty list when no detections are present (grid decays).

        Returns
        -------
        np.ndarray
            Updated self.grid with shape (bev_h, bev_w), dtype float32.
            Values in [0.0, 1.0] representing obstacle density.
        """
        new_frame = np.zeros_like(self.grid)
        for bx, by in foot_bev_pts:
            ix, iy = int(round(bx)), int(round(by))
            if 0 <= iy < self.bev_h and 0 <= ix < self.bev_w:
                cv2.circle(new_frame, (ix, iy), self.radius, 1.0, -1)
        self.grid = (1.0 - self.alpha) * self.grid + self.alpha * new_frame
        return self.grid

    def reset(self) -> None:
        """Zero out the entire EMA grid."""
        self.grid[:] = 0.0
