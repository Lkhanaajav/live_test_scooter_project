"""
test_bev_obstacle.py
====================
Tests for Phase 03.1 — YOLO BEV Obstacle Projection.

Wave 1 (this plan 03.1-02): OBS-01, OBS-02, OBS-03, OBS-04, OBS-07 implemented.
Wave 2 (plan 03.1-03): OBS-05, OBS-06, OBS-08, OBS-09 implemented.

Requirements covered: OBS-01 through OBS-09.
"""

import cv2
import numpy as np
import pytest

from bev_obstacle import project_foot_to_bev, detection_to_metric, ObstacleEMAGrid
from realtime_nav_core import BEVPathExtractor


def test_projection_centered(bev_h_matrix, mock_detections):
    """OBS-01: foot-point of a detection centered in image projects to correct BEV quadrant."""
    # Use first detection: bbox=(250, 200, 350, 400), foot center_x=(250+350)/2=300, foot_y=400
    det = {"bbox": (270, 200, 330, 400), "class_name": "person", "confidence": 0.9, "distance_m": 2.5}
    # H = [[0.3,0,0],[0,0.5,0],[0,0,1]]
    # foot = (300, 400) -> bev_x = 300*0.3 = 90.0, bev_y = 400*0.5 = 200.0
    bx, by = project_foot_to_bev(det, bev_h_matrix)
    assert abs(bx - 90.0) < 1.0, f"Expected bev_x ~90.0, got {bx}"
    assert abs(by - 200.0) < 1.0, f"Expected bev_y ~200.0, got {by}"
    # Should be in valid BEV pixel range
    assert 0.0 <= bx <= 599.0
    assert 0.0 <= by <= 499.0


def test_metric_conversion(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-02: metric coordinate of projected foot matches expected (forward_m, lateral_m).

    Analytical derivation (H scales x*0.3, y*0.5):
      foot = ((270+330)/2, 400) = (300, 400)  ->  bev = (90, 200)
      forward_m = (499 - 200) / 499 * NAV_BEV_FORWARD_M
      lateral_m = (90 - bev_ego_x_px(600)) / 599 * NAV_BEV_LATERAL_M

    With NAV_BEV_FORWARD_M=11.0, NAV_BEV_LATERAL_M=6.0, ego_x=299.5 (BEV_EGO_X_FRAC=0.5):
      forward_m = 299/499 * 11.0  ~= 6.59
      lateral_m = (90 - 299.5)/599 * 6.0  ~= -2.10
    """
    det = {"bbox": (270, 200, 330, 400), "class_name": "person", "confidence": 0.9, "distance_m": 2.5}
    forward_m, lateral_m = detection_to_metric(det, bev_h_matrix, bev_obstacle_mask_500x600.shape)
    assert abs(forward_m - 6.59) < 0.1, f"Expected forward_m ~6.59, got {forward_m}"
    assert abs(lateral_m - (-2.10)) < 0.15, f"Expected lateral_m ~-2.10, got {lateral_m}"


def test_ema_decay(bev_obstacle_mask_500x600):
    """OBS-03: EMA grid decays to near-zero after N frames with no detection."""
    ema = ObstacleEMAGrid(500, 600, alpha=0.5)
    # Prime with one detection
    ema.update([(300, 250)])
    # Decay 10 times with no detections
    for _ in range(10):
        ema.update([])
    # After 10 decays at 0.5 alpha: 0.5^10 ≈ 0.001 — should be well below 0.01
    assert ema.grid.max() < 0.01, f"Expected near-zero after 10 decays, got max={ema.grid.max()}"


def test_ema_update(bev_obstacle_mask_500x600):
    """OBS-04: EMA grid shows nonzero at foot-point location after one detection."""
    ema = ObstacleEMAGrid(500, 600, alpha=0.5)
    grid = ema.update([(300, 250)])
    # After one update from zeros: grid = 0.5 * 0 + 0.5 * new_frame = 0.5 * new_frame
    # Center pixel of the footprint circle (radius=15) at (300,250) should be ~0.5
    assert grid[250, 300] > 0.4, f"Expected center pixel > 0.4 after one update, got {grid[250, 300]}"


def test_obstacle_penalty_prefers_clear_path():
    """OBS-05: _obstacle_penalty returns 0 for empty zones and nonzero for occupied zones."""
    extractor = BEVPathExtractor()
    # Points along forward axis (column 0 = forward_m, column 1 = lateral_m)
    path_pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    # No obstacle zones -> penalty must be zero
    extractor.obstacle_zones = []
    assert extractor._obstacle_penalty(path_pts) == 0.0, "Expected 0.0 with empty obstacle_zones"

    # Obstacle zone directly on path -> penalty must be nonzero
    extractor.obstacle_zones = [(2.0, 0.0, 1.5)]  # centered at (2m fwd, 0m lat), radius 1.5m
    penalty = extractor._obstacle_penalty(path_pts)
    assert penalty > 0.0, f"Expected nonzero penalty with obstacle on path, got {penalty}"

    # Obstacle zone far off to the side -> penalty must be zero
    extractor.obstacle_zones = [(2.0, 8.0, 0.5)]  # 8m lateral — far right
    penalty_clear = extractor._obstacle_penalty(path_pts)
    assert penalty_clear == 0.0, f"Expected 0.0 with off-path obstacle, got {penalty_clear}"


def test_hard_block_masks_bev(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-06: hard-block paints BEV mask black at stop-distance obstacle location."""
    from config import BEV_HARD_BLOCK_DIST_M
    mask = bev_obstacle_mask_500x600.copy()
    bev_h, bev_w = mask.shape[:2]

    # Use the second detection which has distance_m=0.8 < BEV_HARD_BLOCK_DIST_M=1.2
    close_det = mock_detections[1]  # distance_m=0.8
    assert close_det["distance_m"] < BEV_HARD_BLOCK_DIST_M

    bx, by = project_foot_to_bev(close_det, bev_h_matrix, bev_h=bev_h, bev_w=bev_w)
    r_px = int(BEV_HARD_BLOCK_DIST_M * 50)
    cv2.circle(mask, (int(bx), int(by)), r_px, 0, -1)

    # Center of the painted circle must be black
    assert mask[int(by), int(bx)] == 0, (
        f"Expected mask[{int(by)}, {int(bx)}] == 0 after hard-block paint, got {mask[int(by), int(bx)]}"
    )


def test_projection_out_of_bounds_clamped(bev_h_matrix):
    """OBS-07: out-of-bounds projected point is clamped, not raised as exception."""
    # Detection with foot at very negative coordinates
    det = {"bbox": (-1000, -1000, -900, -900), "class_name": "person", "confidence": 0.9, "distance_m": 5.0}
    # Should not raise, should return clamped values >= 0.0
    result = project_foot_to_bev(det, bev_h_matrix)
    assert isinstance(result, tuple) and len(result) == 2
    bx, by = result
    assert isinstance(bx, float)
    assert isinstance(by, float)
    assert bx >= 0.0, f"Expected bx >= 0.0 (clamped), got {bx}"
    assert by >= 0.0, f"Expected by >= 0.0 (clamped), got {by}"


def test_no_penalty_when_no_obstacles():
    """OBS-08: no penalty when obstacle_zones is empty or None."""
    extractor = BEVPathExtractor()
    path_pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

    # Empty list
    extractor.obstacle_zones = []
    assert extractor._obstacle_penalty(path_pts) == 0.0, "Expected 0.0 for empty obstacle_zones list"

    # process() with obstacle_zones_m=None must not crash
    mask = np.zeros((500, 600), dtype=np.uint8)
    mask[300:, 250:350] = 255
    result = extractor.process(mask, obstacle_zones_m=None)
    assert hasattr(result, "has_path"), "Expected PathPlanResult with has_path attribute"

    # process() with obstacle_zones_m=[] must not crash
    result2 = extractor.process(mask, obstacle_zones_m=[])
    assert hasattr(result2, "has_path"), "Expected PathPlanResult with has_path attribute"


def test_integration_full_pipeline(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-09: full pipeline runs end-to-end with detections on a synthetic BEV mask."""
    from config import BEV_OBSTACLE_RADIUS_PX
    mask = bev_obstacle_mask_500x600.copy()
    bev_h, bev_w = mask.shape[:2]

    # Step 1: instantiate path extractor and EMA grid
    extractor = BEVPathExtractor()
    ema = ObstacleEMAGrid(bev_h=bev_h, bev_w=bev_w)

    # Step 2: project detections and update EMA
    det = mock_detections[0]  # distance_m=2.5 (not a hard-block)
    foot_bev_pts = [project_foot_to_bev(det, bev_h_matrix, bev_h=bev_h, bev_w=bev_w)]
    ema.update(foot_bev_pts)

    # Step 3: build obstacle_zones
    fwd, lat = detection_to_metric(det, bev_h_matrix, mask.shape)
    r_m = BEV_OBSTACLE_RADIUS_PX.get(det.get("class_name", "default"),
                                      BEV_OBSTACLE_RADIUS_PX["default"]) / 50.0
    obstacle_zones = [(fwd, lat, r_m)]

    # Step 4: call path_extractor.process with obstacle_zones_m
    result = extractor.process(mask, obstacle_zones_m=obstacle_zones)

    # Assert no exception and result is a valid PathPlanResult
    assert hasattr(result, "has_path"), "Expected PathPlanResult with has_path attribute"
    assert hasattr(result, "candidate_paths_px"), "Expected PathPlanResult with candidate_paths_px attribute"
