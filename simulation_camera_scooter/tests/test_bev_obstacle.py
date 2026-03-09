"""
test_bev_obstacle.py
====================
Tests for Phase 03.1 — YOLO BEV Obstacle Projection.

Wave 1 (this plan 03.1-02): OBS-01, OBS-02, OBS-03, OBS-04, OBS-07 implemented.
Wave 2 (plan 03.1-03): OBS-05, OBS-06, OBS-08, OBS-09 to be implemented.

Requirements covered: OBS-01 through OBS-09.
"""

import numpy as np
import pytest

from bev_obstacle import project_foot_to_bev, detection_to_metric, ObstacleEMAGrid


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
    """OBS-02: metric coordinate of projected foot matches expected (forward_m, lateral_m)."""
    det = {"bbox": (270, 200, 330, 400), "class_name": "person", "confidence": 0.9, "distance_m": 2.5}
    # bev_x=90, bev_y=200 in a 500x600 BEV
    # forward_m = (499 - 200) / 499 * 10.0 = 299/499 * 10 ≈ 5.99
    # lateral_m = (90/599 - 0.5) * 10.0 = (0.1502 - 0.5) * 10 ≈ -3.50
    forward_m, lateral_m = detection_to_metric(det, bev_h_matrix, bev_obstacle_mask_500x600.shape)
    assert abs(forward_m - 5.99) < 0.1, f"Expected forward_m ~5.99, got {forward_m}"
    assert abs(lateral_m - (-3.50)) < 0.1, f"Expected lateral_m ~-3.50, got {lateral_m}"


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
    """OBS-05: candidate path through obstacle zone has higher cost than clear path."""
    assert False, "not implemented"


def test_hard_block_masks_bev(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-06: hard-block paints BEV mask black at stop-distance obstacle location."""
    assert False, "not implemented"


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
    assert False, "not implemented"


def test_integration_full_pipeline(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-09: full pipeline runs end-to-end with detections on a synthetic BEV mask."""
    assert False, "not implemented"
