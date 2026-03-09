"""
test_bev_obstacle.py
====================
Wave 0 stub tests for Phase 03.1 — YOLO BEV Obstacle Projection.

All 9 tests fail with AssertionError("not implemented") until Wave 1 and Wave 2
implement bev_obstacle.py and integrate it into realtime_nav_core.py.

Requirements covered: OBS-01 through OBS-09.
"""

import numpy as np
import pytest


def test_projection_centered(bev_h_matrix, mock_detections):
    """OBS-01: foot-point of a detection centered in image projects to correct BEV quadrant."""
    assert False, "not implemented"


def test_metric_conversion(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-02: metric coordinate of projected foot matches expected (forward_m, lateral_m)."""
    assert False, "not implemented"


def test_ema_decay(bev_obstacle_mask_500x600):
    """OBS-03: EMA grid decays to near-zero after N frames with no detection."""
    assert False, "not implemented"


def test_ema_update(bev_obstacle_mask_500x600, mock_detections):
    """OBS-04: EMA grid shows nonzero at foot-point location after one detection."""
    assert False, "not implemented"


def test_obstacle_penalty_prefers_clear_path():
    """OBS-05: candidate path through obstacle zone has higher cost than clear path."""
    assert False, "not implemented"


def test_hard_block_masks_bev(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-06: hard-block paints BEV mask black at stop-distance obstacle location."""
    assert False, "not implemented"


def test_projection_out_of_bounds_clamped(bev_h_matrix):
    """OBS-07: out-of-bounds projected point is clamped, not raised as exception."""
    assert False, "not implemented"


def test_no_penalty_when_no_obstacles():
    """OBS-08: no penalty when obstacle_zones is empty or None."""
    assert False, "not implemented"


def test_integration_full_pipeline(bev_h_matrix, bev_obstacle_mask_500x600, mock_detections):
    """OBS-09: full pipeline runs end-to-end with detections on a synthetic BEV mask."""
    assert False, "not implemented"
