"""
test_bev_predictor.py
=====================
Unit tests for BEVPredictiveTracker.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bev_predictor import BEVPredictiveTracker, _compute_iou
from realtime_nav_core import CubicPathModel, PathPlanResult


def _make_straight_path_model(length_m=4.5):
    """Helper: straight path model y=0."""
    x_grid = np.linspace(0.0, length_m, 120, dtype=np.float64)
    y_grid = np.zeros(120, dtype=np.float64)
    s_grid = x_grid.copy()
    kappa_grid = np.zeros(120, dtype=np.float64)
    coeff = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return CubicPathModel(coeff=coeff, x_grid=x_grid, y_grid=y_grid,
                          s_grid=s_grid, kappa_grid=kappa_grid)


def _make_curved_path_model(a2=0.05, length_m=4.5):
    """Helper: curved path y = a2*x^2."""
    x_grid = np.linspace(0.0, length_m, 120, dtype=np.float64)
    y_grid = a2 * x_grid ** 2
    ds = np.sqrt(np.diff(x_grid) ** 2 + np.diff(y_grid) ** 2)
    s_grid = np.concatenate(([0.0], np.cumsum(ds)))
    dy = 2.0 * a2 * x_grid
    ddy = 2.0 * a2
    kappa_grid = ddy / (1.0 + dy ** 2) ** 1.5
    coeff = np.array([0.0, 0.0, a2, 0.0], dtype=np.float64)
    return CubicPathModel(coeff=coeff, x_grid=x_grid, y_grid=y_grid,
                          s_grid=s_grid, kappa_grid=kappa_grid)


def _make_path_result(has_path=True, path_model=None, path_m=None):
    """Minimal PathPlanResult for testing."""
    if path_m is None:
        path_m = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    return PathPlanResult(
        has_path=has_path,
        path_model=path_model,
        best_path_m=path_m,
        best_path_px=np.zeros((0, 2), dtype=np.int32),
        candidate_paths_m=[path_m],
        candidate_paths_px=[np.zeros((0, 2), dtype=np.int32)],
        candidate_lengths_m=[3.0],
        best_idx=0,
        skeleton_px=np.zeros((10, 10), dtype=np.uint8),
        graph_nodes=2,
        graph_edges=1,
        t_skeleton_ms=0.0,
        t_path_ms=0.0,
    )


@pytest.fixture
def tracker():
    return BEVPredictiveTracker()


@pytest.fixture
def bev_mask_with_corridor():
    """500x600 float32 BEV mask with a straight corridor in center."""
    # BEV_SIZE = (600, 500) → width=600, height=500
    mask = np.zeros((500, 600), dtype=np.float32)
    mask[:, 250:350] = 1.0
    return mask


# ------------------------------------------------------------------
# Warp tests
# ------------------------------------------------------------------
class TestWarpBEVMask:
    def test_straight_forward(self, tracker, bev_mask_with_corridor):
        """Forward motion shifts mask downward (new blank rows at top)."""
        dx_m = 1.0  # 1 meter forward
        warped = tracker.warp_bev_mask(bev_mask_with_corridor, dx_m, 0.0, 0.0)

        # Top rows should be empty (new terrain)
        shift_px = int(dx_m * tracker._px_per_m_fwd)
        assert warped.shape == bev_mask_with_corridor.shape
        # Top rows should be zero (shifted in from border)
        assert np.all(warped[:shift_px - 1, :] < 0.01)
        # Bottom part should have the corridor
        assert np.sum(warped > 0.5) > 0

    def test_lateral_shift(self, tracker, bev_mask_with_corridor):
        """Lateral motion shifts mask horizontally.

        In warpAffine, the matrix maps dst→src, so positive R[0,2] shifts
        the image content LEFT (ego moved right → world shifts left).
        """
        dy_m = 1.0  # 1 meter right
        warped = tracker.warp_bev_mask(bev_mask_with_corridor, 0.0, dy_m, 0.0)

        # Original corridor center at col 300.
        # warpAffine shifts content left by dy * px_per_m_lat pixels.
        mid_row = warped.shape[0] // 2
        orig_row = bev_mask_with_corridor[mid_row, :]
        new_row = warped[mid_row, :]
        orig_center = np.average(np.arange(len(orig_row)), weights=orig_row + 1e-9)
        new_center = np.average(np.arange(len(new_row)), weights=new_row + 1e-9)
        # Corridor should have shifted significantly in the lateral direction
        assert abs(new_center - orig_center) > 20

    def test_rotation(self, tracker, bev_mask_with_corridor):
        """Small rotation rotates around ego (bottom center)."""
        dtheta = math.radians(5.0)
        warped = tracker.warp_bev_mask(bev_mask_with_corridor, 0.0, 0.0, dtheta)
        assert warped.shape == bev_mask_with_corridor.shape
        # Should still have significant mask content
        assert np.sum(warped > 0.5) > 1000

    def test_zero_displacement_identity(self, tracker, bev_mask_with_corridor):
        """Zero ego displacement returns approximately the same mask."""
        warped = tracker.warp_bev_mask(bev_mask_with_corridor, 0.0, 0.0, 0.0)
        diff = np.abs(warped - bev_mask_with_corridor)
        assert np.max(diff) < 0.01


# ------------------------------------------------------------------
# Path warp tests
# ------------------------------------------------------------------
class TestWarpPath:
    def test_straight_forward(self, tracker):
        """Forward motion shifts path points backward (closer to ego)."""
        path = np.array([[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
        warped = tracker.warp_path(path, dx_m=1.0, dy_m=0.0, dtheta_rad=0.0)
        # After 1m forward, point at x=1 should be at x≈0
        assert len(warped) >= 3
        assert warped[0, 0] < 0.1  # was x=1, now ~0

    def test_trim_behind_ego(self, tracker):
        """Points that end up behind ego (x<0) are trimmed."""
        path = np.array([[0.5, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        warped = tracker.warp_path(path, dx_m=1.0, dy_m=0.0, dtheta_rad=0.0)
        # x=0.5 - 1.0 = -0.5 → should be trimmed
        assert all(warped[:, 0] >= 0.0)

    def test_with_turn(self, tracker):
        """Path points rotate when ego turns."""
        path = np.array([[1, 0], [2, 0], [3, 0]], dtype=np.float32)
        dtheta = math.radians(10.0)
        warped = tracker.warp_path(path, dx_m=0.0, dy_m=0.0, dtheta_rad=dtheta)
        # Points should have non-zero lateral component after rotation
        assert len(warped) == 3
        assert abs(warped[1, 1]) > 0.01  # lateral displacement from rotation

    def test_empty_path(self, tracker):
        """Empty or None path returns empty array."""
        result = tracker.warp_path(None, 1.0, 0.0, 0.0)
        assert len(result) == 0
        result = tracker.warp_path(np.zeros((0, 2), dtype=np.float32), 1.0, 0.0, 0.0)
        assert len(result) == 0


# ------------------------------------------------------------------
# Should-skip tests
# ------------------------------------------------------------------
class TestShouldSkip:
    def test_no_stored_state(self, tracker):
        """Cannot skip without stored state."""
        assert tracker.should_skip(speed=1.0, dt=0.1) is False

    def test_straight_allows_skip(self, tracker):
        """Low curvature allows skipping."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 240:360] = 1.0
        tracker.last_occ_ratio = 0.20
        tracker.last_curvature = 0.0  # perfectly straight
        tracker.prediction_confidence = 0.9
        tracker.cumulative_skip_count = 0
        assert tracker.should_skip(speed=1.0, dt=0.1) is True

    def test_sharp_turn_no_skip(self, tracker):
        """High curvature blocks skipping."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 240:360] = 1.0
        tracker.last_occ_ratio = 0.20
        tracker.last_curvature = 0.5  # sharp
        tracker.prediction_confidence = 0.9
        tracker.cumulative_skip_count = 0
        assert tracker.should_skip(speed=1.0, dt=0.1) is False

    def test_low_confidence_no_skip(self, tracker):
        """Low prediction confidence blocks skipping."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 240:360] = 1.0
        tracker.last_occ_ratio = 0.20
        tracker.last_curvature = 0.0
        tracker.prediction_confidence = 0.3  # below floor
        tracker.cumulative_skip_count = 0
        assert tracker.should_skip(speed=1.0, dt=0.1) is False

    def test_max_consecutive_skips(self, tracker):
        """Exceeding max consecutive skips blocks further skipping."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 240:360] = 1.0
        tracker.last_occ_ratio = 0.20
        tracker.last_curvature = 0.0
        tracker.prediction_confidence = 0.9
        tracker.cumulative_skip_count = 3  # already at max for straight
        assert tracker.should_skip(speed=1.0, dt=0.1) is False

    def test_gentle_turn_limited_skip(self, tracker):
        """Gentle turn allows 1 skip only."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 240:360] = 1.0
        tracker.last_occ_ratio = 0.20
        tracker.last_curvature = 0.2  # gentle turn (between 0.10 and 0.30)
        tracker.prediction_confidence = 0.9
        tracker.cumulative_skip_count = 0
        assert tracker.should_skip(speed=1.0, dt=0.1) is True
        tracker.cumulative_skip_count = 1
        assert tracker.should_skip(speed=1.0, dt=0.1) is False

    def test_low_occupancy_no_skip(self, tracker):
        """Very sparse BEV occupancy blocks skip to force compute-frame refresh."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_occ_ratio = 0.001
        tracker.last_curvature = 0.0
        tracker.prediction_confidence = 0.95
        tracker.cumulative_skip_count = 0
        assert tracker.should_skip(speed=1.0, dt=0.1) is False


# ------------------------------------------------------------------
# Compute / skip frame handlers
# ------------------------------------------------------------------
class TestOnComputeFrame:
    def test_first_frame(self, tracker):
        """First compute frame stores state without blending."""
        mask = np.zeros((500, 600), dtype=np.uint8)
        mask[:, 250:350] = 255
        path_model = _make_straight_path_model()
        pr = _make_path_result(has_path=True, path_model=path_model)

        result = tracker.on_compute_frame(mask, pr, speed=1.0, dt=0.1, steer_deg=0.0)
        assert result.dtype == np.uint8
        assert tracker.last_bev_mask is not None
        assert tracker.cumulative_skip_count == 0

    def test_blend_after_skips(self, tracker):
        """After skip frames, compute frame blends predicted + real."""
        # Set up stored state
        stored = np.zeros((500, 600), dtype=np.float32)
        stored[:, 250:350] = 1.0
        tracker.last_bev_mask = stored
        tracker.last_path_model = _make_straight_path_model()
        tracker.last_path_points_m = np.array([[0, 0], [2, 0]], dtype=np.float32)
        tracker.cumulative_skip_count = 2
        tracker.prediction_confidence = 0.8

        real_mask = np.zeros((500, 600), dtype=np.uint8)
        real_mask[:, 260:360] = 255  # slightly shifted
        pr = _make_path_result(has_path=True, path_model=_make_straight_path_model())

        result = tracker.on_compute_frame(real_mask, pr, speed=1.0, dt=0.1, steer_deg=0.0)
        assert result.dtype == np.uint8
        assert tracker.cumulative_skip_count == 0
        assert 0.0 < tracker.prediction_confidence <= 1.0


class TestOnSkipFrame:
    def test_skip_returns_mask_and_path(self, tracker):
        """Skip frame returns predicted mask and path."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 250:350] = 1.0
        tracker.last_path_points_m = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32
        )
        tracker.last_path_model = _make_straight_path_model()

        mask, path = tracker.on_skip_frame(speed=1.0, dt=0.1, steer_deg=0.0)
        assert mask.dtype == np.uint8
        assert mask.shape == (500, 600)
        assert path is not None
        assert len(path) >= 2

    def test_skip_increments_count(self, tracker):
        """Skip frame increments cumulative_skip_count."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_bev_mask[:, 250:350] = 1.0
        tracker.last_path_points_m = np.array([[1, 0], [2, 0]], dtype=np.float32)
        tracker.last_path_model = _make_straight_path_model()
        tracker.cumulative_skip_count = 0

        tracker.on_skip_frame(speed=1.0, dt=0.1, steer_deg=0.0)
        assert tracker.cumulative_skip_count == 1

        tracker.on_skip_frame(speed=1.0, dt=0.1, steer_deg=0.0)
        assert tracker.cumulative_skip_count == 2

    def test_skip_drops_path_on_empty_predicted_mask(self, tracker):
        """Sparse/empty predicted mask should return no path and lower confidence."""
        tracker.last_bev_mask = np.zeros((500, 600), dtype=np.float32)
        tracker.last_path_points_m = np.array([[1, 0], [2, 0]], dtype=np.float32)
        tracker.last_path_model = _make_straight_path_model()
        tracker.prediction_confidence = 0.9
        mask, path = tracker.on_skip_frame(speed=1.0, dt=0.1, steer_deg=0.0)
        assert mask.dtype == np.uint8
        assert path is None
        assert tracker.prediction_confidence <= 0.25


# ------------------------------------------------------------------
# Ego displacement
# ------------------------------------------------------------------
class TestEgoDisplacement:
    def test_with_path_model(self, tracker):
        """Path-based displacement follows the path curve."""
        model = _make_straight_path_model()
        dx, dy, dtheta = tracker.compute_ego_displacement(1.0, 0.5, model, 0.0)
        assert abs(dx - 0.5) < 0.01  # 1.0 m/s * 0.5s = 0.5m forward
        assert abs(dy) < 0.01
        assert abs(dtheta) < 0.01

    def test_dead_reckoning_straight(self, tracker):
        """Dead reckoning with zero steer goes straight."""
        dx, dy, dtheta = tracker.compute_ego_displacement(1.0, 0.5, None, 0.0)
        assert abs(dx - 0.5) < 0.01
        assert abs(dy) < 0.01
        assert abs(dtheta) < 0.01

    def test_zero_speed(self, tracker):
        """Zero speed gives zero displacement."""
        dx, dy, dtheta = tracker.compute_ego_displacement(0.0, 0.1, None, 10.0)
        assert dx == 0.0 and dy == 0.0 and dtheta == 0.0


# ------------------------------------------------------------------
# IoU helper
# ------------------------------------------------------------------
class TestComputeIoU:
    def test_identical_masks(self):
        a = np.ones((10, 10), dtype=np.float32)
        assert _compute_iou(a, a) == 1.0

    def test_disjoint_masks(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32)
        assert _compute_iou(a, b) == 0.0
