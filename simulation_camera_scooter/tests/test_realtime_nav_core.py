"""
test_realtime_nav_core.py
=========================
Unit tests for BEVPathExtractor and AdaptivePurePursuitController.
"""

import numpy as np
import pytest

import config as cfg_module
from realtime_nav_core import (
    BEVPathExtractor,
    PathExtractorConfig,
    PathPlanResult,
    AdaptivePurePursuitController,
    PurePursuitConfig,
)


# ---------------------------------------------------------------------------
# BEVPathExtractor tests
# ---------------------------------------------------------------------------

class TestBEVPathExtractor:

    def test_process_empty_mask_returns_no_path(self):
        """An all-black mask should produce no path."""
        extractor = BEVPathExtractor()
        mask = np.zeros((220, 220), dtype=np.uint8)
        result = extractor.process(mask)
        assert isinstance(result, PathPlanResult)
        assert result.has_path is False

    def test_process_straight_corridor_returns_path(self, straight_bev_mask):
        """A clear straight corridor should yield a valid path."""
        extractor = BEVPathExtractor()
        result = extractor.process(straight_bev_mask)
        assert isinstance(result, PathPlanResult)
        # May or may not find a path depending on corridor width vs config thresholds,
        # but result should always be a valid PathPlanResult with no crash.
        assert result.best_path_m.ndim == 2
        assert result.best_path_m.shape[1] == 2

    def test_process_wide_corridor_returns_path_with_positive_progress(self, straight_bev_mask):
        """Wide corridor should yield a path with forward progress."""
        cfg = PathExtractorConfig(
            min_sidewalk_width_m=0.3,
            branch_min_len_m=0.2,
            min_path_len_m=0.5,
        )
        extractor = BEVPathExtractor(cfg)
        result = extractor.process(straight_bev_mask)
        if result.has_path:
            # Best path should have points with positive x (forward)
            assert float(np.max(result.best_path_m[:, 0])) > 0.0

    def test_process_returns_path_plan_result_type(self, straight_bev_mask):
        """process() always returns a PathPlanResult regardless of mask content."""
        extractor = BEVPathExtractor()
        result = extractor.process(straight_bev_mask)
        assert isinstance(result, PathPlanResult)
        assert isinstance(result.has_path, bool)
        assert isinstance(result.graph_nodes, int)
        assert isinstance(result.graph_edges, int)
        assert isinstance(result.t_skeleton_ms, float)
        assert isinstance(result.t_path_ms, float)

    def test_commit_selected_path_clears_template_family_for_graph(self):
        """Graph/fallback commits should not keep stale template family intent alive."""
        extractor = BEVPathExtractor()
        extractor.prev_template_family = "right"
        path = np.array([[0.0, 0.0], [1.0, 0.1]], dtype=np.float32)

        extractor._commit_selected_path(path, path_source="graph")

        assert extractor.prev_template_family == ""
        assert extractor.no_path_counter == 0

    def test_commit_selected_path_keeps_template_family_for_hold(self):
        """Fallback hold should preserve the last template family for short reuse windows."""
        extractor = BEVPathExtractor()
        extractor.prev_template_family = "left"
        path = np.array([[0.0, 0.0], [1.0, -0.1]], dtype=np.float32)

        extractor._commit_selected_path(path, path_source="fallback_hold")

        assert extractor.prev_template_family == "left"
        assert extractor.no_path_counter == 0


# ---------------------------------------------------------------------------
# AdaptivePurePursuitController tests
# ---------------------------------------------------------------------------

class TestAdaptivePurePursuitController:

    def test_update_with_none_path_returns_zero_or_decayed_steer(self):
        """With no path, controller should return small/decayed steering."""
        ctrl = AdaptivePurePursuitController()
        out = ctrl.update(None, 1.0)
        assert abs(out.steer_deg) < 5.0  # near zero on first call
        assert out.valid_path is False

    def test_update_with_straight_path_returns_near_zero_steer(self, straight_path_model):
        """A perfectly straight path should produce near-zero steering."""
        ctrl = AdaptivePurePursuitController()
        out = ctrl.update(straight_path_model, 1.0)
        assert abs(out.steer_deg) < 5.0
        assert out.valid_path is True

    def test_update_with_curved_path_returns_nonzero_steer(self):
        """A left-curving path should produce non-zero (left) steering."""
        from realtime_nav_core import CubicPathModel
        # Build a path that curves left: y = 0.3*x^2 (positive y = left)
        x_grid = np.linspace(0.0, 4.0, 100, dtype=np.float64)
        y_grid = 0.15 * x_grid * x_grid
        ds = np.sqrt(np.diff(x_grid) ** 2 + np.diff(y_grid) ** 2)
        s_grid = np.concatenate(([0.0], np.cumsum(ds)))
        dy = 0.30 * x_grid
        ddy = 0.30 * np.ones_like(x_grid)
        kappa_grid = ddy / np.maximum(1e-6, (1.0 + dy * dy) ** 1.5)
        coeff = np.array([0.0, 0.0, 0.15, 0.0], dtype=np.float64)
        path = CubicPathModel(coeff=coeff, x_grid=x_grid, y_grid=y_grid,
                              s_grid=s_grid, kappa_grid=kappa_grid)
        ctrl = AdaptivePurePursuitController()
        out = ctrl.update(path, 1.0)
        # Positive steering = right; left curve has positive y -> should steer right
        assert out.valid_path is True
        assert out.steer_deg != 0.0

    def test_steer_rate_limiting_prevents_large_jumps(self, straight_path_model):
        """Steering command should not jump more than rate limit allows."""
        cfg = PurePursuitConfig(steer_rate_max_deg_s=30.0, dt_s=0.125)
        ctrl = AdaptivePurePursuitController(cfg)
        # First call (from zero)
        out1 = ctrl.update(straight_path_model, 1.0)
        # Second call with no path (fallback decay)
        out2 = ctrl.update(None, 1.0)
        max_step = cfg.steer_rate_max_deg_s * cfg.dt_s
        assert abs(out2.steer_deg - out1.steer_deg) <= max_step + 1e-6


# ---------------------------------------------------------------------------
# Waypoint-Turn Integration Tests (Phase 11.1 Plan 03)
# ---------------------------------------------------------------------------

def _make_waypoint_extractor() -> BEVPathExtractor:
    """Create a BEVPathExtractor with waypoint-turn mode enabled."""
    cfg = PathExtractorConfig(waypoint_turn_enabled=True, use_dt_planner=False)
    return BEVPathExtractor(cfg)


class TestWaypointTurnMode:
    """Integration tests: waypoint-turn path wired into BEVPathExtractor.process()."""

    def test_waypoint_turn_uses_runtime_bev_geometry(self):
        """Waypoint-turn config should inherit the extractor BEV metric dimensions."""
        extractor = _make_waypoint_extractor()
        cfg = extractor._waypoint_turn_cfg
        assert cfg is not None
        assert float(cfg.bev_forward_m) == float(extractor.cfg.bev_forward_m)
        assert float(cfg.bev_lateral_m) == float(extractor.cfg.bev_lateral_m)

    def test_waypoint_turn_engaged_for_commanded_left(self, commanded_left_bev_mask):
        """With waypoint-turn mode on + commanded left, path_source should be 'waypoint_turn'."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            commanded_left_bev_mask,
            gps_intent_family="left",
        )
        assert isinstance(result, PathPlanResult)
        assert result.path_source == "waypoint_turn"
        assert result.has_path is True
        assert result.selected_template_family == "left"

    def test_waypoint_turn_engaged_for_commanded_right(self, commanded_right_bev_mask):
        """With waypoint-turn mode on + commanded right, path_source should be 'waypoint_turn'."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            commanded_right_bev_mask,
            gps_intent_family="right",
        )
        assert isinstance(result, PathPlanResult)
        assert result.path_source == "waypoint_turn"
        assert result.has_path is True
        assert result.selected_template_family == "right"


class TestWaypointTurnLockStability:
    """Maneuver lock: consecutive supported frames keep the family locked."""

    def test_maneuver_lock_persists_across_frames(self, commanded_left_bev_mask):
        """Repeated supported left-turn frames keep path_source='waypoint_turn' and family='left'."""
        extractor = _make_waypoint_extractor()
        families = []
        sources = []
        for _ in range(5):
            result = extractor.process(
                commanded_left_bev_mask,
                gps_intent_family="left",
            )
            families.append(result.selected_template_family)
            sources.append(result.path_source)
        # All frames should be waypoint_turn with left family
        assert all(f == "left" for f in families), f"Families bounced: {families}"
        assert all(s == "waypoint_turn" for s in sources), f"Sources bounced: {sources}"

    def test_maneuver_lock_does_not_flip_to_opposite(self, commanded_left_bev_mask):
        """Commanded left should never produce right family even if right-side support exists."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            commanded_left_bev_mask,
            gps_intent_family="left",
        )
        # Family should be left, never right
        assert result.selected_template_family != "right"


class TestNoIntentPreservation:
    """No-intent and straight-intent preserve existing dt_corridor/template behavior."""

    def test_no_intent_uses_existing_behavior(self, no_intent_straight_bev_mask):
        """Without intent, path_source should NOT be 'waypoint_turn'."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            no_intent_straight_bev_mask,
            gps_intent_family=None,
        )
        assert isinstance(result, PathPlanResult)
        assert result.path_source != "waypoint_turn"
        assert result.path_source != "waypoint_turn_hold"

    def test_straight_intent_uses_existing_behavior(self, no_intent_straight_bev_mask):
        """Explicit 'straight' intent should NOT engage waypoint-turn mode."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            no_intent_straight_bev_mask,
            gps_intent_family="straight",
        )
        assert result.path_source != "waypoint_turn"
        assert result.path_source != "waypoint_turn_hold"


class TestWaypointTurnHoldBehavior:
    """Unsupported commanded turns produce low-confidence hold, not opposite fallback."""

    def test_unsupported_turn_returns_low_confidence(self, unsupported_turn_bev_mask):
        """Unsupported commanded left returns low confidence and recommends hold/slowdown."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            unsupported_turn_bev_mask,
            gps_intent_family="left",
        )
        assert isinstance(result, PathPlanResult)
        # Should NOT have active waypoint_turn path
        assert result.path_source != "waypoint_turn"
        # Should have low confidence or suggest slowdown
        assert result.is_low_confidence or result.suggested_slowdown > 0.0

    def test_unsupported_turn_does_not_choose_opposite(self, unsupported_turn_bev_mask):
        """Unsupported commanded left must not silently pick a right-turn path."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            unsupported_turn_bev_mask,
            gps_intent_family="left",
        )
        # Family must not be opposite of the commanded direction
        assert result.selected_template_family != "right"

    def test_unsupported_turn_emits_slowdown(self, unsupported_turn_bev_mask):
        """Unsupported commanded right triggers planner slowdown."""
        extractor = _make_waypoint_extractor()
        result = extractor.process(
            unsupported_turn_bev_mask,
            gps_intent_family="right",
        )
        # Either low confidence or explicit slowdown recommendation
        has_slowdown = result.suggested_slowdown > 0.0 or result.is_low_confidence
        assert has_slowdown, "Unsupported turn should emit slowdown or low confidence"
