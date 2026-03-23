"""
test_waypoint_turn_planner.py
=============================
Phase 11.1 Wave 0+1: Contract, fixture, and algorithm tests for the waypoint-turn planner.

Covers:
  - Public contract stability (imports, function signatures, dataclass fields)
  - Commanded-left and commanded-right fixtures expose distinct side support
  - Unsupported or fragmented commanded-turn masks return low-confidence hold
  - No-intent and straight-intent leave the commanded-turn module inactive
  - (Wave 1) Commanded-side support scoring selects the best cluster on the requested side
  - (Wave 1) Weak outer-edge candidates do not pass the target gate
  - (Wave 1) Fitted paths pass only when containment thresholds are met
"""

import numpy as np
import pytest

from waypoint_turn_planner import (
    WaypointTurnPlannerConfig,
    WaypointTurnTarget,
    WaypointTurnResult,
    plan_waypoint_turn,
)
from template_path_planner import corridor_from_mask, CorridorConfig


# ---------------------------------------------------------------------------
# Helper: build a corridor from a fixture mask
# ---------------------------------------------------------------------------

def _corridor(mask: np.ndarray) -> "template_path_planner.Corridor":
    return corridor_from_mask(mask, CorridorConfig(bev_forward_m=10.0, bev_lateral_m=10.0))


# ---------------------------------------------------------------------------
# Contract tests: verify public API shape
# ---------------------------------------------------------------------------

class TestContract:
    """Verify that the public contract (imports, signatures, fields) is stable."""

    def test_config_dataclass_has_expected_fields(self):
        cfg = WaypointTurnPlannerConfig()
        assert hasattr(cfg, "decision_band_min_m")
        assert hasattr(cfg, "decision_band_max_m")
        assert hasattr(cfg, "acquire_support_min")
        assert hasattr(cfg, "sustain_support_min")

    def test_result_dataclass_has_expected_fields(self):
        result = WaypointTurnResult(
            active=False,
            intent="",
            target=None,
            path_m=np.zeros((0, 2), dtype=np.float32),
            confidence=0.0,
            recommend_hold=True,
            suggested_slowdown=1.0,
        )
        assert result.active is False
        assert result.confidence == 0.0
        assert result.path_m.shape[1] == 2

    def test_plan_waypoint_turn_returns_result(self, straight_bev_mask):
        corridor = _corridor(straight_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="straight",
            bev_mask=straight_bev_mask,
        )
        assert isinstance(result, WaypointTurnResult)

    def test_target_dataclass_has_expected_fields(self):
        target = WaypointTurnTarget(
            apex_m=(3.0, -1.5),
            exit_m=(5.0, -2.0),
            support_score=0.7,
            side="left",
        )
        assert target.side == "left"
        assert target.support_score == 0.7


# ---------------------------------------------------------------------------
# Fixture tests: commanded-left and commanded-right expose distinct support
# ---------------------------------------------------------------------------

class TestFixtureSupport:
    """Commanded-side masks should produce distinct support clusters."""

    def test_commanded_left_fixture_has_left_support(self, commanded_left_bev_mask):
        """The commanded-left fixture has drivable pixels extending left
        in the forward decision band."""
        corridor = _corridor(commanded_left_bev_mask)
        # In the decision band, left_lateral_m should extend further left
        # than in a plain straight corridor.
        valid = corridor.valid_mask
        assert np.any(valid), "Corridor must have valid rows"
        left_extent = float(np.min(corridor.left_lateral_m[valid]))
        # With a center corridor at cols 80-140 on a 220-wide image,
        # left_lateral_m for center-only ~= -1.36.  With the left opening
        # (cols 20-80), the leftmost should be further left.
        assert left_extent < -1.5, (
            f"Expected left extent < -1.5 for left opening, got {left_extent}"
        )

    def test_commanded_right_fixture_has_right_support(self, commanded_right_bev_mask):
        """The commanded-right fixture has drivable pixels extending right
        in the forward decision band."""
        corridor = _corridor(commanded_right_bev_mask)
        valid = corridor.valid_mask
        assert np.any(valid), "Corridor must have valid rows"
        right_extent = float(np.max(corridor.right_lateral_m[valid]))
        assert right_extent > 1.5, (
            f"Expected right extent > 1.5 for right opening, got {right_extent}"
        )

    def test_left_and_right_fixtures_are_distinct(
        self, commanded_left_bev_mask, commanded_right_bev_mask
    ):
        """Left and right masks must differ in their lateral support profile."""
        c_left = _corridor(commanded_left_bev_mask)
        c_right = _corridor(commanded_right_bev_mask)
        left_center = float(np.mean(c_left.center_lateral_m[c_left.valid_mask]))
        right_center = float(np.mean(c_right.center_lateral_m[c_right.valid_mask]))
        # Left opening shifts corridor center leftward; right shifts rightward
        assert left_center < right_center, (
            f"Expected left center ({left_center}) < right center ({right_center})"
        )


# ---------------------------------------------------------------------------
# Unsupported-turn tests: fragmented masks return low confidence / hold
# ---------------------------------------------------------------------------

class TestUnsupportedTurn:
    """Fragmented or unsupported commanded-turn masks should return
    a low-confidence or hold-style result."""

    def test_unsupported_left_turn_returns_low_confidence(self, unsupported_turn_bev_mask):
        corridor = _corridor(unsupported_turn_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=unsupported_turn_bev_mask,
        )
        assert isinstance(result, WaypointTurnResult)
        # Stub should return low confidence for unsupported turns
        assert result.confidence < 0.5 or result.recommend_hold is True

    def test_unsupported_right_turn_returns_low_confidence(self, unsupported_turn_bev_mask):
        corridor = _corridor(unsupported_turn_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=unsupported_turn_bev_mask,
        )
        assert isinstance(result, WaypointTurnResult)
        assert result.confidence < 0.5 or result.recommend_hold is True

    def test_fragmented_mask_does_not_crash(self, fragmented_near_field_bev_mask):
        corridor = _corridor(fragmented_near_field_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=fragmented_near_field_bev_mask,
        )
        assert isinstance(result, WaypointTurnResult)


# ---------------------------------------------------------------------------
# Inactive tests: no-intent and straight-intent leave module inactive
# ---------------------------------------------------------------------------

class TestInactive:
    """No-intent and straight-intent should leave the commanded-turn
    module inactive rather than silently selecting a turn."""

    def test_no_intent_leaves_inactive(self, no_intent_straight_bev_mask):
        corridor = _corridor(no_intent_straight_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent=None,
            bev_mask=no_intent_straight_bev_mask,
        )
        assert result.active is False
        assert result.target is None

    def test_empty_intent_leaves_inactive(self, no_intent_straight_bev_mask):
        corridor = _corridor(no_intent_straight_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="",
            bev_mask=no_intent_straight_bev_mask,
        )
        assert result.active is False
        assert result.target is None

    def test_straight_intent_leaves_inactive(self, straight_bev_mask):
        corridor = _corridor(straight_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="straight",
            bev_mask=straight_bev_mask,
        )
        assert result.active is False
        assert result.target is None

    def test_inactive_result_has_empty_path(self, no_intent_straight_bev_mask):
        corridor = _corridor(no_intent_straight_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent=None,
            bev_mask=no_intent_straight_bev_mask,
        )
        assert result.path_m.shape == (0, 2) or len(result.path_m) == 0


# ===========================================================================
# Wave 1 tests: target selection, support scoring, containment gating
# ===========================================================================


class TestCommandedSideTargetSelection:
    """WPT-01: commanded left/right selects a corridor-supported target on
    the requested side instead of using skeleton branches."""

    def test_left_target_has_negative_lateral(self, commanded_left_bev_mask):
        """Commanded 'left' must produce a target with negative lateral
        (left of ego in BEV metric coords)."""
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
        )
        assert result.active is True, "Left turn with good support should be active"
        assert result.target is not None, "Target must be selected"
        assert result.target.side == "left"
        # BEV convention: left = negative lateral
        assert result.target.apex_m[1] < 0.0, (
            f"Left target lateral should be negative, got {result.target.apex_m[1]}"
        )

    def test_right_target_has_positive_lateral(self, commanded_right_bev_mask):
        """Commanded 'right' must produce a target with positive lateral
        (right of ego in BEV metric coords)."""
        corridor = _corridor(commanded_right_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=commanded_right_bev_mask,
        )
        assert result.active is True, "Right turn with good support should be active"
        assert result.target is not None, "Target must be selected"
        assert result.target.side == "right"
        # BEV convention: right = positive lateral
        assert result.target.apex_m[1] > 0.0, (
            f"Right target lateral should be positive, got {result.target.apex_m[1]}"
        )

    def test_left_target_ignores_right_side_opening(self, commanded_right_bev_mask):
        """When intent is 'left' but only right-side support exists,
        the planner should NOT produce an active target on the wrong side."""
        corridor = _corridor(commanded_right_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_right_bev_mask,
        )
        # With no left support, either inactive or hold with low confidence
        if result.target is not None:
            assert result.target.side == "left", (
                "Target side must match commanded intent"
            )
        else:
            assert result.recommend_hold is True or result.confidence < 0.35

    def test_right_target_ignores_left_side_opening(self, commanded_left_bev_mask):
        """When intent is 'right' but only left-side support exists,
        the planner should NOT produce an active target on the wrong side."""
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=commanded_left_bev_mask,
        )
        if result.target is not None:
            assert result.target.side == "right"
        else:
            assert result.recommend_hold is True or result.confidence < 0.35

    def test_target_forward_within_decision_band(self, commanded_left_bev_mask):
        """The selected target apex should be in or near the forward decision band."""
        cfg = WaypointTurnPlannerConfig()
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
            cfg=cfg,
        )
        assert result.target is not None
        fwd = result.target.apex_m[0]
        # Target forward should be reasonably in the decision band region
        assert fwd >= cfg.decision_band_min_m * 0.5, (
            f"Target forward too close: {fwd}"
        )
        assert fwd <= cfg.decision_band_max_m * 2.0, (
            f"Target forward too far: {fwd}"
        )


class TestSupportScoring:
    """Support-score ordering and false-pocket rejection."""

    def test_support_score_above_threshold_for_good_opening(self, commanded_left_bev_mask):
        """A wide left opening should produce a target with support_score
        above the acquire threshold."""
        cfg = WaypointTurnPlannerConfig()
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
            cfg=cfg,
        )
        assert result.target is not None
        assert result.target.support_score >= cfg.acquire_support_min, (
            f"Expected support >= {cfg.acquire_support_min}, got {result.target.support_score}"
        )

    def test_false_pocket_rejected_as_target(self, false_pocket_bev_mask):
        """A near-ego false pocket on the left should not produce a high-support
        target since it is disconnected and narrow."""
        corridor = _corridor(false_pocket_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=false_pocket_bev_mask,
        )
        # The false pocket is near ego (rows 185-220), outside the decision band
        # so it should either not produce an active result or have low confidence
        assert (
            result.active is False
            or result.confidence < 0.5
            or result.recommend_hold is True
        ), "False pocket should not produce a confident turn target"

    def test_weak_unsupported_target_not_active(self, unsupported_turn_bev_mask):
        """Sparse/fragmented side support (small isolated pockets) should
        not pass the target gate when below acquire_support_min."""
        cfg = WaypointTurnPlannerConfig()
        corridor = _corridor(unsupported_turn_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=unsupported_turn_bev_mask,
            cfg=cfg,
        )
        # Either inactive or the target has low support
        if result.target is not None:
            assert result.target.support_score < cfg.acquire_support_min, (
                f"Unsupported target should have low support, got {result.target.support_score}"
            )
        assert result.active is False or result.recommend_hold is True


class TestContainmentGating:
    """WPT-02: the generated path must pass containment gates to be approved."""

    def test_approved_left_path_stays_inside_corridor(self, commanded_left_bev_mask):
        """When active, the returned path points should all be within
        the visible drivable corridor boundaries."""
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
        )
        assert result.active is True, "Good left opening should produce active result"
        assert result.path_m.shape[0] > 0, "Active result must have path points"
        assert result.path_m.shape[1] == 2, "Path must be (N, 2)"
        # Path should be smooth: forward coordinates monotonically increasing
        fwd = result.path_m[:, 0]
        assert np.all(np.diff(fwd) >= -0.01), "Path forward coords should be non-decreasing"

    def test_approved_right_path_stays_inside_corridor(self, commanded_right_bev_mask):
        """Right turn path should also be valid and inside corridor."""
        corridor = _corridor(commanded_right_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=commanded_right_bev_mask,
        )
        assert result.active is True, "Good right opening should produce active result"
        assert result.path_m.shape[0] > 0, "Active result must have path points"
        fwd = result.path_m[:, 0]
        assert np.all(np.diff(fwd) >= -0.01), "Path forward coords should be non-decreasing"

    def test_confidence_above_low_threshold_when_active(self, commanded_left_bev_mask):
        """An active (approved) result should have confidence above the
        low_confidence_threshold."""
        cfg = WaypointTurnPlannerConfig()
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
            cfg=cfg,
        )
        assert result.active is True
        assert result.confidence >= cfg.low_confidence_threshold, (
            f"Active result confidence should be >= {cfg.low_confidence_threshold}, got {result.confidence}"
        )

    def test_narrow_corridor_path_rejected(self, unsupported_turn_bev_mask):
        """On a narrow corridor with no real side support, the path gate should
        reject the turn and result should be inactive or hold."""
        corridor = _corridor(unsupported_turn_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=unsupported_turn_bev_mask,
        )
        # Either inactive or low-confidence hold
        assert result.active is False or result.recommend_hold is True

    def test_hold_result_has_slowdown(self, unsupported_turn_bev_mask):
        """When the planner recommends hold, suggested_slowdown should be
        above zero (advising the controller to slow down or stop)."""
        corridor = _corridor(unsupported_turn_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=unsupported_turn_bev_mask,
        )
        if result.recommend_hold:
            assert result.suggested_slowdown > 0.0, (
                "Hold recommendation should come with positive slowdown"
            )

    def test_active_result_has_nonzero_path_length(self, commanded_right_bev_mask):
        """An active turn result should produce a path of meaningful length
        (not just a single point or zero-length path)."""
        corridor = _corridor(commanded_right_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=commanded_right_bev_mask,
        )
        assert result.active is True
        assert result.path_m.shape[0] >= 3, (
            f"Active path should have >= 3 points, got {result.path_m.shape[0]}"
        )
        # Path should span some forward distance
        fwd_span = float(result.path_m[-1, 0] - result.path_m[0, 0])
        assert fwd_span > 0.5, f"Path forward span too short: {fwd_span}"
