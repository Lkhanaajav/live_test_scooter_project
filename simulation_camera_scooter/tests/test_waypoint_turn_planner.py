"""
test_waypoint_turn_planner.py
=============================
Phase 11.1 Wave 0: Contract and fixture tests for the waypoint-turn planner.

Covers:
  - Public contract stability (imports, function signatures, dataclass fields)
  - Commanded-left and commanded-right fixtures expose distinct side support
  - Unsupported or fragmented commanded-turn masks return low-confidence hold
  - No-intent and straight-intent leave the commanded-turn module inactive

These tests are intentionally stub-level: they verify the contract and
placeholder behavior, not the full algorithm (which arrives in later waves).
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
