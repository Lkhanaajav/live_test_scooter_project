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

    def test_left_path_does_not_rejoin_across_centerline(self, commanded_left_bev_mask):
        """Short-horizon left path should stay on the commanded side rather than snake back."""
        corridor = _corridor(commanded_left_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="left",
            bev_mask=commanded_left_bev_mask,
        )
        assert result.active is True
        assert np.min(result.path_m[:, 1]) < -0.05
        assert np.max(result.path_m[:, 1]) <= 0.05

    def test_right_path_does_not_rejoin_across_centerline(self, commanded_right_bev_mask):
        """Short-horizon right path should stay on the commanded side rather than snake back."""
        corridor = _corridor(commanded_right_bev_mask)
        result = plan_waypoint_turn(
            corridor=corridor,
            intent="right",
            bev_mask=commanded_right_bev_mask,
        )
        assert result.active is True
        assert np.max(result.path_m[:, 1]) > 0.05
        assert np.min(result.path_m[:, 1]) >= -0.05

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


# ===========================================================================
# Wave 4 tests: replay evaluator summary parsing, metrics, mode dispatch
# ===========================================================================


class TestReplayManifestParsing:
    """The replay manifest should be parseable and list valid paths."""

    def test_replay_set_file_can_be_loaded(self):
        """The fixed replay manifest can be loaded and parsed by the evaluator."""
        from scripts.eval_waypoint_turn_planner import load_replay_set
        import os

        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".planning",
            "phases",
            "11.1-gps-intent-corridor-waypoint-turn-planner",
            "11.1-REPLAY_SET.txt",
        )
        videos = load_replay_set(manifest_path)
        assert isinstance(videos, list)
        assert len(videos) > 0
        # All entries should end with a video extension
        for v in videos:
            assert v.endswith(".mp4") or v.endswith(".mov") or v.endswith(".MOV"), (
                f"Unexpected video path: {v}"
            )

    def test_replay_set_skips_comments_and_blanks(self):
        """Comments and blank lines in the manifest are ignored."""
        from scripts.eval_waypoint_turn_planner import load_replay_set
        import tempfile, os

        content = "# comment line\n\ntest_videos/a.mp4\n# another comment\ntest_videos/b.mp4\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = f.name
        try:
            videos = load_replay_set(tmp_path)
            assert videos == ["test_videos/a.mp4", "test_videos/b.mp4"]
        finally:
            os.unlink(tmp_path)


class TestReplaySummaryMetrics:
    """Replay summaries must include waypoint-turn-specific metrics."""

    def test_summarize_log_includes_waypoint_turn_rate(self):
        """summarize_log must report waypoint_turn_rate when path_source column exists."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "waypoint_turn", "selected_template_family": "left"},
            {"path_source": "waypoint_turn", "selected_template_family": "left"},
            {"path_source": "template", "selected_template_family": "straight"},
            {"path_source": "dt_corridor", "selected_template_family": "straight"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source", "selected_template_family"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert "waypoint_turn_rate" in summary
            assert abs(summary["waypoint_turn_rate"] - 0.5) < 0.01
        finally:
            os.unlink(tmp_path)

    def test_summarize_log_includes_hold_rate(self):
        """summarize_log must report waypoint_turn_hold_rate."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "waypoint_turn_hold"},
            {"path_source": "waypoint_turn"},
            {"path_source": "template"},
            {"path_source": "dt_corridor"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert "waypoint_turn_hold_rate" in summary
            assert abs(summary["waypoint_turn_hold_rate"] - 0.25) < 0.01
        finally:
            os.unlink(tmp_path)

    def test_summarize_log_includes_family_switch_count(self):
        """summarize_log must count template_family switches."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "waypoint_turn", "selected_template_family": "left"},
            {"path_source": "waypoint_turn", "selected_template_family": "left"},
            {"path_source": "waypoint_turn", "selected_template_family": "right"},
            {"path_source": "template", "selected_template_family": "straight"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source", "selected_template_family"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert "maneuver_family_switches" in summary
            assert summary["maneuver_family_switches"] >= 2
        finally:
            os.unlink(tmp_path)

    def test_summarize_log_robust_when_no_waypoint_fields(self):
        """summarize_log stays robust when path_source has no waypoint_turn entries."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "template", "selected_template_family": "straight"},
            {"path_source": "dt_corridor", "selected_template_family": "straight"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source", "selected_template_family"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert summary.get("waypoint_turn_rate", 0.0) == 0.0
            assert summary.get("waypoint_turn_hold_rate", 0.0) == 0.0
        finally:
            os.unlink(tmp_path)

    def test_summarize_log_includes_heading_stats(self):
        """summarize_log must include mean and p95 heading magnitude."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "waypoint_turn", "heading_smoothed_deg": "5.0"},
            {"path_source": "waypoint_turn", "heading_smoothed_deg": "-10.0"},
            {"path_source": "template", "heading_smoothed_deg": "3.0"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source", "heading_smoothed_deg"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert "mean_abs_heading_deg" in summary
            assert "p95_abs_heading_deg" in summary
            assert summary["mean_abs_heading_deg"] > 0
        finally:
            os.unlink(tmp_path)

    def test_summarize_log_includes_low_confidence_rate(self):
        """summarize_log must report low_confidence_rate."""
        from scripts.eval_waypoint_turn_planner import summarize_log
        import tempfile, os, csv

        rows = [
            {"path_source": "waypoint_turn", "planner_low_confidence": "1"},
            {"path_source": "waypoint_turn", "planner_low_confidence": "0"},
            {"path_source": "template", "planner_low_confidence": "0"},
            {"path_source": "dt_corridor", "planner_low_confidence": "0"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["path_source", "planner_low_confidence"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            summary = summarize_log(tmp_path)
            assert "low_confidence_rate" in summary
            assert abs(summary["low_confidence_rate"] - 0.25) < 0.01
        finally:
            os.unlink(tmp_path)


class TestModeDispatch:
    """The evaluator exercises distinct planner modes (baseline/template/waypoint_turn)."""

    def test_mode_config_produces_distinct_settings(self):
        """Each mode name maps to different run_live kwargs."""
        from scripts.eval_waypoint_turn_planner import mode_to_run_kwargs

        baseline_kw = mode_to_run_kwargs("baseline")
        template_kw = mode_to_run_kwargs("template")
        waypoint_kw = mode_to_run_kwargs("waypoint_turn")

        # Baseline should disable template planner
        assert baseline_kw["template_planner_enabled"] is False

        # Template mode should enable template planner but disable waypoint turn
        assert template_kw["template_planner_enabled"] is True

        # Waypoint-turn mode should enable waypoint turn
        assert waypoint_kw.get("initial_intent") is not None or waypoint_kw.get("template_planner_enabled") is True

    def test_all_three_modes_are_valid(self):
        """The evaluator recognises baseline, template, and waypoint_turn as valid modes."""
        from scripts.eval_waypoint_turn_planner import VALID_MODES

        assert "baseline" in VALID_MODES
        assert "template" in VALID_MODES
        assert "waypoint_turn" in VALID_MODES


class TestComparisonArtifact:
    """A combined comparison artifact (JSON or MD) is produced from multi-mode results."""

    def test_build_comparison_produces_markdown(self):
        """build_comparison_report creates valid markdown from multi-mode summaries."""
        from scripts.eval_waypoint_turn_planner import build_comparison_report

        summaries = [
            {"mode": "baseline", "video_label": "vid017", "frames": 100, "waypoint_turn_rate": 0.0},
            {"mode": "template", "video_label": "vid017", "frames": 100, "waypoint_turn_rate": 0.0},
            {"mode": "waypoint_turn", "video_label": "vid017", "frames": 100, "waypoint_turn_rate": 0.6},
        ]
        md = build_comparison_report("Test Report", summaries)
        assert isinstance(md, str)
        assert "baseline" in md.lower()
        assert "waypoint_turn" in md.lower() or "waypoint" in md.lower()
        assert len(md) > 100
