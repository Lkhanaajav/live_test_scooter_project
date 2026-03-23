"""
waypoint_turn_planner.py
========================
Phase 11.1 Wave 0: Contract stub for the GPS-intent corridor waypoint turn planner.

This module defines the public interface for commanded-turn path planning.
Later waves will implement the full algorithm (side-support extraction,
target selection, smooth path fitting, maneuver hysteresis).

Current stub behavior:
  - No-intent and straight-intent: returns inactive result immediately.
  - Left/right intent: returns a low-confidence placeholder result
    (no actual target extraction yet).

Design rules:
  - GPS or route logic provides maneuver intent; vision fits geometry.
  - This module is additive -- it does not modify realtime_nav_core.py.
  - The existing Phase 11 template-approval planner is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from template_path_planner import Corridor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointTurnPlannerConfig:
    """Configuration for the waypoint-turn planner.

    All distances are in meters. Thresholds are dimensionless [0, 1].
    """
    # Forward decision band where turn openings are searched
    decision_band_min_m: float = 1.8
    decision_band_max_m: float = 3.2

    # Support thresholds for target acquisition and sustain
    acquire_support_min: float = 0.40
    sustain_support_min: float = 0.25

    # Path containment thresholds
    path_support_min: float = 0.50

    # Confidence floor below which hold is recommended
    low_confidence_threshold: float = 0.35

    # BEV geometry (must match corridor config)
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaypointTurnTarget:
    """A candidate waypoint target on the commanded side.

    Attributes
    ----------
    apex_m : tuple[float, float]
        (forward_m, lateral_m) of the turn apex in BEV metric coords.
    exit_m : tuple[float, float]
        (forward_m, lateral_m) of the exit anchor after the apex.
    support_score : float
        Fraction of corridor rows in the decision band that support this target.
    side : str
        "left" or "right".
    """
    apex_m: Tuple[float, float]
    exit_m: Tuple[float, float]
    support_score: float
    side: str


@dataclass
class WaypointTurnResult:
    """Result of the waypoint-turn planner for one frame.

    Attributes
    ----------
    active : bool
        True if the commanded-turn module produced a path (intent is left/right
        and corridor support was sufficient).
    intent : str
        The intent that was passed in ("left", "right", "straight", or "").
    target : WaypointTurnTarget | None
        The selected waypoint target, or None if inactive / unsupported.
    path_m : np.ndarray
        (N, 2) array of (forward_m, lateral_m) path points, or (0, 2) if inactive.
    confidence : float
        Overall confidence in [0, 1].
    recommend_hold : bool
        True if the planner recommends holding position (unsupported turn).
    suggested_slowdown : float
        Slowdown factor in [0, 1] (0 = full speed, 1 = stop).
    """
    active: bool
    intent: str
    target: Optional[WaypointTurnTarget]
    path_m: np.ndarray
    confidence: float
    recommend_hold: bool
    suggested_slowdown: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_waypoint_turn(
    corridor: Corridor,
    intent: Optional[str],
    bev_mask: np.ndarray,
    *,
    cfg: Optional[WaypointTurnPlannerConfig] = None,
    prev_target: Optional[WaypointTurnTarget] = None,
    prev_result: Optional[WaypointTurnResult] = None,
) -> WaypointTurnResult:
    """Plan a waypoint-conditioned turn path inside the BEV corridor.

    Parameters
    ----------
    corridor : Corridor
        Row-wise corridor geometry from ``corridor_from_mask()``.
    intent : str | None
        GPS/route maneuver intent: "left", "right", "straight", or None/"".
    bev_mask : np.ndarray
        Binary BEV mask (H, W) uint8, 255 = drivable.
    cfg : WaypointTurnPlannerConfig | None
        Configuration. Uses defaults if None.
    prev_target : WaypointTurnTarget | None
        Previous frame's selected target for hysteresis (Wave 2+).
    prev_result : WaypointTurnResult | None
        Previous frame's full result for continuity (Wave 2+).

    Returns
    -------
    WaypointTurnResult
        Turn planner output. ``active=False`` when intent is not a commanded
        turn or when the turn is unsupported.
    """
    cfg = cfg or WaypointTurnPlannerConfig()
    normalized_intent = str(intent or "").strip().lower()

    # No-intent or straight-intent: module stays inactive
    if normalized_intent not in ("left", "right"):
        return WaypointTurnResult(
            active=False,
            intent=normalized_intent,
            target=None,
            path_m=np.zeros((0, 2), dtype=np.float32),
            confidence=0.0,
            recommend_hold=False,
            suggested_slowdown=0.0,
        )

    # --- Commanded turn (left or right) ---
    # Wave 0 stub: return a low-confidence placeholder.
    # Full implementation (side-support extraction, target selection,
    # smooth path fitting) will be added in later waves.
    return WaypointTurnResult(
        active=False,
        intent=normalized_intent,
        target=None,
        path_m=np.zeros((0, 2), dtype=np.float32),
        confidence=0.0,
        recommend_hold=True,
        suggested_slowdown=1.0,
    )
