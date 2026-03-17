"""
heading.py
==========
Heading computation, command classification, and speed profiling.

Research impl: Temporal heading filter (Idea 3).
  compute_heading_smooth() wraps compute_heading() with a circular EMA filter
  to prevent single-frame heading spikes from flipping command classification.
  Papers: Regulated Pure Pursuit (arXiv:2305.20026)
"""

import math

from config import (
    HEADING_STRAIGHT_THRESH,
    HEADING_TURN_THRESH,
    SPEED_MAX,
    SPEED_TURN,
    SPEED_SHARP_TURN,
    SPEED_OBSTACLE_NEAR,
    SPEED_STOP,
    OBSTACLE_STOP_M,
    OBSTACLE_CLOSE_M,
    COLOR_STRAIGHT,
    COLOR_LEFT,
    COLOR_RIGHT,
    COLOR_SHARP,
    HEADING_SMOOTH_ENABLED,
)

# Research impl: lazy import to avoid circular dependency
try:
    from path_smoother import HeadingTemporalFilter as _HeadingTemporalFilter
    _HAS_HEADING_FILTER = True
except ImportError:
    _HAS_HEADING_FILTER = False
    _HeadingTemporalFilter = None  # type: ignore[assignment,misc]

# Module-level heading filter instance (shared across calls)
_heading_filter = None  # type: ignore[assignment]


def _get_heading_filter():
    """Lazy-init the module-level heading filter."""
    global _heading_filter
    if _heading_filter is None and _HAS_HEADING_FILTER and HEADING_SMOOTH_ENABLED:
        _heading_filter = _HeadingTemporalFilter()
    return _heading_filter


def compute_heading(path_pts):
    """
    Compute heading angle from a BEV path.
    Returns angle in degrees: 0 = straight ahead, negative = left, positive = right.
    """
    if len(path_pts) < 2:
        return 0.0
    start = [float(v) for v in path_pts[0]]
    idx = min(len(path_pts) - 1, max(1, len(path_pts) * 2 // 5))
    end = [float(v) for v in path_pts[idx]]
    dx = end[0] - start[0]
    dy = start[1] - end[1]
    if dy <= 0:
        return 0.0
    angle_rad = math.atan2(dx, dy)
    return math.degrees(angle_rad)


def compute_heading_smooth(path_pts, confidence: float = 1.0) -> float:
    """
    Research impl: Heading with temporal circular EMA filter.

    Computes raw heading via compute_heading() then applies the module-level
    HeadingTemporalFilter. Returns the smoothed heading in degrees.

    If HEADING_SMOOTH_ENABLED=False or path_smoother not available, falls
    back to raw compute_heading() (fully backward compatible).

    Parameters
    ----------
    path_pts : sequence of (x, y)
        BEV path points (same format as compute_heading).
    confidence : float
        Path confidence [0,1] to modulate filter alpha.

    Returns
    -------
    float
        Smoothed heading angle in degrees.
    """
    raw = compute_heading(path_pts)
    filt = _get_heading_filter()
    if filt is None:
        return raw
    return filt.update(raw, confidence)


def reset_heading_filter() -> None:
    """Reset the module-level heading temporal filter (call on scene restart)."""
    filt = _get_heading_filter()
    if filt is not None:
        filt.reset()


def heading_to_command(angle_deg):
    """Convert heading angle to command string and color."""
    abs_angle = abs(angle_deg)
    if abs_angle < HEADING_STRAIGHT_THRESH:
        return "STRAIGHT", COLOR_STRAIGHT
    elif abs_angle < HEADING_TURN_THRESH:
        if angle_deg < 0:
            return "LEFT", COLOR_LEFT
        else:
            return "RIGHT", COLOR_RIGHT
    else:
        if angle_deg < 0:
            return "SHARP LEFT", COLOR_SHARP
        else:
            return "SHARP RIGHT", COLOR_SHARP


def compute_speed(angle_deg, min_obstacle_dist, has_path):
    """
    Compute target speed based on heading angle and nearest obstacle distance.
    Returns speed in m/s.
    """
    if not has_path:
        # Don't fully stop -- slow crawl forward while path is temporarily lost
        return SPEED_OBSTACLE_NEAR

    # Obstacle override
    if min_obstacle_dist is not None:
        if min_obstacle_dist < OBSTACLE_STOP_M:
            return SPEED_STOP
        elif min_obstacle_dist < OBSTACLE_CLOSE_M:
            return SPEED_OBSTACLE_NEAR

    # Speed from heading
    abs_angle = abs(angle_deg)
    if abs_angle < HEADING_STRAIGHT_THRESH:
        return SPEED_MAX
    elif abs_angle < HEADING_TURN_THRESH:
        # Linear interpolation between SPEED_TURN and SPEED_MAX
        t = (abs_angle - HEADING_STRAIGHT_THRESH) / (HEADING_TURN_THRESH - HEADING_STRAIGHT_THRESH)
        return SPEED_MAX - t * (SPEED_MAX - SPEED_TURN)
    else:
        return SPEED_SHARP_TURN


def apply_planner_speed_limit(speed_mps, suggested_slowdown, is_low_confidence, cautious_speed_mps=SPEED_TURN):
    """
    Reduce speed based on planner-native slowdown guidance.
    `suggested_slowdown` is expected in [0, 1], where 1 means "slow aggressively".
    """
    base = float(speed_mps)
    slowdown = max(0.0, min(1.0, float(suggested_slowdown)))
    if slowdown <= 1e-6:
        return base

    cautious = max(0.05, float(cautious_speed_mps))
    low_conf = bool(is_low_confidence)
    gain = 0.85 if low_conf else 0.65
    floor_scale = 0.20 if low_conf else 0.35
    cap = max(cautious * floor_scale, base * (1.0 - gain * slowdown))
    return float(min(base, cap))
