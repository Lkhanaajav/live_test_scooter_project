"""
heading.py
==========
Heading computation, command classification, and speed profiling.
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
)


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
