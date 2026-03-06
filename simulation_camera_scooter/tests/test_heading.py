"""
test_heading.py
===============
Unit tests for compute_heading, heading_to_command, and compute_speed.
"""

import pytest

from heading import compute_heading, heading_to_command, compute_speed
from config import (
    HEADING_STRAIGHT_THRESH,
    HEADING_TURN_THRESH,
    SPEED_MAX,
    SPEED_TURN,
    SPEED_SHARP_TURN,
    SPEED_OBSTACLE_NEAR,
    SPEED_STOP,
)


class TestComputeHeading:

    def test_straight_path_gives_zero_heading(self):
        """Vertical path (straight ahead) should return ~0 degrees."""
        path = [(110, 220), (110, 100)]  # x=constant, y decreasing = forward
        angle = compute_heading(path)
        assert abs(angle) < 5.0

    def test_left_curve_gives_negative_heading(self):
        """Path curving left (x decreasing) should return negative heading."""
        # Start at (110, 220), end at (60, 100) -> moving left
        path = [(110, 220), (90, 180), (70, 140), (60, 100)]
        angle = compute_heading(path)
        assert angle < 0.0

    def test_right_curve_gives_positive_heading(self):
        """Path curving right (x increasing) should return positive heading."""
        path = [(110, 220), (130, 180), (150, 140), (160, 100)]
        angle = compute_heading(path)
        assert angle > 0.0

    def test_single_point_returns_zero(self):
        """Less than 2 points should return 0."""
        assert compute_heading([]) == 0.0
        assert compute_heading([(110, 110)]) == 0.0


class TestHeadingToCommand:

    def test_heading_to_command_straight(self):
        """Small angle should map to STRAIGHT."""
        cmd, color = heading_to_command(0.0)
        assert cmd == "STRAIGHT"

    def test_heading_to_command_left(self):
        """Moderate negative angle should map to LEFT."""
        cmd, color = heading_to_command(-(HEADING_STRAIGHT_THRESH + 5))
        assert cmd == "LEFT"

    def test_heading_to_command_right(self):
        """Moderate positive angle should map to RIGHT."""
        cmd, color = heading_to_command(HEADING_STRAIGHT_THRESH + 5)
        assert cmd == "RIGHT"

    def test_heading_to_command_sharp_left(self):
        """Large negative angle should map to SHARP LEFT."""
        cmd, color = heading_to_command(-(HEADING_TURN_THRESH + 10))
        assert cmd == "SHARP LEFT"

    def test_heading_to_command_sharp_right(self):
        """Large positive angle should map to SHARP RIGHT."""
        cmd, color = heading_to_command(HEADING_TURN_THRESH + 10)
        assert cmd == "SHARP RIGHT"

    def test_returns_tuple_of_string_and_tuple(self):
        """heading_to_command should return (str, tuple)."""
        cmd, color = heading_to_command(0.0)
        assert isinstance(cmd, str)
        assert isinstance(color, tuple)
        assert len(color) == 3


class TestComputeSpeed:

    def test_compute_speed_stops_near_obstacle(self):
        """Obstacle within stop distance should return SPEED_STOP."""
        speed = compute_speed(0.0, min_obstacle_dist=0.5, has_path=True)
        assert speed == SPEED_STOP

    def test_compute_speed_slows_near_obstacle(self):
        """Obstacle within close range should return SPEED_OBSTACLE_NEAR."""
        speed = compute_speed(0.0, min_obstacle_dist=2.0, has_path=True)
        assert speed == SPEED_OBSTACLE_NEAR

    def test_compute_speed_max_on_straight_clear(self):
        """Straight heading, no obstacle, valid path -> max speed."""
        speed = compute_speed(0.0, min_obstacle_dist=None, has_path=True)
        assert speed == SPEED_MAX

    def test_compute_speed_sharp_turn(self):
        """Sharp turn heading -> SPEED_SHARP_TURN."""
        speed = compute_speed(HEADING_TURN_THRESH + 15, min_obstacle_dist=None, has_path=True)
        assert speed == SPEED_SHARP_TURN

    def test_compute_speed_no_path_returns_crawl(self):
        """No path -> slow crawl (SPEED_OBSTACLE_NEAR), not full stop."""
        speed = compute_speed(0.0, min_obstacle_dist=None, has_path=False)
        assert speed == SPEED_OBSTACLE_NEAR
        assert speed > SPEED_STOP

    def test_compute_speed_turn_interpolated(self):
        """Speed during a turn should be between SPEED_TURN and SPEED_MAX."""
        mid_angle = (HEADING_STRAIGHT_THRESH + HEADING_TURN_THRESH) / 2.0
        speed = compute_speed(mid_angle, min_obstacle_dist=None, has_path=True)
        assert SPEED_TURN <= speed <= SPEED_MAX
