import numpy as np

from image_path_planner import CameraDtPlanner, CameraMidpointPlanner, ImagePathPlannerConfig


def _make_corridor(width: int = 160, height: int = 96, x0: int = 48, x1: int = 112) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, x0:x1] = 255
    return mask


def test_camera_midpoint_planner_finds_straight_path():
    mask = _make_corridor()
    planner = CameraMidpointPlanner(ImagePathPlannerConfig(work_size=(160, 96), min_valid_rows=8, min_forward_span_px=20))
    result = planner.plan(mask)
    assert result.has_path
    assert len(result.path_px) >= 8
    assert abs(result.heading_deg) < 1.0
    assert result.mean_clearance_px > 10.0


def test_camera_dt_planner_tracks_curved_corridor():
    height, width = 96, 160
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        center = int(48 + 0.6 * (height - 1 - y))
        mask[y, max(0, center - 24):min(width, center + 24)] = 255

    planner = CameraDtPlanner(ImagePathPlannerConfig(work_size=(160, 96), min_valid_rows=8, min_forward_span_px=20))
    result = planner.plan(mask)
    assert result.has_path
    assert len(result.path_px) >= 8
    assert result.heading_deg > 0.5
    assert result.mean_clearance_px > 8.0
