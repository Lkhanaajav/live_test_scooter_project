"""
conftest.py
===========
Shared pytest fixtures for scooter navigation tests.
"""

import sys
import os

import numpy as np
import pytest

# Ensure the project root is on the path so modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from realtime_nav_core import BEVPathExtractor, PathExtractorConfig, CubicPathModel


@pytest.fixture(autouse=True)
def _pin_ego_center():
    """Pin BEV_EGO_X_FRAC to 0.5 so tests don't depend on calibration files."""
    original = config.BEV_EGO_X_FRAC
    config.BEV_EGO_X_FRAC = 0.5
    yield
    config.BEV_EGO_X_FRAC = original


@pytest.fixture
def straight_bev_mask():
    """220x220 uint8 mask with a straight vertical corridor in the center."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    # A 60-pixel-wide corridor running from bottom to top (straight ahead)
    mask[:, 80:140] = 255
    return mask


@pytest.fixture
def curved_bev_mask():
    """220x220 uint8 mask with a corridor that curves to the left."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    for row in range(220):
        # Lateral shift increases linearly from bottom to top (left curve)
        shift = int((220 - row) * 0.3)
        cx = 110 + shift
        left = max(0, cx - 30)
        right = min(220, cx + 30)
        mask[row, left:right] = 255
    return mask


@pytest.fixture
def wide_straight_bev_mask():
    """220x220 mask with a wider straight corridor for ambiguous-template tests."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    mask[:, 60:160] = 255
    return mask


@pytest.fixture
def right_curved_bev_mask():
    """220x220 uint8 mask with a corridor that curves to the right."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    for row in range(220):
        shift = int((220 - row) * 0.25)
        cx = 110 - shift
        left = max(0, cx - 30)
        right = min(220, cx + 30)
        mask[row, left:right] = 255
    return mask


@pytest.fixture
def fragmented_near_field_bev_mask():
    """Straight corridor with broken near-field support and missing rows."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    mask[:, 85:135] = 255
    mask[188:220, :] = 0
    mask[150:168, :] = 0
    mask[112:124, 85:118] = 0
    return mask


@pytest.fixture
def false_pocket_bev_mask():
    """Straight corridor with a false side pocket near ego on the left."""
    mask = np.zeros((220, 220), dtype=np.uint8)
    mask[:, 86:136] = 255
    mask[185:220, 18:62] = 255
    mask[185:220, 62:86] = 0
    return mask


@pytest.fixture
def straight_path_model():
    """Pre-built CubicPathModel for a perfectly straight line (y=0)."""
    x_grid = np.linspace(0.0, 4.5, 120, dtype=np.float64)
    y_grid = np.zeros(120, dtype=np.float64)
    # Arc length = x for a straight line
    s_grid = x_grid.copy()
    kappa_grid = np.zeros(120, dtype=np.float64)
    coeff = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return CubicPathModel(
        coeff=coeff,
        x_grid=x_grid,
        y_grid=y_grid,
        s_grid=s_grid,
        kappa_grid=kappa_grid,
    )


@pytest.fixture
def bev_h_matrix():
    """Simple scale homography: image -> BEV pixels. Maps (x_img, y_img) -> (x_img*0.3, y_img*0.5).
    Not a real perspective matrix — chosen so projection math is analytically verifiable in tests.
    With this H, a detection with foot at image (300, 400) projects to BEV (90, 200)."""
    return np.array([[0.3, 0.0, 0.0],
                     [0.0, 0.5, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


@pytest.fixture
def bev_obstacle_mask_500x600():
    """Fully white 500x600 BEV mask (numpy HxW = 500 rows x 600 cols)."""
    return np.ones((500, 600), dtype=np.uint8) * 255


@pytest.fixture
def mock_detections():
    """Two detection dicts representing a person at safe distance and one at stop distance."""
    return [
        {
            "bbox": (250, 200, 350, 400),
            "class_name": "person",
            "confidence": 0.9,
            "distance_m": 2.5,
        },
        {
            "bbox": (280, 350, 320, 480),
            "class_name": "person",
            "confidence": 0.8,
            "distance_m": 0.8,
        },
    ]


# ---------------------------------------------------------------------------
# Phase 11.1: Waypoint-turn planner fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def commanded_left_bev_mask():
    """220x220 BEV mask: straight corridor + left-side turn opening.

    Forward decision band (rows ~60-120, i.e. ~3-5 m forward) has drivable
    surface extending to the left, creating a commanded-left support cluster.
    """
    mask = np.zeros((220, 220), dtype=np.uint8)
    # Straight corridor (center)
    mask[:, 80:140] = 255
    # Left turn opening in the forward decision band (rows 40-120)
    mask[40:120, 20:80] = 255
    return mask


@pytest.fixture
def commanded_right_bev_mask():
    """220x220 BEV mask: straight corridor + right-side turn opening.

    Forward decision band has drivable surface extending to the right,
    creating a commanded-right support cluster.
    """
    mask = np.zeros((220, 220), dtype=np.uint8)
    # Straight corridor (center)
    mask[:, 80:140] = 255
    # Right turn opening in the forward decision band (rows 40-120)
    mask[40:120, 140:200] = 255
    return mask


@pytest.fixture
def unsupported_turn_bev_mask():
    """220x220 BEV mask with fragmented side support.

    Narrow corridor with small disconnected pockets on both sides --
    not enough contiguous support for a commanded turn in either direction.
    """
    mask = np.zeros((220, 220), dtype=np.uint8)
    # Narrow center corridor
    mask[:, 95:125] = 255
    # Small isolated pockets (too small / disconnected for turn support)
    mask[80:90, 40:60] = 255
    mask[100:108, 160:175] = 255
    return mask


@pytest.fixture
def no_intent_straight_bev_mask():
    """220x220 BEV mask: clean straight corridor, no turn openings.

    Used for verifying that no-intent and straight-intent leave
    the commanded-turn module inactive.
    """
    mask = np.zeros((220, 220), dtype=np.uint8)
    mask[:, 75:145] = 255
    return mask
