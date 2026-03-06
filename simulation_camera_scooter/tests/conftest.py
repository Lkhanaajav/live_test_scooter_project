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

from realtime_nav_core import BEVPathExtractor, PathExtractorConfig, CubicPathModel


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
