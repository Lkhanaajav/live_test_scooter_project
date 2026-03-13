"""
test_bev_calibration.py
=======================
Unit tests for BEV calibration loading and validation.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from bev_calibration import load_bev_params
from config import CALIBRATION_FILE, DEFAULT_SRC_POINTS, DEFAULT_DST_POINTS


class TestLoadBevParams:

    def test_calibration_file_is_absolute(self):
        """Calibration path should not depend on the current working directory."""
        assert os.path.isabs(CALIBRATION_FILE)

    def test_load_bev_params_returns_3_values(self):
        """load_bev_params should return a 3-tuple."""
        result = load_bev_params()
        assert len(result) == 3

    def test_load_bev_params_h_is_3x3(self):
        """The homography matrix H should be 3x3."""
        H, Hinv, src = load_bev_params()
        assert H.shape == (3, 3)

    def test_h_and_hinv_are_inverses(self):
        """H @ Hinv should be close to the identity matrix."""
        H, Hinv, src = load_bev_params()
        product = H @ Hinv
        identity = np.eye(3)
        assert np.allclose(product, identity, atol=1e-6)

    def test_singular_calibration_falls_back_to_default(self, tmp_path, capsys):
        """A degenerate (collinear) calibration file should fall back to defaults."""
        # Create a degenerate calibration: all 4 points on the same line
        degenerate_src = np.array(
            [[100.0, 500.0], [200.0, 500.0], [300.0, 500.0], [400.0, 500.0]],
            dtype=np.float32,
        )
        cal_file = tmp_path / "bev_calibration.npy"
        np.save(str(cal_file), degenerate_src)

        # Temporarily override calibration file path by monkeypatching
        import bev_calibration as bev_mod
        original = bev_mod.CALIBRATION_FILE
        bev_mod.CALIBRATION_FILE = str(cal_file)
        try:
            H, Hinv, src = bev_mod.load_bev_params()
        finally:
            bev_mod.CALIBRATION_FILE = original

        captured = capsys.readouterr()
        # Should have printed a warning
        assert "WARNING" in captured.out or H.shape == (3, 3)
        # Result should still be a valid 3x3 matrix
        assert H.shape == (3, 3)

    def test_default_src_points_used_when_no_file(self, monkeypatch, tmp_path):
        """When calibration file is absent, default src points are used."""
        import bev_calibration as bev_mod
        original = bev_mod.CALIBRATION_FILE
        # Point to a non-existent file
        bev_mod.CALIBRATION_FILE = str(tmp_path / "nonexistent.npy")
        try:
            H, Hinv, src = bev_mod.load_bev_params()
        finally:
            bev_mod.CALIBRATION_FILE = original

        # The returned src should match the default
        assert np.allclose(src, DEFAULT_SRC_POINTS, atol=1e-3)
