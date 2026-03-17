"""
path_smoother.py
================
Research impl: Temporal Path Smoothing (Idea 3).

Two classes are provided:

  PathTemporalSmoother  — EMA on cubic polynomial coefficients [a0,a1,a2,a3].
  HeadingTemporalFilter — Circular EMA on heading angle in degrees.

Both implement adaptive alpha (confidence-weighted) and reset conditions to
avoid filter lag during genuine topology changes.

Papers:
  Trajectory Prediction Survey (arXiv:2503.03262)
  Regulated Pure Pursuit smoother (arXiv:2305.20026)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from config import (
    PATH_SMOOTH_ENABLED,
    PATH_SMOOTH_MIN_ALPHA,
    PATH_SMOOTH_MAX_ALPHA,
    PATH_SMOOTH_RESET_THRESH,
    HEADING_SMOOTH_ENABLED,
    HEADING_SMOOTH_ALPHA,
    HEADING_SMOOTH_RESET_DEG,
)


# ---------------------------------------------------------------------------
# PathTemporalSmoother
# ---------------------------------------------------------------------------

class PathTemporalSmoother:
    """
    Research impl: EMA smoother on cubic path polynomial coefficients.

    Reduces inter-frame coefficient jitter caused by segmentation noise
    without introducing lag during genuine path changes (turns, topology).

    Alpha is confidence-adaptive:
      alpha = clip(confidence * 1.3, PATH_SMOOTH_MIN_ALPHA, PATH_SMOOTH_MAX_ALPHA)
    High alpha → fast tracking (trusts new measurement)
    Low alpha  → heavy smoothing (inertia, useful during low-confidence frames)

    Reset conditions (new_coeffs used directly, no smoothing):
      1. Path source changed (e.g. graph -> template or vice versa)
      2. Max per-coefficient jump > PATH_SMOOTH_RESET_THRESH
      3. First call after construction or explicit reset()

    Papers: Trajectory Prediction Survey (arXiv:2503.03262)
    """

    def __init__(
        self,
        min_alpha: float = PATH_SMOOTH_MIN_ALPHA,
        max_alpha: float = PATH_SMOOTH_MAX_ALPHA,
        reset_thresh: float = PATH_SMOOTH_RESET_THRESH,
        enabled: bool = PATH_SMOOTH_ENABLED,
    ) -> None:
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.reset_thresh = float(reset_thresh)
        self.enabled = bool(enabled)

        self._prev_smoothed: Optional[np.ndarray] = None
        self._prev_source: str = ""
        self._initialized: bool = False

    def smooth(
        self,
        coeffs: np.ndarray,
        confidence: float,
        path_source: str,
    ) -> np.ndarray:
        """
        Apply EMA smoothing to cubic coefficients.

        Parameters
        ----------
        coeffs : np.ndarray, shape (4,)
            New cubic coefficients [a0, a1, a2, a3].
        confidence : float, range [0, 1]
            Path planning confidence from corridor/template scoring.
        path_source : str
            Source identifier (e.g. "template", "graph", "fallback_hold").
            A change in path_source category triggers a reset.

        Returns
        -------
        np.ndarray, shape (4,)
            Smoothed cubic coefficients.
        """
        new_c = np.asarray(coeffs, dtype=np.float64).copy()

        if not self.enabled or len(new_c) < 4:
            return new_c.astype(np.float64)

        # Determine path source category for reset detection
        source_cat = _path_source_category(path_source)
        prev_cat = _path_source_category(self._prev_source)
        source_changed = (source_cat != prev_cat) and self._initialized

        # Check for large coefficient jump (topology flip)
        large_jump = False
        if self._initialized and self._prev_smoothed is not None:
            delta = np.abs(new_c - self._prev_smoothed)
            large_jump = bool(np.max(delta) > self.reset_thresh)

        # Reset condition
        if not self._initialized or source_changed or large_jump:
            self._prev_smoothed = new_c.copy()
            self._prev_source = path_source
            self._initialized = True
            return new_c.astype(np.float64)

        # Adaptive alpha: higher confidence → higher alpha → faster tracking
        alpha = float(np.clip(float(confidence) * 1.3, self.min_alpha, self.max_alpha))

        # EMA smoothing
        smoothed = alpha * new_c + (1.0 - alpha) * self._prev_smoothed
        self._prev_smoothed = smoothed.copy()
        self._prev_source = path_source
        return smoothed.astype(np.float64)

    def reset(self) -> None:
        """Reset smoother state (e.g. after large scene change)."""
        self._prev_smoothed = None
        self._prev_source = ""
        self._initialized = False


# ---------------------------------------------------------------------------
# HeadingTemporalFilter
# ---------------------------------------------------------------------------

class HeadingTemporalFilter:
    """
    Research impl: Circular EMA on heading angle (degrees).

    Prevents single-frame heading spikes from triggering wrong command
    classifications (STRAIGHT / LEFT / RIGHT / SHARP).

    Uses circular delta to handle the ±180° wrap-around:
      delta = (heading_deg - prev) wrapped to [-180, 180]
      smoothed = prev + alpha_h * delta

    Reset condition: |delta| > HEADING_SMOOTH_RESET_DEG (default 45°)
    This fires on topology flips (skeletonization reversal, path source change)
    and prevents the filter from averaging across a discontinuity.

    Papers: Regulated Pure Pursuit smoother (arXiv:2305.20026)
    """

    def __init__(
        self,
        alpha: float = HEADING_SMOOTH_ALPHA,
        reset_deg: float = HEADING_SMOOTH_RESET_DEG,
        enabled: bool = HEADING_SMOOTH_ENABLED,
    ) -> None:
        self.alpha = float(alpha)
        self.reset_deg = float(reset_deg)
        self.enabled = bool(enabled)

        self._prev_heading: Optional[float] = None
        self._initialized: bool = False

    def update(self, heading_deg: float, confidence: float = 1.0) -> float:
        """
        Apply circular EMA to incoming heading angle.

        Parameters
        ----------
        heading_deg : float
            Raw heading angle in degrees. 0 = straight, + = right, - = left.
        confidence : float, range [0, 1]
            Optional confidence to further modulate alpha.
            confidence=1.0 means no modulation (use default alpha).

        Returns
        -------
        float
            Smoothed heading angle in degrees.
        """
        h = float(heading_deg)

        if not self.enabled:
            return h

        if not self._initialized or self._prev_heading is None:
            self._prev_heading = h
            self._initialized = True
            return h

        # Circular delta: wrap to [-180, 180]
        delta = h - self._prev_heading
        delta = float((delta + 180.0) % 360.0 - 180.0)

        # Reset on large heading jump (topology flip, path family change)
        if abs(delta) > self.reset_deg:
            self._prev_heading = h
            return h

        # Confidence-modulated alpha (optional; confidence=1.0 → no change)
        alpha = float(np.clip(self.alpha * float(confidence), 0.1, 0.95))

        smoothed = self._prev_heading + alpha * delta
        # Normalise to [-180, 180]
        smoothed = float((smoothed + 180.0) % 360.0 - 180.0)
        self._prev_heading = smoothed
        return smoothed

    def reset(self) -> None:
        """Reset filter state."""
        self._prev_heading = None
        self._initialized = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_source_category(source: str) -> str:
    """
    Map path_source string to a stable category for reset detection.

    "template" and "fallback_hold" that followed a template are treated as
    the same category (no reset between them). Graph-derived sources form
    a separate category.
    """
    s = str(source).lower().strip()
    if s in {"template"}:
        return "template"
    if s in {"graph", "endpoint_arc", "dt_corridor"}:
        return "graph"
    if s in {"fallback_centerline", "fallback_skeleton", "fallback_hold"}:
        return "fallback"
    return "other"
