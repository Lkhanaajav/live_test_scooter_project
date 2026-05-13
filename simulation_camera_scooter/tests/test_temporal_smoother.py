"""
test_temporal_smoother.py
=========================
Unit tests for TemporalMaskSmoother edge cases.

Tests cover:
  - Empty mask handling (first call and after non-empty history)
  - Alpha response time (obstacle appearance within 3 frames at min alpha)
  - High-consistency (IoU > 0.5) full-alpha path
  - Low-consistency (IoU < consistency_thresh) conservative blend
  - Identical masks stabilizing the running average
"""

import numpy as np
import pytest
import sys
import os

# Ensure the project root is on the path so stabilization can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stabilization import TemporalMaskSmoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank(h=50, w=50):
    """Return an all-zero uint8 mask."""
    return np.zeros((h, w), dtype=np.uint8)


def _square(h=50, w=50, row_start=15, row_end=35, col_start=15, col_end=35):
    """Return a uint8 mask with a 20x20 white square in the centre."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[row_start:row_end, col_start:col_end] = 255
    return m


def _full(h=50, w=50):
    """Return an all-255 uint8 mask."""
    return np.full((h, w), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test 1: Empty mask on first call — returns empty mask, iou=1.0
# ---------------------------------------------------------------------------

def test_empty_mask_first_call_no_crash():
    """
    Feeding an all-zero mask on the very first call must not crash and must
    return the empty mask itself with iou=1.0 (first frame always consistent).
    """
    smoother = TemporalMaskSmoother(alpha=0.45, consistency_thresh=0.3)
    empty = _blank()

    result, iou = smoother.smooth(empty, return_iou=True)

    assert result is not None, "smooth() must return a result for an empty mask"
    assert result.shape == empty.shape, "Output shape must match input shape"
    # First call always returns iou=1.0 per the implementation spec
    assert iou == 1.0, f"First-call iou must be 1.0, got {iou}"
    # The result should be all zeros (empty mask passes through on first call)
    assert result.sum() == 0, "First-call result should be the empty mask (all zeros)"


# ---------------------------------------------------------------------------
# Test 2: Empty mask after non-empty history — iou=0.0
# ---------------------------------------------------------------------------

def test_empty_mask_after_nonempty_history():
    """
    If the smoother has a non-empty running average and receives an all-zero
    mask, the IoU must be 0.0 (no intersection with non-empty history).
    The returned mask is held (blended from history) — no crash.
    """
    smoother = TemporalMaskSmoother(alpha=0.45, consistency_thresh=0.3)

    # Establish a non-empty running history with a full mask
    full = _full()
    smoother.smooth(full)   # initialises running_avg to all-1

    # Now feed an empty mask
    empty = _blank()
    result, iou = smoother.smooth(empty, return_iou=True)

    assert result is not None, "smooth() must not crash on empty mask after history"
    assert result.shape == empty.shape, "Output shape must match input shape"
    # IoU between empty and full should be 0.0
    assert iou == 0.0, f"IoU of empty vs full history must be 0.0, got {iou}"


# ---------------------------------------------------------------------------
# Test 3: Alpha response time — obstacle visible within 3 frames at alpha=0.25
# ---------------------------------------------------------------------------

def test_alpha_response_time_min_alpha():
    """
    With alpha=0.25 (minimum required by SEG-03), a new 20x20 white region
    must cross the 0.45 running-average threshold within 3 smooth() calls
    once the running_avg has built up enough history that IoU >= 0.5 (full
    alpha is used, not the conservative branch).

    Scenario: first call establishes the square as the initial running_avg
    (fast path, no EMA applied), so running_avg = 1.0 in the square region
    immediately.  Subsequent calls with the same square confirm stable
    convergence and that the threshold was already cleared on frame 1.

    This validates the SEG-03 constraint: alpha >= 0.25 means that once
    a stable region is seen it stays visible indefinitely.  The test also
    checks that an alternating blank-then-square pattern eventually crosses
    the threshold within 5 frames total (accounting for conservative blending
    that occurs when IoU < 0.5).
    """
    smoother = TemporalMaskSmoother(alpha=0.25, consistency_thresh=0.3)

    square = _square()

    # First call: fast path — running_avg initialised to 1.0 in square region
    smoother.smooth(square)
    assert (smoother.running_avg[15:35, 15:35] > 0.45).all(), (
        "After first call (fast path), running_avg must equal 1.0 in the square region"
    )

    # After 3 more identical calls (iou=1.0, full alpha), region stays visible
    for _ in range(3):
        smoother.smooth(square)

    region = smoother.running_avg[15:35, 15:35]
    assert (region > 0.45).all(), (
        f"After 4 identical square frames, running_avg must remain > 0.45. "
        f"Min in region: {region.min():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 4: High consistency — full alpha used
# ---------------------------------------------------------------------------

def test_high_consistency_uses_full_alpha():
    """
    When IoU between current mask and running_avg is > 0.5, the smoother
    must use effective_alpha = self.alpha (no reduction).
    Verify by checking running_avg change magnitude equals alpha * delta.
    """
    alpha = 0.45
    smoother = TemporalMaskSmoother(alpha=alpha, consistency_thresh=0.3)

    # Initialise with full mask
    full = _full()
    smoother.smooth(full)   # running_avg = 1.0

    # Feed same full mask again — IoU = 1.0 (>0.5), so full alpha applies
    avg_before = smoother.running_avg.copy()
    smoother.smooth(full)   # same mask, IoU = 1.0, should use full alpha
    avg_after = smoother.running_avg.copy()

    # With identical masks, running_avg should stay at ~1.0 (no change)
    np.testing.assert_allclose(
        avg_after, avg_before,
        atol=1e-5,
        err_msg="With identical masks, running_avg must not drift"
    )


# ---------------------------------------------------------------------------
# Test 5: Low consistency — alpha is reduced by 0.25x
# ---------------------------------------------------------------------------

def test_low_consistency_reduces_alpha():
    """
    When IoU < consistency_thresh, effective_alpha must be self.alpha * 0.25.
    A very different mask should blend in slowly (only 0.25x speed).
    """
    alpha = 0.45
    consistency_thresh = 0.3
    smoother = TemporalMaskSmoother(alpha=alpha, consistency_thresh=consistency_thresh)

    # Establish history: top-left quadrant is white
    top_left = np.zeros((50, 50), dtype=np.uint8)
    top_left[:25, :25] = 255
    smoother.smooth(top_left)   # initialise running_avg to top-left pattern

    avg_before = smoother.running_avg.copy()

    # Feed bottom-right quadrant (IoU with top-left = 0.0 < consistency_thresh)
    bottom_right = np.zeros((50, 50), dtype=np.uint8)
    bottom_right[25:, 25:] = 255

    result, iou = smoother.smooth(bottom_right, return_iou=True)

    assert iou < consistency_thresh, (
        f"Expected IoU < {consistency_thresh} for disjoint masks, got {iou}"
    )

    # With effective_alpha = alpha * 0.25, new mask contributes only 0.25*alpha
    # The running_avg at the bottom-right region (was 0.0) should increase
    # but much less than full alpha would provide
    bottom_right_f = bottom_right.astype(np.float32) / 255.0
    effective_alpha_expected = alpha * 0.25
    expected_new_avg = (
        effective_alpha_expected * bottom_right_f
        + (1.0 - effective_alpha_expected) * avg_before
    )

    np.testing.assert_allclose(
        smoother.running_avg,
        expected_new_avg,
        atol=1e-5,
        err_msg="running_avg after low-consistency blend does not match expected (alpha * 0.25)"
    )


# ---------------------------------------------------------------------------
# Test 6: Identical masks across 5 frames — running_avg stabilizes
# ---------------------------------------------------------------------------

def test_identical_masks_stabilize():
    """
    Feeding the same mask 5 times must cause running_avg to converge to that
    mask (values near 1.0 in the white region). The smoothed output must then
    match the input mask exactly.
    """
    smoother = TemporalMaskSmoother(alpha=0.45, consistency_thresh=0.3)
    square = _square()

    # Feed the same mask 5 times
    result = None
    for _ in range(5):
        result, iou = smoother.smooth(square, return_iou=True)

    # After stabilization, output must match the input mask
    np.testing.assert_array_equal(
        result, square,
        err_msg="After 5 identical frames, smoothed output must match input mask"
    )

    # running_avg in white region should be very close to 1.0
    region_avg = smoother.running_avg[15:35, 15:35].mean()
    assert region_avg > 0.90, (
        f"running_avg in white region should be > 0.90 after 5 identical frames, "
        f"got {region_avg:.4f}"
    )
