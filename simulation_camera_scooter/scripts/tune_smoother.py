#!/usr/bin/env python3
"""
tune_smoother.py
================
Sweep EMA alpha x consistency_thresh parameter grid on the demo video.

Loads raw masks from the best checkpoint once (single GPU pass), then
replays the masks through every (alpha, thresh) combination without
rerunning GPU inference.

Usage
-----
    python scripts/tune_smoother.py --video test_video_mar3_1_h264.mp4
    python scripts/tune_smoother.py --video test_video_mar3_1_h264.mp4 --max-frames 500
    python scripts/tune_smoother.py --help
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/ subdirectory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config as cfg
from fast_road_detector import Config, FastRoadDetector
from stabilization import TemporalMaskSmoother

# ---------------------------------------------------------------------------
# Parameter sweep grid (SEG-03: alpha floor = 0.25)
# ---------------------------------------------------------------------------
ALPHAS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65]
THRESHOLDS = [0.20, 0.25, 0.30, 0.40, 0.50]


def load_raw_masks(video_path: str, max_frames: int) -> list:
    """
    Run the best-checkpoint detector on the video and collect raw uint8 masks.

    Returns a list of np.ndarray (uint8, H x W, values in {0, 255}).
    If fewer than 10 frames are readable, prints an error and exits.
    """
    detector_cfg = Config(
        video_path=video_path,
        model_dir=cfg.MODEL_DIR,
        road_id=cfg.ROAD_ID,
        conf_thresh=0.6,
        frame_step=1,
        use_gpu=True,
        enable_logging=False,
        enable_edge_cleaning=False,
        enable_simple_smoothing=False,
    )

    print(f"Loading model from: {cfg.MODEL_DIR}")
    try:
        detector = FastRoadDetector(detector_cfg)
    except Exception as exc:
        print(f"ERROR: Failed to load FastRoadDetector: {exc}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    raw_masks = []
    frame_idx = 0
    print(f"Collecting raw masks (up to {max_frames} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            mask_255, _ = detector.process_frame(frame)
        except Exception as exc:
            print(f"  [WARN] process_frame error at frame {frame_idx}: {exc}")
            frame_idx += 1
            continue

        raw_masks.append(mask_255)
        frame_idx += 1

        if frame_idx >= max_frames:
            break

    cap.release()

    if len(raw_masks) < 10:
        print(
            f"ERROR: Only {len(raw_masks)} frames collected — video not readable "
            f"or too short (need >= 10).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Collected {len(raw_masks)} raw masks.\n")
    return raw_masks


def evaluate_combination(
    raw_masks: list,
    alpha: float,
    consistency_thresh: float,
) -> dict:
    """
    Replay stored raw masks through a fresh TemporalMaskSmoother and
    compute stability metrics.

    Returns:
        pct_stable:  % of frames (excl. first) where returned IoU >= 0.85
        pct_failure: % of frames where returned IoU == 0.0
    """
    smoother = TemporalMaskSmoother(alpha=alpha, consistency_thresh=consistency_thresh)

    n_stable = 0
    n_failure = 0
    n_total = 0

    for i, mask in enumerate(raw_masks):
        _, iou = smoother.smooth(mask, return_iou=True)

        if i == 0:
            # First frame: smoother returns iou=1.0 by convention (fast-path init)
            # Skip — no meaningful comparison yet
            continue

        n_total += 1
        if iou >= 0.85:
            n_stable += 1
        if iou == 0.0:
            n_failure += 1

    pct_stable = 100.0 * n_stable / n_total if n_total > 0 else 0.0
    pct_failure = 100.0 * n_failure / n_total if n_total > 0 else 0.0

    return {"pct_stable": pct_stable, "pct_failure": pct_failure, "n_frames": n_total}


def print_grid_table(results: list) -> None:
    """Print sweep results sorted by pct_stable descending."""
    header = f"{'alpha':>6} | {'c_thresh':>8} | {'pct_stable':>10} | {'pct_failure':>11}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['alpha']:>6.2f} | "
            f"{r['thresh']:>8.2f} | "
            f"{r['pct_stable']:>9.1f}% | "
            f"{r['pct_failure']:>10.1f}%"
        )
    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep EMA alpha x consistency_thresh to maximize smoothed segmentation stability."
    )
    parser.add_argument(
        "--video",
        default=os.path.join(_REPO_ROOT, "test_video_mar3_1_h264.mp4"),
        help="Path to demo video (default: test_video_mar3_1_h264.mp4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to collect for the sweep (default: 500)",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        candidate = os.path.join(_REPO_ROOT, video_path)
        if os.path.isfile(candidate):
            video_path = candidate
        else:
            print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Tune smoother settings:")
    print(f"  video     : {video_path}")
    print(f"  max_frames: {args.max_frames}")
    print(f"  alphas    : {ALPHAS}")
    print(f"  thresholds: {THRESHOLDS}")
    print(f"  grid size : {len(ALPHAS) * len(THRESHOLDS)} combinations")
    print()

    # --- Step 1: Single GPU pass to collect raw masks ---
    raw_masks = load_raw_masks(video_path, args.max_frames)

    # --- Step 2: Sweep (CPU-only replay, no GPU needed) ---
    print("Running parameter sweep...")
    results = []
    for alpha in ALPHAS:
        for thresh in THRESHOLDS:
            stats = evaluate_combination(raw_masks, alpha, thresh)
            results.append(
                {
                    "alpha": alpha,
                    "thresh": thresh,
                    "pct_stable": stats["pct_stable"],
                    "pct_failure": stats["pct_failure"],
                    "n_frames": stats["n_frames"],
                }
            )

    # Sort by pct_stable descending; tie-break by alpha descending (better responsiveness)
    results.sort(key=lambda r: (r["pct_stable"], r["alpha"]), reverse=True)

    # --- Step 3: Print grid table ---
    print_grid_table(results)

    # --- Step 4: Report best params ---
    best = results[0]
    best_alpha = best["alpha"]
    best_thresh = best["thresh"]
    best_pct = best["pct_stable"]

    print(
        f"BEST PARAMS: alpha={best_alpha}, consistency_thresh={best_thresh} "
        f"=> {best_pct:.1f}% stable"
    )

    if best_pct >= 90.0:
        print("SEG-01 TARGET MET")
    else:
        print(
            f"BEST ACHIEVED: {best_pct:.1f}% — below 90% target. "
            f"Document in SUMMARY."
        )

    # --- Step 5: SEG-03 constraint check ---
    if best_alpha >= 0.25:
        print(f"SEG-03 CHECK: alpha={best_alpha} >= 0.25 — PASS")
    else:
        print(f"SEG-03 CHECK: alpha={best_alpha} < 0.25 — FAIL — alpha too low")


if __name__ == "__main__":
    main()
