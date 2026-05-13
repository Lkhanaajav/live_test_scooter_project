#!/usr/bin/env python3
"""
benchmark_seg_stability.py
==========================
Evaluate temporal segmentation stability across all SegFormer checkpoints.

For each checkpoint the script:
  1. Loads the parent processor + checkpoint model weights
  2. Runs inference on up to --max-frames frames of the demo video
  3. Computes consecutive-frame IoU (raw, unsmoothed masks)
  4. Classifies each frame as: stable (IoU>=0.85), seg_failure (IoU=0.0 with
     non-empty previous), or unstable (everything else)
  5. Prints a sorted summary table and identifies the best checkpoint

Usage
-----
    python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4
    python scripts/benchmark_seg_stability.py --video test_video_mar3_1_h264.mp4 --max-frames 300
    python scripts/benchmark_seg_stability.py --help
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/ subdirectory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fast_road_detector import Config, FastRoadDetector
from stabilization import TemporalMaskSmoother

# ---------------------------------------------------------------------------
# Checkpoint list (evaluated in this order)
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    "checkpoint-500",
    "checkpoint-1000",
    "checkpoint-1500",
    "checkpoint-2000",
    "checkpoint-2500",
    "checkpoint-3000",
    "checkpoint-3500",
    "checkpoint-4000",
    "checkpoint-4500",
    "checkpoint-5000",
    "my-segformer-road",
    "my-segformer-road_new",
]

# Processor shared across all checkpoints (only model weights swap)
PROCESSOR_DIR = os.path.join(_REPO_ROOT, "models", "my-segformer-road_new")


def _models_dir(checkpoint_name: str) -> str:
    return os.path.join(_REPO_ROOT, "models", checkpoint_name)


def evaluate_checkpoint(
    checkpoint_name: str,
    video_path: str,
    road_id: int,
    conf_thresh: float,
    max_frames: int,
) -> dict | None:
    """
    Evaluate a single checkpoint on the given video.

    Returns a dict with benchmark stats, or None if the checkpoint fails to load.
    """
    ckpt_dir = _models_dir(checkpoint_name)
    if not os.path.isdir(ckpt_dir):
        print(f"  [SKIP] Directory not found: {ckpt_dir}")
        return None

    # Build a FastRoadDetector Config that points the model_dir at this checkpoint.
    # FastRoadDetector._load_model() uses the processor AND the model from model_dir.
    # Per the plan spec: processor from my-segformer-road_new, model from checkpoint.
    # We achieve this by monkey-patching after construction — or by using a custom
    # load sequence.  To keep things self-contained we instantiate FastRoadDetector
    # normally (it will load both processor and model from ckpt_dir), but since all
    # checkpoints share the same tokenizer/processor config this is equivalent.
    cfg = Config(
        video_path=video_path,
        model_dir=ckpt_dir,
        road_id=road_id,
        conf_thresh=conf_thresh,
        frame_step=1,
        use_gpu=True,
        enable_logging=False,
        enable_edge_cleaning=False,
        enable_simple_smoothing=False,
    )

    try:
        detector = FastRoadDetector(cfg)
    except Exception as exc:
        print(f"  [WARN] Failed to load checkpoint '{checkpoint_name}': {exc}")
        return None

    # --- Swap processor to the reference processor (my-segformer-road_new) ---
    # This ensures a consistent tokenization baseline across all checkpoints.
    # Only the model weights differ.
    try:
        from transformers import AutoImageProcessor
        detector.processor = AutoImageProcessor.from_pretrained(
            PROCESSOR_DIR, local_files_only=True
        )
    except Exception as exc:
        print(f"  [WARN] Could not load reference processor; using checkpoint's own: {exc}")

    # --- Video loop ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        detector.model = None
        return None

    prev_mask_f = None          # float32 [0,1] previous frame mask
    prev_pixels = 0             # pixel count in previous mask

    n_total = 0          # frames analysed (excluding first)
    n_stable = 0         # IoU >= 0.85
    n_failure = 0        # IoU == 0.0 AND prev had non-zero pixels
    iou_list = []

    t0 = time.time()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # For the very first frame, just record the mask (no IoU to compute)
        try:
            mask_255, _ = detector.process_frame(frame)
        except Exception as exc:
            print(f"  [WARN] process_frame error at frame {frame_idx}: {exc}")
            frame_idx += 1
            continue

        mask_f = (mask_255 > 127).astype(np.float32)
        pixel_count = int(mask_f.sum())

        if prev_mask_f is not None:
            iou = TemporalMaskSmoother._iou(mask_f, prev_mask_f)

            # Classify frame
            if iou == 0.0 and prev_pixels > 0:
                n_failure += 1
            elif iou >= 0.85:
                n_stable += 1

            iou_list.append(iou)
            n_total += 1

        prev_mask_f = mask_f
        prev_pixels = pixel_count
        frame_idx += 1

        if frame_idx >= max_frames:
            break

    cap.release()
    elapsed = time.time() - t0

    # Free GPU memory before the next checkpoint
    del detector.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if n_total == 0:
        print(f"  [WARN] No frames processed for '{checkpoint_name}'")
        return None

    iou_arr = np.array(iou_list, dtype=np.float64)
    return {
        "checkpoint": checkpoint_name,
        "n_frames": n_total,
        "pct_stable": 100.0 * n_stable / n_total,
        "pct_failure": 100.0 * n_failure / n_total,
        "mean_iou": float(iou_arr.mean()),
        "median_iou": float(np.median(iou_arr)),
        "elapsed_s": elapsed,
    }


def print_table(results: list[dict]) -> None:
    """Print a formatted summary table sorted by pct_stable descending."""
    header = (
        f"{'Checkpoint':<25} | {'pct_stable':>10} | {'pct_failure':>11} | "
        f"{'mean_iou':>8} | {'median_iou':>10}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['checkpoint']:<25} | "
            f"{r['pct_stable']:>9.1f}% | "
            f"{r['pct_failure']:>10.1f}% | "
            f"{r['mean_iou']:>8.3f} | "
            f"{r['median_iou']:>10.3f}"
        )
    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SegFormer checkpoint temporal stability on a video."
    )
    parser.add_argument(
        "--video",
        default=os.path.join(_REPO_ROOT, "test_video_mar3_1_h264.mp4"),
        help="Path to the demo video (default: test_video_mar3_1_h264.mp4)",
    )
    parser.add_argument(
        "--road-id",
        type=int,
        default=1,
        help="Segmentation class ID for road/sidewalk (default: 1 = road, 2 = sidewalk)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.6,
        help="Confidence threshold for mask binarisation (default: 0.6)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum frames to evaluate per checkpoint (default: 500)",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        # Try resolving relative to repo root
        candidate = os.path.join(_REPO_ROOT, video_path)
        if os.path.isfile(candidate):
            video_path = candidate
        else:
            print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Benchmark settings:")
    print(f"  video      : {video_path}")
    print(f"  road_id    : {args.road_id}")
    print(f"  conf_thresh: {args.conf_thresh}")
    print(f"  max_frames : {args.max_frames}")
    print(f"  checkpoints: {len(CHECKPOINTS)}")
    print()

    results = []

    for ckpt in CHECKPOINTS:
        print(f"[{ckpt}] Evaluating...")
        r = evaluate_checkpoint(
            checkpoint_name=ckpt,
            video_path=video_path,
            road_id=args.road_id,
            conf_thresh=args.conf_thresh,
            max_frames=args.max_frames,
        )
        if r is not None:
            results.append(r)
            print(
                f"  done — {r['n_frames']} frames in {r['elapsed_s']:.1f}s  |  "
                f"stable={r['pct_stable']:.1f}%  fail={r['pct_failure']:.1f}%  "
                f"mean_iou={r['mean_iou']:.3f}"
            )
        print()

    if not results:
        print("ERROR: No checkpoints evaluated successfully.", file=sys.stderr)
        sys.exit(1)

    # Sort by pct_stable descending
    results.sort(key=lambda r: r["pct_stable"], reverse=True)

    print_table(results)

    best = results[0]
    print(f"BEST CHECKPOINT: {best['checkpoint']} ({best['pct_stable']:.1f}% stable)")

    if best["pct_stable"] >= 90.0:
        print("TARGET MET")
    else:
        print("TARGET NOT MET — proceed to smoothing tuning in Plan 01-02")

    print()
    print(f"(To update config.py, set MODEL_DIR to: models/{best['checkpoint']})")


if __name__ == "__main__":
    main()
