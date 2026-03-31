#!/usr/bin/env python3
"""
Fresh Thesis Evaluation Framework
==================================

Clean-room implementation for generating ALL thesis experimental data.
Does NOT reuse any existing eval scripts.

Produces per-frame CSVs suitable for:
  - Statistical tests (Friedman, Wilcoxon, bootstrap CI)
  - Table generation (mean ± SD)
  - Figure generation (box plots, violin plots, time series)

Modes:
  1. seg_compare     — Compare baseline vs candidate segmentation on GT masks
  2. planner_compare — Run all 5 planners on GT-masked frames
  3. bev_fragility   — Measure BEV path extraction success rate on full videos
  4. template_vs_skeleton — Head-to-head on full videos (no GT needed)
  5. bev_occupancy   — Measure BEV mask occupancy distribution
  6. full_replay     — Full pipeline replay for temporal stability

Usage:
    python scripts/thesis_eval_fresh.py seg_compare
    python scripts/thesis_eval_fresh.py planner_compare
    python scripts/thesis_eval_fresh.py bev_fragility --videos VID_017 IMG_1878
    python scripts/thesis_eval_fresh.py template_vs_skeleton --max-frames 500
    python scripts/thesis_eval_fresh.py bev_occupancy
    python scripts/thesis_eval_fresh.py full_replay --model candidate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent

# Add simulation_camera_scooter to path
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

# ── Lazy imports (resolved after sys.path setup) ─────────────────────────

def _import_modules():
    """Import project modules. Called once at startup."""
    import config as cfg
    from bev_calibration import load_bev_params
    from fast_road_detector import Config as DetectorConfig, FastRoadDetector
    from masks import clean_bev_mask_enhanced
    from template_path_planner import (
        CorridorConfig,
        TemplatePlannerConfig,
        approve_template_bank,
        corridor_from_mask,
    )
    return {
        "cfg": cfg,
        "load_bev_params": load_bev_params,
        "DetectorConfig": DetectorConfig,
        "FastRoadDetector": FastRoadDetector,
        "clean_bev_mask_enhanced": clean_bev_mask_enhanced,
        "CorridorConfig": CorridorConfig,
        "TemplatePlannerConfig": TemplatePlannerConfig,
        "approve_template_bank": approve_template_bank,
        "corridor_from_mask": corridor_from_mask,
    }


# ── Output directories ───────────────────────────────────────────────────

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "thesis_eval_fresh"
VIDEOS_DIR = SIM_DIR / "test_videos"
GT_IMAGES_DIR = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "images"
GT_MASKS_DIR = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "masks"

BASELINE_MODEL = SIM_DIR / "models" / "my-segformer-road"
CANDIDATE_MODEL = PROJECT_ROOT / "outputs" / "training" / "binary_segformer_oneformer_teacher" / "best_checkpoint"


# ── Data classes for results ──────────────────────────────────────────────

@dataclass
class SegFrameResult:
    """Per-frame segmentation metrics against ground truth."""
    video: str
    frame: str
    model: str  # "baseline" or "candidate"
    iou: float
    precision: float
    recall: float
    f1: float
    infer_ms: float
    pred_pixels: int
    gt_pixels: int

@dataclass
class PlannerFrameResult:
    """Per-frame planner metrics against ground truth."""
    video: str
    frame: str
    mask_type: str      # "predicted" or "oracle"
    planner: str        # "bev_skeleton", "bev_dt_full", "bev_dt_near", "img_midpoint", "img_dt"
    has_path: bool
    inside_gt_ratio: float
    center_error_px: float
    forward_span_px: float
    runtime_ms: float
    path_source: str
    path_points: int

@dataclass
class BevFragilityResult:
    """Per-frame BEV path extraction outcome."""
    video: str
    frame_idx: int
    model: str
    has_path: bool
    path_source: str
    mask_occ_ratio: float
    bev_mask_pixels: int

@dataclass
class TemplateVsSkeletonResult:
    """Per-frame template vs skeleton heading comparison."""
    video: str
    frame_idx: int
    template_heading_deg: float
    template_confidence: float
    template_family: str
    skeleton_heading_deg: float
    skeleton_has_path: bool
    heading_delta_deg: float

@dataclass
class BevOccupancyResult:
    """Per-frame BEV occupancy measurement."""
    video: str
    frame_idx: int
    model: str
    camera_mask_pixels: int
    camera_mask_ratio: float
    bev_mask_pixels: int
    bev_mask_ratio: float
    bev_total_pixels: int


# ── Utility functions ─────────────────────────────────────────────────────

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_precision_recall_f1(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for binary masks."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_center_error_and_alignment(
    path_px: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute lateral center error and inside-GT ratio for a path.

    Args:
        path_px: [N, 2] path points in pixel coordinates (x, y).
        gt_mask: Binary GT mask (H, W), >0 = road.

    Returns:
        (mean_center_error_px, inside_gt_ratio, forward_span_px)
    """
    if path_px is None or len(path_px) == 0:
        return float("nan"), 0.0, 0.0

    h, w = gt_mask.shape[:2]
    gt_bin = gt_mask > 0
    inside_count = 0
    center_errors = []

    for pt in path_px:
        px, py = int(round(pt[0])), int(round(pt[1]))
        # Clamp to image bounds
        py_c = max(0, min(py, h - 1))
        px_c = max(0, min(px, w - 1))

        # Inside GT check
        if gt_bin[py_c, px_c]:
            inside_count += 1

        # Compute center error: distance from path point to row center of GT
        row = gt_bin[py_c, :]
        if row.any():
            cols = np.where(row)[0]
            gt_center = (cols[0] + cols[-1]) / 2.0
            center_errors.append(abs(px - gt_center))

    inside_ratio = inside_count / len(path_px) if len(path_px) > 0 else 0.0
    mean_center_err = float(np.mean(center_errors)) if center_errors else float("nan")
    ys = path_px[:, 1]
    forward_span = float(ys.max() - ys.min()) if len(ys) > 1 else 0.0

    return mean_center_err, inside_ratio, forward_span


def load_segmentation_model(model_dir: Path, use_gpu: bool = True):
    """Load a SegFormer model for inference."""
    mods = _import_modules()
    config = mods["DetectorConfig"](
        model_dir=str(model_dir),
        use_gpu=use_gpu,
        enable_logging=False,
        enable_simple_smoothing=False,
    )
    return mods["FastRoadDetector"](config)


def segment_frame(
    detector, frame_bgr: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run segmentation and return binary mask + inference time in ms."""
    t0 = time.perf_counter()
    mask, _ = detector.process_frame(frame_bgr, return_overlay=False)
    t1 = time.perf_counter()
    return mask, (t1 - t0) * 1000.0


def frame_to_bev(
    mask: np.ndarray,
    H: np.ndarray,
    bev_size: Tuple[int, int],
) -> np.ndarray:
    """Warp a camera-space mask to BEV using homography."""
    bev = cv2.warpPerspective(mask, H, bev_size, flags=cv2.INTER_NEAREST)
    return bev


def write_csv(path: Path, rows: List[dict]) -> None:
    """Write a list of dicts to CSV."""
    if not rows:
        print(f"  [skip] No data to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {path} ({len(rows)} rows)")


def find_videos(names: Optional[List[str]] = None) -> List[Path]:
    """Find video files in test_videos directory."""
    if not VIDEOS_DIR.exists():
        print(f"  [error] Videos directory not found: {VIDEOS_DIR}")
        return []
    all_vids = sorted(
        p for p in VIDEOS_DIR.iterdir()
        if p.suffix.lower() in (".mov", ".mp4", ".avi")
    )
    if names:
        name_set = set(names)
        return [v for v in all_vids if v.stem in name_set or any(n in v.stem for n in name_set)]
    return all_vids


def find_gt_frames() -> List[Tuple[str, str, Path, Path]]:
    """Find all ground truth frame pairs (image + mask).

    Returns:
        List of (video_name, frame_name, image_path, mask_path)
    """
    if not GT_IMAGES_DIR.exists() or not GT_MASKS_DIR.exists():
        print(f"  [warn] GT directories not found:")
        print(f"         images: {GT_IMAGES_DIR}")
        print(f"         masks:  {GT_MASKS_DIR}")
        print(f"         Create these with hand-annotated masks to enable GT evaluation.")
        return []

    frames = []
    for video_dir in sorted(GT_IMAGES_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        mask_dir = GT_MASKS_DIR / video_dir.name
        if not mask_dir.is_dir():
            continue
        for img_path in sorted(video_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                frames.append((video_dir.name, img_path.stem, img_path, mask_path))
    return frames


# ══════════════════════════════════════════════════════════════════════════
# MODE 1: Segmentation Comparison (requires GT masks)
# ══════════════════════════════════════════════════════════════════════════

def run_seg_compare(args: argparse.Namespace) -> None:
    """Compare baseline vs candidate segmentation on GT-masked frames."""
    print("\n=== Segmentation Comparison ===")
    gt_frames = find_gt_frames()
    if not gt_frames:
        print("  [abort] No GT frames found. Annotate masks first.")
        return

    print(f"  Found {len(gt_frames)} GT frames across {len(set(f[0] for f in gt_frames))} videos")

    models = {
        "baseline": BASELINE_MODEL,
        "candidate": CANDIDATE_MODEL,
    }

    results: List[dict] = []

    for model_name, model_dir in models.items():
        if not model_dir.exists():
            print(f"  [skip] Model not found: {model_dir}")
            continue

        print(f"\n  Loading {model_name} model from {model_dir.name}...")
        detector = load_segmentation_model(model_dir, use_gpu=args.gpu)

        for video, frame_name, img_path, mask_path in gt_frames:
            frame_bgr = cv2.imread(str(img_path))
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if frame_bgr is None or gt_mask is None:
                print(f"  [skip] Could not read {img_path.name}")
                continue

            pred_mask, infer_ms = segment_frame(detector, frame_bgr)

            # Resize GT to match prediction if needed
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            iou = compute_iou(pred_mask, gt_mask)
            prec, rec, f1 = compute_precision_recall_f1(pred_mask, gt_mask)

            r = SegFrameResult(
                video=video,
                frame=frame_name,
                model=model_name,
                iou=round(iou, 6),
                precision=round(prec, 6),
                recall=round(rec, 6),
                f1=round(f1, 6),
                infer_ms=round(infer_ms, 2),
                pred_pixels=int((pred_mask > 0).sum()),
                gt_pixels=int((gt_mask > 0).sum()),
            )
            results.append(asdict(r))

        # Release model memory
        del detector

    out_path = OUTPUT_ROOT / "seg_compare_per_frame.csv"
    write_csv(out_path, results)

    # Print summary
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n  Summary:")
        print(df.groupby("model")[["iou", "precision", "recall", "f1", "infer_ms"]].agg(["mean", "std"]).round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════
# MODE 2: Planner Comparison (requires GT masks)
# ══════════════════════════════════════════════════════════════════════════

def run_planner_compare(args: argparse.Namespace) -> None:
    """Run all 5 planners on GT-masked frames."""
    print("\n=== Planner Comparison ===")
    gt_frames = find_gt_frames()
    if not gt_frames:
        print("  [abort] No GT frames found. Annotate masks first.")
        return

    mods = _import_modules()
    cfg = mods["cfg"]

    print(f"  Found {len(gt_frames)} GT frames")
    print(f"  Loading candidate model for predicted masks...")
    detector = load_segmentation_model(CANDIDATE_MODEL, use_gpu=args.gpu)

    # Load BEV calibration
    H, Hinv, src_pts = mods["load_bev_params"](current_frame_size=None)
    bev_w, bev_h = cfg.BEV_SIZE
    m_per_px = cfg.NAV_BEV_LATERAL_M / bev_w

    results: List[dict] = []

    for video, frame_name, img_path, mask_path in gt_frames:
        frame_bgr = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame_bgr is None or gt_mask is None:
            continue

        # Get predicted mask from candidate model
        pred_mask, _ = segment_frame(detector, frame_bgr)

        # For each mask type (predicted + oracle)
        for mask_type, camera_mask in [("predicted", pred_mask), ("oracle", gt_mask)]:

            # === Image-space planners ===
            try:
                from image_path_planner import CameraMidpointPlanner, CameraDtPlanner

                # Image-space midpoint
                t0 = time.perf_counter()
                mid_planner = CameraMidpointPlanner()
                mid_result = mid_planner.plan(camera_mask)
                t_mid = (time.perf_counter() - t0) * 1000.0

                if mid_result is not None and mid_result.path_px is not None and len(mid_result.path_px) > 0:
                    ce, igr, fspan = compute_center_error_and_alignment(mid_result.path_px, gt_mask)
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="img_midpoint", has_path=True,
                        inside_gt_ratio=round(igr, 4), center_error_px=round(ce, 2),
                        forward_span_px=round(fspan, 1), runtime_ms=round(t_mid, 2),
                        path_source="img_midpoint", path_points=len(mid_result.path_px),
                    )))
                else:
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="img_midpoint", has_path=False,
                        inside_gt_ratio=0.0, center_error_px=float("nan"),
                        forward_span_px=0.0, runtime_ms=round(t_mid, 2),
                        path_source="none", path_points=0,
                    )))

                # Image-space DT
                t0 = time.perf_counter()
                dt_planner = CameraDtPlanner()
                dt_result = dt_planner.plan(camera_mask)
                t_dt = (time.perf_counter() - t0) * 1000.0

                if dt_result is not None and dt_result.path_px is not None and len(dt_result.path_px) > 0:
                    ce, igr, fspan = compute_center_error_and_alignment(dt_result.path_px, gt_mask)
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="img_dt", has_path=True,
                        inside_gt_ratio=round(igr, 4), center_error_px=round(ce, 2),
                        forward_span_px=round(fspan, 1), runtime_ms=round(t_dt, 2),
                        path_source="img_dt", path_points=len(dt_result.path_px),
                    )))
                else:
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="img_dt", has_path=False,
                        inside_gt_ratio=0.0, center_error_px=float("nan"),
                        forward_span_px=0.0, runtime_ms=round(t_dt, 2),
                        path_source="none", path_points=0,
                    )))
            except ImportError as e:
                print(f"  [warn] Image planners not available: {e}")

            # === BEV planners ===
            # Warp mask to BEV
            bev_mask_raw = frame_to_bev(camera_mask, H, (bev_w, bev_h))
            bev_mask = mods["clean_bev_mask_enhanced"](bev_mask_raw, m_per_px=m_per_px)

            # BEV skeleton-graph
            from realtime_nav_core import BEVPathExtractor, PathExtractorConfig
            try:
                skel_cfg = PathExtractorConfig(
                    bev_forward_m=cfg.NAV_BEV_FORWARD_M,
                    bev_lateral_m=cfg.NAV_BEV_LATERAL_M,
                    use_dt_planner=False,
                    waypoint_turn_enabled=False,
                )
                skel_ext = BEVPathExtractor(skel_cfg)
                t0 = time.perf_counter()
                skel_result = skel_ext.process(bev_mask)
                t_skel = (time.perf_counter() - t0) * 1000.0

                # Convert BEV path to camera for GT comparison
                if skel_result.has_path and skel_result.best_path_px is not None:
                    bev_pts = skel_result.best_path_px.astype(np.float32)
                    cam_pts = cv2.perspectiveTransform(
                        bev_pts.reshape(-1, 1, 2), Hinv
                    ).reshape(-1, 2)
                    ce, igr, fspan = compute_center_error_and_alignment(cam_pts, gt_mask)
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="bev_skeleton", has_path=True,
                        inside_gt_ratio=round(igr, 4), center_error_px=round(ce, 2),
                        forward_span_px=round(fspan, 1), runtime_ms=round(t_skel, 2),
                        path_source=skel_result.path_source, path_points=len(bev_pts),
                    )))
                else:
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="bev_skeleton", has_path=False,
                        inside_gt_ratio=0.0, center_error_px=float("nan"),
                        forward_span_px=0.0, runtime_ms=round(t_skel, 2),
                        path_source="none", path_points=0,
                    )))
            except Exception as e:
                print(f"  [warn] BEV skeleton failed on {frame_name}: {e}")

            # BEV template-approval
            try:
                corridor = mods["corridor_from_mask"](bev_mask)
                t0 = time.perf_counter()
                tpl_result = mods["approve_template_bank"](
                    corridor,
                    prev_path_m=None,
                    prev_template_family=None,
                    gps_intent_family="straight",
                )
                t_tpl = (time.perf_counter() - t0) * 1000.0

                if tpl_result.approved and tpl_result.selected_template is not None:
                    tpl_pts_m = tpl_result.selected_template.pts_m
                    # Convert metric to BEV px then to camera
                    bev_pts = np.column_stack([
                        cfg.bev_ego_x_px(bev_w) + tpl_pts_m[:, 1] * cfg.BEV_PIXELS_PER_METER_LATERAL,
                        bev_h - tpl_pts_m[:, 0] * cfg.BEV_PIXELS_PER_METER_FORWARD,
                    ]).astype(np.float32)
                    cam_pts = cv2.perspectiveTransform(
                        bev_pts.reshape(-1, 1, 2), Hinv
                    ).reshape(-1, 2)
                    ce, igr, fspan = compute_center_error_and_alignment(cam_pts, gt_mask)
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="bev_template", has_path=True,
                        inside_gt_ratio=round(igr, 4), center_error_px=round(ce, 2),
                        forward_span_px=round(fspan, 1), runtime_ms=round(t_tpl, 2),
                        path_source="template", path_points=len(tpl_pts_m),
                    )))
                else:
                    results.append(asdict(PlannerFrameResult(
                        video=video, frame=frame_name, mask_type=mask_type,
                        planner="bev_template", has_path=False,
                        inside_gt_ratio=0.0, center_error_px=float("nan"),
                        forward_span_px=0.0, runtime_ms=round(t_tpl, 2),
                        path_source="none", path_points=0,
                    )))
            except Exception as e:
                print(f"  [warn] BEV template failed on {frame_name}: {e}")

    del detector
    out_path = OUTPUT_ROOT / "planner_compare_per_frame.csv"
    write_csv(out_path, results)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n  Summary (predicted masks):")
        pred_df = df[df["mask_type"] == "predicted"]
        print(pred_df.groupby("planner")[["center_error_px", "inside_gt_ratio", "runtime_ms"]].agg(["mean", "std"]).round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════
# MODE 3: BEV Fragility (no GT needed)
# ══════════════════════════════════════════════════════════════════════════

def run_bev_fragility(args: argparse.Namespace) -> None:
    """Measure BEV path extraction success rate on full videos."""
    print("\n=== BEV Fragility Analysis ===")
    mods = _import_modules()
    cfg = mods["cfg"]

    videos = find_videos(args.videos)
    if not videos:
        print("  [abort] No videos found.")
        return

    models = {}
    if args.model in ("both", "baseline"):
        models["baseline"] = BASELINE_MODEL
    if args.model in ("both", "candidate"):
        models["candidate"] = CANDIDATE_MODEL

    H, Hinv, _ = mods["load_bev_params"]()
    bev_w, bev_h = cfg.BEV_SIZE
    m_per_px = cfg.NAV_BEV_LATERAL_M / bev_w

    results: List[dict] = []

    for model_name, model_dir in models.items():
        print(f"\n  Model: {model_name} ({model_dir.name})")
        detector = load_segmentation_model(model_dir, use_gpu=args.gpu)

        from realtime_nav_core import BEVPathExtractor, PathExtractorConfig
        bev_cfg = PathExtractorConfig(
            bev_forward_m=cfg.NAV_BEV_FORWARD_M,
            bev_lateral_m=cfg.NAV_BEV_LATERAL_M,
            use_dt_planner=False,
            waypoint_turn_enabled=False,
        )

        for vid_path in videos:
            print(f"  Processing {vid_path.name}...")
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                print(f"    [skip] Cannot open {vid_path.name}")
                continue

            extractor = BEVPathExtractor(bev_cfg)
            frame_idx = 0
            stride = args.stride

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % stride != 0:
                    continue
                if args.max_frames > 0 and frame_idx // stride > args.max_frames:
                    break

                mask, _ = segment_frame(detector, frame)
                bev_raw = frame_to_bev(mask, H, (bev_w, bev_h))
                bev_clean = mods["clean_bev_mask_enhanced"](bev_raw, m_per_px=m_per_px)

                result = extractor.process(bev_clean)

                results.append(asdict(BevFragilityResult(
                    video=vid_path.stem,
                    frame_idx=frame_idx,
                    model=model_name,
                    has_path=result.has_path,
                    path_source=result.path_source,
                    mask_occ_ratio=round(result.mask_occ_ratio, 6),
                    bev_mask_pixels=int((bev_clean > 0).sum()),
                )))

            cap.release()
            processed = len([r for r in results if r["video"] == vid_path.stem and r["model"] == model_name])
            valid = len([r for r in results if r["video"] == vid_path.stem and r["model"] == model_name and r["has_path"]])
            print(f"    {processed} frames, {valid} with valid path ({100*valid/max(processed,1):.1f}%)")

        del detector

    out_path = OUTPUT_ROOT / "bev_fragility_per_frame.csv"
    write_csv(out_path, results)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n  Summary:")
        summary = df.groupby("model").agg(
            total_frames=("has_path", "count"),
            valid_path_pct=("has_path", lambda x: f"{100*x.mean():.1f}%"),
            mean_occ_ratio=("mask_occ_ratio", "mean"),
        )
        print(summary.to_string())


# ══════════════════════════════════════════════════════════════════════════
# MODE 4: BEV Occupancy Distribution (no GT needed)
# ══════════════════════════════════════════════════════════════════════════

def run_bev_occupancy(args: argparse.Namespace) -> None:
    """Measure BEV mask occupancy distribution across all videos."""
    print("\n=== BEV Occupancy Distribution ===")
    mods = _import_modules()
    cfg = mods["cfg"]

    videos = find_videos(args.videos)
    if not videos:
        print("  [abort] No videos found.")
        return

    models = {}
    if args.model in ("both", "baseline"):
        models["baseline"] = BASELINE_MODEL
    if args.model in ("both", "candidate"):
        models["candidate"] = CANDIDATE_MODEL

    H, _, _ = mods["load_bev_params"]()
    bev_w, bev_h = cfg.BEV_SIZE
    bev_total = bev_w * bev_h

    results: List[dict] = []

    for model_name, model_dir in models.items():
        print(f"\n  Model: {model_name}")
        detector = load_segmentation_model(model_dir, use_gpu=args.gpu)

        for vid_path in videos:
            print(f"  Processing {vid_path.name}...")
            cap = cv2.VideoCapture(str(vid_path))
            if not cap.isOpened():
                continue

            frame_idx = 0
            stride = args.stride

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % stride != 0:
                    continue
                if args.max_frames > 0 and frame_idx // stride > args.max_frames:
                    break

                mask, _ = segment_frame(detector, frame)
                cam_pixels = int((mask > 0).sum())
                cam_total = mask.shape[0] * mask.shape[1]

                bev_mask = frame_to_bev(mask, H, (bev_w, bev_h))
                bev_pixels = int((bev_mask > 0).sum())

                results.append(asdict(BevOccupancyResult(
                    video=vid_path.stem,
                    frame_idx=frame_idx,
                    model=model_name,
                    camera_mask_pixels=cam_pixels,
                    camera_mask_ratio=round(cam_pixels / cam_total, 6),
                    bev_mask_pixels=bev_pixels,
                    bev_mask_ratio=round(bev_pixels / bev_total, 6),
                    bev_total_pixels=bev_total,
                )))

            cap.release()

        del detector

    out_path = OUTPUT_ROOT / "bev_occupancy_per_frame.csv"
    write_csv(out_path, results)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n  Summary:")
        print(df.groupby("model")[["camera_mask_ratio", "bev_mask_ratio"]].describe().round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh Thesis Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  seg_compare           Compare baseline vs candidate segmentation (needs GT masks)
  planner_compare       Run all planners on GT frames (needs GT masks)
  bev_fragility         Measure BEV path extraction success rate
  bev_occupancy         Measure BEV mask occupancy distribution

Examples:
  python scripts/thesis_eval_fresh.py seg_compare
  python scripts/thesis_eval_fresh.py bev_fragility --model both --stride 4
  python scripts/thesis_eval_fresh.py bev_occupancy --model candidate --max-frames 500
        """,
    )
    parser.add_argument("mode", choices=[
        "seg_compare", "planner_compare",
        "bev_fragility", "bev_occupancy",
    ])
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU for inference (default: True)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument("--model", choices=["baseline", "candidate", "both"],
                        default="both", help="Which model(s) to evaluate")
    parser.add_argument("--stride", type=int, default=4,
                        help="Frame stride for video processing (default: 4)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0 = all)")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific video names to process")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Thesis Evaluation Framework — Mode: {args.mode}")
    print(f"Output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "seg_compare": run_seg_compare,
        "planner_compare": run_planner_compare,
        "bev_fragility": run_bev_fragility,
        "bev_occupancy": run_bev_occupancy,
    }

    t_start = time.time()
    dispatch[args.mode](args)
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
