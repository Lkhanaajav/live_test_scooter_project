#!/usr/bin/env python3
"""
eval_simple_road.py
===================
Evaluate the simplified simple-road pipeline vs the full baseline on test videos.

Computes per-frame metrics:
  - mask_coverage: fraction of BEV mask that is drivable
  - mask_iou_prev: IoU of mask with previous frame (stability)
  - heading_deg: steering heading angle
  - lateral_m: centerline lateral offset at lookahead
  - path_jump: frame-to-frame heading change (smoothness)

Saves:
  - outputs/baseline/<video>/  — baseline overlay video + metrics JSON
  - outputs/improved/<video>/  — simple-road overlay video + metrics JSON
  - outputs/comparisons/<video>/ — side-by-side comparison video
  - EVALUATION_REPORT.md (updated at end)

Usage:
    cd simulation_camera_scooter
    python scripts/eval_simple_road.py
    python scripts/eval_simple_road.py --videos test_video_june_03_3.mp4
    python scripts/eval_simple_road.py --max-frames 300
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_DIR, BEV_SIZE, TRIM_BOTTOM,
    DEFAULT_SRC_POINTS, DEFAULT_DST_POINTS,
    CALIBRATION_FILE,
    ROAD_ID,
    MASK_SMOOTH_ALPHA, MASK_SMOOTH_CONSISTENCY_THRESH, BEV_SMOOTH_ALPHA,
    NAV_BEV_FORWARD_M, NAV_BEV_LATERAL_M,
    MORPH_ENHANCED,
    DT_CORRIDOR_ENABLED,
    DT_CORRIDOR_SG_WINDOW,
    PATH_SMOOTH_ENABLED,
    HEADING_SMOOTH_ENABLED,
)
from masks import clean_bev_mask_enhanced, clean_sidewalk_mask, select_main_component
from simple_road_pipeline import SimpleRoadConfig, SimpleRoadProcessor
from stabilization import TemporalMaskSmoother

try:
    from safe_corridor import DtSafeCorridor
    _HAS_DT = True
except ImportError:
    _HAS_DT = False

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# BEV helper
# ---------------------------------------------------------------------------

def _load_homography() -> np.ndarray:
    """Load BEV homography from calibration file, or use defaults."""
    if os.path.exists(CALIBRATION_FILE):
        try:
            src_pts = np.load(CALIBRATION_FILE).astype(np.float32)
            if src_pts.shape == (4, 2):
                dst_pts = DEFAULT_DST_POINTS.astype(np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts)
                return H.astype(np.float32)
        except Exception as e:
            print(f"[BEV] Calibration load failed: {e}, using defaults")
    # Default homography
    H, _ = cv2.findHomography(
        DEFAULT_SRC_POINTS.astype(np.float32),
        DEFAULT_DST_POINTS.astype(np.float32),
    )
    return H.astype(np.float32)


def apply_bev(frame: np.ndarray, H: np.ndarray, bev_size=(600, 600)) -> np.ndarray:
    """Warp frame to BEV using homography."""
    return cv2.warpPerspective(frame, H, bev_size)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def load_segformer(model_dir: str, device: str = "cuda"):
    """Load SegFormer model + processor."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    model.eval()
    return model, processor, device


def segment_frame(frame_bgr: np.ndarray, model, processor, device: str, road_id: int = 1, conf_thresh: float = 0.6) -> np.ndarray:
    """Run SegFormer on a frame, return binary mask (0/255)."""
    import torch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(rgb, return_tensors="pt").to(device)
    H, W = frame_bgr.shape[:2]
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)[0]
    probs = up.softmax(0)[road_id].cpu().numpy()
    mask = (probs > conf_thresh).astype(np.uint8) * 255
    return mask


# ---------------------------------------------------------------------------
# Baseline path extraction (using existing DT corridor)
# ---------------------------------------------------------------------------

class BaselineProcessor:
    """
    Baseline: uses MORPH_ENHANCED mask cleaning + DT corridor (current config).
    """

    def __init__(self) -> None:
        self._prev_bev_mask: Optional[np.ndarray] = None
        self._prev_heading: Optional[float] = None
        # BEV m/px
        self._m_per_px = float(NAV_BEV_FORWARD_M) / float(BEV_SIZE[1])
        if _HAS_DT:
            self._dt = DtSafeCorridor(
                bev_forward_m=NAV_BEV_FORWARD_M,
                bev_lateral_m=NAV_BEV_LATERAL_M,
                sg_window=DT_CORRIDOR_SG_WINDOW,
                lateral_drift_px=30,
            )
        else:
            self._dt = None

    def process_bev_mask(self, raw_bev_mask_255: np.ndarray):
        """Process BEV mask with baseline settings."""
        H, W = raw_bev_mask_255.shape

        # Temporal BEV mask smoothing
        bev_smooth = raw_bev_mask_255
        if self._prev_bev_mask is not None and self._prev_bev_mask.shape == bev_smooth.shape:
            iou = _mask_iou(bev_smooth, self._prev_bev_mask)
            if iou >= float(MASK_SMOOTH_CONSISTENCY_THRESH):
                alpha = float(BEV_SMOOTH_ALPHA)
                blended = alpha * (bev_smooth > 0).astype(np.float32) + (1.0 - alpha) * (self._prev_bev_mask > 0).astype(np.float32)
                bev_smooth = (blended > 0.5).astype(np.uint8) * 255
        self._prev_bev_mask = bev_smooth.copy()

        # Mask cleaning
        if MORPH_ENHANCED:
            clean = clean_bev_mask_enhanced(bev_smooth, self._m_per_px)
        else:
            clean = clean_sidewalk_mask(bev_smooth)

        # Coverage and IoU
        coverage = float(np.count_nonzero(clean)) / max(1.0, float(clean.size))
        iou_prev = _mask_iou(clean, self._prev_bev_mask) if self._prev_bev_mask is not None else 1.0

        # Centerline
        heading_deg = 0.0
        lateral_m = 0.0
        confidence = 0.0
        valid = False

        if self._dt is not None and int(np.count_nonzero(clean)) > 100:
            result = self._dt.extract(clean)
            if result.centerline_m is not None and len(result.centerline_m) >= 2:
                cm = result.centerline_m
                n = len(cm)
                i0 = max(0, n - 6)
                vec = cm[-1] - cm[i0]
                import math as _math
                heading_rad = _math.atan2(float(vec[1]), max(1e-6, float(vec[0])))
                heading_deg = _math.degrees(heading_rad)
                lateral_m = float(np.interp(1.5, cm[:, 0], cm[:, 1])) if float(cm[-1, 0]) >= 1.5 else float(cm[-1, 1])
                confidence = float(result.confidence)
                valid = True

        return {
            "clean_mask": clean,
            "coverage": coverage,
            "iou_prev": iou_prev,
            "heading_deg": heading_deg,
            "lateral_m": lateral_m,
            "confidence": confidence,
            "valid": valid,
        }

    def reset(self) -> None:
        self._prev_bev_mask = None
        self._prev_heading = None


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0)
    b_bin = (b > 0)
    inter = int(np.count_nonzero(a_bin & b_bin))
    union = int(np.count_nonzero(a_bin | b_bin))
    return float(inter) / max(1.0, float(union))


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------

class MetricAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.coverage: List[float] = []
        self.iou_prev: List[float] = []
        self.heading: List[float] = []
        self.lateral: List[float] = []
        self.path_jump: List[float] = []  # frame-to-frame heading change
        self._prev_heading: Optional[float] = None

    def update(self, coverage, iou_prev, heading_deg, lateral_m):
        self.coverage.append(float(coverage))
        self.iou_prev.append(float(iou_prev))
        self.heading.append(float(heading_deg))
        self.lateral.append(float(lateral_m))
        if self._prev_heading is not None:
            jump = abs(float(heading_deg) - self._prev_heading)
            if jump > 180.0:
                jump = 360.0 - jump
            self.path_jump.append(jump)
        self._prev_heading = float(heading_deg)

    def summary(self) -> Dict:
        def _stats(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            arr = np.array(vals, dtype=np.float32)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        return {
            "name": self.name,
            "n_frames": len(self.coverage),
            "mask_coverage": _stats(self.coverage),
            "mask_iou_prev": _stats(self.iou_prev),
            "heading_deg": _stats(self.heading),
            "lateral_m": _stats(self.lateral),
            "path_jump_deg": _stats(self.path_jump),
        }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_overlay(
    frame_bgr: np.ndarray,
    bev_frame: np.ndarray,
    clean_mask: np.ndarray,
    centerline_px: Optional[np.ndarray],
    heading_deg: float,
    lateral_m: float,
    template_family: str,
    label: str,
    coverage: float,
    iou_prev: float,
) -> np.ndarray:
    """Draw BEV mask + centerline + stats overlay."""
    bev_vis = bev_frame.copy()
    # Tint clean mask green
    green_overlay = np.zeros_like(bev_vis)
    green_overlay[clean_mask > 0] = (0, 200, 0)
    bev_vis = cv2.addWeighted(bev_vis, 0.6, green_overlay, 0.4, 0)

    # Draw centerline
    if centerline_px is not None and len(centerline_px) >= 2:
        pts = centerline_px[:, [1, 0]].astype(np.int32)  # (col, row) = (x, y)
        for i in range(len(pts) - 1):
            cv2.line(bev_vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 255), 2)

    # Heading indicator at bottom-center
    H, W = bev_vis.shape[:2]
    cx = W // 2
    cy = H - 10
    heading_rad = float(heading_deg) * (3.14159 / 180.0)
    arrow_len = 50
    ex = int(cx + arrow_len * float(np.sin(heading_rad)))
    ey = int(cy - arrow_len * float(np.cos(heading_rad)))
    cv2.arrowedLine(bev_vis, (cx, cy), (ex, ey), (0, 0, 255), 3, tipLength=0.3)

    # Stats text
    color = (0, 255, 0) if "straight" in template_family else (0, 165, 255)
    cv2.putText(bev_vis, f"{label}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bev_vis, f"hdg:{heading_deg:+.1f}d lat:{lateral_m:+.2f}m", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(bev_vis, f"cov:{coverage:.2f} iou:{iou_prev:.2f}", (5, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(bev_vis, f"tmpl:{template_family}", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return bev_vis


def make_side_by_side(frame_bgr, bev_baseline, bev_improved, out_w=1280, out_h=480):
    """Create a side-by-side comparison frame."""
    target_h = out_h
    fw = int(frame_bgr.shape[1] * target_h / max(1, frame_bgr.shape[0]))
    f_resized = cv2.resize(frame_bgr, (fw, target_h))

    bw = (out_w - fw) // 2
    bl = cv2.resize(bev_baseline, (bw, target_h))
    bi = cv2.resize(bev_improved, (bw, target_h))

    combo = np.hstack([f_resized, bl, bi])
    if combo.shape[1] < out_w:
        pad = out_w - combo.shape[1]
        combo = np.hstack([combo, np.zeros((target_h, pad, 3), dtype=np.uint8)])
    elif combo.shape[1] > out_w:
        combo = combo[:, :out_w, :]

    cv2.putText(combo, "Camera", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(combo, "Baseline", (fw + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(combo, "Simple Road", (fw + bw + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
    return combo


# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    model, processor, device: str,
    output_dir: str,
    max_frames: int = 0,
    frame_step: int = 2,
) -> Dict:
    """
    Run both baseline and simple-road pipeline on a video.
    Returns combined metrics dict.
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    # Output dirs
    base_dir = os.path.join(output_dir, "baseline", video_name)
    impr_dir = os.path.join(output_dir, "improved", video_name)
    comp_dir = os.path.join(output_dir, "comparisons", video_name)
    for d in [base_dir, impr_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    # Load homography
    H_bev = _load_homography()

    # Processors
    baseline_proc = BaselineProcessor()
    simple_proc = SimpleRoadProcessor(SimpleRoadConfig())
    mask_smoother = TemporalMaskSmoother()

    # Metrics
    base_metrics = MetricAccumulator("baseline")
    impr_metrics = MetricAccumulator("simple_road")

    # Video I/O
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {}

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps / max(1, frame_step)
    print(f"  Video: {W_in}x{H_in} @{fps:.1f}fps, {total_frames} frames")

    # Video writers (600x600 BEV)
    bev_w, bev_h = BEV_SIZE
    comp_w, comp_h = 1280, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_base = cv2.VideoWriter(os.path.join(base_dir, f"{video_name}_baseline.mp4"), fourcc, out_fps, (bev_w, bev_h))
    writer_impr = cv2.VideoWriter(os.path.join(impr_dir, f"{video_name}_improved.mp4"), fourcc, out_fps, (bev_w, bev_h))
    writer_comp = cv2.VideoWriter(os.path.join(comp_dir, f"{video_name}_comparison.mp4"), fourcc, out_fps, (comp_w, comp_h))

    frame_idx = 0
    processed = 0
    t_seg_total = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames > 0 and processed >= max_frames:
                break
            if frame_idx % max(1, frame_step) != 0:
                frame_idx += 1
                continue

            t_seg = time.time()

            # --- Segmentation ---
            seg_mask = segment_frame(frame, model, processor, device, road_id=ROAD_ID)

            # --- BEV warp ---
            bev_frame = apply_bev(frame, H_bev, BEV_SIZE)
            bev_mask_raw = apply_bev(seg_mask, H_bev, BEV_SIZE)

            # --- Temporal smoothing on raw BEV mask (baseline gets smoothed mask too) ---
            bev_mask_smoothed = mask_smoother.smooth(bev_mask_raw.copy())

            t_seg_total += time.time() - t_seg

            # --- Baseline processing ---
            b_result = baseline_proc.process_bev_mask(bev_mask_smoothed.copy())
            base_metrics.update(
                b_result["coverage"],
                b_result["iou_prev"],
                b_result["heading_deg"],
                b_result["lateral_m"],
            )

            # --- Simple road processing ---
            s_result = simple_proc.process_bev_mask(bev_mask_smoothed.copy())
            impr_metrics.update(
                s_result.mask_coverage,
                s_result.mask_iou_prev,
                s_result.heading_deg,
                s_result.lateral_m_at_lookahead,
            )

            # --- Visualization ---
            bev_vis_base = draw_overlay(
                frame, bev_frame, b_result["clean_mask"],
                centerline_px=None,  # baseline doesn't return px
                heading_deg=b_result["heading_deg"],
                lateral_m=b_result["lateral_m"],
                template_family="dt_baseline",
                label=f"Baseline [f{frame_idx}]",
                coverage=b_result["coverage"],
                iou_prev=b_result["iou_prev"],
            )
            bev_vis_impr = draw_overlay(
                frame, bev_frame, s_result.clean_mask,
                centerline_px=s_result.centerline_px,
                heading_deg=s_result.heading_deg,
                lateral_m=s_result.lateral_m_at_lookahead,
                template_family=s_result.template_family,
                label=f"SimpleRoad [f{frame_idx}]",
                coverage=s_result.mask_coverage,
                iou_prev=s_result.mask_iou_prev,
            )

            combo = make_side_by_side(frame, bev_vis_base, bev_vis_impr, out_w=comp_w, out_h=comp_h)

            writer_base.write(bev_vis_base)
            writer_impr.write(bev_vis_impr)
            writer_comp.write(combo)

            processed += 1
            if processed % 30 == 0:
                seg_ms = (t_seg_total / max(1, processed)) * 1000
                print(f"  Frame {frame_idx}/{total_frames} processed={processed}, seg_ms={seg_ms:.1f}")

            frame_idx += 1

    finally:
        cap.release()
        writer_base.release()
        writer_impr.release()
        writer_comp.release()

    base_sum = base_metrics.summary()
    impr_sum = impr_metrics.summary()

    # Save metrics JSON
    combined = {
        "video": video_name,
        "video_path": video_path,
        "frames_processed": processed,
        "baseline": base_sum,
        "simple_road": impr_sum,
        "improvement": _compute_improvement(base_sum, impr_sum),
    }
    metrics_path = os.path.join(comp_dir, f"{video_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Print summary
    _print_comparison(video_name, base_sum, impr_sum)
    return combined


def _compute_improvement(base: Dict, impr: Dict) -> Dict:
    """Compute delta metrics (positive = improvement for simple road)."""
    def _delta(key, sub="mean"):
        b = base.get(key, {}).get(sub, 0.0)
        i = impr.get(key, {}).get(sub, 0.0)
        return round(i - b, 4)

    return {
        # Higher IoU = more stable mask
        "mask_iou_prev_delta": _delta("mask_iou_prev"),
        # Lower heading std = smoother direction
        "heading_std_delta": round(
            impr.get("heading_deg", {}).get("std", 0.0) - base.get("heading_deg", {}).get("std", 0.0), 4
        ),
        # Lower path_jump std = fewer sudden jumps
        "path_jump_std_delta": round(
            impr.get("path_jump_deg", {}).get("std", 0.0) - base.get("path_jump_deg", {}).get("std", 0.0), 4
        ),
        # Lower path_jump mean = smoother
        "path_jump_mean_delta": round(
            impr.get("path_jump_deg", {}).get("mean", 0.0) - base.get("path_jump_deg", {}).get("mean", 0.0), 4
        ),
    }


def _print_comparison(name: str, base: Dict, impr: Dict) -> None:
    print(f"\n  --- Comparison: {name} ---")
    metrics = [
        ("mask_coverage", "Mask coverage", "mean"),
        ("mask_iou_prev", "Mask stability (IoU/prev)", "mean"),
        ("mask_iou_prev", "Mask stability std", "std"),
        ("heading_deg", "Heading mean (deg)", "mean"),
        ("heading_deg", "Heading std (deg)", "std"),
        ("path_jump_deg", "Path jump mean (deg)", "mean"),
        ("path_jump_deg", "Path jump std (deg)", "std"),
    ]
    for key, label, sub in metrics:
        bv = base.get(key, {}).get(sub, 0.0)
        iv = impr.get(key, {}).get(sub, 0.0)
        diff = iv - bv
        sym = "^" if diff > 0 else ("v" if diff < 0 else "=")
        # For stability metrics, lower is better (except coverage and iou)
        better = ""
        if key in ("mask_iou_prev", "mask_coverage"):
            better = " [BETTER]" if diff > 0.005 else (" [WORSE]" if diff < -0.005 else "")
        else:
            better = " [BETTER]" if diff < -0.1 else (" [WORSE]" if diff > 0.1 else "")
        print(f"  {label:35s} base={bv:7.3f}  impr={iv:7.3f}  {sym}{diff:+.3f}{better}")


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def write_evaluation_report(all_results: List[Dict], output_dir: str) -> None:
    """Append simple-road results to EVALUATION_REPORT.md"""
    report_path = os.path.join(
        Path(__file__).parent.parent.parent,
        "EVALUATION_REPORT.md"
    )

    lines = [
        "\n\n---\n",
        "## Simple Road Pipeline Evaluation — 2026-03-18\n\n",
        "**Branch**: `simplify-pipeline-simple-roads`  \n",
        "**Approach**: Post-processing improvements only (no model retraining)  \n",
        "**Key changes**: 3-template bank, stronger temporal inertia, longer SG centerline, ",
        "straight preference bias, near-field aggressive morphological close\n\n",
        "### Results per Video\n\n",
        "| Video | Frames | BaseCov | ImpCov | BaseIoU | ImpIoU | BaseHdgStd | ImpHdgStd | BaseJump | ImpJump | Better? |\n",
        "|-------|--------|---------|--------|---------|--------|------------|-----------|----------|---------|--------|\n",
    ]

    for r in all_results:
        if not r:
            continue
        vid = r.get("video", "?")
        n = r.get("frames_processed", 0)
        b = r.get("baseline", {})
        i = r.get("simple_road", {})
        imp = r.get("improvement", {})

        bcov = b.get("mask_coverage", {}).get("mean", 0.0)
        icov = i.get("mask_coverage", {}).get("mean", 0.0)
        biou = b.get("mask_iou_prev", {}).get("mean", 0.0)
        iiou = i.get("mask_iou_prev", {}).get("mean", 0.0)
        bhstd = b.get("heading_deg", {}).get("std", 0.0)
        ihstd = i.get("heading_deg", {}).get("std", 0.0)
        bjump = b.get("path_jump_deg", {}).get("mean", 0.0)
        ijump = i.get("path_jump_deg", {}).get("mean", 0.0)

        # Better if: IOU up, heading std down, jump down
        improvements = sum([
            iiou > biou + 0.01,
            ihstd < bhstd - 0.3,
            ijump < bjump - 0.2,
        ])
        better = "++++" if improvements >= 3 else ("++" if improvements == 2 else ("+" if improvements == 1 else "~"))

        lines.append(
            f"| {vid[:20]} | {n} | {bcov:.3f} | {icov:.3f} | {biou:.3f} | {iiou:.3f} | "
            f"{bhstd:.2f} | {ihstd:.2f} | {bjump:.2f} | {ijump:.2f} | {better} |\n"
        )

    lines += [
        "\n### Metric Explanations\n",
        "- **Cov**: Mask coverage (fraction of BEV that is drivable) — higher usually means better detection\n",
        "- **IoU**: Mask IoU with previous frame — higher = more stable mask\n",
        "- **HdgStd**: Heading angle standard deviation (deg) — lower = smoother steering\n",
        "- **Jump**: Mean frame-to-frame heading change (deg) — lower = fewer sudden turns\n",
        "\n### Simple Road Config Used\n",
        "```\n",
        "mask_ema_alpha: 0.50       (was 0.65)\n",
        "close_kernel_near: 15x15  (was 7x7)\n",
        "dt_sg_window: 15           (was 9)\n",
        "dt_lateral_drift_px: 20    (was 30)\n",
        "templates: 3               (was 7)\n",
        "straight_preference_margin: 0.08 (was 0.03)\n",
        "heading_smooth_alpha: 0.35  (was 0.50)\n",
        "```\n",
    ]

    # Read existing report
    existing = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            existing = f.read()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(existing + "".join(lines))

    print(f"\n  EVALUATION_REPORT.md updated: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate simple-road pipeline vs baseline")
    parser.add_argument("--videos", nargs="*", help="Specific video paths (default: auto-discover)")
    parser.add_argument("--max-frames", type=int, default=300, help="Max frames per video (0=all)")
    parser.add_argument("--frame-step", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: project outputs/)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    proj_root = script_dir.parent.parent
    sim_dir = script_dir.parent

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(proj_root / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Discover videos
    if args.videos:
        video_paths = []
        for v in args.videos:
            if os.path.exists(v):
                video_paths.append(v)
            else:
                # Try relative to sim dir and test_videos/
                candidates = [
                    str(sim_dir / v),
                    str(sim_dir / "test_videos" / v),
                    str(proj_root / v),
                ]
                found = next((c for c in candidates if os.path.exists(c)), None)
                if found:
                    video_paths.append(found)
                else:
                    print(f"  WARNING: Cannot find video: {v}")
    else:
        # Auto-discover: prioritize june video (simpler), then IMG_1877, others
        priority = [
            str(sim_dir / "test_video_june_03_3.mp4"),
            str(sim_dir / "test_videos" / "IMG_1877.MOV"),
            str(sim_dir / "test_videos" / "IMG_1876.MOV"),
            str(sim_dir / "test_videos" / "IMG_1878.MOV"),
            str(sim_dir / "test_videos" / "IMG_1921.MOV"),
            str(sim_dir / "test_videos" / "IMG_1922.MOV"),
            str(sim_dir / "test_videos" / "IMG_1924.MOV"),
        ]
        video_paths = [p for p in priority if os.path.exists(p)]

    if not video_paths:
        print("ERROR: No test videos found.")
        sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to process:")
    for vp in video_paths:
        print(f"  {vp}")

    # Load model
    print(f"\nLoading SegFormer from: {MODEL_DIR}")
    model_dir = MODEL_DIR if os.path.isdir(MODEL_DIR) else str(sim_dir / "models" / "my-segformer-road")
    import torch
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    model, processor, device = load_segformer(model_dir, device)
    print("Model loaded.\n")

    all_results = []
    for vp in video_paths:
        result = process_video(
            vp, model, processor, device,
            output_dir=output_dir,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
        )
        all_results.append(result)

    # Write report
    write_evaluation_report(all_results, output_dir)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
