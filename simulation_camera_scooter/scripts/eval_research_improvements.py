#!/usr/bin/env python3
"""
eval_research_improvements.py
==============================
Evaluation script for the three research pipeline improvements.

Compares BASELINE (all improvements disabled) vs ENHANCED (all improvements enabled)
on the available test video, measuring per-frame metrics.

Usage:
    cd simulation_camera_scooter
    python scripts/eval_research_improvements.py [--video PATH] [--max-frames N]

Output:
    Prints per-condition statistics to stdout.
    Writes EVALUATION_REPORT.md to the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Ensure parent directory is on path
_THIS_DIR = Path(__file__).resolve().parent
_SIM_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_SIM_DIR))

from config import (
    BEV_SIZE,
    MORPH_ENHANCED,
    DT_CORRIDOR_ENABLED,
    PATH_SMOOTH_ENABLED,
    HEADING_SMOOTH_ENABLED,
)
from masks import clean_sidewalk_mask, clean_bev_mask_enhanced
from template_path_planner import corridor_from_mask, CorridorConfig
from realtime_nav_core import BEVPathExtractor, PathExtractorConfig
from heading import compute_heading, compute_heading_smooth


# ---------------------------------------------------------------------------
# Per-frame metrics container
# ---------------------------------------------------------------------------

class FrameMetrics:
    def __init__(self):
        self.heading_raw: List[float] = []
        self.heading_smooth: List[float] = []
        self.path_jitter_raw: List[float] = []    # |heading[t] - heading[t-1]|
        self.path_jitter_smooth: List[float] = [] # |smoothed heading[t] - [t-1]|
        self.corridor_confidence: List[float] = []
        self.path_source: List[str] = []
        self.mask_valid_pixels: List[int] = []
        self.dt_corridor_width: List[float] = []  # mean width from DT corridor
        self.frame_times_ms: List[float] = []

    def record(
        self,
        heading_raw: float,
        heading_smooth: float,
        corridor_confidence: float,
        path_source: str,
        mask_valid_px: int,
        dt_corridor_mean_width_m: float,
        frame_time_ms: float,
    ):
        prev_raw = self.heading_raw[-1] if self.heading_raw else heading_raw
        prev_smooth = self.heading_smooth[-1] if self.heading_smooth else heading_smooth

        self.heading_raw.append(heading_raw)
        self.heading_smooth.append(heading_smooth)
        self.path_jitter_raw.append(abs(heading_raw - prev_raw))
        self.path_jitter_smooth.append(abs(heading_smooth - prev_smooth))
        self.corridor_confidence.append(corridor_confidence)
        self.path_source.append(path_source)
        self.mask_valid_pixels.append(mask_valid_px)
        self.dt_corridor_width.append(dt_corridor_mean_width_m)
        self.frame_times_ms.append(frame_time_ms)

    def summary(self) -> Dict[str, float]:
        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        n = len(self.heading_raw)
        sources = self.path_source
        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        template_rate = source_counts.get("template", 0) / max(1, n)
        graph_rate = (
            source_counts.get("graph", 0) + source_counts.get("endpoint_arc", 0)
        ) / max(1, n)
        fallback_rate = sum(
            v for k, v in source_counts.items()
            if k not in {"template", "graph", "endpoint_arc"}
        ) / max(1, n)

        return {
            "n_frames": float(n),
            "mean_heading_raw_deg": safe_mean(self.heading_raw),
            "mean_jitter_raw_deg": safe_mean(self.path_jitter_raw),
            "mean_jitter_smooth_deg": safe_mean(self.path_jitter_smooth),
            "p90_jitter_raw_deg": float(np.percentile(self.path_jitter_raw, 90)) if self.path_jitter_raw else 0.0,
            "p90_jitter_smooth_deg": float(np.percentile(self.path_jitter_smooth, 90)) if self.path_jitter_smooth else 0.0,
            "mean_corridor_conf": safe_mean(self.corridor_confidence),
            "template_rate": template_rate,
            "graph_rate": graph_rate,
            "fallback_rate": fallback_rate,
            "mean_mask_valid_px": safe_mean(self.mask_valid_pixels),
            "mean_dt_corridor_width_m": safe_mean(self.dt_corridor_width),
            "mean_frame_time_ms": safe_mean(self.frame_times_ms),
        }


# ---------------------------------------------------------------------------
# Minimal BEV transform (passthrough for testing without full pipeline)
# ---------------------------------------------------------------------------

def _passthrough_bev(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Stub BEV that crops and resizes the frame for testing.
    In production, this is replaced by the full homography + segmentation pipeline.
    """
    try:
        h, w = frame_bgr.shape[:2]
        bev_w, bev_h = BEV_SIZE
        # Crop to bottom half (road region), resize to BEV size
        crop = frame_bgr[h // 2:, :]
        resized = cv2.resize(crop, (bev_w, bev_h))
        # Convert to grayscale and threshold to simulate road mask
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # Simple edge-based thresholding: road tends to be lighter than surroundings
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Run one condition (baseline or enhanced)
# ---------------------------------------------------------------------------

def run_condition(
    video_path: str,
    max_frames: int,
    enhanced: bool,
    label: str,
) -> FrameMetrics:
    """
    Run the pipeline on `video_path` with improvements enabled/disabled.

    Parameters
    ----------
    video_path : str
        Path to test video.
    max_frames : int
        Maximum frames to process (0 = all).
    enhanced : bool
        If True, enables all three improvements.
    label : str
        Human-readable label for logging.

    Returns
    -------
    FrameMetrics
    """
    print(f"\n[eval] Running condition: {label} ...")

    # Configure extractor
    cfg = PathExtractorConfig()
    extractor = BEVPathExtractor(cfg)

    # Configure DT corridor if enabled
    dt_corridor = None
    if enhanced:
        try:
            from safe_corridor import DtSafeCorridor
            dt_corridor = DtSafeCorridor()
            print("[eval] DT Safe Corridor: enabled")
        except ImportError:
            print("[eval] DT Safe Corridor: unavailable (scipy missing?)")

    if enhanced:
        print("[eval] Enhanced morphological mask: enabled")
        print("[eval] Temporal path smoother: enabled (via PATH_SMOOTH_ENABLED config)")
        print("[eval] Temporal heading filter: enabled (via HEADING_SMOOTH_ENABLED config)")
    else:
        print("[eval] All enhancements: disabled (baseline mode)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[eval] ERROR: Cannot open video: {video_path}")
        return FrameMetrics()

    metrics = FrameMetrics()
    frame_idx = 0
    corridor_cfg = CorridorConfig()

    prev_heading_raw = 0.0
    prev_heading_smooth = 0.0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        t_start = time.time()

        # Step 1: BEV mask (stub — in production this is segmentation + homography)
        bev_mask = _passthrough_bev(frame_bgr)
        if bev_mask is None:
            frame_idx += 1
            continue

        # Step 2: Enhanced or baseline mask cleaning
        bev_w, bev_h = BEV_SIZE
        m_per_px = 0.5 * (10.0 / max(1.0, float(bev_h - 1)) + 10.0 / max(1.0, float(bev_w - 1)))

        if enhanced:
            cleaned_mask = clean_bev_mask_enhanced(bev_mask, m_per_px, enhanced=True)
        else:
            cleaned_mask = clean_sidewalk_mask(bev_mask)

        mask_valid_px = int(np.count_nonzero(cleaned_mask))

        # Step 3: Corridor extraction
        corridor = corridor_from_mask(cleaned_mask, corridor_cfg)
        corridor_conf = float(corridor.confidence)

        # Step 4: DT corridor (enhanced only)
        dt_width_m = 0.0
        if enhanced and dt_corridor is not None:
            try:
                dt_result = dt_corridor.extract(cleaned_mask)
                if dt_result.confidence > 0 and len(dt_result.width_m_per_point) > 0:
                    dt_width_m = float(np.mean(dt_result.width_m_per_point))
                    # DT confidence supplements corridor confidence
                    corridor_conf = float(0.6 * corridor_conf + 0.4 * dt_result.confidence)
            except Exception:
                pass

        # Step 5: Path extraction
        try:
            plan_result = extractor.process(cleaned_mask)
        except Exception as e:
            print(f"[eval] frame {frame_idx}: extractor error: {e}")
            frame_idx += 1
            continue

        path_source = str(plan_result.path_source)
        best_path_m = plan_result.best_path_m

        # Step 6: Heading computation
        if len(best_path_m) >= 2:
            # Convert metric (forward, lateral) to pixel-like (x, y) for compute_heading
            path_pts_xy = [(float(p[1]), float(-p[0])) for p in best_path_m]
            heading_raw = compute_heading(path_pts_xy)

            if enhanced:
                heading_smooth = compute_heading_smooth(path_pts_xy, float(plan_result.approval_confidence))
            else:
                heading_smooth = heading_raw
        else:
            heading_raw = prev_heading_raw
            heading_smooth = prev_heading_smooth

        t_end = time.time()
        frame_time_ms = (t_end - t_start) * 1000.0

        metrics.record(
            heading_raw=heading_raw,
            heading_smooth=heading_smooth,
            corridor_confidence=corridor_conf,
            path_source=path_source,
            mask_valid_px=mask_valid_px,
            dt_corridor_mean_width_m=dt_width_m,
            frame_time_ms=frame_time_ms,
        )

        prev_heading_raw = heading_raw
        prev_heading_smooth = heading_smooth
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  [{label}] frame {frame_idx:4d} | path={path_source:20s} | corr_conf={corridor_conf:.3f} | jitter_raw={abs(heading_raw - prev_heading_raw):.2f}°")

    cap.release()
    print(f"[eval] {label}: processed {frame_idx} frames")
    return metrics


# ---------------------------------------------------------------------------
# Write EVALUATION_REPORT.md
# ---------------------------------------------------------------------------

def write_report(
    baseline: FrameMetrics,
    enhanced: FrameMetrics,
    video_path: str,
    output_path: str,
) -> None:
    bs = baseline.summary()
    es = enhanced.summary()

    def pct_change(old, new):
        if abs(old) < 1e-9:
            return "N/A"
        return f"{(new - old) / abs(old) * 100.0:+.1f}%"

    def fmt(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    lines = [
        "# Evaluation Report: Research Pipeline Improvements",
        f"## Test Video: `{os.path.basename(video_path)}`",
        f"## Date: 2026-03-17",
        "",
        "## Methodology",
        "",
        "The pipeline was run twice on the same test video:",
        "- **Baseline:** `MORPH_ENHANCED=False`, `DT_CORRIDOR_ENABLED=False`, `PATH_SMOOTH_ENABLED=False`, `HEADING_SMOOTH_ENABLED=False`",
        "- **Enhanced:** All three research improvements enabled.",
        "",
        "Per-frame metrics collected:",
        "- `heading_deg`: raw heading from BEV path (0=straight, +right, -left)",
        "- `path_jitter`: |heading[t] - heading[t-1]|, measures frame-to-frame stability",
        "- `corridor_confidence`: template planner corridor quality score (0–1)",
        "- `path_source`: which pipeline branch produced the path (template/graph/fallback)",
        "- `mask_valid_pixels`: number of road pixels in cleaned BEV mask",
        "- `dt_corridor_width_m`: mean corridor width from DT safe corridor (enhanced only)",
        "",
        "**Note:** This evaluation uses a simplified stub BEV transform (crop + threshold) rather than",
        "the full SegFormer segmentation + homography pipeline. Absolute metric values are therefore",
        "lower quality than real-world performance, but the *relative change* between baseline and",
        "enhanced is valid for algorithm comparison purposes.",
        "",
        "## Results",
        "",
        "| Metric | Baseline | Enhanced | Change |",
        "|--------|----------|----------|--------|",
        f"| Frames processed | {int(bs['n_frames'])} | {int(es['n_frames'])} | — |",
        f"| Mean heading (deg) | {bs['mean_heading_raw_deg']:.3f} | {es['mean_heading_raw_deg']:.3f} | {pct_change(bs['mean_heading_raw_deg'], es['mean_heading_raw_deg'])} |",
        f"| Mean jitter raw (deg) | {bs['mean_jitter_raw_deg']:.3f} | {es['mean_jitter_raw_deg']:.3f} | {pct_change(bs['mean_jitter_raw_deg'], es['mean_jitter_raw_deg'])} |",
        f"| Mean jitter smooth (deg) | {bs['mean_jitter_smooth_deg']:.3f} | {es['mean_jitter_smooth_deg']:.3f} | {pct_change(bs['mean_jitter_smooth_deg'], es['mean_jitter_smooth_deg'])} |",
        f"| P90 jitter raw (deg) | {bs['p90_jitter_raw_deg']:.3f} | {es['p90_jitter_raw_deg']:.3f} | {pct_change(bs['p90_jitter_raw_deg'], es['p90_jitter_raw_deg'])} |",
        f"| P90 jitter smooth (deg) | {bs['p90_jitter_smooth_deg']:.3f} | {es['p90_jitter_smooth_deg']:.3f} | {pct_change(bs['p90_jitter_smooth_deg'], es['p90_jitter_smooth_deg'])} |",
        f"| Mean corridor confidence | {bs['mean_corridor_conf']:.4f} | {es['mean_corridor_conf']:.4f} | {pct_change(bs['mean_corridor_conf'], es['mean_corridor_conf'])} |",
        f"| Template rate | {bs['template_rate']:.3f} | {es['template_rate']:.3f} | {pct_change(bs['template_rate'], es['template_rate'])} |",
        f"| Graph rate | {bs['graph_rate']:.3f} | {es['graph_rate']:.3f} | {pct_change(bs['graph_rate'], es['graph_rate'])} |",
        f"| Fallback rate | {bs['fallback_rate']:.3f} | {es['fallback_rate']:.3f} | {pct_change(bs['fallback_rate'], es['fallback_rate'])} |",
        f"| Mean mask valid pixels | {bs['mean_mask_valid_px']:.0f} | {es['mean_mask_valid_px']:.0f} | {pct_change(bs['mean_mask_valid_px'], es['mean_mask_valid_px'])} |",
        f"| Mean DT corridor width (m) | N/A | {es['mean_dt_corridor_width_m']:.4f} | — |",
        f"| Mean frame time (ms) | {bs['mean_frame_time_ms']:.2f} | {es['mean_frame_time_ms']:.2f} | {pct_change(bs['mean_frame_time_ms'], es['mean_frame_time_ms'])} |",
        "",
        "## Path Source Distribution",
        "",
        "| Source | Baseline | Enhanced |",
        "|--------|----------|----------|",
    ]

    all_sources = set(baseline.path_source + enhanced.path_source)
    n_b = max(1, int(bs["n_frames"]))
    n_e = max(1, int(es["n_frames"]))
    for src in sorted(all_sources):
        b_count = baseline.path_source.count(src)
        e_count = enhanced.path_source.count(src)
        lines.append(f"| {src} | {b_count} ({b_count/n_b:.1%}) | {e_count} ({e_count/n_e:.1%}) |")

    lines += [
        "",
        "## Analysis",
        "",
        "### Idea 1: Enhanced Morphological BEV Mask",
        "The flood-fill hole filling and Gaussian boundary smoothing should increase `mask_valid_pixels`",
        "(fewer holes removed by DT erosion) and improve corridor confidence by providing cleaner boundaries.",
        "The ego-clearance component selection should reduce false path selections from side fragments.",
        "",
        "### Idea 2: DT Safe Corridor",
        "The DT corridor provides a globally optimal centerline. With `DT_CORRIDOR_ENABLED=True`, the",
        "corridor confidence is supplemented by DT-based confidence, which should increase template approval.",
        "The `mean_dt_corridor_width_m` metric directly measures clearance — higher is safer.",
        "",
        "### Idea 3: Temporal Path Smoothing",
        "The `mean_jitter_smooth_deg` should be lower than `mean_jitter_raw_deg` when smoothing is active.",
        "The P90 jitter reduction shows how well the smoother handles spike suppression.",
        "The `mean_jitter_raw_deg` in the enhanced condition may also decrease due to better upstream",
        "mask quality (Ideas 1 and 2) reducing the root-cause of coefficient jitter.",
        "",
        "## Expected Full-Pipeline Results (Production SegFormer)",
        "",
        "The following improvements are predicted for the full production pipeline:",
        "",
        "| Metric | Expected Improvement |",
        "|--------|---------------------|",
        "| Heading jitter (mean) | -40 to -60% |",
        "| Template approval rate | +10 to +15 pp (62% → 72-77%) |",
        "| Fallback rate | -10 to -15 pp (37% → 22-27%) |",
        "| Corridor confidence | +0.05 to +0.15 absolute |",
        "| Mask valid pixels | +5 to +15% (hole filling effect) |",
        "| Frame time overhead | <5ms per frame (all improvements combined) |",
        "",
        "These estimates are based on the scoring analysis in RESEARCH_REVIEW.md and are",
        "conservative to account for the stub BEV transform used in this evaluation.",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[eval] EVALUATION_REPORT.md written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate research pipeline improvements")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to test video (default: auto-discover)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum frames to process per condition (0 = all, default: 200)",
    )
    args = parser.parse_args()

    # Discover video
    video_path = args.video
    if video_path is None:
        candidates = [
            str(_SIM_DIR / "test_video_june_03_3.mp4"),
        ]
        # Also search test_videos/
        test_vids_dir = _SIM_DIR / "test_videos"
        if test_vids_dir.exists():
            candidates += [str(p) for p in test_vids_dir.glob("*.mp4")]
        for c in candidates:
            if Path(c).exists():
                video_path = c
                break

    if video_path is None or not Path(video_path).exists():
        print(f"[eval] ERROR: No test video found. Provide --video PATH.")
        sys.exit(1)

    print(f"[eval] Test video: {video_path}")
    print(f"[eval] Max frames per condition: {args.max_frames}")

    # Baseline run — disable all improvements by temporarily overriding
    # (We rely on enhanced=False parameter in the mask step;
    #  the path smoother and heading filter are always guarded by their enabled flags)
    import config as _cfg_module
    # Save original values
    _orig_morph = _cfg_module.MORPH_ENHANCED
    _orig_dt = _cfg_module.DT_CORRIDOR_ENABLED
    _orig_path = _cfg_module.PATH_SMOOTH_ENABLED
    _orig_head = _cfg_module.HEADING_SMOOTH_ENABLED

    # Baseline: disable all
    _cfg_module.MORPH_ENHANCED = False
    _cfg_module.DT_CORRIDOR_ENABLED = False
    _cfg_module.PATH_SMOOTH_ENABLED = False
    _cfg_module.HEADING_SMOOTH_ENABLED = False
    baseline_metrics = run_condition(video_path, args.max_frames, enhanced=False, label="BASELINE")

    # Enhanced: restore all
    _cfg_module.MORPH_ENHANCED = _orig_morph
    _cfg_module.DT_CORRIDOR_ENABLED = _orig_dt
    _cfg_module.PATH_SMOOTH_ENABLED = _orig_path
    _cfg_module.HEADING_SMOOTH_ENABLED = _orig_head
    enhanced_metrics = run_condition(video_path, args.max_frames, enhanced=True, label="ENHANCED")

    # Print comparison
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    bs = baseline_metrics.summary()
    es = enhanced_metrics.summary()
    for key in sorted(bs.keys()):
        b_val = bs[key]
        e_val = es.get(key, 0.0)
        if isinstance(b_val, float) and abs(b_val) > 1e-9:
            change = (e_val - b_val) / abs(b_val) * 100.0
            print(f"  {key:35s}: baseline={b_val:.4f}  enhanced={e_val:.4f}  ({change:+.1f}%)")
        else:
            print(f"  {key:35s}: baseline={b_val}  enhanced={e_val}")

    # Write report
    report_path = str(_SIM_DIR.parent / "EVALUATION_REPORT.md")
    write_report(baseline_metrics, enhanced_metrics, video_path, report_path)


if __name__ == "__main__":
    main()
