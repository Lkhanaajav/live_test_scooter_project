#!/usr/bin/env python3
"""
eval_waypoint_turn_planner.py
=============================
Phase 11.1 replay helper: compare baseline, Phase 11 template, and Phase 11.1
waypoint-turn behavior on the same recordings.

Reuses the eval_template_planner.py log-and-summarize pattern but adds
waypoint-turn-specific metrics:
  - waypoint_turn_rate / waypoint_turn_hold_rate
  - maneuver_family_switches
  - low_confidence_rate
  - path_source distribution
  - heading statistics (mean, p95)

Usage:
    cd simulation_camera_scooter
    python scripts/eval_waypoint_turn_planner.py \\
        --replay-set .planning/phases/11.1-gps-intent-corridor-waypoint-turn-planner/11.1-REPLAY_SET.txt \\
        --modes baseline template waypoint_turn \\
        --output-dir outputs/evaluation/waypoint_turn_phase11_1

    # Or run specific modes on specific videos:
    python scripts/eval_waypoint_turn_planner.py \\
        --videos test_videos/VID_20260319_155939_00_017.mp4 \\
        --modes waypoint_turn \\
        --max-frames 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Valid modes
# ---------------------------------------------------------------------------

VALID_MODES = {"baseline", "template", "waypoint_turn"}

# Default model path for replay (best current model)
_DEFAULT_MODEL_DIR = str(
    Path(PROJECT_DIR).parent / "outputs" / "training"
    / "binary_segformer_old400_img1931_vid017_020" / "best_checkpoint"
)


# ---------------------------------------------------------------------------
# Accepted threshold snapshot (Phase 11.1 Plan 04, 2026-03-24)
# ---------------------------------------------------------------------------
# These are the config.py values at acceptance time. The evaluator logs them
# into the comparison artifact so future replays can verify which thresholds
# were active during a given evaluation run.

def _snapshot_waypoint_thresholds() -> Dict:
    """Capture current waypoint-turn config values for provenance logging."""
    try:
        import config as _cfg
        return {
            "WAYPOINT_DECISION_BAND_MIN_M": _cfg.WAYPOINT_DECISION_BAND_MIN_M,
            "WAYPOINT_DECISION_BAND_MAX_M": _cfg.WAYPOINT_DECISION_BAND_MAX_M,
            "WAYPOINT_ACQUIRE_SUPPORT_MIN": _cfg.WAYPOINT_ACQUIRE_SUPPORT_MIN,
            "WAYPOINT_SUSTAIN_SUPPORT_MIN": _cfg.WAYPOINT_SUSTAIN_SUPPORT_MIN,
            "WAYPOINT_TARGET_UPDATE_ALPHA": _cfg.WAYPOINT_TARGET_UPDATE_ALPHA,
            "WAYPOINT_PATH_CONTAINMENT_MIN": _cfg.WAYPOINT_PATH_CONTAINMENT_MIN,
            "WAYPOINT_NEAR_CONTAINMENT_MIN": _cfg.WAYPOINT_NEAR_CONTAINMENT_MIN,
            "WAYPOINT_NEAR_FIELD_M": _cfg.WAYPOINT_NEAR_FIELD_M,
            "WAYPOINT_LOW_CONFIDENCE_THRESHOLD": _cfg.WAYPOINT_LOW_CONFIDENCE_THRESHOLD,
            "WAYPOINT_SLOWDOWN_FLOOR": _cfg.WAYPOINT_SLOWDOWN_FLOOR,
            "WAYPOINT_HOLD_SLOWDOWN": _cfg.WAYPOINT_HOLD_SLOWDOWN,
            "WAYPOINT_PATH_HORIZON_M": _cfg.WAYPOINT_PATH_HORIZON_M,
            "WAYPOINT_PATH_DS_M": _cfg.WAYPOINT_PATH_DS_M,
            "WAYPOINT_EXIT_REJOIN_FACTOR": _cfg.WAYPOINT_EXIT_REJOIN_FACTOR,
            "WAYPOINT_TURN_ENABLED": _cfg.WAYPOINT_TURN_ENABLED,
            "WAYPOINT_LOCK_SUSTAIN_FRAMES": _cfg.WAYPOINT_LOCK_SUSTAIN_FRAMES,
            "WAYPOINT_LOCK_RELEASE_FRAMES": _cfg.WAYPOINT_LOCK_RELEASE_FRAMES,
        }
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def load_replay_set(manifest_path: str) -> List[str]:
    """Load a replay set manifest file.

    Reads lines from the manifest, ignoring comments (lines starting with #)
    and blank lines. Returns a list of video paths (relative or absolute).
    """
    lines: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Mode -> run_live kwargs mapping
# ---------------------------------------------------------------------------

def mode_to_run_kwargs(mode: str) -> Dict:
    """Map a mode name to the run_live keyword overrides for that mode.

    Returns a dict of kwargs that configure the runtime for the requested
    comparison mode.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Valid: {sorted(VALID_MODES)}")

    common = dict(
        headless=True,
        enable_detection=False,
        enable_logging=True,
        save_video=False,
    )

    if mode == "baseline":
        return {
            **common,
            "template_planner_enabled": False,
            "use_dt_planner": True,
            "initial_intent": None,
        }
    elif mode == "template":
        return {
            **common,
            "template_planner_enabled": True,
            "use_dt_planner": False,
            "initial_intent": None,
        }
    elif mode == "waypoint_turn":
        return {
            **common,
            "template_planner_enabled": True,
            "use_dt_planner": False,
            "initial_intent": None,
        }
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Log summary with waypoint-turn-specific metrics
# ---------------------------------------------------------------------------

def _switch_count(values: list) -> int:
    """Count the number of transitions in a list of strings."""
    vals = [str(v) for v in values if str(v)]
    if len(vals) < 2:
        return 0
    return sum(1 for a, b in zip(vals[:-1], vals[1:]) if a != b)


def summarize_log(csv_path: str) -> Dict:
    """Summarize a single replay log CSV into metrics dict.

    Includes both standard metrics (from eval_template_planner pattern) and
    waypoint-turn-specific metrics:
    - waypoint_turn_rate: fraction of frames with path_source == "waypoint_turn"
    - waypoint_turn_hold_rate: fraction with path_source == "waypoint_turn_hold"
    - maneuver_family_switches: number of selected_template_family transitions
    - low_confidence_rate: fraction of frames with planner_low_confidence == 1
    - path_source_counts: dict of path_source -> count
    - heading stats: mean_abs_heading_deg, p95_abs_heading_deg
    """
    if _HAS_PANDAS:
        return _summarize_log_pandas(csv_path)
    return _summarize_log_csv(csv_path)


def _summarize_log_pandas(csv_path: str) -> Dict:
    """Pandas-based log summarization."""
    df = pd.read_csv(csv_path)
    out: Dict = {
        "csv_path": str(csv_path),
        "frames": int(len(df)),
    }
    if len(df) == 0:
        out.update(
            waypoint_turn_rate=0.0,
            waypoint_turn_hold_rate=0.0,
            maneuver_family_switches=0,
            low_confidence_rate=0.0,
            mean_abs_heading_deg=0.0,
            p95_abs_heading_deg=0.0,
            has_path_rate=0.0,
            mean_slowdown=0.0,
            template_rate=0.0,
            path_source_counts={},
            template_family_counts={},
        )
        return out

    path_source = df.get("path_source", pd.Series([""] * len(df))).fillna("").astype(str)
    template_family = df.get("selected_template_family", pd.Series([""] * len(df))).fillna("").astype(str)

    heading = pd.to_numeric(
        df.get("heading_smoothed_deg", pd.Series(dtype=float)),
        errors="coerce",
    ).abs().dropna()

    has_path = pd.to_numeric(
        df.get("has_path", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    low_conf = pd.to_numeric(
        df.get("planner_low_confidence", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    slowdown = pd.to_numeric(
        df.get("planner_slowdown", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    approval_conf = pd.to_numeric(
        df.get("approval_confidence", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    occ = pd.to_numeric(
        df.get("bev_mask_occ_ratio", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    n = len(df)
    out.update(
        # Waypoint-turn specific metrics
        waypoint_turn_rate=float((path_source == "waypoint_turn").mean()),
        waypoint_turn_hold_rate=float((path_source == "waypoint_turn_hold").mean()),
        maneuver_family_switches=_switch_count(template_family.tolist()),
        # Standard metrics
        has_path_rate=float(has_path.mean()),
        low_confidence_rate=float(low_conf.mean()),
        mean_slowdown=float(slowdown.mean()),
        mean_abs_heading_deg=float(heading.mean()) if len(heading) else 0.0,
        p95_abs_heading_deg=float(heading.quantile(0.95)) if len(heading) else 0.0,
        max_abs_heading_deg=float(heading.max()) if len(heading) else 0.0,
        template_rate=float((path_source == "template").mean()),
        dt_corridor_rate=float((path_source == "dt_corridor").mean()),
        mean_approval_confidence=float(approval_conf.mean()),
        mean_bev_occ_ratio=float(occ.mean()),
        path_source_switches=_switch_count(path_source.tolist()),
        path_source_counts=path_source.value_counts().to_dict(),
        template_family_counts=template_family.value_counts().to_dict(),
    )
    return out


def _summarize_log_csv(csv_path: str) -> Dict:
    """Fallback CSV-based summarization without pandas."""
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    out: Dict = {"csv_path": str(csv_path), "frames": n}
    if n == 0:
        out.update(
            waypoint_turn_rate=0.0,
            waypoint_turn_hold_rate=0.0,
            maneuver_family_switches=0,
            low_confidence_rate=0.0,
            mean_abs_heading_deg=0.0,
            p95_abs_heading_deg=0.0,
            has_path_rate=0.0,
            mean_slowdown=0.0,
            template_rate=0.0,
            path_source_counts={},
            template_family_counts={},
        )
        return out

    path_sources = [str(r.get("path_source", "")).strip() for r in rows]
    families = [str(r.get("selected_template_family", "")).strip() for r in rows]

    headings_abs: List[float] = []
    for r in rows:
        try:
            headings_abs.append(abs(float(r.get("heading_smoothed_deg", 0))))
        except (ValueError, TypeError):
            pass

    low_conf_vals: List[float] = []
    for r in rows:
        try:
            low_conf_vals.append(float(r.get("planner_low_confidence", 0)))
        except (ValueError, TypeError):
            low_conf_vals.append(0.0)

    slowdown_vals: List[float] = []
    for r in rows:
        try:
            slowdown_vals.append(float(r.get("planner_slowdown", 0)))
        except (ValueError, TypeError):
            slowdown_vals.append(0.0)

    has_path_vals: List[float] = []
    for r in rows:
        try:
            has_path_vals.append(float(r.get("has_path", 0)))
        except (ValueError, TypeError):
            has_path_vals.append(0.0)

    # Path source counts
    ps_counts: Dict[str, int] = {}
    for ps in path_sources:
        ps_counts[ps] = ps_counts.get(ps, 0) + 1

    tf_counts: Dict[str, int] = {}
    for tf in families:
        tf_counts[tf] = tf_counts.get(tf, 0) + 1

    wt_count = sum(1 for s in path_sources if s == "waypoint_turn")
    wth_count = sum(1 for s in path_sources if s == "waypoint_turn_hold")
    tpl_count = sum(1 for s in path_sources if s == "template")

    headings_sorted = sorted(headings_abs) if headings_abs else [0.0]
    p95_idx = max(0, int(len(headings_sorted) * 0.95) - 1)

    out.update(
        waypoint_turn_rate=float(wt_count) / max(1, n),
        waypoint_turn_hold_rate=float(wth_count) / max(1, n),
        maneuver_family_switches=_switch_count(families),
        has_path_rate=float(sum(has_path_vals)) / max(1, n),
        low_confidence_rate=float(sum(low_conf_vals)) / max(1, n),
        mean_slowdown=float(sum(slowdown_vals)) / max(1, n),
        mean_abs_heading_deg=float(sum(headings_abs)) / max(1, len(headings_abs)),
        p95_abs_heading_deg=float(headings_sorted[p95_idx]),
        max_abs_heading_deg=float(max(headings_abs)) if headings_abs else 0.0,
        template_rate=float(tpl_count) / max(1, n),
        path_source_switches=_switch_count(path_sources),
        path_source_counts=ps_counts,
        template_family_counts=tf_counts,
    )
    return out


# ---------------------------------------------------------------------------
# Comparison report builder
# ---------------------------------------------------------------------------

def build_comparison_report(title: str, results: List[Dict]) -> str:
    """Build a Markdown comparison report from multi-mode summaries.

    Parameters
    ----------
    title : str
        Report title.
    results : list of dict
        Each dict is a summarize_log output augmented with 'mode' and 'video_label'.

    Returns
    -------
    str
        Markdown text.
    """
    lines = [f"# {title}", ""]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Video | Mode | Frames | WPT Rate | Hold Rate | Low Conf | "
        "Family Sw | Mean Hdg | P95 Hdg | Has Path |"
    )
    lines.append(
        "|-------|------|--------|----------|-----------|----------|"
        "----------|----------|---------|----------|"
    )

    for item in results:
        video = item.get("video_label", "?")
        mode = item.get("mode", "?")
        frames = item.get("frames", 0)
        wpt_rate = 100.0 * float(item.get("waypoint_turn_rate", 0.0))
        hold_rate = 100.0 * float(item.get("waypoint_turn_hold_rate", 0.0))
        low_conf = 100.0 * float(item.get("low_confidence_rate", 0.0))
        family_sw = int(item.get("maneuver_family_switches", 0))
        mean_hdg = float(item.get("mean_abs_heading_deg", 0.0))
        p95_hdg = float(item.get("p95_abs_heading_deg", 0.0))
        has_path = 100.0 * float(item.get("has_path_rate", 0.0))
        lines.append(
            f"| {video} | {mode} | {frames} | {wpt_rate:.1f}% | {hold_rate:.1f}% | "
            f"{low_conf:.1f}% | {family_sw} | {mean_hdg:.2f} | {p95_hdg:.2f} | {has_path:.1f}% |"
        )

    lines.append("")

    # Per-mode detail sections
    for item in results:
        video = item.get("video_label", "?")
        mode = item.get("mode", "?")
        lines.append(f"## {video} :: {mode}")
        lines.append(f"- CSV: `{item.get('csv_path', '?')}`")
        lines.append(f"- Frames: {item.get('frames', 0)}")
        lines.append(f"- Has-path rate: {100.0 * float(item.get('has_path_rate', 0.0)):.1f}%")
        lines.append(f"- Mean abs heading: {float(item.get('mean_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- P95 abs heading: {float(item.get('p95_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- Waypoint-turn rate: {100.0 * float(item.get('waypoint_turn_rate', 0.0)):.1f}%")
        lines.append(f"- Waypoint-turn hold rate: {100.0 * float(item.get('waypoint_turn_hold_rate', 0.0)):.1f}%")
        lines.append(f"- Low-confidence rate: {100.0 * float(item.get('low_confidence_rate', 0.0)):.1f}%")
        lines.append(f"- Mean slowdown: {float(item.get('mean_slowdown', 0.0)):.3f}")
        lines.append(f"- Template rate: {100.0 * float(item.get('template_rate', 0.0)):.1f}%")
        lines.append(f"- Maneuver-family switches: {int(item.get('maneuver_family_switches', 0))}")
        lines.append(f"- Path-source switches: {int(item.get('path_source_switches', 0))}")

        ps_counts = item.get("path_source_counts", {})
        if ps_counts:
            lines.append(f"- Path-source distribution: {json.dumps(ps_counts, sort_keys=True)}")

        tf_counts = item.get("template_family_counts", {})
        if tf_counts:
            lines.append(f"- Template-family distribution: {json.dumps(tf_counts, sort_keys=True)}")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run a single replay case
# ---------------------------------------------------------------------------

def _safe_name(value: str) -> str:
    base = Path(value).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return cleaned or "video"


def _latest_csv(log_dir: Path) -> Path:
    csvs = sorted(log_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")
    return csvs[-1]


def run_case(
    *,
    video_path: Path,
    mode: str,
    output_root: Path,
    max_frames: Optional[int],
    stride: int,
    model_dir: str,
    seg_conf_thresh: float,
    intent_schedule_path: Optional[str] = None,
) -> Dict:
    """Run one video+mode replay case and return the summary dict."""
    from live_heading_demo import run_live

    video_label = _safe_name(video_path.name)
    mode_dir = output_root / video_label / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    log_dir = mode_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    kwargs = mode_to_run_kwargs(mode)
    kwargs.update(
        video_path=str(video_path),
        stride=int(stride),
        enable_logging=True,
        log_dir=str(log_dir),
        model_dir=model_dir,
        seg_conf_thresh=float(seg_conf_thresh),
    )
    if max_frames is not None:
        kwargs["max_frames"] = int(max_frames)
    if intent_schedule_path:
        kwargs["intent_schedule_path"] = intent_schedule_path

    run_live(**kwargs)

    csv_path = _latest_csv(log_dir)
    summary = summarize_log(str(csv_path))
    summary["mode"] = mode
    summary["video_label"] = video_label
    summary["video_path"] = str(video_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11.1 waypoint-turn three-mode replay comparison."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--replay-set",
        help="Path to replay manifest file (one video per line, # comments ignored)",
    )
    group.add_argument(
        "--videos",
        nargs="+",
        help="One or more video paths to replay",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "template", "waypoint_turn"],
        choices=sorted(VALID_MODES),
        help="Modes to compare (default: all three)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation/waypoint_turn_phase11_1",
        help="Directory for logs and summary artifacts",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Max processed frames per video")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--model-dir",
        default=_DEFAULT_MODEL_DIR,
        help="Segmentation model checkpoint directory",
    )
    parser.add_argument("--seg-conf-thresh", type=float, default=0.6, help="Segmentation confidence threshold")
    parser.add_argument(
        "--intent-schedule",
        default=None,
        help="Path to JSON intent schedule for waypoint-turn mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve video paths
    if args.replay_set:
        raw_videos = load_replay_set(args.replay_set)
        sim_dir = PROJECT_DIR
        video_paths: List[Path] = []
        for v in raw_videos:
            candidate = Path(v)
            if candidate.exists():
                video_paths.append(candidate)
            elif (sim_dir / v).exists():
                video_paths.append(sim_dir / v)
            else:
                print(f"  WARNING: Cannot find video: {v}")
    else:
        video_paths = [Path(v) for v in args.videos]

    if not video_paths:
        print("ERROR: No videos found.")
        return 1

    print(f"Found {len(video_paths)} video(s):")
    for vp in video_paths:
        print(f"  {vp}")

    modes = list(args.modes)
    print(f"Modes: {modes}")
    print(f"Output: {output_root}")
    print(f"Model: {args.model_dir}")

    results: List[Dict] = []
    for vp in video_paths:
        if not vp.exists():
            print(f"  SKIP: {vp} not found")
            continue
        for mode in modes:
            print(f"\n--- {vp.name} :: {mode} ---")
            try:
                summary = run_case(
                    video_path=vp,
                    mode=mode,
                    output_root=output_root,
                    max_frames=args.max_frames,
                    stride=args.stride,
                    model_dir=args.model_dir,
                    seg_conf_thresh=float(args.seg_conf_thresh),
                    intent_schedule_path=args.intent_schedule if mode == "waypoint_turn" else None,
                )
                results.append(summary)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                results.append({
                    "mode": mode,
                    "video_label": _safe_name(vp.name),
                    "video_path": str(vp),
                    "frames": 0,
                    "error": str(exc),
                })

    # Capture threshold provenance for the artifact
    threshold_snapshot = _snapshot_waypoint_thresholds()

    # Write combined artifacts
    summary_json = output_root / "summary.json"
    summary_md = output_root / "comparison.md"
    thresholds_json = output_root / "thresholds.json"

    artifact = {
        "thresholds": threshold_snapshot,
        "modes": modes,
        "results": results,
    }
    summary_json.write_text(
        json.dumps(artifact, indent=2, default=str), encoding="utf-8"
    )
    thresholds_json.write_text(
        json.dumps(threshold_snapshot, indent=2), encoding="utf-8"
    )
    md_text = build_comparison_report(
        "Phase 11.1 Waypoint-Turn Replay Comparison", results
    )
    summary_md.write_text(md_text, encoding="utf-8")

    print(f"\n[eval] Wrote {summary_json}")
    print(f"[eval] Wrote {summary_md}")
    print(f"[eval] Wrote {thresholds_json}")
    print(f"\n{'=' * 60}")
    print("WAYPOINT-TURN REPLAY COMPARISON COMPLETE")
    print(f"Output: {output_root}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
