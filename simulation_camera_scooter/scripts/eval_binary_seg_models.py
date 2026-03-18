#!/usr/bin/env python3
"""
Replay all available test videos with the baseline model and a candidate binary
SegFormer checkpoint, then summarize segmentation/path-planning metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent
if str(SIM_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SIM_DIR))

from config import MODEL_DIR
from live_heading_demo import run_live


DEFAULT_VIDEOS_DIR = SIM_DIR / "test_videos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "binary_model_replay"


@dataclass(frozen=True)
class EvalCase:
    label: str
    model_dir: str
    seg_conf_thresh: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default=str(DEFAULT_VIDEOS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--baseline-label", default="baseline_current")
    parser.add_argument("--baseline-model", default=str(MODEL_DIR))
    parser.add_argument("--baseline-thresh", type=float, default=0.5)
    parser.add_argument("--candidate-label", default="candidate_binary")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-thresh", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--enable-detection", action="store_true")
    parser.add_argument("--planner-mode", choices=["dijkstra", "dfs"], default="dijkstra")
    return parser.parse_args()


def safe_name(value: str) -> str:
    base = Path(value).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return cleaned or "video"


def latest_csv(log_dir: Path) -> Path:
    csvs = sorted(log_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")
    return csvs[-1]


def list_videos(videos_dir: Path) -> list[Path]:
    videos = sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v"}])
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}")
    return videos


def switch_count(series: pd.Series) -> int:
    vals = [str(v) for v in series.fillna("").astype(str) if str(v)]
    if len(vals) < 2:
        return 0
    return sum(1 for a, b in zip(vals[:-1], vals[1:]) if a != b)


def summarize_log(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    out: dict[str, object] = {
        "csv_path": str(csv_path),
        "frames": int(len(df)),
    }
    if df.empty:
        return out

    seg_iou = pd.to_numeric(df.get("seg_iou", pd.Series(dtype=float)), errors="coerce").dropna()
    heading = pd.to_numeric(df.get("heading_smoothed_deg", pd.Series(dtype=float)), errors="coerce").dropna()
    abs_heading = heading.abs()
    heading_delta = heading.diff().abs().dropna()
    has_path = pd.to_numeric(df.get("has_path", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    occ = pd.to_numeric(df.get("bev_mask_occ_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    low_conf = pd.to_numeric(df.get("planner_low_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    slowdown = pd.to_numeric(df.get("planner_slowdown", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    sidewalk_px = pd.to_numeric(df.get("sidewalk_mask_pixels", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    corridor_conf = pd.to_numeric(df.get("corridor_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    path_source = df.get("path_source", pd.Series([""] * len(df))).fillna("").astype(str)
    stability_mode = df.get("stability_mode", pd.Series([""] * len(df))).fillna("").astype(str)

    def _mean(series: pd.Series) -> float:
        return float(series.mean()) if len(series) else 0.0

    def _p95(series: pd.Series) -> float:
        return float(series.quantile(0.95)) if len(series) else 0.0

    out.update(
        mean_seg_iou=_mean(seg_iou),
        p10_seg_iou=float(seg_iou.quantile(0.10)) if len(seg_iou) else 0.0,
        unstable_rate=float((stability_mode != "NORMAL").mean()) if len(stability_mode) else 0.0,
        has_path_rate=float(has_path.mean()),
        mean_abs_heading_deg=_mean(abs_heading),
        p95_abs_heading_deg=_p95(abs_heading),
        mean_heading_delta_deg=_mean(heading_delta),
        p95_heading_delta_deg=_p95(heading_delta),
        mean_bev_occ_ratio=_mean(occ),
        bev_occ_cv=float(occ.std() / max(1e-6, occ.mean())) if len(occ) else 0.0,
        mean_sidewalk_pixels=_mean(sidewalk_px),
        sidewalk_pixels_cv=float(sidewalk_px.std() / max(1e-6, sidewalk_px.mean())) if len(sidewalk_px) else 0.0,
        mean_corridor_conf=_mean(corridor_conf),
        low_confidence_rate=float(low_conf.mean()),
        mean_slowdown=_mean(slowdown),
        template_rate=float((path_source == "template").mean()),
        dt_corridor_rate=float((path_source == "dt_corridor").mean()),
        graph_rate=float(path_source.isin(["graph", "endpoint_arc"]).mean()),
        fallback_rate=float(path_source.astype(str).str.startswith("fallback").mean()),
        path_source_switches=switch_count(path_source),
        path_source_counts=path_source.value_counts().to_dict(),
    )
    return out


def run_case(
    *,
    case: EvalCase,
    video_path: Path,
    output_root: Path,
    max_frames: int | None,
    save_video: bool,
    enable_detection: bool,
    planner_mode: str,
) -> dict[str, object]:
    video_label = safe_name(video_path.name)
    case_dir = output_root / case.label / video_label
    logs_dir = case_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_video = case_dir / f"{video_label}_{case.label}.mp4"
    run_live(
        video_path=str(video_path),
        save_video=bool(save_video),
        output_video_path=str(output_video) if save_video else None,
        enable_detection=bool(enable_detection),
        enable_logging=True,
        log_dir=str(logs_dir),
        headless=True,
        planner_mode=planner_mode,
        template_planner_enabled=True,
        max_frames=max_frames,
        model_dir=case.model_dir,
        seg_conf_thresh=case.seg_conf_thresh,
    )

    summary = summarize_log(latest_csv(logs_dir))
    summary.update(
        case_label=case.label,
        model_dir=case.model_dir,
        seg_conf_thresh=case.seg_conf_thresh,
        video_path=str(video_path),
        video_label=video_label,
        output_video=str(output_video) if save_video else None,
    )
    return summary


def build_markdown_summary(results: list[dict[str, object]], cases: list[EvalCase]) -> str:
    lines = ["# Binary Segmentation Replay Summary", ""]
    for item in results:
        lines.append(f"## {item['video_label']} :: {item['case_label']}")
        lines.append(f"- CSV: `{item['csv_path']}`")
        if item.get("output_video"):
            lines.append(f"- Video: `{item['output_video']}`")
        lines.append(f"- Model: `{item['model_dir']}`")
        lines.append(f"- Threshold: {float(item['seg_conf_thresh']):.2f}")
        lines.append(f"- Frames: {int(item['frames'])}")
        lines.append(f"- Mean seg IoU: {float(item.get('mean_seg_iou', 0.0)):.4f}")
        lines.append(f"- Unstable rate: {100.0 * float(item.get('unstable_rate', 0.0)):.1f}%")
        lines.append(f"- Has-path rate: {100.0 * float(item.get('has_path_rate', 0.0)):.1f}%")
        lines.append(f"- Mean |heading|: {float(item.get('mean_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- Mean heading delta: {float(item.get('mean_heading_delta_deg', 0.0)):.3f} deg")
        lines.append(f"- Mean BEV occupancy: {100.0 * float(item.get('mean_bev_occ_ratio', 0.0)):.2f}%")
        lines.append(f"- Mean sidewalk pixels: {float(item.get('mean_sidewalk_pixels', 0.0)):.1f}")
        lines.append(f"- Mean corridor confidence: {float(item.get('mean_corridor_conf', 0.0)):.3f}")
        lines.append(f"- Template / DT / Fallback: {100.0 * float(item.get('template_rate', 0.0)):.1f}% / {100.0 * float(item.get('dt_corridor_rate', 0.0)):.1f}% / {100.0 * float(item.get('fallback_rate', 0.0)):.1f}%")
        lines.append("")

    lines.append("## Pairwise Deltas")
    lines.append("")
    baseline_label = cases[0].label
    candidate_label = cases[1].label
    by_key = {(item["video_label"], item["case_label"]): item for item in results}
    for video_label in sorted({item["video_label"] for item in results}):
        base = by_key.get((video_label, baseline_label))
        cand = by_key.get((video_label, candidate_label))
        if not base or not cand:
            continue
        lines.append(f"### {video_label}")
        lines.append(f"- Mean seg IoU delta: {float(cand.get('mean_seg_iou', 0.0)) - float(base.get('mean_seg_iou', 0.0)):+.4f}")
        lines.append(f"- Unstable-rate delta: {100.0 * (float(cand.get('unstable_rate', 0.0)) - float(base.get('unstable_rate', 0.0))):+.1f} pp")
        lines.append(f"- Has-path-rate delta: {100.0 * (float(cand.get('has_path_rate', 0.0)) - float(base.get('has_path_rate', 0.0))):+.1f} pp")
        lines.append(f"- Mean heading-delta change: {float(cand.get('mean_heading_delta_deg', 0.0)) - float(base.get('mean_heading_delta_deg', 0.0)):+.3f} deg")
        lines.append(f"- Corridor-confidence delta: {float(cand.get('mean_corridor_conf', 0.0)) - float(base.get('mean_corridor_conf', 0.0)):+.3f}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    videos_dir = Path(args.videos_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos = list_videos(videos_dir)
    cases = [
        EvalCase(args.baseline_label, args.baseline_model, float(args.baseline_thresh)),
        EvalCase(args.candidate_label, args.candidate_model, float(args.candidate_thresh)),
    ]

    results: list[dict[str, object]] = []
    for video_path in videos:
        for case in cases:
            print(f"[eval] {video_path.name} :: {case.label}")
            results.append(
                run_case(
                    case=case,
                    video_path=video_path,
                    output_root=output_root,
                    max_frames=args.max_frames,
                    save_video=bool(args.save_video),
                    enable_detection=bool(args.enable_detection),
                    planner_mode=args.planner_mode,
                )
            )

    json_path = output_root / "summary.json"
    md_path = output_root / "summary.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_summary(results, cases), encoding="utf-8")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
