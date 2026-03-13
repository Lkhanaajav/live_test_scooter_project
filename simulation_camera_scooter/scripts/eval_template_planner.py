#!/usr/bin/env python3
"""
Replay helper for comparing the graph-first planner baseline against the Phase 11
template-approval planner on the same recorded videos.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_DIR))

from live_heading_demo import run_live


def _safe_name(value: str) -> str:
    base = Path(value).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return cleaned or "video"


def _latest_csv(log_dir: Path) -> Path:
    csvs = sorted(log_dir.glob("run_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")
    return csvs[-1]


def _collect_rendered_video(search_roots: Iterable[Path], dst_video: Path) -> Path | None:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        candidate = Path(root) / "heading_demo_output.mp4"
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    src_video = max(candidates, key=lambda p: p.stat().st_mtime)
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_video), str(dst_video))
    return dst_video


def summarize_log(csv_path: str | Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    out: dict[str, object] = {
        "csv_path": str(csv_path),
        "frames": int(len(df)),
    }
    if len(df) == 0:
        return out

    heading = pd.to_numeric(df.get("heading_smoothed_deg", pd.Series(dtype=float)), errors="coerce").abs().dropna()
    speed = pd.to_numeric(df.get("speed_smoothed_mps", pd.Series(dtype=float)), errors="coerce").dropna()
    has_path = pd.to_numeric(df.get("has_path", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    occ = pd.to_numeric(df.get("bev_mask_occ_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    low_conf = pd.to_numeric(df.get("planner_low_confidence", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    slowdown = pd.to_numeric(df.get("planner_slowdown", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    path_source = df.get("path_source", pd.Series([""] * len(df))).fillna("")
    template_family = df.get("selected_template_family", pd.Series([""] * len(df))).fillna("")

    def _switch_count(series: pd.Series) -> int:
        vals = [str(v) for v in series if str(v)]
        if len(vals) < 2:
            return 0
        return sum(1 for a, b in zip(vals[:-1], vals[1:]) if a != b)

    out.update(
        has_path_rate=float(has_path.mean()),
        mean_abs_heading_deg=float(heading.mean()) if len(heading) else 0.0,
        p95_abs_heading_deg=float(heading.quantile(0.95)) if len(heading) else 0.0,
        max_abs_heading_deg=float(heading.max()) if len(heading) else 0.0,
        high_heading_frames=int((heading > 8.0).sum()) if len(heading) else 0,
        mean_speed_mps=float(speed.mean()) if len(speed) else 0.0,
        low_confidence_rate=float(low_conf.mean()),
        mean_slowdown=float(slowdown.mean()),
        template_rate=float((path_source == "template").mean()),
        graph_rate=float((path_source == "graph").mean()),
        fallback_rate=float(path_source.astype(str).str.startswith("fallback").mean()),
        path_source_switches=_switch_count(path_source.astype(str)),
        template_family_switches=_switch_count(template_family.astype(str)),
        mean_bev_occ_ratio=float(occ.mean()),
        low_occ_rate=float((occ < 0.05).mean()),
        path_source_counts=path_source.astype(str).value_counts().to_dict(),
        template_family_counts=template_family.astype(str).value_counts().to_dict(),
    )
    return out


def build_markdown_summary(title: str, results: list[dict[str, object]]) -> str:
    lines = [f"# {title}", ""]
    for item in results:
        lines.append(f"## {item['video_label']} :: {item['mode']}")
        lines.append(f"- CSV: `{item['csv_path']}`")
        lines.append(f"- Frames: {item['frames']}")
        lines.append(f"- Has-path rate: {100.0 * float(item.get('has_path_rate', 0.0)):.1f}%")
        lines.append(f"- Mean abs heading: {float(item.get('mean_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- P95 abs heading: {float(item.get('p95_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- Max abs heading: {float(item.get('max_abs_heading_deg', 0.0)):.3f} deg")
        lines.append(f"- High-heading frames (>8 deg): {int(item.get('high_heading_frames', 0))}")
        lines.append(f"- Mean speed: {float(item.get('mean_speed_mps', 0.0)):.3f} m/s")
        lines.append(f"- Low-confidence rate: {100.0 * float(item.get('low_confidence_rate', 0.0)):.1f}%")
        lines.append(f"- Mean slowdown: {float(item.get('mean_slowdown', 0.0)):.3f}")
        lines.append(f"- Template rate: {100.0 * float(item.get('template_rate', 0.0)):.1f}%")
        lines.append(f"- Graph rate: {100.0 * float(item.get('graph_rate', 0.0)):.1f}%")
        lines.append(f"- Fallback rate: {100.0 * float(item.get('fallback_rate', 0.0)):.1f}%")
        lines.append(f"- Path-source switches: {int(item.get('path_source_switches', 0))}")
        lines.append(f"- Template-family switches: {int(item.get('template_family_switches', 0))}")
        lines.append(f"- Mean BEV occupancy: {100.0 * float(item.get('mean_bev_occ_ratio', 0.0)):.2f}%")
        lines.append(f"- Low-occupancy rate (<5%): {100.0 * float(item.get('low_occ_rate', 0.0)):.1f}%")
        lines.append("")
    return "\n".join(lines)


def run_case(
    *,
    video_path: Path,
    mode: str,
    output_root: Path,
    max_frames: int | None,
    stride: int,
    planner_mode: str,
    enable_detection: bool,
    save_video: bool,
) -> dict[str, object]:
    video_label = _safe_name(video_path.name)
    mode_dir = output_root / video_label / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    log_dir = mode_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_live(
        video_path=str(video_path),
        save_video=bool(save_video),
        stride=int(stride),
        enable_detection=bool(enable_detection),
        enable_logging=True,
        log_dir=str(log_dir),
        headless=True,
        planner_mode=planner_mode,
        template_planner_enabled=(mode == "template_phase11"),
        max_frames=max_frames,
    )

    if save_video:
        dst_video = mode_dir / f"{video_label}_{mode}.mp4"
        _collect_rendered_video((Path.cwd(), PROJECT_DIR), dst_video)

    csv_path = _latest_csv(log_dir)
    summary = summarize_log(csv_path)
    summary["mode"] = mode
    summary["video_label"] = video_label
    summary["video_path"] = str(video_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", required=True, help="One or more video paths to replay")
    parser.add_argument("--output-root", default="simulation_camera_scooter/eval_runs/template_planner",
                        help="Directory for logs and summaries")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional processed-frame limit per video")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--planner-mode", default="dijkstra", choices=["dijkstra", "dfs"],
                        help="Graph baseline search mode")
    parser.add_argument("--enable-detection", action="store_true",
                        help="Enable YOLO during replay (disabled by default for planner-focused comparison)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save the combined output video for each replayed mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for video in args.videos:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(video)
        for mode in ("graph_baseline", "template_phase11"):
            results.append(
                run_case(
                    video_path=video_path,
                    mode=mode,
                    output_root=output_root,
                    max_frames=args.max_frames,
                    stride=args.stride,
                    planner_mode=args.planner_mode,
                    enable_detection=bool(args.enable_detection),
                    save_video=bool(args.save_video),
                )
            )

    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_md.write_text(build_markdown_summary("Template Planner Replay Summary", results), encoding="utf-8")
    print(f"[eval] Wrote {summary_json}")
    print(f"[eval] Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
