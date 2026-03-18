#!/usr/bin/env python3
"""
Replay one segmentation model on all test videos and save output videos + logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent
if str(SIM_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SIM_DIR))

from live_heading_demo import run_live


DEFAULT_VIDEOS_DIR = SIM_DIR / "test_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seg-conf-thresh", type=float, default=0.6)
    parser.add_argument("--videos-dir", default=str(DEFAULT_VIDEOS_DIR))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--planner-mode", choices=["dijkstra", "dfs"], default="dijkstra")
    parser.add_argument("--max-frames", type=int, default=None)
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


def summarize_log(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    out: dict[str, object] = {"csv_path": str(csv_path), "frames": int(len(df))}
    if df.empty:
        return out
    for col in ["seg_iou", "corridor_confidence", "bev_mask_occ_ratio"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    heading = pd.to_numeric(df.get("heading_smoothed_deg"), errors="coerce")
    out.update(
        mean_seg_iou=float(df["seg_iou"].dropna().mean()),
        mean_corridor_conf=float(df["corridor_confidence"].fillna(0.0).mean()),
        mean_bev_occ_ratio=float(df["bev_mask_occ_ratio"].fillna(0.0).mean()),
        mean_heading_delta_deg=float(heading.diff().abs().dropna().mean()) if heading.notna().any() else 0.0,
    )
    return out


def main() -> int:
    args = parse_args()
    videos_dir = Path(args.videos_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    videos = sorted([p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v"}])
    if not videos:
        raise FileNotFoundError(f"No videos in {videos_dir}")

    results = []
    for video_path in videos:
        video_label = safe_name(video_path.name)
        run_dir = output_root / video_label
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out_video = run_dir / f"{video_label}_{args.label}.mp4"

        print(f"[replay] {video_path.name}")
        run_live(
            video_path=str(video_path),
            save_video=True,
            output_video_path=str(out_video),
            enable_detection=False,
            enable_logging=True,
            log_dir=str(log_dir),
            headless=True,
            planner_mode=args.planner_mode,
            template_planner_enabled=True,
            max_frames=args.max_frames,
            model_dir=args.model_dir,
            seg_conf_thresh=args.seg_conf_thresh,
        )

        result = summarize_log(latest_csv(log_dir))
        result.update(
            video_label=video_label,
            video_path=str(video_path),
            output_video=str(out_video),
            model_dir=args.model_dir,
            label=args.label,
            seg_conf_thresh=args.seg_conf_thresh,
        )
        results.append(result)

    (output_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[replay] wrote {output_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
