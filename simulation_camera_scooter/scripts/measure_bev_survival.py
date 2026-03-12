"""
Measure BEV calibration quality from a run log.

Usage:
    python scripts/measure_bev_survival.py [logs/run_YYYYMMDD_HHMMSS.csv]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bev_calibration import load_bev_params  # noqa: E402


def _latest_log() -> str:
    logs = sorted(glob.glob(os.path.join(ROOT, "logs", "run_*.csv")))
    if not logs:
        raise FileNotFoundError("No run logs found under logs/.")
    return logs[-1]


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    if warn:
        return "WARN"
    return "FAIL"


def analyze(csv_path: str) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")

    bev = pd.to_numeric(df["bev_mask_pixels"], errors="coerce")
    sidewalk = pd.to_numeric(df["sidewalk_mask_pixels"], errors="coerce")
    has_path = pd.to_numeric(df["has_path"], errors="coerce")
    heading = pd.to_numeric(df["heading_smoothed_deg"], errors="coerce")

    survival = bev / sidewalk.where(sidewalk > 0)
    path_frames = heading[has_path == 1]
    heading_jump = path_frames.diff().abs().dropna()

    H, _, src = load_bev_params()
    cond = float(np.linalg.cond(H))

    path_source_counts = {}
    if "path_source" in df.columns:
        path_source_counts = df["path_source"].fillna("").value_counts().to_dict()

    return {
        "csv_path": csv_path,
        "rows": int(len(df)),
        "cond": cond,
        "src": src.tolist(),
        "survival_mean": float(survival.mean()),
        "survival_min": float(survival.min()),
        "has_path_rate": float(has_path.mean()),
        "heading_jump_mean": float(heading_jump.mean()) if not heading_jump.empty else 0.0,
        "heading_jump_max": float(heading_jump.max()) if not heading_jump.empty else 0.0,
        "reversals": int((heading_jump > 90.0).sum()) if not heading_jump.empty else 0,
        "path_source_counts": path_source_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", help="Path to a run CSV. Defaults to latest log.")
    args = parser.parse_args()

    csv_path = args.csv_path or _latest_log()
    metrics = analyze(csv_path)

    bev01 = metrics["cond"] < 1e6
    bev02 = metrics["survival_mean"] >= 0.50
    bev02_warn = 0.40 <= metrics["survival_mean"] < 0.50
    path01 = metrics["has_path_rate"] >= 0.60
    path03 = metrics["reversals"] == 0

    print(f"CSV: {metrics['csv_path']}")
    print(f"Rows: {metrics['rows']}")
    print(f"Homography cond(H): {metrics['cond']:.3e}")
    print(f"Source points: {metrics['src']}")
    print(f"Mean survival: {metrics['survival_mean']:.1%}")
    print(f"Min survival:  {metrics['survival_min']:.1%}")
    print(f"has_path rate: {metrics['has_path_rate']:.1%}")
    print(f"Mean heading jump: {metrics['heading_jump_mean']:.3f} deg/frame")
    print(f"Max heading jump:  {metrics['heading_jump_max']:.3f} deg/frame")
    print(f"Reversals > 90 deg: {metrics['reversals']}")
    if metrics["path_source_counts"]:
        print(f"path_source counts: {metrics['path_source_counts']}")

    print()
    print(f"BEV-01 no-warning calibration: {_status(bev01)}")
    print(f"BEV-02 pixel survival >= 50%: {_status(bev02, warn=bev02_warn)}")
    print(f"PATH-01 has_path >= 60%: {_status(path01)}")
    print(f"PATH-03 no heading reversals: {_status(path03)}")
    print("PATH-02 centerline quality: MANUAL/visual sign-off required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
