#!/usr/bin/env python3
"""
Interactive BEV calibration on exported frame examples with a live segmentation preview.

Features:
  - browse exported examples with n/p
  - click 4 source points on the left image
  - see the warped segmentation mask update on the right
  - save calibration directly to simulation_camera_scooter/bev_calibration.npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = PROJECT_ROOT / "simulation_camera_scooter"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from config import (  # noqa: E402
    BEV_SIZE,
    CALIBRATION_FILE,
    CALIBRATION_META_FILE,
    DEFAULT_DST_POINTS,
    bev_ego_x_px,
)


@dataclass(frozen=True)
class FrameExample:
    frame_id: int
    raw_path: Path
    overlay_path: Path | None
    mask_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        default=str(PROJECT_ROOT / "outputs" / "evaluation" / "img_1931_manual_bev"),
        help="Directory containing frame_*_rgb.png, *_seg_overlay.png, *_sidewalk_mask.png",
    )
    return parser.parse_args()


def _discover_examples(examples_dir: Path) -> list[FrameExample]:
    pat = re.compile(r"(.+)_frame_(\d+)_rgb\.png$", re.IGNORECASE)
    examples: list[FrameExample] = []
    for raw_path in sorted(examples_dir.glob("*_rgb.png")):
        m = pat.match(raw_path.name)
        if not m:
            continue
        stem_prefix = m.group(1)
        frame_id = int(m.group(2))
        stem = f"{stem_prefix}_frame_{frame_id:04d}"
        overlay_path = examples_dir / f"{stem}_seg_overlay.png"
        mask_path = examples_dir / f"{stem}_sidewalk_mask.png"
        examples.append(
            FrameExample(
                frame_id=frame_id,
                raw_path=raw_path,
                overlay_path=overlay_path if overlay_path.exists() else None,
                mask_path=mask_path if mask_path.exists() else None,
            )
        )
    return examples


def _load_example(example: FrameExample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = cv2.imread(str(example.raw_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Failed to read {example.raw_path}")

    if example.overlay_path is not None:
        display = cv2.imread(str(example.overlay_path), cv2.IMREAD_COLOR)
        if display is None:
            display = raw.copy()
    else:
        display = raw.copy()

    mask = None
    if example.mask_path is not None:
        mask = cv2.imread(str(example.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros(raw.shape[:2], dtype=np.uint8)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return raw, display, mask


def _width_ratio(points: np.ndarray) -> float:
    bl, br, tr, tl = points
    bottom_w = float(np.linalg.norm(br - bl))
    top_w = max(1e-6, float(np.linalg.norm(tr - tl)))
    return bottom_w / top_w


def _make_bev_preview(mask: np.ndarray, points: list[list[int]], ego_x_frac: float) -> tuple[np.ndarray, str]:
    bev_h, bev_w = BEV_SIZE[1], BEV_SIZE[0]
    vis = np.full((bev_h, bev_w, 3), 24, dtype=np.uint8)
    status = "Click 4 points to preview warped segmentation"
    if len(points) != 4:
        return vis, status

    src = np.array(points, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, DEFAULT_DST_POINTS.astype(np.float32))
    warped = cv2.warpPerspective(mask, H, BEV_SIZE)
    vis[warped > 0] = (80, 200, 80)

    ego_x = int(round(float(np.clip(ego_x_frac, 0.05, 0.95)) * max(1.0, float(bev_w - 1))))
    ego_y = bev_h - 1
    cv2.circle(vis, (ego_x, ego_y), 11, (0, 200, 255), -1)
    cv2.circle(vis, (ego_x, ego_y), 11, (255, 255, 255), 2)

    cond = float(np.linalg.cond(H))
    ratio = _width_ratio(src)
    status = f"cond={cond:.2e} | width_ratio={ratio:.2f}"
    color = (0, 255, 0) if cond < 1e6 else (0, 165, 255)
    cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        "Goal: top pair wider than before, keep them on sidewalk edges",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis, status


def main() -> int:
    args = parse_args()
    examples_dir = Path(args.examples_dir).resolve()
    examples = _discover_examples(examples_dir)
    if not examples:
        raise SystemExit(f"No frame examples found in {examples_dir}")

    idx = 0
    points: list[list[int]] = []
    ego_x_frac = 0.5
    if CALIBRATION_META_FILE and Path(CALIBRATION_META_FILE).exists():
        try:
            ego_x_frac = float(json.loads(Path(CALIBRATION_META_FILE).read_text()).get("ego_x_frac", 0.5))
        except Exception:
            ego_x_frac = 0.5

    title = "BEV Example Calibration"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1800, 900)

    current_display_w = [0]

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= current_display_w[0]:
            return
        if len(points) < 4:
            points.append([int(x), int(y)])

    cv2.setMouseCallback(title, on_click)

    while True:
        example = examples[idx]
        raw, display, mask = _load_example(example)
        current_display_w[0] = int(display.shape[1])

        left = display.copy()
        labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
        for i, pt in enumerate(points):
            cv2.circle(left, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(
                left,
                f"{i+1}:{labels[i]}",
                (pt[0] + 10, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(left, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(left, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

        header = (
            f"frame={example.frame_id} | points={len(points)}/4 | "
            "n/p next-prev  r reset  s save  q quit"
        )
        cv2.rectangle(left, (0, 0), (left.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(left, header, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        bev_vis, status = _make_bev_preview(mask, points, ego_x_frac)
        bev_panel = cv2.resize(
            bev_vis,
            (int(left.shape[0] * bev_vis.shape[1] / bev_vis.shape[0]), left.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        combined = np.hstack((left, bev_panel))
        cv2.imshow(title, combined)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
        elif key == ord("n"):
            idx = (idx + 1) % len(examples)
            points.clear()
        elif key == ord("p"):
            idx = (idx - 1) % len(examples)
            points.clear()
        elif key == ord("s") and len(points) == 4:
            src = np.array(points, dtype=np.float32)
            np.save(CALIBRATION_FILE, src)
            meta = {
                "ego_x_frac": float(np.clip(ego_x_frac, 0.05, 0.95)),
                "source_frame_width": int(raw.shape[1]),
                "source_frame_height": int(raw.shape[0]),
                "source_frame_id": int(example.frame_id),
                "bev_width": int(BEV_SIZE[0]),
                "bev_height": int(BEV_SIZE[1]),
                "dst_points": DEFAULT_DST_POINTS.astype(float).tolist(),
            }
            Path(CALIBRATION_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            ratio = _width_ratio(src)
            H = cv2.getPerspectiveTransform(src, DEFAULT_DST_POINTS.astype(np.float32))
            cond = float(np.linalg.cond(H))
            print(
                f"Saved calibration from frame {example.frame_id} to {CALIBRATION_FILE}\n"
                f"  cond={cond:.2e} width_ratio={ratio:.2f}\n"
                f"  meta={meta}"
            )

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
