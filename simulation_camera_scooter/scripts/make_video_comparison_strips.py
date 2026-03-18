#!/usr/bin/env python3
"""
Build simple comparison strips from baseline/candidate replay videos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "binary_model_replay_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparisons" / "binary_model_replay_full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--baseline-label", default="baseline_current")
    parser.add_argument("--candidate-label", default="candidate_binary")
    parser.add_argument("--baseline-root", default=None, help="Optional explicit baseline case directory")
    parser.add_argument("--candidate-root", default=None, help="Optional explicit candidate case directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (8, 8), (360, 42), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def resize_h(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    width = int(round(height * w / max(1, h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_root = Path(args.baseline_root).resolve() if args.baseline_root else (eval_root / args.baseline_label)
    candidate_root = Path(args.candidate_root).resolve() if args.candidate_root else (eval_root / args.candidate_label)
    video_labels = sorted(set(p.name for p in baseline_root.iterdir() if p.is_dir()) & set(p.name for p in candidate_root.iterdir() if p.is_dir()))

    sample_fracs = (0.2, 0.5, 0.8)

    for video_label in video_labels:
        base_video = next((baseline_root / video_label).glob("*.mp4"), None)
        cand_video = next((candidate_root / video_label).glob("*.mp4"), None)
        if base_video is None or cand_video is None:
            continue

        cap = cv2.VideoCapture(str(base_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0:
            continue

        columns = []
        for frac in sample_fracs:
            frame_idx = min(max(int(total_frames * frac), 0), max(0, total_frames - 1))
            base_frame = annotate(read_frame(base_video, frame_idx), f"baseline  frame {frame_idx}")
            cand_frame = annotate(read_frame(cand_video, frame_idx), f"candidate frame {frame_idx}")
            base_frame = resize_h(base_frame, 280)
            cand_frame = resize_h(cand_frame, 280)
            width = max(base_frame.shape[1], cand_frame.shape[1])
            base_pad = cv2.copyMakeBorder(base_frame, 0, 0, 0, width - base_frame.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cand_pad = cv2.copyMakeBorder(cand_frame, 0, 0, 0, width - cand_frame.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            columns.append(np.vstack([base_pad, cand_pad]))

        strip = np.hstack(columns)
        out_path = output_dir / f"{video_label}_comparison.jpg"
        cv2.imwrite(str(out_path), strip)
        print(f"[compare] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
