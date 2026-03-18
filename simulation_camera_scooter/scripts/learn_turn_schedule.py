#!/usr/bin/env python3
"""
Learn likely left/right turn windows from an unguided replay CSV log and write a
frame-range intent schedule JSON for live_heading_demo.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Replay CSV log to analyze")
    parser.add_argument("--output", required=True, help="Output JSON schedule path")
    parser.add_argument("--heading-thresh", type=float, default=10.0,
                        help="Absolute heading threshold in degrees for a strong turn vote")
    parser.add_argument("--family-heading-thresh", type=float, default=6.0,
                        help="Absolute heading threshold when template family already indicates a turn")
    parser.add_argument("--min-turn-frames", type=int, default=18,
                        help="Minimum kept turn segment length in frames")
    parser.add_argument("--bridge-gap", type=int, default=10,
                        help="Merge same-direction turn segments across gaps up to this many frames")
    parser.add_argument("--pad-before", type=int, default=12,
                        help="Expand each kept segment backward by this many frames")
    parser.add_argument("--pad-after", type=int, default=12,
                        help="Expand each kept segment forward by this many frames")
    return parser.parse_args()


def _row_family(row: pd.Series, heading_thresh: float, family_heading_thresh: float) -> str | None:
    command = str(row.get("command", "")).strip().upper()
    family = str(row.get("selected_template_family", "")).strip().lower()
    path_source = str(row.get("path_source", "")).strip().lower()
    heading = pd.to_numeric(row.get("heading_smoothed_deg"), errors="coerce")
    if pd.isna(heading):
        heading = 0.0
    heading = float(heading)

    left_votes = 0
    right_votes = 0

    if command == "LEFT":
        left_votes += 2
    elif command == "RIGHT":
        right_votes += 2

    if family == "left" and abs(heading) >= family_heading_thresh:
        left_votes += 1
    elif family == "right" and abs(heading) >= family_heading_thresh:
        right_votes += 1

    if heading <= -heading_thresh:
        left_votes += 1
    elif heading >= heading_thresh:
        right_votes += 1

    if path_source == "template":
        if family == "left":
            left_votes += 1
        elif family == "right":
            right_votes += 1

    if left_votes >= 2 and left_votes > right_votes:
        return "left"
    if right_votes >= 2 and right_votes > left_votes:
        return "right"
    return None


def _extract_segments(labels: list[str | None]) -> list[dict]:
    segments: list[dict] = []
    active_label: str | None = None
    start_idx = 0
    for idx, label in enumerate(labels):
        if label == active_label:
            continue
        if active_label in {"left", "right"}:
            segments.append({"intent": active_label, "start_frame": start_idx, "end_frame": idx - 1})
        active_label = label
        start_idx = idx
    if active_label in {"left", "right"}:
        segments.append({"intent": active_label, "start_frame": start_idx, "end_frame": len(labels) - 1})
    return segments


def _merge_segments(segments: list[dict], bridge_gap: int) -> list[dict]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = int(seg["start_frame"]) - int(prev["end_frame"]) - 1
        if seg["intent"] == prev["intent"] and gap <= bridge_gap:
            prev["end_frame"] = int(seg["end_frame"])
        else:
            merged.append(dict(seg))
    return merged


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    output_path = Path(args.output).resolve()

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")

    labels = [
        _row_family(row, args.heading_thresh, args.family_heading_thresh)
        for _, row in df.iterrows()
    ]
    segments = _extract_segments(labels)
    segments = _merge_segments(segments, bridge_gap=max(0, int(args.bridge_gap)))

    kept: list[dict] = []
    frame_max = max(0, len(df) - 1)
    for seg in segments:
        length = int(seg["end_frame"]) - int(seg["start_frame"]) + 1
        if length < int(args.min_turn_frames):
            continue
        kept.append(
            {
                "intent": seg["intent"],
                "start_frame": max(0, int(seg["start_frame"]) - int(args.pad_before)),
                "end_frame": min(frame_max, int(seg["end_frame"]) + int(args.pad_after)),
                "label": "auto_learned_turn",
            }
        )

    payload = {
        "source_csv": str(csv_path),
        "events": kept,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[schedule] wrote {output_path}")
    if kept:
        for item in kept:
            print(f"[schedule] {item['intent']:>5s} frames {item['start_frame']}..{item['end_frame']}")
    else:
        print("[schedule] no sustained turn windows detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
