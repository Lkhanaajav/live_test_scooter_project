"""
Export row-wise sidewalk boundary targets from mask records.

Usage:
    python scripts/export_boundary_targets.py ^
      --records models/drivable-segformer-b0/train_records.jsonl ^
      --output metrics/boundary_train_records.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from boundary_targets import build_boundary_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Input JSONL with image_path/mask_path records.")
    parser.add_argument("--output", required=True, help="Output JSONL for boundary targets.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max record count.")
    parser.add_argument("--row-step", type=int, default=4)
    parser.add_argument("--min-width-px", type=int, default=4)
    parser.add_argument("--bottom-crop-px", type=int, default=0)
    parser.add_argument("--skip-missing", action="store_true", help="Skip records with missing/unreadable masks.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    total = 0
    exported = 0
    skipped = 0

    with open(args.records, "r", encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            if args.limit > 0 and exported >= args.limit:
                break

            rec = json.loads(line)
            mask_path = rec.get("mask_path", "")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                if args.skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Could not read mask: {mask_path}")

            boundary_rec = build_boundary_record(
                mask,
                name=rec.get("name", ""),
                image_path=rec.get("image_path", ""),
                mask_path=mask_path,
                source=rec.get("source", ""),
                row_step=args.row_step,
                min_width_px=args.min_width_px,
                bottom_crop_px=args.bottom_crop_px,
            )
            dst.write(json.dumps(boundary_rec) + "\n")
            exported += 1

    print(f"Exported {exported} boundary records to {args.output}")
    print(f"Total input records seen: {total}")
    print(f"Skipped missing masks: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
