"""
Evaluate a trained tiny row-wise sidewalk-boundary model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from boundary_inference import BoundaryPathConfig, decode_boundary_prediction  # noqa: E402
from boundary_dataset import BoundaryRecordDataset, boundary_collate  # noqa: E402
from boundary_model import TinyBoundaryNet, boundary_loss, boundary_metrics  # noqa: E402


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bev-forward-m", type=float, default=10.0)
    parser.add_argument("--bev-lateral-m", type=float, default=10.0)
    parser.add_argument("--path-sample-ds-m", type=float, default=0.25)
    parser.add_argument("--valid-threshold", type=float, default=0.5)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--min-forward-span-m", type=float, default=1.5)
    parser.add_argument("--min-width-m", type=float, default=0.6)
    parser.add_argument("--max-width-m", type=float, default=4.0)
    parser.add_argument("--min-near-field-valid-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(_resolve(args.checkpoint), map_location="cpu")
    image_size = tuple(int(v) for v in ckpt["image_size"])
    ds = BoundaryRecordDataset(_resolve(args.records), image_size=image_size)
    model = TinyBoundaryNet(num_rows=int(ckpt["num_rows"])).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=boundary_collate,
    )

    path_cfg = BoundaryPathConfig(
        bev_forward_m=args.bev_forward_m,
        bev_lateral_m=args.bev_lateral_m,
        path_sample_ds_m=args.path_sample_ds_m,
        valid_threshold=args.valid_threshold,
        low_confidence_threshold=args.low_confidence_threshold,
        min_forward_span_m=args.min_forward_span_m,
        min_width_m=args.min_width_m,
        max_width_m=args.max_width_m,
        min_near_field_valid_ratio=args.min_near_field_valid_ratio,
    )
    agg = defaultdict(float)
    batches = 0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            batch_t = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred = model(batch_t["image"])
            _, parts = boundary_loss(pred, batch_t)
            metrics = boundary_metrics(pred, batch_t)
            for key, value in {**parts, **metrics}.items():
                agg[key] += float(value)
            for idx in range(batch_t["image"].shape[0]):
                decoded = decode_boundary_prediction(
                    {
                        "left_x": pred["left_x"][idx],
                        "right_x": pred["right_x"][idx],
                        "valid_prob": pred["valid_prob"][idx],
                    },
                    rows_y=batch_t["rows_y"][idx],
                    image_size=image_size,
                    cfg=path_cfg,
                )
                agg["path_has_path_rate"] += float(decoded.has_path)
                agg["path_low_conf_rate"] += float(decoded.is_low_confidence)
                agg["path_confidence"] += float(decoded.confidence)
                agg["path_suggested_slowdown"] += float(decoded.suggested_slowdown)
                agg["path_forward_span_m"] += float(decoded.forward_span_m)
                agg["path_mean_width_m"] += float(decoded.mean_width_m)
                agg["path_valid_rows"] += float(decoded.valid_row_count)
                samples += 1
            batches += 1

    result = {k: v / max(1, batches) for k, v in agg.items()}
    for key in (
        "path_has_path_rate",
        "path_low_conf_rate",
        "path_confidence",
        "path_suggested_slowdown",
        "path_forward_span_m",
        "path_mean_width_m",
        "path_valid_rows",
    ):
        if key in result and samples > 0:
            result[key] = agg[key] / float(samples)
    result["num_samples"] = len(ds)
    result["checkpoint"] = _resolve(args.checkpoint)
    print(json.dumps(result, indent=2))

    if args.output_json:
        with open(_resolve(args.output_json), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
