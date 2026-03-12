"""
Train a tiny row-wise sidewalk-boundary model from exported boundary JSONL.
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

from boundary_dataset import BoundaryRecordDataset, boundary_collate  # noqa: E402
from boundary_model import TinyBoundaryNet, boundary_loss, boundary_metrics  # noqa: E402


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def _run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    agg = defaultdict(float)
    batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        batch_t = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        pred = model(image)
        loss, parts = boundary_loss(pred, batch_t)
        metrics = boundary_metrics(pred, batch_t)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for key, value in {**parts, **metrics}.items():
            agg[key] += float(value)
        batches += 1

    return {k: v / max(1, batches) for k, v in agg.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--val-records", required=True)
    parser.add_argument("--output-dir", default="models/tiny-boundary-net")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    train_ds = BoundaryRecordDataset(_resolve(args.train_records), image_size=(args.image_width, args.image_height))
    val_ds = BoundaryRecordDataset(_resolve(args.val_records), image_size=(args.image_width, args.image_height))
    if train_ds.num_rows != val_ds.num_rows:
        raise ValueError("train/val record row counts do not match")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=boundary_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=boundary_collate,
    )

    model = TinyBoundaryNet(num_rows=train_ds.num_rows).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(_resolve(args.output_dir), exist_ok=True)
    history = []
    best_val = None

    for epoch in range(1, args.epochs + 1):
        train_stats = _run_epoch(model, train_loader, optimizer, args.device)
        with torch.no_grad():
            val_stats = _run_epoch(model, val_loader, None, args.device)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        print(
            f"[epoch {epoch}] "
            f"train_loss={train_stats['loss_total']:.4f} "
            f"val_loss={val_stats['loss_total']:.4f} "
            f"val_center_mae={val_stats['center_mae']:.4f} "
            f"val_valid_acc={val_stats['valid_acc']:.4f}"
        )

        if best_val is None or val_stats["loss_total"] < best_val:
            best_val = val_stats["loss_total"]
            ckpt = {
                "model_state": model.state_dict(),
                "num_rows": train_ds.num_rows,
                "image_size": [args.image_width, args.image_height],
                "epoch": epoch,
                "val_loss": best_val,
            }
            torch.save(ckpt, _resolve(os.path.join(args.output_dir, "best.pt")))

    with open(_resolve(os.path.join(args.output_dir, "history.json")), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
