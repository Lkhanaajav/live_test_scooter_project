#!/usr/bin/env python3
"""
cityscapes_miou_segformer_b0.py
===============================
Evaluate SegFormer-B0 on Cityscapes and report mIoU + per-class IoU.

Example:
    python scripts/cityscapes_miou_segformer_b0.py \
      --cityscapes-root "D:/datasets/cityscapes" \
      --output-json metrics/cityscapes_miou_segformer_b0.json \
      --output-csv metrics/cityscapes_miou_segformer_b0_per_class.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Cityscapes
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


CITYSCAPES_CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# Cityscapes original label IDs -> train IDs used for standard 19-class mIoU.
CITYSCAPES_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
CITYSCAPES_ID_TO_TRAIN_ID[7] = 0
CITYSCAPES_ID_TO_TRAIN_ID[8] = 1
CITYSCAPES_ID_TO_TRAIN_ID[11] = 2
CITYSCAPES_ID_TO_TRAIN_ID[12] = 3
CITYSCAPES_ID_TO_TRAIN_ID[13] = 4
CITYSCAPES_ID_TO_TRAIN_ID[17] = 5
CITYSCAPES_ID_TO_TRAIN_ID[19] = 6
CITYSCAPES_ID_TO_TRAIN_ID[20] = 7
CITYSCAPES_ID_TO_TRAIN_ID[21] = 8
CITYSCAPES_ID_TO_TRAIN_ID[22] = 9
CITYSCAPES_ID_TO_TRAIN_ID[23] = 10
CITYSCAPES_ID_TO_TRAIN_ID[24] = 11
CITYSCAPES_ID_TO_TRAIN_ID[25] = 12
CITYSCAPES_ID_TO_TRAIN_ID[26] = 13
CITYSCAPES_ID_TO_TRAIN_ID[27] = 14
CITYSCAPES_ID_TO_TRAIN_ID[28] = 15
CITYSCAPES_ID_TO_TRAIN_ID[31] = 16
CITYSCAPES_ID_TO_TRAIN_ID[32] = 17
CITYSCAPES_ID_TO_TRAIN_ID[33] = 18


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


@dataclass
class EvalStats:
    model_id: str
    split: str
    num_images: int
    mean_iou: float
    per_class_iou: list[float]
    elapsed_seconds: float


class CityscapesTrainIdDataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.dataset = Cityscapes(
            root=root,
            split=split,
            mode="fine",
            target_type="semantic",
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image_pil, target_pil = self.dataset[idx]
        target_raw = np.asarray(target_pil, dtype=np.uint8)
        target_train = CITYSCAPES_ID_TO_TRAIN_ID[target_raw].astype(np.uint8)
        return image_pil.convert("RGB"), target_train


class IoUMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        valid = (target >= 0) & (target < self.num_classes)
        target_valid = target[valid]
        pred_valid = pred[valid]
        if target_valid.size == 0:
            return
        flat = self.num_classes * target_valid + pred_valid
        hist = np.bincount(flat, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self) -> tuple[np.ndarray, float]:
        diag = np.diag(self.confusion).astype(np.float64)
        gt = self.confusion.sum(axis=1).astype(np.float64)
        pred = self.confusion.sum(axis=0).astype(np.float64)
        denom = gt + pred - diag
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = diag / denom
        mean_iou = float(np.nanmean(iou))
        return iou, mean_iou


def _collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def evaluate(
    cityscapes_root: str,
    model_id: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: str,
    max_images: int | None,
    local_files_only: bool,
) -> EvalStats:
    dataset = CityscapesTrainIdDataset(cityscapes_root, split=split)
    if max_images is not None:
        max_images = max(1, min(max_images, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, range(max_images))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    if int(model.config.num_labels) != 19:
        raise ValueError(
            f"Expected a 19-class Cityscapes model, got num_labels={model.config.num_labels}"
        )

    meter = IoUMeter(num_classes=19)
    processed = 0
    t0 = time.time()

    with torch.no_grad():
        for images, targets in loader:
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            logits = model(pixel_values=pixel_values).logits
            h, w = targets[0].shape
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for pred, target in zip(preds, targets):
                meter.update(pred=pred, target=target)
                processed += 1

            if processed % 25 == 0:
                print(f"[progress] {processed} images processed")

    per_class_iou, mean_iou = meter.compute()
    elapsed = time.time() - t0
    return EvalStats(
        model_id=model_id,
        split=split,
        num_images=processed,
        mean_iou=mean_iou,
        per_class_iou=[float(x) for x in per_class_iou.tolist()],
        elapsed_seconds=elapsed,
    )


def _write_outputs(stats: EvalStats, output_json: str, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    payload = {
        "model_id": stats.model_id,
        "split": stats.split,
        "num_images": stats.num_images,
        "mean_iou": stats.mean_iou,
        "per_class_iou": {
            name: iou for name, iou in zip(CITYSCAPES_CLASS_NAMES, stats.per_class_iou)
        },
        "elapsed_seconds": stats.elapsed_seconds,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou"])
        for name, iou in zip(CITYSCAPES_CLASS_NAMES, stats.per_class_iou):
            writer.writerow([name, f"{iou:.6f}"])
        writer.writerow(["mean_iou", f"{stats.mean_iou:.6f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SegFormer-B0 Cityscapes mIoU (19-class train IDs)."
    )
    parser.add_argument(
        "--cityscapes-root",
        required=True,
        help="Cityscapes root directory containing leftImg8bit/ and gtFine/.",
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        help="HF model ID or local path (default: official SegFormer-B0 Cityscapes checkpoint).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Cityscapes split (default: val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers (default: 2).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick runs (e.g., 100). Default: full split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model artifacts from local cache/files only.",
    )
    parser.add_argument(
        "--output-json",
        default="metrics/cityscapes_miou_segformer_b0.json",
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--output-csv",
        default="metrics/cityscapes_miou_segformer_b0_per_class.csv",
        help="Output CSV per-class IoU path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cityscapes_root = _resolve_path(args.cityscapes_root)
    output_json = _resolve_path(args.output_json)
    output_csv = _resolve_path(args.output_csv)

    left_dir = os.path.join(cityscapes_root, "leftImg8bit")
    gt_dir = os.path.join(cityscapes_root, "gtFine")
    if not (os.path.isdir(left_dir) and os.path.isdir(gt_dir)):
        raise FileNotFoundError(
            f"Cityscapes root missing expected folders: {left_dir} and/or {gt_dir}"
        )

    print("Cityscapes mIoU evaluation")
    print(f"  cityscapes_root : {cityscapes_root}")
    print(f"  model_id        : {args.model_id}")
    print(f"  split           : {args.split}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  device          : {args.device}")
    print(f"  max_images      : {args.max_images if args.max_images else 'full split'}")
    print(f"  local_only      : {args.local_files_only}")
    print()

    stats = evaluate(
        cityscapes_root=cityscapes_root,
        model_id=args.model_id,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_images=args.max_images,
        local_files_only=args.local_files_only,
    )
    _write_outputs(stats=stats, output_json=output_json, output_csv=output_csv)

    print("Evaluation complete")
    print(f"  images evaluated: {stats.num_images}")
    print(f"  mIoU            : {stats.mean_iou:.4f}")
    print(f"  elapsed seconds : {stats.elapsed_seconds:.1f}")
    print(f"  json            : {output_json}")
    print(f"  csv             : {output_csv}")


if __name__ == "__main__":
    main()
