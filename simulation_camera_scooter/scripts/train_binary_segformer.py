#!/usr/bin/env python3
"""
Fine-tune a SegFormer checkpoint for binary drivable-mask segmentation.

Inputs:
  - extracted frame JPGs
  - binary PNG masks with matching relative paths

Outputs:
  - best checkpoint saved with Hugging Face-compatible files
  - training history CSV + JSON summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent


DEFAULT_IMAGES_ROOT = SIM_DIR / "annotation_frames"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "outputs" / "pseudo_labels" / "oneformer_cityscapes_swin_large_binary" / "masks"
DEFAULT_INIT_MODEL = SIM_DIR / "models" / "my-segformer-road"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training" / "binary_segformer_oneformer_teacher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks-root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--init-model", default=str(DEFAULT_INIT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--val-every", type=int, default=5, help="Use every Nth frame per video folder for validation")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    return parser.parse_args()


def choose_device(force: str | None) -> str:
    if force:
        return force
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path
    rel_path: str
    video_id: str


def discover_pairs(images_root: Path, masks_root: Path) -> list[SamplePair]:
    pairs: list[SamplePair] = []
    for image_path in sorted(images_root.rglob("*.jpg")):
        rel = image_path.relative_to(images_root)
        mask_path = masks_root / rel.with_suffix(".png")
        if not mask_path.exists():
            continue
        parts = rel.parts
        video_id = parts[0] if parts else "unknown"
        pairs.append(
            SamplePair(
                image_path=image_path,
                mask_path=mask_path,
                rel_path=str(rel).replace("\\", "/"),
                video_id=video_id,
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No matching image/mask pairs found under {images_root} and {masks_root}")
    return pairs


def split_pairs(pairs: list[SamplePair], val_every: int) -> tuple[list[SamplePair], list[SamplePair]]:
    grouped: dict[str, list[SamplePair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.video_id].append(pair)

    train_pairs: list[SamplePair] = []
    val_pairs: list[SamplePair] = []
    for video_id, items in sorted(grouped.items()):
        items = sorted(items, key=lambda p: p.rel_path)
        for idx, item in enumerate(items):
            if val_every > 0 and idx % val_every == 0:
                val_pairs.append(item)
            else:
                train_pairs.append(item)
    if not train_pairs or not val_pairs:
        raise ValueError("Train/validation split failed; adjust --val-every")
    return train_pairs, val_pairs


class BinarySegDataset(Dataset):
    def __init__(
        self,
        samples: list[SamplePair],
        processor: AutoImageProcessor,
        target_hw: tuple[int, int],
        train: bool,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.target_hw = target_hw
        self.train = train
        self.color_jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_mask(self, sample: SamplePair) -> tuple[Image.Image, Image.Image]:
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.train:
            return image, mask
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.7:
            image = self.color_jitter(image)
        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        image, mask = self._load_image_mask(sample)
        image, mask = self._augment(image, mask)

        target_h, target_w = self.target_hw
        image = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)

        proc = self.processor(images=image, return_tensors="pt")
        pixel_values = proc["pixel_values"][0]
        labels = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "rel_path": sample.rel_path,
            "video_id": sample.video_id,
        }


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    rel_paths = [str(item["rel_path"]) for item in batch]
    video_ids = [str(item["video_id"]) for item in batch]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "rel_paths": rel_paths,
        "video_ids": video_ids,
    }


def compute_class_weights(samples: list[SamplePair], target_hw: tuple[int, int]) -> torch.Tensor:
    pos_pixels = 0
    total_pixels = 0
    target_h, target_w = target_hw
    for sample in samples:
        with Image.open(sample.mask_path).convert("L") as mask:
            mask = mask.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
            arr = (np.array(mask) > 127).astype(np.uint8)
        pos_pixels += int(arr.sum())
        total_pixels += int(arr.size)
    neg_pixels = max(1, total_pixels - pos_pixels)
    pos_pixels = max(1, pos_pixels)
    pos_weight = min(6.0, neg_pixels / float(pos_pixels))
    return torch.tensor([1.0, pos_weight], dtype=torch.float32)


def dice_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)[:, 1]
    target = labels.float()
    intersection = (probs * target).sum(dim=(1, 2))
    union = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return 1.0 - dice.mean()


def forward_loss(
    model: SegformerForSemanticSegmentation,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(pixel_values=pixel_values)
    logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    ce = F.cross_entropy(logits, labels, weight=class_weights)
    dice = dice_loss_from_logits(logits, labels)
    return logits, ce + dice


def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits, loss = forward_loss(model, pixel_values, labels, class_weights)
            total_loss += float(loss.item()) * int(labels.shape[0])
            preds = logits.argmax(dim=1)
            tp += int(((preds == 1) & (labels == 1)).sum().item())
            fp += int(((preds == 1) & (labels == 0)).sum().item())
            fn += int(((preds == 0) & (labels == 1)).sum().item())
            tn += int(((preds == 0) & (labels == 0)).sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    iou = tp / max(1, tp + fp + fn)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    mean_loss = total_loss / max(1, len(loader.dataset))
    return {
        "loss": mean_loss,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
    }


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: SegformerForSemanticSegmentation,
    processor: AutoImageProcessor,
    metrics: dict[str, float],
    history: list[dict[str, float | int]],
    args: argparse.Namespace,
    train_count: int,
    val_count: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    payload = {
        "metrics": metrics,
        "train_count": train_count,
        "val_count": val_count,
        "args": vars(args),
        "history": history,
    }
    (checkpoint_dir / "training_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    images_root = Path(args.images_root).resolve()
    masks_root = Path(args.masks_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_csv = output_dir / "history.csv"
    summary_json = output_dir / "summary.json"
    device = choose_device(args.device)
    target_hw = (int(args.image_height), int(args.image_width))

    print(f"[train] device={device}")
    print(f"[train] images_root={images_root}")
    print(f"[train] masks_root={masks_root}")
    print(f"[train] init_model={args.init_model}")
    print(f"[train] output_dir={output_dir}")

    processor = AutoImageProcessor.from_pretrained(args.init_model, local_files_only=True)
    processor.size = {"height": target_hw[0], "width": target_hw[1]}
    processor.do_resize = True

    model = SegformerForSemanticSegmentation.from_pretrained(args.init_model, local_files_only=True)
    model.to(device)

    pairs = discover_pairs(images_root, masks_root)
    train_pairs, val_pairs = split_pairs(pairs, args.val_every)
    class_weights = compute_class_weights(train_pairs, target_hw).to(device)

    train_ds = BinarySegDataset(train_pairs, processor, target_hw=target_hw, train=True)
    val_ds = BinarySegDataset(val_pairs, processor, target_hw=target_hw, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history: list[dict[str, float | int]] = []
    best_iou = -1.0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = forward_loss(model, pixel_values, labels, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size = int(labels.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_items += batch_size

        val_metrics = evaluate(model, val_loader, class_weights, device)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(1, train_items),
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        print(
            f"[train] epoch={epoch:02d} "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_iou={epoch_metrics['val_iou']:.4f} "
            f"val_precision={epoch_metrics['val_precision']:.4f} "
            f"val_recall={epoch_metrics['val_recall']:.4f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                checkpoint_dir=output_dir / "best_checkpoint",
                model=model,
                processor=processor,
                metrics=val_metrics,
                history=history,
                args=args,
                train_count=len(train_pairs),
                val_count=len(val_pairs),
            )

    save_checkpoint(
        checkpoint_dir=output_dir / "last_checkpoint",
        model=model,
        processor=processor,
        metrics=history[-1] if history else {},
        history=history,
        args=args,
        train_count=len(train_pairs),
        val_count=len(val_pairs),
    )

    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_iou", "val_precision", "val_recall", "val_accuracy", "lr"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "images_root": str(images_root),
        "masks_root": str(masks_root),
        "output_dir": str(output_dir),
        "init_model": str(args.init_model),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "class_weights": class_weights.detach().cpu().tolist(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "target_hw": list(target_hw),
        "history": history,
        "best_val_iou": best_iou,
        "total_runtime_s": time.time() - started_at,
        "val_videos": sorted({p.video_id for p in val_pairs}),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[train] wrote {history_csv}")
    print(f"[train] wrote {summary_json}")
    print(f"[train] best checkpoint: {output_dir / 'best_checkpoint'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
