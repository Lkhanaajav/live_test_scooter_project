#!/usr/bin/env python3
"""
Sweep segmentation thresholds against the pseudo-label validation split.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent


DEFAULT_IMAGES_ROOT = SIM_DIR / "annotation_frames"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "outputs" / "pseudo_labels" / "oneformer_cityscapes_swin_large_binary" / "masks"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "training" / "binary_segformer_oneformer_teacher" / "best_checkpoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks-root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root).resolve()
    masks_root = Path(args.masks_root).resolve()

    pairs = []
    for img in sorted(images_root.rglob("*.jpg")):
        rel = img.relative_to(images_root)
        mask = masks_root / rel.with_suffix(".png")
        if mask.exists():
            pairs.append((rel.parts[0], rel, img, mask))

    by_vid = defaultdict(list)
    for item in pairs:
        by_vid[item[0]].append(item)
    val = []
    for _, items in sorted(by_vid.items()):
        for idx, item in enumerate(sorted(items, key=lambda x: str(x[1]))):
            if args.val_every > 0 and idx % args.val_every == 0:
                val.append(item)

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to(args.device)
    model.eval()

    thresholds = [round(x, 2) for x in np.arange(0.35, 0.71, 0.05)]
    metrics = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    for _, _, img_path, mask_path in val:
        image = Image.open(img_path).convert("RGB").resize((args.width, args.height), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((args.width, args.height), Image.Resampling.NEAREST)
        gt = (np.array(mask) > 127)
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)[0]
        prob = logits.softmax(0)[1].detach().cpu().numpy()

        for t in thresholds:
            pred = prob > t
            metrics[t]["tp"] += int(np.logical_and(pred, gt).sum())
            metrics[t]["fp"] += int(np.logical_and(pred, ~gt).sum())
            metrics[t]["fn"] += int(np.logical_and(~pred, gt).sum())

    print(f"validation_images={len(val)}")
    best_t = None
    best_iou = -1.0
    for t in thresholds:
        m = metrics[t]
        iou = m["tp"] / max(1, (m["tp"] + m["fp"] + m["fn"]))
        prec = m["tp"] / max(1, (m["tp"] + m["fp"]))
        rec = m["tp"] / max(1, (m["tp"] + m["fn"]))
        print(f"thresh={t:.2f} iou={iou:.4f} prec={prec:.4f} rec={rec:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_t = t
    print(f"best_threshold={best_t:.2f} best_iou={best_iou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
