#!/usr/bin/env python3
"""
Generate binary drivable pseudo-labels from extracted annotation frames.

Teacher:
  - OneFormer semantic segmentation checkpoint
Target:
  - drivable=1, non-drivable=0

By default this script treats Cityscapes `road` and `sidewalk` as drivable and
collapses everything else to background.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent


DEFAULT_INPUT_ROOT = SIM_DIR / "annotation_frames"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "pseudo_labels" / "oneformer_cityscapes_swin_large_binary"
DEFAULT_MODEL_ID = "shi-labs/oneformer_cityscapes_swin_large"
DEFAULT_DRIVABLE_LABELS = ("road", "sidewalk")


@dataclass
class ImageResult:
    image_rel: str
    mask_rel: str
    preview_rel: str | None
    width: int
    height: int
    drivable_pixels: int
    drivable_ratio: float
    road_pixels: int
    sidewalk_pixels: int
    runtime_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Root containing extracted JPG frames")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Where masks and metadata will be written")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Teacher checkpoint")
    parser.add_argument(
        "--drivable-labels",
        nargs="+",
        default=list(DEFAULT_DRIVABLE_LABELS),
        help="Semantic labels to collapse into the binary drivable class",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images")
    parser.add_argument("--save-previews", action="store_true", help="Save overlay previews next to masks")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", None], help="Force device selection")
    return parser.parse_args()


def choose_device(force: str | None) -> str:
    if force:
        return force
    return "cuda" if torch.cuda.is_available() else "cpu"


def list_images(input_root: Path) -> list[Path]:
    images = sorted(input_root.rglob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No JPG files found under {input_root}")
    return images


def semantic_scores_to_map(
    outputs: object,
    target_size_hw: tuple[int, int],
) -> torch.Tensor:
    class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
    mask_probs = outputs.masks_queries_logits.sigmoid()
    seg_scores = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    resized = torch.nn.functional.interpolate(
        seg_scores,
        size=target_size_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0]


def overlay_mask(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    green = np.zeros_like(image_rgb)
    green[..., 1] = 255
    alpha = 0.45
    active = mask_u8 > 0
    overlay[active] = (
        (1.0 - alpha) * overlay[active].astype(np.float32)
        + alpha * green[active].astype(np.float32)
    ).astype(np.uint8)
    return overlay


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    masks_root = output_root / "masks"
    previews_root = output_root / "previews"
    meta_dir = output_root / "metadata"
    output_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if args.save_previews:
        previews_root.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    image_paths = list_images(input_root)
    if args.limit is not None:
        image_paths = image_paths[: int(args.limit)]

    print(f"[pseudo] input_root={input_root}")
    print(f"[pseudo] output_root={output_root}")
    print(f"[pseudo] images={len(image_paths)}")
    print(f"[pseudo] model={args.model_id}")
    print(f"[pseudo] device={device}")

    processor = OneFormerProcessor.from_pretrained(args.model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()

    id2label = {
        int(k): str(v)
        for k, v in model.config.id2label.items()
    }
    label2id = {v: k for k, v in id2label.items()}
    drivable_ids = []
    for label in args.drivable_labels:
        if label not in label2id:
            raise KeyError(f"Label '{label}' not found in teacher config labels: {sorted(label2id)}")
        drivable_ids.append(int(label2id[label]))
    drivable_ids = sorted(set(drivable_ids))

    started_at = time.time()
    results: list[ImageResult] = []

    for idx, image_path in enumerate(image_paths, start=1):
        t0 = time.time()
        rel_image = image_path.relative_to(input_root)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            image_np = np.array(image)

        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            seg_scores = semantic_scores_to_map(outputs, target_size_hw=(height, width))
            pred = seg_scores.argmax(dim=0).cpu().numpy().astype(np.uint8)

        binary_mask = np.isin(pred, drivable_ids).astype(np.uint8) * 255
        road_pixels = int(np.count_nonzero(pred == label2id.get("road", -1)))
        sidewalk_pixels = int(np.count_nonzero(pred == label2id.get("sidewalk", -1)))
        drivable_pixels = int(np.count_nonzero(binary_mask))
        drivable_ratio = drivable_pixels / float(binary_mask.size)

        mask_path = masks_root / rel_image.with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary_mask, mode="L").save(mask_path)

        preview_rel = None
        if args.save_previews:
            preview = overlay_mask(image_np, binary_mask)
            preview_path = previews_root / rel_image.with_suffix(".jpg")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            preview_rel = str(preview_path.relative_to(output_root))

        result = ImageResult(
            image_rel=str(rel_image).replace("\\", "/"),
            mask_rel=str(mask_path.relative_to(output_root)).replace("\\", "/"),
            preview_rel=preview_rel.replace("\\", "/") if preview_rel else None,
            width=width,
            height=height,
            drivable_pixels=drivable_pixels,
            drivable_ratio=drivable_ratio,
            road_pixels=road_pixels,
            sidewalk_pixels=sidewalk_pixels,
            runtime_ms=(time.time() - t0) * 1000.0,
        )
        results.append(result)

        if idx == 1 or idx % 25 == 0 or idx == len(image_paths):
            print(
                f"[pseudo] {idx:4d}/{len(image_paths):4d}  "
                f"{result.image_rel}  "
                f"drive={100.0 * drivable_ratio:5.1f}%  "
                f"t={result.runtime_ms:6.1f} ms"
            )

    csv_path = meta_dir / "pseudo_label_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_rel",
                "mask_rel",
                "preview_rel",
                "width",
                "height",
                "drivable_pixels",
                "drivable_ratio",
                "road_pixels",
                "sidewalk_pixels",
                "runtime_ms",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(item.__dict__)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_id": args.model_id,
        "device": device,
        "drivable_labels": list(args.drivable_labels),
        "drivable_ids": drivable_ids,
        "images": len(results),
        "mean_drivable_ratio": float(np.mean([r.drivable_ratio for r in results])) if results else 0.0,
        "mean_runtime_ms": float(np.mean([r.runtime_ms for r in results])) if results else 0.0,
        "total_runtime_s": time.time() - started_at,
        "label_map": id2label,
        "csv_index": str(csv_path.relative_to(output_root)).replace("\\", "/"),
    }
    summary_path = meta_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[pseudo] wrote {csv_path}")
    print(f"[pseudo] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
