"""
Cityscapes sidewalk-class IoU evaluation for the thesis SegFormer-B0 model.

Metric: Intersection-over-Union for the sidewalk class (Cityscapes label=8).
Also reports road+sidewalk combined IoU (label 7 or 8) as secondary metric.

Usage:
    python eval_cityscapes.py
    python eval_cityscapes.py --max-images 50   # quick smoke test
    python eval_cityscapes.py --resume           # skip already-evaluated images
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── config ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_SCRIPT_DIR, "models", "drivable-segformer-b0")

# Cityscapes original label IDs
CITYSCAPES_SIDEWALK_ID = 8
CITYSCAPES_ROAD_ID = 7

# Inference resolution (resize before model, then upsample back for comparison)
# 640x320 keeps aspect ratio of 2048x1024 and fits in GPU memory easily
INFER_W, INFER_H = 640, 320

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── IoU accumulator ───────────────────────────────────────────────────────────
class IoUMeter:
    """Pixel-level TP/FP/FN accumulator for binary IoU."""
    def __init__(self, name):
        self.name = name
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """pred_mask, gt_mask: bool arrays of same shape."""
        self.tp += int((pred_mask & gt_mask).sum())
        self.fp += int((pred_mask & ~gt_mask).sum())
        self.fn += int((~pred_mask & gt_mask).sum())

    @property
    def iou(self):
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else float("nan")


# ── model loader ──────────────────────────────────────────────────────────────
def load_model(model_dir):
    print(f"Loading model from {model_dir} on {DEVICE}...")
    processor = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.eval()
    model.to(DEVICE)
    print("Model loaded.")
    return processor, model


# ── single-image inference ────────────────────────────────────────────────────
@torch.no_grad()
def predict(processor, model, pil_image: Image.Image) -> np.ndarray:
    """Returns binary mask at INFER_H x INFER_W (not original resolution)."""
    resized = pil_image.resize((INFER_W, INFER_H), Image.BILINEAR)

    inputs = processor(images=resized, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    # logits: (1, num_labels, H/4, W/4) — upsample to INFER resolution
    logits = outputs.logits
    pred = torch.nn.functional.interpolate(
        logits, size=(INFER_H, INFER_W), mode="bilinear", align_corners=False
    )
    pred_label = pred.argmax(dim=1).squeeze(0).cpu().numpy()  # (INFER_H, INFER_W)
    return pred_label == 1  # True = sidewalk/road class


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Stop after N images (for quick testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Load existing results and skip already-done images")
    parser.add_argument("--batch-report", type=int, default=25,
                        help="Print running IoU every N images")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_dir.rstrip("/\\"))
    RESULTS_FILE = os.path.join(_SCRIPT_DIR, f"cityscapes_iou_{model_name}.json")
    print(f"Model: {model_name}")
    print(f"Results: {RESULTS_FILE}")

    # Load or init results
    results = {"model": model_name, "images": [], "n": 0}
    sidewalk_meter = IoUMeter("sidewalk")
    road_sidewalk_meter = IoUMeter("road+sidewalk")
    skip_count = 0

    if args.resume and os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        skip_count = results["n"]
        # Reconstruct meters from saved per-image data
        for r in results["images"]:
            sidewalk_meter.tp += r["sw_tp"]
            sidewalk_meter.fp += r["sw_fp"]
            sidewalk_meter.fn += r["sw_fn"]
            road_sidewalk_meter.tp += r["rs_tp"]
            road_sidewalk_meter.fp += r["rs_fp"]
            road_sidewalk_meter.fn += r["rs_fn"]
        print(f"Resuming from image {skip_count} (IoU so far: {sidewalk_meter.iou:.4f})")

    processor, model = load_model(args.model_dir)

    from datasets import load_dataset
    ds = load_dataset("Chris1/cityscapes", split="validation", streaming=True)

    t0 = time.time()
    n_processed = 0

    for idx, sample in enumerate(ds):
        if idx < skip_count:
            continue
        if args.max_images and n_processed >= args.max_images:
            break

        pil_img = sample["image"].convert("RGB")
        seg_arr = np.array(sample["semantic_segmentation"])[:, :, 0]  # R channel = label IDs

        # Resize GT to inference resolution for comparison (much faster than upsampling predictions)
        gt_pil = Image.fromarray(seg_arr).resize((INFER_W, INFER_H), Image.NEAREST)
        seg_arr_small = np.array(gt_pil)
        gt_sidewalk = seg_arr_small == CITYSCAPES_SIDEWALK_ID
        gt_road_sidewalk = (seg_arr_small == CITYSCAPES_SIDEWALK_ID) | (seg_arr_small == CITYSCAPES_ROAD_ID)

        pred_mask = predict(processor, model, pil_img)

        sw_tp = int((pred_mask & gt_sidewalk).sum())
        sw_fp = int((pred_mask & ~gt_sidewalk).sum())
        sw_fn = int((~pred_mask & gt_sidewalk).sum())
        rs_tp = int((pred_mask & gt_road_sidewalk).sum())
        rs_fp = int((pred_mask & ~gt_road_sidewalk).sum())
        rs_fn = int((~pred_mask & gt_road_sidewalk).sum())

        sidewalk_meter.tp += sw_tp
        sidewalk_meter.fp += sw_fp
        sidewalk_meter.fn += sw_fn
        road_sidewalk_meter.tp += rs_tp
        road_sidewalk_meter.fp += rs_fp
        road_sidewalk_meter.fn += rs_fn

        results["images"].append({
            "idx": idx,
            "sw_tp": sw_tp, "sw_fp": sw_fp, "sw_fn": sw_fn,
            "rs_tp": rs_tp, "rs_fp": rs_fp, "rs_fn": rs_fn,
        })
        results["n"] = skip_count + n_processed + 1

        n_processed += 1
        elapsed = time.time() - t0
        fps = n_processed / elapsed

        if n_processed % args.batch_report == 0 or n_processed == 1:
            print(
                f"[{skip_count + n_processed:4d}] "
                f"sidewalk IoU={sidewalk_meter.iou:.4f}  "
                f"road+sw IoU={road_sidewalk_meter.iou:.4f}  "
                f"({fps:.1f} img/s, ~{int((500 - skip_count - n_processed) / max(fps, 0.01))}s left)"
            )
            # Save checkpoint
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f)

    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)

    total = skip_count + n_processed
    print("\n" + "=" * 60)
    print(f"CITYSCAPES EVALUATION — {total} val images")
    print("=" * 60)
    print(f"  Sidewalk class IoU  (GT=label 8):      {sidewalk_meter.iou:.4f}")
    print(f"  Sidewalk precision:                    {sidewalk_meter.precision:.4f}")
    print(f"  Sidewalk recall:                       {sidewalk_meter.recall:.4f}")
    print(f"  Road+sidewalk IoU   (GT=label 7|8):    {road_sidewalk_meter.iou:.4f}")
    print(f"  Road+sw precision:                     {road_sidewalk_meter.precision:.4f}")
    print(f"  Road+sw recall:                        {road_sidewalk_meter.recall:.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
