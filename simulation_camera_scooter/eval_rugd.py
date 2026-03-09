"""
RUGD drivable-surface IoU evaluation for the thesis SegFormer model.

RUGD is captured from a ground robot platform at low camera height — much closer
to scooter-level perspective than Cityscapes (car-level). 18 scenes: trails, parks,
creek, village. 7,442 frames total.

Drivable classes used (scooter-navigable surfaces):
  10 = asphalt   RGB (64, 64, 64)
  11 = gravel    RGB (255, 128, 0)
  23 = concrete  RGB (101, 101, 11)

Usage:
    python eval_rugd.py                           # default: drivable-segformer-b0, 150 frames
    python eval_rugd.py --max-images 50           # quick smoke test
    python eval_rugd.py --model-dir models/my-segformer-road
    python eval_rugd.py --resume                  # resume interrupted run
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── config ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(_SCRIPT_DIR, "models", "drivable-segformer-b0")
RUGD_REPO = "WilliamBonilla62/RUGD"

# RUGD RGB colors for drivable classes (from RUGD_annotation-colormap.txt)
DRIVABLE_COLORS = {
    "asphalt":  (64,  64,  64),
    "gravel":   (255, 128, 0),
    "concrete": (101, 101, 11),
}

# Inference resolution (keeps RUGD 688:550 ≈ 5:4 aspect ratio roughly)
INFER_W, INFER_H = 640, 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── drivable mask from RGB annotation ────────────────────────────────────────
def annotation_to_drivable(ann_rgb: np.ndarray) -> np.ndarray:
    """Convert RUGD RGB annotation to binary drivable mask (True = drivable)."""
    mask = np.zeros(ann_rgb.shape[:2], dtype=bool)
    for name, (r, g, b) in DRIVABLE_COLORS.items():
        mask |= (ann_rgb[:, :, 0] == r) & (ann_rgb[:, :, 1] == g) & (ann_rgb[:, :, 2] == b)
    return mask


# ── IoU accumulator ───────────────────────────────────────────────────────────
class IoUMeter:
    def __init__(self, name):
        self.name = name
        self.tp = self.fp = self.fn = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())

    @property
    def iou(self):
        d = self.tp + self.fp + self.fn
        return self.tp / d if d > 0 else float("nan")

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d > 0 else float("nan")

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d > 0 else float("nan")

    def to_dict(self):
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}

    def from_dict(self, d):
        self.tp, self.fp, self.fn = d["tp"], d["fp"], d["fn"]


# ── model ─────────────────────────────────────────────────────────────────────
def load_model(model_dir):
    print(f"Loading model from {model_dir} on {DEVICE}...")
    processor = SegformerImageProcessor.from_pretrained(model_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.eval().to(DEVICE)
    print("Model loaded.")
    return processor, model


@torch.no_grad()
def predict(processor, model, pil_image: Image.Image) -> np.ndarray:
    """Returns binary drivable mask at INFER_H × INFER_W."""
    resized = pil_image.resize((INFER_W, INFER_H), Image.BILINEAR)
    inputs = processor(images=resized, return_tensors="pt").to(DEVICE)
    logits = model(**inputs).logits
    pred = torch.nn.functional.interpolate(
        logits, size=(INFER_H, INFER_W), mode="bilinear", align_corners=False
    )
    return pred.argmax(dim=1).squeeze(0).cpu().numpy() == 1


# ── frame list builder ────────────────────────────────────────────────────────
def build_frame_list(seed=42):
    """
    Return list of (img_path, ann_path) pairs from RUGD repo.
    Samples evenly across all 18 scenes so results represent full dataset.
    """
    from huggingface_hub import list_repo_files
    all_files = list(list_repo_files(repo_id=RUGD_REPO, repo_type="dataset"))
    ann_files = sorted(f for f in all_files if f.startswith("RUGD_annotations/"))
    img_files = set(f for f in all_files if f.startswith("RUGD_frames"))

    pairs = []
    for ann in ann_files:
        # e.g. RUGD_annotations/creek/creek_00001.png
        #   -> RUGD_frames-with-annotations/creek/creek_00001.png
        parts = ann.split("/")  # ['RUGD_annotations', scene, filename]
        img = f"RUGD_frames-with-annotations/{parts[1]}/{parts[2]}"
        if img in img_files:
            pairs.append((img, ann))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-images", type=int, default=150,
                        help="Number of frames to evaluate (default 150, use 0 for all 7442)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-report", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_dir.rstrip("/\\"))
    results_file = os.path.join(_SCRIPT_DIR, f"rugd_iou_{model_name}.json")
    print(f"Model:   {model_name}")
    print(f"Results: {results_file}")
    print(f"Device:  {DEVICE}")
    print(f"Drivable classes: asphalt, gravel, concrete")

    # Resume support
    meter = IoUMeter("drivable")
    results = {"model": model_name, "frames": [], "n": 0,
               "drivable_classes": list(DRIVABLE_COLORS.keys())}
    skip = 0

    if args.resume and os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        skip = results["n"]
        meter.from_dict(results.get("meter", {"tp": 0, "fp": 0, "fn": 0}))
        print(f"Resuming from frame {skip} (IoU so far: {meter.iou:.4f})")

    # Build frame list
    print("Building frame list from RUGD repo...")
    pairs = build_frame_list(seed=args.seed)
    max_n = len(pairs) if args.max_images == 0 else args.max_images
    pairs = pairs[skip:skip + max_n]
    print(f"Evaluating {len(pairs)} frames (skip={skip})")

    processor, model = load_model(args.model_dir)

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    for i, (img_path, ann_path) in enumerate(pairs):
        # Download frame pair
        local_img = hf_hub_download(repo_id=RUGD_REPO, repo_type="dataset", filename=img_path)
        local_ann = hf_hub_download(repo_id=RUGD_REPO, repo_type="dataset", filename=ann_path)

        pil_img = Image.open(local_img).convert("RGB")
        ann_rgb = np.array(Image.open(local_ann).convert("RGB"))

        # GT drivable mask (resize to inference resolution)
        gt_full = annotation_to_drivable(ann_rgb)
        gt_pil = Image.fromarray(gt_full.astype(np.uint8) * 255).resize(
            (INFER_W, INFER_H), Image.NEAREST
        )
        gt = np.array(gt_pil) > 0

        pred = predict(processor, model, pil_img)

        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        meter.tp += tp; meter.fp += fp; meter.fn += fn

        results["frames"].append({"img": img_path, "tp": tp, "fp": fp, "fn": fn})
        results["n"] = skip + i + 1
        results["meter"] = meter.to_dict()

        n_done = i + 1
        elapsed = time.time() - t0
        fps = n_done / elapsed

        if n_done % args.batch_report == 0 or n_done == 1:
            remaining = len(pairs) - n_done
            print(
                f"[{skip+n_done:4d}] drivable IoU={meter.iou:.4f}  "
                f"prec={meter.precision:.4f}  rec={meter.recall:.4f}  "
                f"({fps:.1f} img/s, ~{int(remaining/max(fps,0.01))}s left)"
            )
            with open(results_file, "w") as f:
                json.dump(results, f)

    # Final save
    with open(results_file, "w") as f:
        json.dump(results, f)

    total = skip + len(pairs)
    print("\n" + "=" * 65)
    print(f"RUGD EVALUATION — {total} frames — model: {model_name}")
    print("=" * 65)
    print(f"  Drivable IoU   (asphalt + gravel + concrete):  {meter.iou:.4f}")
    print(f"  Precision:                                      {meter.precision:.4f}")
    print(f"  Recall:                                         {meter.recall:.4f}")
    print(f"  Drivable classes: {', '.join(DRIVABLE_COLORS.keys())}")
    print("=" * 65)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
