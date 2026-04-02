#!/usr/bin/env python3
"""
Extract frames from VID mp4s and generate starter masks using SegFormer.
Sets up the hand_annotations workspace for the annotation tool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent

VIDEO_DIR = SIM_DIR / "test_videos"
MODEL_DIR = SIM_DIR / "models" / "my-segformer-road"
OUTPUT_IMAGES = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "images"
OUTPUT_MASKS = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "masks"

VIDEOS = [
    "VID_20260319_155939_00_017.mp4",
    "VID_20260319_160039_00_018.mp4",
    "VID_20260319_160139_00_019.mp4",
    "VID_20260319_160240_00_020.mp4",
]

# Target ~55 frames per video for consistency with IMG videos
TARGET_FRAMES_PER_VIDEO = 55


def load_segformer(
    model_dir: Path, device: str
) -> tuple[SegformerForSemanticSegmentation, SegformerImageProcessor]:
    """Load the SegFormer model and processor."""
    print(f"Loading SegFormer from {model_dir.name}...")
    processor = SegformerImageProcessor.from_pretrained(str(model_dir))
    model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, processor


def predict_mask(
    image_bgr: np.ndarray,
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: str,
) -> np.ndarray:
    """Run segmentation and return binary mask (0/255 uint8)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_classes, H', W')
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image_bgr.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    # Class 1 = road/sidewalk -> 255, everything else -> 0
    mask = np.where(pred >= 1, 255, 0).astype(np.uint8)
    return mask


def process_video(
    video_path: Path,
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: str,
    target_frames: int,
) -> int:
    """Extract frames and generate masks for one video."""
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // target_frames)

    img_dir = OUTPUT_IMAGES / video_name
    mask_dir = OUTPUT_MASKS / video_name
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{video_name}: {total_frames} frames, extracting every {step}th (~{total_frames // step} frames)")

    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame_name = f"frame_{frame_idx:06d}"

            # Save image
            cv2.imwrite(str(img_dir / f"{frame_name}.jpg"), frame)

            # Generate mask
            mask = predict_mask(frame, model, processor, device)
            cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask)

            count += 1
            if count % 10 == 0:
                print(f"  {count} frames processed...")

        frame_idx += 1

    cap.release()
    print(f"  Done: {count} frames extracted and masks generated")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--target-frames", type=int, default=TARGET_FRAMES_PER_VIDEO,
        help="Target number of frames per video",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Process only this video (e.g. VID_20260319_155939_00_017)",
    )
    args = parser.parse_args()

    model, processor = load_segformer(MODEL_DIR, args.device)

    total_count = 0
    for video_file in VIDEOS:
        video_path = VIDEO_DIR / video_file
        if not video_path.exists():
            print(f"  [skip] {video_file} not found")
            continue
        if args.video and args.video not in video_file:
            continue
        total_count += process_video(
            video_path, model, processor, args.device, args.target_frames
        )

    print(f"\nAll done. {total_count} total frames ready for annotation.")
    print(f"Images: {OUTPUT_IMAGES}")
    print(f"Masks:  {OUTPUT_MASKS}")
    print(f"\nRun the annotation tool:")
    print(f"  python scripts/annotate_masks.py --video VID_20260319_155939_00_017")


if __name__ == "__main__":
    main()
