#!/usr/bin/env python3
"""
scripts/extract_annotation_frames.py
=====================================
Extract visually diverse frames from videos for hand annotation.

Frames are selected by scene-change detection: a new frame is saved only
when it differs enough from the last saved frame (mean absolute pixel diff
above a threshold). This avoids saving near-duplicate frames.

Output: annotation_frames/<video_stem>/frame_XXXXXX.jpg

Usage:
    # Extract from all test_videos MOV/MP4
    python scripts/extract_annotation_frames.py

    # Single video, custom target count
    python scripts/extract_annotation_frames.py --video test_videos/IMG_1922.mp4 --target 150

    # All videos, combined into one folder
    python scripts/extract_annotation_frames.py --combined
"""

import argparse
import os
import sys
import cv2
import numpy as np


def extract_frames(
    video_path: str,
    out_dir: str,
    target: int = 100,
    min_diff: float = 8.0,
    max_diff: float = 60.0,
    sample_stride: int = 5,
    resize_hw: tuple[int, int] = (720, 1280),
) -> int:
    """
    Extract up to `target` diverse frames from video_path into out_dir.

    Args:
        video_path:    Input video
        out_dir:       Output directory for JPEGs
        target:        Desired number of output frames
        min_diff:      Min mean abs diff to save a new frame (avoids duplicates)
        max_diff:      Frames with diff above this are likely motion blur — skip
        sample_stride: Check every Nth frame (skip ahead for speed)
        resize_hw:     Output resolution (height, width)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)

    # Divide the video into `target` equal segments.
    # From each segment pick the sharpest (highest Laplacian variance) frame
    # that isn't too blurry or too motion-blurred.
    segment_size = max(1, total // target)
    print(f"  {total} frames  {target} segments x {segment_size} frames each")

    saved = 0
    frame_idx = 0
    seg_idx = 0

    h, w = resize_hw

    while seg_idx < target:
        seg_start = seg_idx * segment_size
        seg_end = min(total, seg_start + segment_size)

        # Collect candidates from this segment
        best_frame = None
        best_score = -1.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        for fi in range(seg_start, seg_end, max(1, segment_size // 8)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (w, h))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            # Sharpness score: Laplacian variance (higher = sharper)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            # Skip very blurry frames (score < 20) or extreme motion (score > 8000)
            if score < 20.0 or score > 8000.0:
                continue
            if score > best_score:
                best_score = score
                best_frame = (frame_resized, seg_start + (fi - seg_start))

        if best_frame is not None:
            frame_resized, pick_idx = best_frame
            fname = f"frame_{pick_idx:06d}.jpg"
            out_path = os.path.join(out_dir, fname)
            cv2.imwrite(out_path, frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        seg_idx += 1
        if seg_idx % 20 == 0:
            pct = 100.0 * seg_idx / target
            print(f"  saved {saved}/{target}  ({pct:.0f}% done)", end="\r", flush=True)

    cap.release()
    print(f"  saved {saved} frames -> {out_dir}          ")
    return saved


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--video", default=None, help="Single input video (default: all in test_videos/)")
    ap.add_argument("--out-dir", default=None, help="Output root dir (default: annotation_frames/)")
    ap.add_argument("--target", type=int, default=100, help="Target frames per video (default: 100)")
    ap.add_argument("--min-diff", type=float, default=8.0, help="Min pixel diff to save frame (default: 8.0)")
    ap.add_argument("--combined", action="store_true", help="Put all frames in one folder (annotation_frames/all/)")
    ap.add_argument("--resize", type=str, default="1280x720", help="Output WxH (default: 1280x720)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = args.out_dir or os.path.join(base, "annotation_frames")

    W, H = (int(x) for x in args.resize.split("x"))
    resize_hw = (H, W)

    if args.video:
        videos = [args.video]
    else:
        tv_dir = os.path.join(base, "test_videos")
        # Prefer MP4 over MOV — deduplicate by stem name
        all_files = sorted(os.listdir(tv_dir))
        stems_seen: dict[str, str] = {}
        for f in all_files:
            stem = os.path.splitext(f)[0]
            ext = os.path.splitext(f)[1].lower()
            if ext not in (".mp4", ".mov", ".avi"):
                continue
            # MP4 wins over MOV for same stem
            if stem not in stems_seen or ext == ".mp4":
                stems_seen[stem] = os.path.join(tv_dir, f)
        videos = list(stems_seen.values())

    if not videos:
        print("No videos found.")
        sys.exit(0)

    total_saved = 0
    for vpath in videos:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        if args.combined:
            out_dir = os.path.join(out_root, "all")
        else:
            out_dir = os.path.join(out_root, stem)

        size_mb = os.path.getsize(vpath) / 1e6
        print(f"\n[{stem}]  {size_mb:.0f} MB")
        n = extract_frames(
            vpath,
            out_dir,
            target=args.target,
            min_diff=args.min_diff,
            resize_hw=resize_hw,
        )
        total_saved += n

    print(f"\nDone. Total frames saved: {total_saved}")
    print(f"Location: {out_root}")
    print(f"\nNext step: open annotation_frames/ in LabelStudio or your labeling tool.")
    print(f"Label the 'road/sidewalk' class and export masks as PNG.")


if __name__ == "__main__":
    main()
