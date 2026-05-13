#!/usr/bin/env python3
"""
scripts/convert_videos.py
=========================
Convert MOV (or any OpenCV-readable format) to H264 MP4.

Usage:
    python scripts/convert_videos.py                        # converts all MOV in test_videos/
    python scripts/convert_videos.py --input test_videos/IMG_1878.MOV
    python scripts/convert_videos.py --input test_videos/ --out-dir test_videos/
"""

import argparse
import os
import sys
import cv2


def convert(src_path: str, dst_path: str, show_progress: bool = True) -> bool:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {src_path}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Try H264 first, fall back to mp4v (MPEG-4)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"  ERROR: cannot create writer for {dst_path}")
        cap.release()
        return False

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        i += 1
        if show_progress and i % 300 == 0:
            pct = 100.0 * i / max(1, total)
            print(f"  {i}/{total} frames ({pct:.0f}%)", end="\r", flush=True)

    cap.release()
    out.release()
    if show_progress:
        print(f"  Done — {i} frames written to {dst_path}      ")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None,
                    help="Source file or directory (default: test_videos/)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: same as input)")
    ap.add_argument("--ext", default=".MOV",
                    help="Extension to convert when --input is a dir (default: .MOV)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input or os.path.join(base, "test_videos")

    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.endswith(args.ext) or f.endswith(args.ext.lower())
        ]
    else:
        files = [input_path]

    if not files:
        print(f"No {args.ext} files found in {input_path}")
        sys.exit(0)

    for src in files:
        out_dir = args.out_dir or os.path.dirname(src)
        base_name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(out_dir, base_name + ".mp4")
        size_mb = os.path.getsize(src) / 1e6
        print(f"\n[{os.path.basename(src)}]  {size_mb:.0f} MB  ->  {dst}")
        convert(src, dst)

    print("\nAll done.")


if __name__ == "__main__":
    main()
