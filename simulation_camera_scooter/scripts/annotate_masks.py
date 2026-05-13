#!/usr/bin/env python3
"""
Brush-Based Mask Annotation Tool
==================================

Interactive tool for reviewing and correcting starter masks.
No external models needed — just OpenCV.

Controls:
  LEFT CLICK + DRAG    — Paint sidewalk (add to mask)
  RIGHT CLICK + DRAG   — Erase sidewalk (remove from mask)
  SHIFT + LEFT CLICK   — Draw straight paint line from last click
  SHIFT + RIGHT CLICK  — Draw straight erase line from last click
  L + CLICK            — Line mode: click two endpoints for a straight edge
  SCROLL UP / =        — Increase brush size
  SCROLL DOWN / -      — Decrease brush size
  A                    — Accept starter mask as-is
  SPACE / S            — Save corrected mask and move to next frame
  R                    — Reset to starter mask (undo all edits)
  Z                    — Undo last stroke
  B                    — Go back to previous frame
  T                    — Toggle overlay transparency
  Q / ESC              — Quit (progress is printed)

The tool shows the image with the mask overlaid in green.
A circle follows your cursor showing the brush size.
In line mode, a yellow preview line shows where the edge will go.

Workflow:
  1. Look at the green overlay — if it matches the sidewalk, press A
  2. If edges are wrong, paint (left-click) or erase (right-click)
  3. For clean straight edges: hold SHIFT and click endpoints
  4. Press SPACE to save and move to next frame

Usage:
    python scripts/annotate_masks.py
    python scripts/annotate_masks.py --video VID_20260319_155939_00_017
    python scripts/annotate_masks.py --start-from 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIM_DIR.parent

GT_IMAGES = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "images"
GT_MASKS = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "masks"
CORRECTED_DIR = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "corrected"
PROGRESS_FILE = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "progress.txt"


def find_frames(video_filter: str | None = None) -> list[tuple[str, str, Path, Path]]:
    """Find all frames with starter masks."""
    frames: list[tuple[str, str, Path, Path]] = []
    if not GT_IMAGES.is_dir():
        return frames
    for video_dir in sorted(GT_IMAGES.iterdir()):
        if not video_dir.is_dir():
            continue
        if video_filter and video_dir.name != video_filter:
            continue
        mask_dir = GT_MASKS / video_dir.name
        if not mask_dir.is_dir():
            continue
        for img_path in sorted(video_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                frames.append((video_dir.name, img_path.stem, img_path, mask_path))
    return frames


def load_progress() -> set[str]:
    """Load set of already-completed frame keys (video/frame_name)."""
    done: set[str] = set()
    if PROGRESS_FILE.exists():
        for line in PROGRESS_FILE.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                done.add(line.split("\t")[0])
    return done


def save_progress_entry(video: str, frame_name: str, action: str) -> None:
    """Append a progress entry."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{video}/{frame_name}\t{action}\n")


def make_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create green overlay of mask on image."""
    overlay = image_bgr.copy()
    mask_bool = mask > 0
    overlay[mask_bool, 1] = np.clip(
        overlay[mask_bool, 1].astype(np.int16) + int(255 * alpha), 0, 255
    ).astype(np.uint8)
    overlay[mask_bool, 0] = (overlay[mask_bool, 0] * (1 - alpha * 0.5)).astype(np.uint8)
    overlay[mask_bool, 2] = (overlay[mask_bool, 2] * (1 - alpha * 0.5)).astype(np.uint8)
    # Draw contour
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Brush-based mask annotation tool")
    parser.add_argument("--video", type=str, default=None, help="Annotate specific video only")
    parser.add_argument("--start-from", type=int, default=0, help="Start from frame index")
    parser.add_argument("--skip-done", action="store_true", help="Skip already-completed frames")
    parser.add_argument("--display-width", type=int, default=1400, help="Display window width")
    args = parser.parse_args()

    frames = find_frames(args.video)
    if not frames:
        print(f"No frames found under {GT_IMAGES}")
        print("Run setup_vid_annotations.py first.")
        return

    done_keys = load_progress() if args.skip_done else set()
    total = len(frames)

    print(f"Found {total} frames to annotate")
    if done_keys:
        print(f"  ({len(done_keys)} already completed — use --skip-done to skip them)")
    print()
    print("Controls:")
    print("  LEFT DRAG   = paint sidewalk    RIGHT DRAG = erase")
    print("  SHIFT+CLICK = straight line from last point (paint or erase)")
    print("  L           = toggle line mode (click two points for edge)")
    print("  +/-         = brush size        T = toggle transparency")
    print("  A = accept as-is   SPACE/S = save & next")
    print("  R = reset to starter   Z = undo last stroke")
    print("  B = go back   Q/ESC = quit")
    print()

    brush_radius = 15
    alpha = 0.4
    saved_count = 0
    accepted_count = 0
    skipped_count = 0

    idx = args.start_from
    mouse_x, mouse_y = -1, -1
    is_painting = False
    is_erasing = False

    # Line tool state
    last_img_pt: tuple[int, int] | None = None  # last click in image coords
    line_mode = False
    line_start: tuple[int, int] | None = None  # first endpoint in image coords
    line_is_erase = False

    while 0 <= idx < total:
        video, frame_name, img_path, mask_path = frames[idx]
        frame_key = f"{video}/{frame_name}"

        if args.skip_done and frame_key in done_keys:
            skipped_count += 1
            idx += 1
            continue

        image_bgr = cv2.imread(str(img_path))
        starter_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or starter_mask is None:
            print(f"  [skip] Cannot read {img_path.name}")
            idx += 1
            continue

        h, w = image_bgr.shape[:2]
        scale = args.display_width / w
        disp_w = args.display_width
        disp_h = int(h * scale)

        current_mask = starter_mask.copy()
        undo_stack: list[np.ndarray] = []
        last_img_pt = None
        line_start = None

        window_name = "Mask Annotator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_w, disp_h)

        def to_img_coords(x: int, y: int) -> tuple[int, int]:
            """Convert display coords to image coords."""
            img_x = max(0, min(int(x / scale), w - 1))
            img_y = max(0, min(int(y / scale), h - 1))
            return img_x, img_y

        def draw_thick_line(
            mask: np.ndarray,
            pt1: tuple[int, int],
            pt2: tuple[int, int],
            value: int,
            radius: int,
        ) -> None:
            """Draw a line with brush thickness on the mask."""
            cv2.line(mask, pt1, pt2, value, thickness=radius * 2)

        def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
            nonlocal mouse_x, mouse_y, is_painting, is_erasing
            nonlocal current_mask, last_img_pt, line_start, line_is_erase

            mouse_x, mouse_y = x, y
            img_x, img_y = to_img_coords(x, y)
            img_r = max(1, int(brush_radius / scale))
            shift_held = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)

            if event == cv2.EVENT_LBUTTONDOWN:
                undo_stack.append(current_mask.copy())

                if line_mode:
                    # Line mode: first click = start, second click = draw
                    if line_start is None:
                        line_start = (img_x, img_y)
                        line_is_erase = False
                    else:
                        draw_thick_line(current_mask, line_start, (img_x, img_y), 255, img_r)
                        line_start = None
                elif shift_held and last_img_pt is not None:
                    # Shift+click: straight line from last point
                    draw_thick_line(current_mask, last_img_pt, (img_x, img_y), 255, img_r)
                    last_img_pt = (img_x, img_y)
                else:
                    # Normal brush
                    is_painting = True
                    cv2.circle(current_mask, (img_x, img_y), img_r, 255, -1)
                    last_img_pt = (img_x, img_y)

            elif event == cv2.EVENT_RBUTTONDOWN:
                undo_stack.append(current_mask.copy())

                if line_mode:
                    if line_start is None:
                        line_start = (img_x, img_y)
                        line_is_erase = True
                    else:
                        draw_thick_line(current_mask, line_start, (img_x, img_y), 0, img_r)
                        line_start = None
                elif shift_held and last_img_pt is not None:
                    draw_thick_line(current_mask, last_img_pt, (img_x, img_y), 0, img_r)
                    last_img_pt = (img_x, img_y)
                else:
                    is_erasing = True
                    cv2.circle(current_mask, (img_x, img_y), img_r, 0, -1)
                    last_img_pt = (img_x, img_y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if is_painting:
                    cv2.circle(current_mask, (img_x, img_y), img_r, 255, -1)
                    last_img_pt = (img_x, img_y)
                elif is_erasing:
                    cv2.circle(current_mask, (img_x, img_y), img_r, 0, -1)
                    last_img_pt = (img_x, img_y)

            elif event == cv2.EVENT_LBUTTONUP:
                is_painting = False
            elif event == cv2.EVENT_RBUTTONUP:
                is_erasing = False

        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            # Build display
            overlay = make_overlay(image_bgr, current_mask, alpha)

            # HUD text
            progress_text = f"[{idx+1}/{total}] {video}/{frame_name}"
            mode_str = "LINE MODE" if line_mode else "BRUSH"
            stats_text = f"Saved: {saved_count}  Accepted: {accepted_count}  Brush: {brush_radius}px  [{mode_str}]"
            cv2.putText(overlay, progress_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, stats_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if line_mode:
                cv2.putText(overlay, "L=exit line mode | Click two points for straight edge",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw brush cursor
            if 0 <= mouse_x < disp_w and 0 <= mouse_y < disp_h:
                cursor_color = (0, 255, 255) if line_mode else (0, 255, 255)
                cv2.circle(overlay, (mouse_x, mouse_y), brush_radius, cursor_color, 1)

            # Draw line preview in line mode
            if line_mode and line_start is not None and mouse_x >= 0:
                start_disp = (int(line_start[0] * scale), int(line_start[1] * scale))
                # Draw start point marker
                cv2.circle(overlay, start_disp, 6, (0, 255, 255), -1)
                cv2.circle(overlay, start_disp, 6, (0, 0, 0), 2)
                # Draw preview line
                line_color = (0, 0, 255) if line_is_erase else (0, 255, 0)
                cv2.line(overlay, start_disp, (mouse_x, mouse_y), line_color, 2)

            # Draw shift-line preview: show last point marker
            if not line_mode and last_img_pt is not None:
                lp_disp = (int(last_img_pt[0] * scale), int(last_img_pt[1] * scale))
                cv2.circle(overlay, lp_disp, 3, (255, 255, 0), -1)

            disp = cv2.resize(overlay, (disp_w, disp_h))
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                cv2.destroyAllWindows()
                print(f"\nDone. Saved: {saved_count}, Accepted: {accepted_count}, Skipped: {skipped_count}")
                return

            elif key == ord(' ') or key == ord('s'):  # Save and next
                cv2.imwrite(str(mask_path), current_mask)
                corrected_path = CORRECTED_DIR / video
                corrected_path.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(corrected_path / f"{frame_name}.png"), current_mask)
                save_progress_entry(video, frame_name, "saved")
                saved_count += 1
                edits = len(undo_stack)
                print(f"  [{idx+1}] SAVED {frame_key} ({edits} strokes)")
                idx += 1
                break

            elif key == ord('a'):  # Accept as-is
                save_progress_entry(video, frame_name, "accepted")
                accepted_count += 1
                print(f"  [{idx+1}] ACCEPTED {frame_key}")
                idx += 1
                break

            elif key == ord('r'):  # Reset
                current_mask = starter_mask.copy()
                undo_stack.clear()
                last_img_pt = None
                line_start = None

            elif key == ord('z'):  # Undo
                if undo_stack:
                    current_mask = undo_stack.pop()
                line_start = None

            elif key == ord('b'):  # Back
                idx = max(0, idx - 1)
                break

            elif key == ord('l'):  # Toggle line mode
                line_mode = not line_mode
                line_start = None
                if line_mode:
                    print("    LINE MODE ON — click two points for straight edge")
                else:
                    print("    LINE MODE OFF — back to brush")

            elif key == ord('t'):  # Toggle transparency
                alpha = 0.6 if alpha < 0.5 else 0.2

            elif key == ord('=') or key == ord('+'):  # Bigger brush
                brush_radius = min(100, brush_radius + 5)

            elif key == ord('-') or key == ord('_'):  # Smaller brush
                brush_radius = max(3, brush_radius - 5)

        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    print(f"\nAll frames processed. Saved: {saved_count}, Accepted: {accepted_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
