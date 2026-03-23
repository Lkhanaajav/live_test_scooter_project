"""
bev_calibration.py
==================
BEV (Bird's Eye View) calibration tool and parameter loader.
"""

import os
import json
import cv2
import numpy as np

from config import (
    DEFAULT_SRC_POINTS,
    DEFAULT_DST_POINTS,
    BEV_SIZE,
    CALIBRATION_FILE,
    CALIBRATION_META_FILE,
    bev_ego_x_px,
)


def _load_calibration_meta() -> dict:
    """Read calibration metadata from disk each time so runtime stays in sync."""
    if not os.path.exists(CALIBRATION_META_FILE):
        return {}
    try:
        with open(CALIBRATION_META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_calibration(camera_id=0, video_path=None, start_frame=0):
    """Interactive 4-point BEV calibration. Click 4 sidewalk corners.

    For videos: use the trackbar to scrub to a clear, representative frame
    (straight sidewalk visible, no motion blur) before clicking points.

    Click order:
      1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left

    Keys:
      SPACE / trackbar — scrub to a new frame (resets clicked points)
      a/d or ←/→      — move ego anchor
      r               — reset points on current frame
      s / ENTER       — save calibration (requires 4 points)
      q / ESC         — quit without saving
    """
    print("\n=== BEV CALIBRATION MODE ===")
    print("Click 4 points on the sidewalk in order:")
    print("  1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left")
    print("Scrub the trackbar to find a clear frame first.")
    print("Press 'a'/'d' or arrow keys for 1 px fine movement.")
    print("Press 'j'/'l' for 5 px coarse movement.")
    print("You can also click the BEV ego ruler near the bottom-right to set it directly.")
    print("Press 'r' to reset points, 's' or ENTER to save, 'q' to quit.\n")

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video = total_frames > 1

    # Jump to start_frame
    if is_video and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, total_frames - 1))

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame.")
        cap.release()
        return

    current_frame = [int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1]
    points = []
    labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]
    frame_holder = [frame.copy()]
    ego_x_px = [int(round(bev_ego_x_px(BEV_SIZE[0])))]

    def on_trackbar(val):
        cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        ret2, f = cap.read()
        if ret2:
            frame_holder[0] = f.copy()
            current_frame[0] = val
            points.clear()  # reset points when frame changes

    cv2.namedWindow("BEV Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Calibration", 1280, 720)
    if is_video:
        cv2.createTrackbar("Frame", "BEV Calibration", current_frame[0], max(1, total_frames - 1), on_trackbar)

    while True:
        display = frame_holder[0].copy()

        # Draw clicked points
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(display, f"{i+1}: {labels[i]}", (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

        # Draw a small BEV-width ruler that shows the ego anchor position.
        bar_w = min(420, max(180, display.shape[1] // 3))
        bar_h = 18
        bar_x0 = display.shape[1] - bar_w - 20
        bar_y0 = display.shape[0] - 50
        ego_x_bar = int(round(bar_x0 + (ego_x_px[0] / max(1.0, float(BEV_SIZE[0] - 1))) * (bar_w - 1)))
        cv2.rectangle(display, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (80, 80, 80), -1)
        cv2.rectangle(display, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (255, 255, 255), 1)
        cv2.line(display, (ego_x_bar, bar_y0 - 10), (ego_x_bar, bar_y0 + bar_h + 10), (255, 255, 0), 2)
        cv2.putText(display, "BEV ego x", (bar_x0, bar_y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if bar_x0 <= x <= bar_x0 + bar_w and bar_y0 - 14 <= y <= bar_y0 + bar_h + 14:
                frac = float(np.clip((x - bar_x0) / max(1.0, float(bar_w - 1)), 0.0, 1.0))
                ego_x_px[0] = int(round(frac * float(BEV_SIZE[0] - 1)))
                print(f"  Ego anchor x -> {ego_x_px[0]} px")
                return
            if len(points) < 4:
                points.append([x, y])
                print(f"  Point {len(points)}: ({x}, {y}) — {labels[len(points)-1]}")

        cv2.setMouseCallback("BEV Calibration", on_click)

        frame_info = f"Frame {current_frame[0]}/{total_frames}" if is_video else "Live"
        ego_frac = float(ego_x_px[0]) / max(1.0, float(BEV_SIZE[0] - 1))
        info = (
            f"{frame_info} | Points: {len(points)}/4 | bev_ego_x={ego_x_px[0]} px ({ego_frac:.3f}) | "
            "click ruler or a/d/←/→=fine  j/l=coarse  r=reset  s/ENTER=save  q=quit"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(display, info, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("BEV Calibration", display)

        key = cv2.waitKeyEx(30)
        key_low = key & 0xFF
        if key_low in (ord('a'), ord('A')) or key in (2424832, 81):
            ego_x_px[0] = max(0, ego_x_px[0] - 1)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key_low in (ord('d'), ord('D')) or key in (2555904, 83):
            ego_x_px[0] = min(BEV_SIZE[0] - 1, ego_x_px[0] + 1)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key_low in (ord('j'), ord('J')):
            ego_x_px[0] = max(0, ego_x_px[0] - 5)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key_low in (ord('l'), ord('L')):
            ego_x_px[0] = min(BEV_SIZE[0] - 1, ego_x_px[0] + 5)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key_low in (ord('r'), ord('R')):
            points.clear()
            print("  Points reset.")
        elif key_low in (ord('s'), ord('S')) or key in (10, 13):
            if len(points) != 4:
                print(f"  Need 4 points before saving. Current points: {len(points)}/4")
                continue
            src = np.array(points, dtype=np.float32)
            np.save(CALIBRATION_FILE, src)
            print(f"\n  Saved calibration to {CALIBRATION_FILE}")
            print(f"  Source points: {src.tolist()}")
            # Also save H and Hinv as separate .npy for quick loading
            dst = DEFAULT_DST_POINTS
            H = cv2.getPerspectiveTransform(src, dst)
            Hinv = np.linalg.inv(H)
            np.save("bev_H.npy", H)
            np.save("bev_Hinv.npy", Hinv)
            print(f"  Also saved bev_H.npy and bev_Hinv.npy")
            ego_x_frac = float(np.clip(float(ego_x_px[0]) / max(1.0, float(BEV_SIZE[0] - 1)), 0.05, 0.95))
            src_h, src_w = frame_holder[0].shape[:2]
            with open(CALIBRATION_META_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ego_x_frac": ego_x_frac,
                        "source_frame_width": int(src_w),
                        "source_frame_height": int(src_h),
                    },
                    f,
                    indent=2,
                )
            print(
                f"  Saved ego anchor to {CALIBRATION_META_FILE} "
                f"(ego_x_frac={ego_x_frac:.4f}, source={src_w}x{src_h})"
            )
            break
        elif key_low in (ord('q'), ord('Q')) or key == 27:
            print("  Calibration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


def load_bev_params(current_frame_size=None):
    """Load calibration or use defaults. Returns (H, Hinv, src_points)."""
    if os.path.exists(CALIBRATION_FILE):
        src = np.load(CALIBRATION_FILE).astype(np.float32)
        print(f"Loaded BEV calibration from {CALIBRATION_FILE}")
    else:
        src = DEFAULT_SRC_POINTS
        print("Using default BEV calibration (scooter camera). Run --calibrate for iPhone.")

    meta = _load_calibration_meta()
    if current_frame_size is not None:
        try:
            cur_w, cur_h = int(current_frame_size[0]), int(current_frame_size[1])
        except Exception:
            cur_w, cur_h = 0, 0
        src_ref_w = int(meta.get("source_frame_width", 0) or 0)
        src_ref_h = int(meta.get("source_frame_height", 0) or 0)
        if cur_w > 0 and cur_h > 0 and src_ref_w > 0 and src_ref_h > 0:
            if cur_w != src_ref_w or cur_h != src_ref_h:
                sx = float(cur_w) / float(src_ref_w)
                sy = float(cur_h) / float(src_ref_h)
                src = src.copy()
                src[:, 0] *= sx
                src[:, 1] *= sy
                print(
                    f"[BEV] Rescaled calibration points from "
                    f"{src_ref_w}x{src_ref_h} to {cur_w}x{cur_h}"
                )
    dst = DEFAULT_DST_POINTS
    H = cv2.getPerspectiveTransform(src, dst)
    # Validate homography is numerically stable
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        print("[BEV] WARNING: Calibration matrix is near-singular (det={:.2e}). Using defaults.".format(det))
        src = DEFAULT_SRC_POINTS
        H = cv2.getPerspectiveTransform(src, dst)
    cond = np.linalg.cond(H)
    if cond > 1e6:
        print("[BEV] WARNING: Calibration matrix is ill-conditioned (cond={:.1e}). Results may be unstable.".format(cond))
    Hinv = np.linalg.inv(H)
    return H, Hinv, src
