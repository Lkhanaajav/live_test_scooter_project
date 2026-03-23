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


def run_calibration(camera_id=0, video_path=None, start_frame=0):
    """Interactive 4-point BEV calibration. Click 4 sidewalk corners.

    For videos: use the trackbar to scrub to a clear, representative frame
    (straight sidewalk visible, no motion blur) before clicking points.

    Click order:
      1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left

    Keys:
      SPACE / trackbar — scrub to a new frame (resets clicked points)
      r               — reset points on current frame
      s               — save calibration (requires 4 points)
      q / ESC         — quit without saving
    """
    print("\n=== BEV CALIBRATION MODE ===")
    print("Click 4 points on the sidewalk in order:")
    print("  1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left")
    print("Scrub the trackbar to find a clear frame first.")
    print("Press 'r' to reset points, 's' to save, 'q' to quit.\n")

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

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"  Point {len(points)}: ({x}, {y}) — {labels[len(points)-1]}")

    cv2.namedWindow("BEV Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Calibration", 1280, 720)
    cv2.setMouseCallback("BEV Calibration", on_click)
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

        frame_info = f"Frame {current_frame[0]}/{total_frames}" if is_video else "Live"
        info = (
            f"{frame_info} | Points: {len(points)}/4 | bev_ego_x={ego_x_px[0]} px | "
            "a/d=move ego  r=reset  s=save  q=quit"
        )
        cv2.rectangle(display, (0, 0), (display.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(display, info, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("BEV Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('a'):
            ego_x_px[0] = max(0, ego_x_px[0] - 5)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key == ord('d'):
            ego_x_px[0] = min(BEV_SIZE[0] - 1, ego_x_px[0] + 5)
            print(f"  Ego anchor x -> {ego_x_px[0]} px")
        elif key == ord('r'):
            points.clear()
            print("  Points reset.")
        elif key == ord('s') and len(points) == 4:
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
            calib_h, calib_w = frame_holder[0].shape[:2]
            with open(CALIBRATION_META_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ego_x_frac": ego_x_frac,
                        "source_width": int(calib_w),
                        "source_height": int(calib_h),
                    },
                    f,
                    indent=2,
                )
            print(
                f"  Saved ego anchor to {CALIBRATION_META_FILE} "
                f"(ego_x_frac={ego_x_frac:.4f}, source={calib_w}x{calib_h})"
            )
            break
        elif key == ord('q') or key == 27:
            print("  Calibration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


def load_bev_params(frame_size=None):
    """Load calibration or use defaults. Returns (H, Hinv, src_points)."""
    if os.path.exists(CALIBRATION_FILE):
        src = np.load(CALIBRATION_FILE).astype(np.float32)
        print(f"Loaded BEV calibration from {CALIBRATION_FILE}")
    else:
        src = DEFAULT_SRC_POINTS
        print("Using default BEV calibration (scooter camera). Run --calibrate for iPhone.")

    if frame_size is not None and len(frame_size) == 2:
        frame_w = max(1.0, float(frame_size[0]))
        frame_h = max(1.0, float(frame_size[1]))
        source_w = None
        source_h = None
        if os.path.exists(CALIBRATION_META_FILE):
            try:
                with open(CALIBRATION_META_FILE, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                source_w = float(meta.get("source_width") or 0.0) or None
                source_h = float(meta.get("source_height") or 0.0) or None
            except Exception:
                source_w = None
                source_h = None

        if source_w and source_h and (abs(source_w - frame_w) > 1 or abs(source_h - frame_h) > 1):
            sx = frame_w / source_w
            sy = frame_h / source_h
            src = src.copy()
            src[:, 0] *= sx
            src[:, 1] *= sy
            print(
                f"[BEV] Scaled calibration points from {int(source_w)}x{int(source_h)} "
                f"to {int(frame_w)}x{int(frame_h)}"
            )
    dst = DEFAULT_DST_POINTS
    if os.path.exists(CALIBRATION_META_FILE):
        try:
            with open(CALIBRATION_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_bev_w = int(meta.get("bev_width") or 0)
            meta_bev_h = int(meta.get("bev_height") or 0)
            if meta_bev_w and meta_bev_h and (meta_bev_w != int(BEV_SIZE[0]) or meta_bev_h != int(BEV_SIZE[1])):
                print(
                    f"[BEV] WARNING: Calibration was created with BEV_SIZE={meta_bev_w}x{meta_bev_h} "
                    f"but runtime is using {int(BEV_SIZE[0])}x{int(BEV_SIZE[1])}."
                )
        except Exception:
            pass
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
