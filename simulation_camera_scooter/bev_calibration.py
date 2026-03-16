"""
bev_calibration.py
==================
BEV (Bird's Eye View) calibration tool and parameter loader.
"""

import os
import cv2
import numpy as np

from config import (
    DEFAULT_SRC_POINTS,
    DEFAULT_DST_POINTS,
    CALIBRATION_FILE,
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

        frame_info = f"Frame {current_frame[0]}/{total_frames}" if is_video else "Live"
        info = f"{frame_info} | Points: {len(points)}/4 | r=reset  s=save  q=quit"
        cv2.rectangle(display, (0, 0), (display.shape[1], 38), (0, 0, 0), -1)
        cv2.putText(display, info, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("BEV Calibration", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('r'):
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
            break
        elif key == ord('q') or key == 27:
            print("  Calibration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


def load_bev_params():
    """Load calibration or use defaults. Returns (H, Hinv, src_points)."""
    if os.path.exists(CALIBRATION_FILE):
        src = np.load(CALIBRATION_FILE).astype(np.float32)
        print(f"Loaded BEV calibration from {CALIBRATION_FILE}")
    else:
        src = DEFAULT_SRC_POINTS
        print("Using default BEV calibration (scooter camera). Run --calibrate for iPhone.")
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
