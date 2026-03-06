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


def run_calibration(camera_id=0, video_path=None):
    """Interactive 4-point BEV calibration. Click 4 sidewalk corners."""
    print("\n=== BEV CALIBRATION MODE ===")
    print("Click 4 points on the sidewalk in order:")
    print("  1. Bottom-Left  2. Bottom-Right  3. Top-Right  4. Top-Left")
    print("Press 'r' to reset, 's' to save, 'q' to quit.\n")

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("ERROR: Cannot open camera/video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame.")
        cap.release()
        return

    points = []
    labels = ["Bottom-Left", "Bottom-Right", "Top-Right", "Top-Left"]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            print(f"  Point {len(points)}: ({x}, {y}) - {labels[len(points)-1]}")

    cv2.namedWindow("BEV Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("BEV Calibration", on_click)

    while True:
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(display, f"{i+1}: {labels[i]}", (pt[0]+10, pt[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, tuple(points[i]), tuple(points[i+1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
        info = f"Points: {len(points)}/4 | 'r'=reset 's'=save 'q'=quit"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
