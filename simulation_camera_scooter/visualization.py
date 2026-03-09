"""
visualization.py
================
HUD drawing functions for camera and BEV views.
"""

import math

import cv2
import numpy as np

from config import (
    COLOR_OBJ_BOX,
    COLOR_OBJ_WARN,
    OBSTACLE_CLOSE_M,
    PATH_COLORS,
)


def draw_heading_hud(img, command, angle_deg, speed_mps, color, fps, pipeline_ms,
                     detections=None, gps_info=None):
    """Draw heading, speed, obstacles, and GPS on the camera view."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---- Large command text at top center ----
    text = command
    (tw, th), baseline = cv2.getTextSize(text, font, 1.8, 4)
    tx = (w - tw) // 2
    ty = 60
    cv2.rectangle(img, (tx - 15, ty - th - 15), (tx + tw + 15, ty + baseline + 15),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, 1.8, color, 4, cv2.LINE_AA)

    # ---- Angle + speed below command ----
    cmd_line = f"Steer: {angle_deg:+.1f} deg | Speed: {speed_mps:.2f} m/s"
    (cw, ch), _ = cv2.getTextSize(cmd_line, font, 0.7, 2)
    cx = (w - cw) // 2
    cy = ty + 40
    cv2.rectangle(img, (cx - 8, cy - ch - 5), (cx + cw + 8, cy + 8), (0, 0, 0), -1)
    cv2.putText(img, cmd_line, (cx, cy), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- Serial command format ----
    serial_cmd = f"CMD,{angle_deg:+.1f},{speed_mps:.2f}"
    (sw, sh), _ = cv2.getTextSize(serial_cmd, font, 0.55, 1)
    sx = (w - sw) // 2
    sy = cy + 30
    cv2.putText(img, serial_cmd, (sx, sy), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # ---- Direction arrow (with dark outline for visibility) ----
    arrow_cx = w // 2
    arrow_cy = h // 2
    arrow_len = 140
    arrow_angle_rad = math.radians(-angle_deg)
    arrow_ex = int(arrow_cx + arrow_len * math.sin(arrow_angle_rad))
    arrow_ey = int(arrow_cy - arrow_len * math.cos(arrow_angle_rad))
    cv2.arrowedLine(img, (arrow_cx, arrow_cy + 30), (arrow_ex, arrow_ey - 30),
                    (0, 0, 0), 10, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(img, (arrow_cx, arrow_cy + 30), (arrow_ex, arrow_ey - 30),
                    color, 6, cv2.LINE_AA, tipLength=0.3)

    # ---- Object detection boxes ----
    if detections:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            dist = det.get("distance_m", None)
            label = f"{det['class_name']} {det['conf']:.0%}"
            if dist is not None:
                label += f" ~{dist:.1f}m"
            box_color = COLOR_OBJ_WARN if (dist and dist < OBSTACLE_CLOSE_M) else COLOR_OBJ_BOX
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(img, label, (x1, y1 - 8), font, 0.5, box_color, 1, cv2.LINE_AA)

    # ---- GPS info (top-right) ----
    if gps_info:
        gps_lines = gps_info if isinstance(gps_info, list) else [gps_info]
        for i, line in enumerate(gps_lines):
            cv2.putText(img, line, (w - 350, 30 + i * 22),
                        font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # ---- FPS / latency at bottom left ----
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Pipeline: {pipeline_ms:.0f} ms",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (10, h - 20 - i * 25),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    return img


def draw_bev_hud(
    bev_rgb,
    paths,
    best_idx,
    skel,
    bev_sidewalk,
    command,
    angle_deg,
    speed_mps=0.0,
    path_source=None,
    mask_occ_ratio=None,
    obstacle_zones_m=None,
):
    """Draw BEV visualization with paths, heading, and speed."""
    h_bev, w_bev = bev_sidewalk.shape
    # Dark background so sidewalk and paths pop
    vis = np.full((h_bev, w_bev, 3), (40, 40, 40), dtype=np.uint8)
    # Sidewalk in visible teal/green
    vis[bev_sidewalk > 0] = (80, 160, 80)

    # Skeleton in bright white, thicker for visibility
    if skel is not None:
        skel_thick = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        vis[skel_thick > 0] = (200, 200, 200)

    # Draw all candidate paths
    for idx, (path_pts, plen) in enumerate(paths):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        thickness = 6 if idx == best_idx else 3
        path_np = np.int32(path_pts).reshape(-1, 1, 2)
        cv2.polylines(vis, [path_np], False, color, thickness, cv2.LINE_AA)

    # Draw best path thick and bright green
    if paths and best_idx >= 0:
        best_path = paths[best_idx][0]
        path_np = np.int32(best_path).reshape(-1, 1, 2)
        cv2.polylines(vis, [path_np], False, (0, 255, 0), 10, cv2.LINE_AA)
        # Draw start/end circles
        cv2.circle(vis, tuple(np.int32(best_path[0])), 10, (0, 255, 255), -1)
        cv2.circle(vis, tuple(np.int32(best_path[-1])), 10, (0, 0, 255), -1)

    # Ego indicator at bottom center
    ego_x, ego_y = w_bev // 2, h_bev - 15
    cv2.circle(vis, (ego_x, ego_y), 12, (0, 200, 255), -1)
    cv2.circle(vis, (ego_x, ego_y), 12, (255, 255, 255), 2)

    # Direction arrow from ego
    arrow_len = 60
    arrow_rad = math.radians(-angle_deg)
    ax = int(ego_x + arrow_len * math.sin(arrow_rad))
    ay = int(ego_y - arrow_len * math.cos(arrow_rad))
    cv2.arrowedLine(vis, (ego_x, ego_y), (ax, ay), (0, 200, 255), 4,
                    cv2.LINE_AA, tipLength=0.35)

    # Semi-transparent black bar for text
    cv2.rectangle(vis, (0, 0), (w_bev, 90), (0, 0, 0), -1)

    # Command + speed text
    cv2.putText(vis, f"{command} ({angle_deg:+.1f} deg)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Speed: {speed_mps:.2f} m/s", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    if path_source is not None or mask_occ_ratio is not None:
        ptxt = str(path_source) if path_source is not None else "-"
        otxt = f"{float(mask_occ_ratio) * 100.0:.1f}%" if mask_occ_ratio is not None else "-"
        cv2.putText(vis, f"Src: {ptxt} | Occ: {otxt}", (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 230, 170), 1, cv2.LINE_AA)

    # Obstacle exclusion zone circles (Phase 03.1) -- only drawn when provided
    if obstacle_zones_m:
        from config import NAV_BEV_FORWARD_M, NAV_BEV_LATERAL_M, BEV_HARD_BLOCK_DIST_M
        for fwd_m, lat_m, rad_m in obstacle_zones_m:
            col = int((lat_m / max(1e-6, NAV_BEV_LATERAL_M) + 0.5) * max(1, w_bev - 1))
            row = int((1.0 - fwd_m / max(1e-6, NAV_BEV_FORWARD_M)) * max(1, h_bev - 1))
            r_px = max(3, int(rad_m * h_bev / max(1e-6, NAV_BEV_FORWARD_M)))
            col = int(np.clip(col, 0, w_bev - 1))
            row = int(np.clip(row, 0, h_bev - 1))
            # Red = stop zone, orange = approaching
            color = (0, 0, 255) if fwd_m < BEV_HARD_BLOCK_DIST_M else (0, 165, 255)
            cv2.circle(vis, (col, row), r_px, color, 2)
            cv2.circle(vis, (col, row), 3, color, -1)

    return vis
