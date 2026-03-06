"""
masks.py
========
BEV mask cleaning, splitting, grass suppression, and component selection.
"""

import cv2
import numpy as np

from config import ROAD_ID, SIDEWALK_ID, DT_CORE_THRESH


def split_masks(model_output):
    """Split segmentation output into sidewalk and road masks (0/255)."""
    m = model_output.astype(np.uint8)
    if set(np.unique(m).tolist()) <= {0, 255}:
        sidewalk = (m > 0).astype(np.uint8) * 255
        road = np.zeros_like(sidewalk)
    else:
        sidewalk = (m == SIDEWALK_ID).astype(np.uint8) * 255
        road = (m == ROAD_ID).astype(np.uint8) * 255
    return sidewalk, road


def suppress_grass_in_mask(frame_bgr, sidewalk_mask,
                           lower_hsv=(30, 40, 40), upper_hsv=(90, 255, 255)):
    """
    Lightweight color-based filter to remove grass-like regions from the sidewalk mask.

    Assumes:
      - frame_bgr is a BGR image from OpenCV
      - sidewalk_mask is a uint8 mask (0/255) where 255 = sidewalk/road
    """
    try:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    except Exception as e:
        print(f"[Mask] HSV filter failed: {e}")
        return sidewalk_mask

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    grass_mask = cv2.inRange(hsv, lower, upper)

    out = sidewalk_mask.copy()
    out[grass_mask > 0] = 0
    return out


def clean_sidewalk_mask(bev_mask_255, dt_thresh=DT_CORE_THRESH):
    """Morphological cleaning + distance-transform core extraction."""
    mask = bev_mask_255.copy().astype(np.uint8)
    # [TIER2] Near-field aggressive closing
    h = mask.shape[0]
    nf_y = h // 2
    k21 = np.ones((21, 21), np.uint8)
    near = mask[nf_y:, :].copy()
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, k21, iterations=1)
    mask[nf_y:, :] = near
    # [/TIER2]
    k7 = np.ones((7, 7), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5, iterations=1)
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    core = (dist > float(dt_thresh)).astype(np.uint8) * 255
    core = cv2.dilate(core, k5, iterations=1)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k5, iterations=1)
    return core


def select_main_component(mask_255, bottom_band_px=45, center_weight=0.35):
    """Select the most likely navigable connected component."""
    bin_ = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if num <= 1:
        return mask_255
    H, W = mask_255.shape
    bottom_band_px = int(max(1, min(H, bottom_band_px)))
    bottom_slice = labels[max(0, H - bottom_band_px):, :]
    center_x = W / 2.0
    best_label, best_score = None, -1.0
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        touches_bottom = bool(np.any(bottom_slice == idx))
        cx, _ = centroids[idx]
        center_bonus = 1.0 - min(1.0, abs(cx - center_x) / (center_x + 1e-6))
        score = area
        if touches_bottom:
            score += area * 0.5
        score += area * float(center_weight) * max(0.0, center_bonus)
        if score > best_score:
            best_score = score
            best_label = idx
    if best_label is None:
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_255, dtype=np.uint8)
    out[labels == best_label] = 255
    return out


def anchor_ego_to_mask(
    bev_mask_255,
    bev_size=(600, 500),
    search_half_width_px=70,
    search_up_px=90,
    max_bridge_px=75,
    bridge_thickness_px=1,
):
    """[TIER2] Add a thin ego bridge to nearest nearby mask pixel.

    This avoids injecting a synthetic bottom rectangle that can create side
    "wings" and fake curvature near the start of the path.
    """
    bev_w, bev_h = bev_size
    ego_x = int(bev_w // 2)
    ego_y = int(bev_h - 1)
    mask = (bev_mask_255 > 0)

    x0 = max(0, ego_x - int(search_half_width_px))
    x1 = min(bev_w, ego_x + int(search_half_width_px) + 1)
    y0 = max(0, ego_y - int(search_up_px))
    y1 = min(bev_h, ego_y + 1)
    roi = mask[y0:y1, x0:x1]

    ys, xs = np.where(roi)
    if len(ys) == 0:
        return bev_mask_255

    pts_y = ys + y0
    pts_x = xs + x0
    d2 = (pts_x - ego_x) ** 2 + (pts_y - ego_y) ** 2
    idx = int(np.argmin(d2))
    tx = int(pts_x[idx])
    ty = int(pts_y[idx])
    if float(np.sqrt(d2[idx])) > float(max_bridge_px):
        return bev_mask_255

    out = bev_mask_255.copy()
    cv2.line(out, (ego_x, ego_y), (tx, ty), 255, int(max(1, bridge_thickness_px)))
    return out


def ego_connected_mask(bev_mask_255, bev_size=(600, 500), dilation_k=15):
    """[TIER2] Return only sidewalk connected to ego (flood fill from bottom-center).

    Guardrail: if flood-fill result is tiny/anchor-only with almost no forward reach,
    fall back to the original BEV mask to avoid collapsing path extraction.
    """
    bev_w, bev_h = bev_size
    ego_x, ego_y = bev_w // 2, bev_h - 1   # (300, 499)
    k = np.ones((dilation_k, dilation_k), np.uint8)
    dilated = cv2.dilate(bev_mask_255, k, iterations=1)
    if dilated[ego_y, ego_x] == 0:
        r = dilation_k
        y0, x0 = max(0, ego_y - r), max(0, ego_x - r)
        region = dilated[y0:ego_y + 1, x0:ego_x + r + 1]
        ys, xs = np.where(region > 0)
        if len(ys) == 0:
            return bev_mask_255
        dists = (ys + y0 - ego_y) ** 2 + (xs + x0 - ego_x) ** 2
        seed_y = int(ys[np.argmin(dists)]) + y0
        seed_x = int(xs[np.argmin(dists)]) + x0
    else:
        seed_y, seed_x = ego_y, ego_x
    ff = dilated.copy()
    ff_mask_arr = np.zeros((bev_h + 2, bev_w + 2), np.uint8)
    cv2.floodFill(ff, ff_mask_arr, (seed_x, seed_y), 128)

    connected = (ff == 128).astype(np.uint8) * 255
    ys_conn, _ = np.where(connected > 0)
    if len(ys_conn) == 0:
        return bev_mask_255

    connected_area = int(len(ys_conn))
    forward_reach_px = int(ego_y - int(np.min(ys_conn)))
    if connected_area < 1200 or forward_reach_px < 35:
        return bev_mask_255

    return connected
