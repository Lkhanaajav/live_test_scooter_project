"""
masks.py
========
BEV mask cleaning, splitting, grass suppression, and component selection.

Research impl: Enhanced Morphological BEV Mask Pipeline (Idea 1)
  - clean_bev_mask_enhanced() adds flood-fill hole filling, Gaussian edge smoothing,
    and DT ego-clearance component selection on top of the existing pipeline.
  - Papers: Road Seg ADAS (arXiv:2505.12206), Skelite (arXiv:2503.07369)
"""

import cv2
import numpy as np

from config import (
    ROAD_ID,
    SIDEWALK_ID,
    DT_CORE_THRESH,
    MORPH_ENHANCED,
    MORPH_HOLE_FILL_MAX_M2,
    MORPH_GAUSS_SIGMA_PX,
    MORPH_GAUSS_THRESH,
    MORPH_EGO_BAND_FRAC,
)


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


def _flood_fill_holes(mask_255: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    """
    Research impl: Flood-fill enclosed holes from corners.
    Fills any hole enclosed by road pixels whose area <= max_hole_area_px.

    Standard morphological close misses holes that are larger than the kernel
    but smaller than the overall corridor. This approach correctly identifies
    holes by flood-filling from all four corners (= background) and then
    inverting to find truly enclosed regions.

    Technique from: Road Segmentation for ADAS/AD (arXiv:2505.12206)
    """
    h, w = mask_255.shape
    inv = cv2.bitwise_not((mask_255 > 0).astype(np.uint8) * 255)

    # Flood from top-left corner to mark all background pixels reachable from border
    ff = inv.copy()
    seed_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, seed_mask, (0, 0), 128)

    # Any remaining non-zero, non-128 pixel in ff is an enclosed hole
    enclosed = ((inv > 0) & (ff != 128)).astype(np.uint8)
    if int(np.count_nonzero(enclosed)) == 0:
        return mask_255

    # Fill only small holes (avoid filling the far-field open corridor end)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)
    out = mask_255.copy()
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) <= max_hole_area_px:
            out[labels == idx] = 255
    return out


def _select_component_by_ego_clearance(
    mask_255: np.ndarray,
    ego_band_frac: float = 0.20,
) -> np.ndarray:
    """
    Research impl: Select the navigable component using DT ego-clearance score.

    Instead of simply taking the largest-area component, we compute the distance
    transform and select the component whose maximum DT value in the bottom
    ego_band_frac rows is highest. This directly measures path clearance near
    the scooter's current position — the most safety-relevant metric.

    Technique from: Road Segmentation for ADAS/AD (arXiv:2505.12206)
    """
    bin_ = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    if num <= 1:
        return mask_255

    h, w = mask_255.shape
    ego_band_y0 = max(0, int(h * (1.0 - float(ego_band_frac))))

    # Compute EDT once on full mask for efficiency
    dt = cv2.distanceTransform(bin_, cv2.DIST_L2, 5)
    ego_dt = dt[ego_band_y0:, :]

    best_label: int | None = None
    best_score = -1.0

    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        # Ego-clearance score: max DT value in ego band for this component
        comp_ego = (labels[ego_band_y0:, :] == idx).astype(np.float32)
        max_dt_ego = float(np.max(ego_dt * comp_ego)) if np.any(comp_ego > 0) else 0.0

        # Tie-break with area (encourages connected corridor over isolated fragment)
        score = max_dt_ego * 1000.0 + area * 0.001
        if score > best_score:
            best_score = score
            best_label = idx

    if best_label is None:
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    out = np.zeros_like(mask_255, dtype=np.uint8)
    out[labels == best_label] = 255
    return out


def clean_bev_mask_enhanced(
    mask: np.ndarray,
    m_per_px: float,
    min_width_m: float = 0.50,
    enhanced: bool = True,
) -> np.ndarray:
    """
    Research impl: Enhanced morphological BEV mask cleaning pipeline.

    Extends the existing clean_sidewalk_mask() with:
      (a) Flood-fill hole filling from corners — fills enclosed holes that
          standard morphological close cannot reach.
      (b) Gaussian boundary smoothing — reduces jagged edges and 1-pixel
          protrusions from segmentation quantization artifacts.
      (c) DT ego-clearance component selection — picks the navigable
          component by maximum clearance at the scooter position, not by
          largest area.

    When enhanced=False, delegates to the original clean_sidewalk_mask().

    Args:
        mask:        Input BEV mask (any dtype, >0 = road).
        m_per_px:    Meters per pixel (for converting area thresholds).
        min_width_m: Minimum road width in meters for open operation.
        enhanced:    If False, fall back to original clean_sidewalk_mask().

    Returns:
        Cleaned binary mask (uint8, 0/255).

    Papers:
        Road Segmentation for ADAS/AD (arXiv:2505.12206)
        Skelite topology pruning (arXiv:2503.07369)
    """
    mask_255 = (mask > 0).astype(np.uint8) * 255

    if not enhanced:
        # Research impl: fallback path — original pipeline
        return clean_sidewalk_mask(mask_255)

    # --- Step 1: Standard morphological close + open (same as existing) ---
    k7 = np.ones((7, 7), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, k7, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k5, iterations=1)

    # --- Step 2: [NEW] Flood-fill enclosed hole filling ---
    # Holes smaller than MORPH_HOLE_FILL_MAX_M2 are filled.
    # This addresses mask holes from segmentation that standard close misses.
    max_hole_area_px = max(
        1,
        int(round(float(MORPH_HOLE_FILL_MAX_M2) / max(1e-12, float(m_per_px) ** 2))),
    )
    cleaned = _flood_fill_holes(cleaned, max_hole_area_px)

    # --- Step 3: [NEW] Gaussian boundary smoothing before re-binarize ---
    # Smooth jagged mask edges by blurring and re-thresholding.
    # sigma=MORPH_GAUSS_SIGMA_PX is conservative (~2 cm at typical BEV scale).
    sigma = max(0.5, float(MORPH_GAUSS_SIGMA_PX))
    blur_float = cv2.GaussianBlur(
        (cleaned > 0).astype(np.float32),
        (0, 0),
        sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    cleaned = (blur_float >= float(MORPH_GAUSS_THRESH)).astype(np.uint8) * 255

    # --- Step 4: [NEW] Component selection by DT ego-clearance score ---
    # Select the component that provides the most clearance near the ego position,
    # rather than simply the largest component.
    cleaned = _select_component_by_ego_clearance(cleaned, float(MORPH_EGO_BAND_FRAC))

    # --- Step 5: Light minimum-width enforcement ---
    # Remove sub-path-width noise while preserving thin genuine corridors.
    if m_per_px > 1e-9:
        half_width_px = max(1, int(round((float(min_width_m) * 0.5) / float(m_per_px))))
        k_w = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * half_width_px + 1, 2 * half_width_px + 1)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_w, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_w, iterations=1)

    return (cleaned > 0).astype(np.uint8) * 255


def anchor_ego_to_mask(
    bev_mask_255,
    bev_size=(600, 500),
    search_half_width_px=70,
    search_up_px=90,
    max_bridge_px=75,
    bridge_thickness_px=1,
    center_bias_px=1.35,
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
    # Bias bridge target toward centerline so the anchor doesn't seed
    # right/left "fake turn" geometry when side fragments are closer.
    score = d2 + float(center_bias_px) * ((pts_x - ego_x) ** 2)
    idx = int(np.argmin(score))
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

    # If flood-filled ego component is much shorter/smaller than global BEV
    # support, it is likely an anchor/triangle fragment; use main component.
    ys_all, _ = np.where(bev_mask_255 > 0)
    if len(ys_all) > 0:
        global_reach_px = int(ego_y - int(np.min(ys_all)))
        global_area = int(len(ys_all))
        short_vs_global = (
            global_reach_px >= 70
            and forward_reach_px < int(max(45, 0.55 * float(global_reach_px)))
        )
        tiny_vs_global = (
            global_area >= 2500
            and connected_area < int(0.35 * float(global_area))
        )
        if short_vs_global or tiny_vs_global:
            return select_main_component(bev_mask_255, bottom_band_px=55, center_weight=0.45)

    return connected
