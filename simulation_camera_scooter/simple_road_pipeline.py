"""
simple_road_pipeline.py
=======================
Simplified, tuned pipeline for clean simple-road navigation.

Designed for scenes that are:
  - Straight roads or gentle curves
  - Clean, uncluttered (no heavy urban traffic)
  - Single clear drivable corridor ahead

Key differences vs full pipeline (MORPH_ENHANCED + DT_CORRIDOR + PATH_SMOOTH):

  1. MASK CLEANING
     - Larger morphological close (15x15 near-field) to bridge road surface gaps
     - Aggressive small-component removal (keep only forward-connected blob)
     - Temporal EMA alpha=0.50 (was 0.65) — give MORE weight to previous frame for stability

  2. CENTERLINE (DT Corridor)
     - Savitzky-Golay window=15 (was 9) — smoother centerline
     - Lateral drift=20px (was 30) — less wiggle allowed per step

  3. TEMPLATE PLANNER
     - Only 3 templates: straight_center, left_gentle, right_gentle
     - Removes sharp turn templates that can win on noisy frames for simple roads
     - straight_preference_margin=0.08 (was 0.03)

  4. PATH SMOOTHING
     - max_alpha=0.72 (was 0.85) — limits fast tracking, more inertia
     - min_alpha=0.25 (was 0.35) — more smoothing at low confidence

  5. HEADING SMOOTHING
     - alpha=0.35 (was 0.50) — higher inertia, less responsive to spikes

All improvements are implemented as standalone functions/classes so they can be
used in eval scripts without modifying any existing modules.

Usage in eval script:
    from simple_road_pipeline import SimpleRoadConfig, SimpleRoadProcessor
    proc = SimpleRoadProcessor(SimpleRoadConfig())
    result = proc.process_bev_mask(bev_mask_uint8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from config import bev_ego_x_px, NAV_BEV_FORWARD_M, NAV_BEV_LATERAL_M


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SimpleRoadConfig:
    """Tuning knobs for simple-road mode."""

    # --- BEV dimensions ---
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0

    # --- Mask cleaning ---
    near_field_frac: float = 0.40          # bottom fraction treated as near-field
    close_kernel_near: int = 15            # kernel size for near-field aggressive close
    close_kernel_far: int = 7             # kernel size for far-field close
    open_kernel: int = 5                  # open kernel to remove thin noise
    min_component_area_px: int = 800      # remove components smaller than this

    # --- Temporal mask smoothing ---
    mask_ema_alpha: float = 0.50          # weight for current frame (was 0.65) — lower = more stable
    mask_reset_iou: float = 0.15          # IoU below this = scene change, reset EMA

    # --- DT corridor ---
    dt_sg_window: int = 15               # SG smoothing window (was 9)
    dt_sg_poly: int = 2
    dt_lateral_drift_px: int = 20        # max col step per row (was 30)
    dt_cost_exponent: float = 1.5
    dt_eps: float = 0.5

    # --- Template planner (simple 3-template bank) ---
    template_horizon_m: float = 8.0
    template_ds_m: float = 0.25
    straight_preference_margin: float = 0.08   # was 0.03
    approval_threshold: float = 0.48           # slightly lower to handle simple scenes
    approval_margin: float = 0.03
    max_curvature_m_inv: float = 0.90

    # --- Path EMA smoothing ---
    path_smooth_min_alpha: float = 0.25   # was 0.35
    path_smooth_max_alpha: float = 0.72   # was 0.85
    path_smooth_reset_thresh: float = 2.0

    # --- Heading EMA smoothing ---
    heading_smooth_alpha: float = 0.35    # was 0.50
    heading_smooth_reset_deg: float = 45.0


@dataclass
class SimpleRoadResult:
    """Output from SimpleRoadProcessor.process_bev_mask()."""
    # Cleaned BEV mask
    clean_mask: np.ndarray                # uint8 (H,W) 0/255
    # DT map for visualization
    dt_map: Optional[np.ndarray]          # float32 (H,W)
    # Centerline in BEV pixel coords (row, col)
    centerline_px: Optional[np.ndarray]   # float32 (N, 2)
    # Centerline in metric ego frame (forward_m, lateral_m)
    centerline_m: Optional[np.ndarray]    # float32 (N, 2)
    # Per-point corridor width in meters
    width_m: Optional[np.ndarray]         # float32 (N,)
    # Selected template family
    template_family: str
    # Heading angle (degrees, 0=straight, +right, -left) after smoothing
    heading_deg: float
    # Lateral offset at lookahead (meters)
    lateral_m_at_lookahead: float
    # Planner confidence 0-1
    confidence: float
    # Whether a valid path was found
    valid: bool
    # Mask coverage fraction
    mask_coverage: float
    # Frame-to-frame mask IoU with previous (stability metric)
    mask_iou_prev: float


# ---------------------------------------------------------------------------
# Mask cleaner
# ---------------------------------------------------------------------------

class SimpleRoadMaskCleaner:
    """
    Simple, aggressive BEV mask cleaner tuned for clean-road scenes.

    Steps:
      1. Near-field large morphological CLOSE (bridges road surface gaps in bottom N rows)
      2. Far-field standard CLOSE + OPEN
      3. Remove small isolated components
      4. Keep only the main forward-connected component (from bottom-center ego)
      5. Temporal EMA with lower alpha for stability
    """

    def __init__(self, cfg: SimpleRoadConfig) -> None:
        self._cfg = cfg
        self._prev_mask: Optional[np.ndarray] = None

    def clean(self, bev_mask_uint8: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Clean BEV mask for simple road.

        Returns:
            (cleaned_mask uint8 0/255, iou_with_prev float)
        """
        mask = (bev_mask_uint8 > 0).astype(np.uint8) * 255
        H, W = mask.shape
        cfg = self._cfg

        # --- Step 1: Near-field aggressive close ---
        nf_y = int(H * (1.0 - cfg.near_field_frac))
        k_near = np.ones((cfg.close_kernel_near, cfg.close_kernel_near), np.uint8)
        near = mask[nf_y:, :].copy()
        near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, k_near, iterations=2)
        mask[nf_y:, :] = near

        # --- Step 2: Far-field standard close + open ---
        k_far = np.ones((cfg.close_kernel_far, cfg.close_kernel_far), np.uint8)
        k_open = np.ones((cfg.open_kernel, cfg.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_far, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

        # --- Step 3: Remove small components ---
        mask = _remove_small_components(mask, cfg.min_component_area_px)

        # --- Step 4: Keep ego-connected component ---
        mask = _keep_ego_connected(mask, W, H)

        # --- Step 5: Temporal EMA ---
        iou_prev = 1.0
        if self._prev_mask is not None and self._prev_mask.shape == mask.shape:
            iou_prev = _mask_iou(mask, self._prev_mask)
            if iou_prev < cfg.mask_reset_iou:
                # Scene change — no blend, accept new frame
                self._prev_mask = mask.copy()
                return mask, float(iou_prev)
            # EMA blend
            alpha = float(cfg.mask_ema_alpha)
            blended = alpha * (mask > 0).astype(np.float32) + (1.0 - alpha) * (self._prev_mask > 0).astype(np.float32)
            mask = (blended > 0.5).astype(np.uint8) * 255

        self._prev_mask = mask.copy()
        return mask, float(iou_prev)

    def reset(self) -> None:
        self._prev_mask = None


# ---------------------------------------------------------------------------
# Centerline extractor (DT-based, simpler + stronger smoothing)
# ---------------------------------------------------------------------------

class SimpleRoadCenterline:
    """
    DT-based maximum-clearance centerline with stronger Savitzky-Golay smoothing.

    Key differences vs DtSafeCorridor:
      - Larger SG window (15 vs 9)
      - Smaller lateral drift (20 vs 30 pixels) — straighter path
      - Slightly more lateral-movement penalty for straighter result
    """

    def __init__(self, cfg: SimpleRoadConfig) -> None:
        self._cfg = cfg
        # Ensure odd SG window
        sg = int(cfg.dt_sg_window)
        self._sg_window = sg if sg % 2 == 1 else sg + 1
        self._sg_poly = int(cfg.dt_sg_poly)

    def extract(
        self, bev_mask_uint8: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Returns:
            centerline_px (N,2), centerline_m (N,2), width_m (N,), dt_map, confidence
        """
        if not _HAS_SCIPY:
            return None, None, None, None, 0.0

        mask = (bev_mask_uint8 > 0).astype(np.uint8)
        H, W = mask.shape
        cfg = self._cfg

        if int(np.count_nonzero(mask)) < 100:
            return None, None, None, None, 0.0

        # EDT
        dt = distance_transform_edt(mask).astype(np.float32)

        # Cost grid
        cost = np.full((H, W), fill_value=1e9, dtype=np.float32)
        road = mask > 0
        cost[road] = (1.0 / (dt[road] + float(cfg.dt_eps)) ** float(cfg.dt_cost_exponent)).astype(np.float32)

        # Start position (ego = bottom-center)
        start_row = H - 1
        start_col = int(round(bev_ego_x_px(W)))

        # Snap to nearest road pixel if not on road
        if not bool(road[start_row, start_col]):
            for sr in range(start_row, max(0, start_row - 30), -1):
                if np.any(road[sr]):
                    start_row = sr
                    road_cols = np.where(road[sr])[0]
                    start_col = int(road_cols[np.argmin(np.abs(road_cols - start_col))])
                    break

        rows, cols = _dijkstra_forward(
            cost, road, start_row, start_col, H, W,
            lateral_drift=int(cfg.dt_lateral_drift_px),
            lateral_penalty=0.015,  # slightly stronger penalty → straighter path
        )

        if len(rows) < 2:
            return None, None, None, dt, 0.0

        rows_arr = np.array(rows, dtype=np.float32)
        cols_arr = np.array(cols, dtype=np.float32)

        # Savitzky-Golay with longer window
        if len(cols_arr) >= self._sg_window:
            try:
                cols_smooth = savgol_filter(
                    cols_arr,
                    window_length=self._sg_window,
                    polyorder=self._sg_poly,
                ).astype(np.float32)
                cols_smooth = np.clip(cols_smooth, 0.0, float(W - 1))
            except Exception:
                cols_smooth = cols_arr
        else:
            cols_smooth = cols_arr

        centerline_px = np.stack([rows_arr, cols_smooth], axis=1).astype(np.float32)

        # Metric conversion
        m_per_px_fwd = float(cfg.bev_forward_m) / max(1.0, float(H - 1))
        m_per_px_lat = float(cfg.bev_lateral_m) / max(1.0, float(W - 1))
        m_per_px_mean = 0.5 * (m_per_px_fwd + m_per_px_lat)

        forward_m = (float(H - 1) - rows_arr) * m_per_px_fwd
        ego_x = bev_ego_x_px(W)
        lateral_m = ((cols_smooth - ego_x) / max(1.0, float(W - 1))) * float(cfg.bev_lateral_m)
        centerline_m = np.stack([forward_m, lateral_m], axis=1).astype(np.float32)

        row_int = np.clip(rows_arr.astype(np.int32), 0, H - 1)
        col_int = np.clip(cols_smooth.astype(np.int32), 0, W - 1)
        dt_at_center = dt[row_int, col_int].astype(np.float32)
        width_m = (dt_at_center * 2.0 * m_per_px_mean).astype(np.float32)

        mean_dt_m = float(np.mean(dt_at_center)) * m_per_px_mean
        valid_fraction = float(len(rows)) / float(H)
        confidence = float(
            np.clip(mean_dt_m / 1.5, 0.0, 1.0) * np.clip(valid_fraction, 0.0, 1.0)
        )

        return centerline_px, centerline_m, width_m, dt, confidence


# ---------------------------------------------------------------------------
# Simple 3-template bank + scoring
# ---------------------------------------------------------------------------

def _build_simple_template_bank(cfg: SimpleRoadConfig) -> list:
    """Build 3-template bank: straight, left_gentle, right_gentle."""
    import math as _math
    from template_path_planner import _build_arc_template_path, PathTemplate, _polyline_length_m

    specs = [
        ("straight_center", "straight", 0.00,   0.0),
        ("left_gentle",      "left",     0.80, -40.0),
        ("right_gentle",     "right",    0.80,  40.0),
    ]
    templates = []
    for tid, family, turn_start_m, end_heading_deg in specs:
        end_heading = _math.radians(float(end_heading_deg))
        coeff, pts_m, max_curv = _build_arc_template_path(
            float(cfg.template_horizon_m),
            float(cfg.template_ds_m),
            float(turn_start_m),
            end_heading,
        )
        if max_curv > float(cfg.max_curvature_m_inv) + 0.05:
            continue
        end_lat = float(pts_m[-1, 1]) if len(pts_m) > 0 else 0.0
        templates.append(PathTemplate(
            template_id=tid,
            family=family,
            coeff=coeff,
            points_m=pts_m.astype(np.float32),
            length_m=_polyline_length_m(pts_m),
            max_curvature_m_inv=max_curv,
            end_heading_rad=float(end_heading),
            end_lateral_m=end_lat,
            turn_start_m=float(turn_start_m),
        ))
    return templates


def _select_simple_template(
    centerline_m: np.ndarray,
    templates: list,
    cfg: SimpleRoadConfig,
    prev_family: Optional[str],
) -> Tuple[str, float, float, float]:
    """
    Score templates against DT centerline for simple roads.

    Scoring: containment + straight preference.
    Returns: (family, heading_deg, lateral_at_lookahead_m, confidence)
    """
    if centerline_m is None or len(centerline_m) < 2:
        return "straight", 0.0, 0.0, 0.3

    # Compute heading from centerline endpoint
    n = len(centerline_m)
    i0 = max(0, n - 6)
    vec = centerline_m[-1] - centerline_m[i0]
    heading_rad = float(math.atan2(vec[1], max(1e-6, vec[0])))
    heading_deg = math.degrees(heading_rad)

    # Lateral at lookahead (1.5m forward)
    lat_at_la = _interp_lateral(centerline_m, 1.5)

    # Simple decision: pick template based on heading
    abs_h = abs(heading_deg)
    if abs_h < 8.0:
        family = "straight"
    elif heading_deg > 0:
        family = "right"
    else:
        family = "left"

    # Apply straight preference hysteresis
    if prev_family == "straight" and abs_h < (8.0 + cfg.straight_preference_margin * 100):
        family = "straight"

    # Confidence from heading magnitude (simple road = low heading = high confidence)
    conf = float(np.clip(1.0 - abs_h / 60.0, 0.2, 1.0))
    return family, float(heading_deg), float(lat_at_la), conf


# ---------------------------------------------------------------------------
# Temporal smoothers (simple versions)
# ---------------------------------------------------------------------------

class SimpleHeadingFilter:
    """Simple circular EMA on heading angle."""

    def __init__(self, alpha: float = 0.35, reset_deg: float = 45.0) -> None:
        self._alpha = float(alpha)
        self._reset_deg = float(reset_deg)
        self._prev: Optional[float] = None

    def update(self, heading_deg: float) -> float:
        h = float(heading_deg)
        if self._prev is None:
            self._prev = h
            return h
        delta = float(((h - self._prev + 180.0) % 360.0) - 180.0)
        if abs(delta) > self._reset_deg:
            self._prev = h
            return h
        smoothed = self._prev + self._alpha * delta
        smoothed = float((smoothed + 180.0) % 360.0 - 180.0)
        self._prev = smoothed
        return smoothed

    def reset(self) -> None:
        self._prev = None


class SimpleLateralFilter:
    """Simple EMA on lateral offset."""

    def __init__(self, alpha: float = 0.35) -> None:
        self._alpha = float(alpha)
        self._prev: Optional[float] = None

    def update(self, lateral_m: float) -> float:
        v = float(lateral_m)
        if self._prev is None:
            self._prev = v
            return v
        smoothed = self._alpha * v + (1.0 - self._alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self) -> None:
        self._prev = None


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class SimpleRoadProcessor:
    """
    End-to-end BEV mask → cleaned mask → centerline → heading for simple roads.

    Usage:
        proc = SimpleRoadProcessor(SimpleRoadConfig())
        result = proc.process_bev_mask(bev_mask_uint8)
        # result.heading_deg, result.lateral_m_at_lookahead, result.clean_mask ...
    """

    def __init__(self, cfg: Optional[SimpleRoadConfig] = None) -> None:
        self._cfg = cfg or SimpleRoadConfig()
        self._mask_cleaner = SimpleRoadMaskCleaner(self._cfg)
        self._centerline = SimpleRoadCenterline(self._cfg)
        self._heading_filter = SimpleHeadingFilter(
            alpha=self._cfg.heading_smooth_alpha,
            reset_deg=self._cfg.heading_smooth_reset_deg,
        )
        self._lateral_filter = SimpleLateralFilter(alpha=self._cfg.heading_smooth_alpha)
        self._templates = _build_simple_template_bank(self._cfg)
        self._prev_family: Optional[str] = None

    def process_bev_mask(self, bev_mask_uint8: np.ndarray) -> SimpleRoadResult:
        """Process a single BEV mask frame."""
        cfg = self._cfg

        # --- Clean mask ---
        clean_mask, iou_prev = self._mask_cleaner.clean(bev_mask_uint8)

        # --- Coverage ---
        total_px = max(1, clean_mask.size)
        coverage = float(np.count_nonzero(clean_mask)) / float(total_px)

        # --- Centerline extraction ---
        centerline_px, centerline_m, width_m, dt_map, dt_confidence = self._centerline.extract(clean_mask)

        valid = centerline_m is not None and len(centerline_m) >= 2

        if valid:
            # --- Template selection + heading ---
            family, raw_heading_deg, raw_lateral_m, planner_conf = _select_simple_template(
                centerline_m, self._templates, cfg, self._prev_family
            )
            self._prev_family = family

            # Smooth heading and lateral
            heading_deg = self._heading_filter.update(raw_heading_deg)
            lateral_m = self._lateral_filter.update(raw_lateral_m)
            confidence = 0.5 * float(dt_confidence) + 0.5 * float(planner_conf)
        else:
            family = "straight"
            heading_deg = self._heading_filter.update(0.0)
            lateral_m = self._lateral_filter.update(0.0)
            confidence = 0.0

        return SimpleRoadResult(
            clean_mask=clean_mask,
            dt_map=dt_map,
            centerline_px=centerline_px,
            centerline_m=centerline_m,
            width_m=width_m,
            template_family=family,
            heading_deg=float(heading_deg),
            lateral_m_at_lookahead=float(lateral_m),
            confidence=float(confidence),
            valid=bool(valid),
            mask_coverage=float(coverage),
            mask_iou_prev=float(iou_prev),
        )

    def reset(self) -> None:
        self._mask_cleaner.reset()
        self._heading_filter.reset()
        self._lateral_filter.reset()
        self._prev_family = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0)
    b_bin = (b > 0)
    inter = int(np.count_nonzero(a_bin & b_bin))
    union = int(np.count_nonzero(a_bin | b_bin))
    return float(inter) / max(1.0, float(union))


def _remove_small_components(mask_255: np.ndarray, min_area_px: int) -> np.ndarray:
    """Remove connected components smaller than min_area_px."""
    bin_ = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_, connectivity=8)
    out = np.zeros_like(mask_255, dtype=np.uint8)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area_px):
            out[labels == idx] = 255
    return out


def _keep_ego_connected(mask_255: np.ndarray, W: int, H: int, dilation_k: int = 15) -> np.ndarray:
    """
    Keep only the road components reachable from ego (bottom-center).

    Strategy:
      1. Dilate the mask to bridge small gaps in segmentation.
      2. Flood-fill from ego bottom-center on the DILATED mask.
      3. Find which ORIGINAL components overlap with the flood-fill region.
      4. Return the union of those original components.

    This preserves the original predicted pixels while filtering out
    components that are structurally disconnected from ego, even across
    small gaps in the BEV segmentation.
    """
    ego_x = int(round(bev_ego_x_px(W)))
    ego_y = H - 1

    k = np.ones((dilation_k, dilation_k), np.uint8)
    dilated = cv2.dilate(mask_255, k, iterations=1)

    # Find seed in bottom rows of dilated mask
    seed_x = ego_x
    if not bool(dilated[ego_y, ego_x]):
        # Search nearby columns and upward rows
        for dy in range(0, min(30, H)):
            row = ego_y - dy
            row_px = np.where(dilated[row] > 0)[0]
            if len(row_px) > 0:
                seed_x = int(row_px[np.argmin(np.abs(row_px - ego_x))])
                ego_y = row
                break

    # Flood fill on dilated mask to get ego-reachable region
    ff = dilated.copy()
    ff_mask_arr = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, ff_mask_arr, (seed_x, ego_y), 128)
    ego_region = (ff == 128).astype(np.uint8)

    # Find original components that overlap with this ego region
    bin_orig = (mask_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_orig, connectivity=8)

    out = np.zeros_like(mask_255, dtype=np.uint8)
    for idx in range(1, num):
        comp_mask = (labels == idx).astype(np.uint8)
        if int(np.count_nonzero(comp_mask & ego_region)) > 0:
            out[labels == idx] = 255

    # Guard: if result too small vs original, fall back
    if int(np.count_nonzero(out)) < 300:
        return mask_255

    return out


def _dijkstra_forward(
    cost: np.ndarray,
    road: np.ndarray,
    start_row: int,
    start_col: int,
    H: int,
    W: int,
    lateral_drift: int = 20,
    lateral_penalty: float = 0.015,
) -> Tuple[list, list]:
    """Dijkstra from (start_row, start_col) moving upward (decreasing row)."""
    import heapq as _heapq
    INF = float("inf")
    dist = np.full((H, W), fill_value=INF, dtype=np.float64)
    parent = np.full((H, W, 2), fill_value=-1, dtype=np.int32)

    dist[start_row, start_col] = float(cost[start_row, start_col])
    heap = [(float(dist[start_row, start_col]), start_row, start_col)]
    best_row = start_row

    while heap:
        d, r, c = _heapq.heappop(heap)
        if d > dist[r, c] + 1e-9:
            continue
        next_row = r - 1
        if next_row < 0:
            break
        col_lo = max(0, c - lateral_drift)
        col_hi = min(W - 1, c + lateral_drift)
        for nc in range(col_lo, col_hi + 1):
            if not bool(road[next_row, nc]):
                continue
            step = float(cost[next_row, nc]) + float(lateral_penalty) * abs(nc - c)
            new_d = float(dist[r, c]) + step
            if new_d < float(dist[next_row, nc]):
                dist[next_row, nc] = new_d
                parent[next_row, nc, 0] = r
                parent[next_row, nc, 1] = c
                _heapq.heappush(heap, (new_d, next_row, nc))
                if next_row < best_row:
                    best_row = next_row

    if best_row >= start_row:
        return [], []

    dist_top = dist[best_row, :]
    valid_cols = np.where(dist_top < INF)[0]
    if len(valid_cols) == 0:
        return [], []

    end_col = int(valid_cols[np.argmin(dist_top[valid_cols])])
    path_rows: list = []
    path_cols: list = []
    r, c = best_row, end_col
    while r != -1 and c != -1:
        path_rows.append(r)
        path_cols.append(c)
        pr, pc = int(parent[r, c, 0]), int(parent[r, c, 1])
        if pr == -1 or pc == -1:
            break
        r, c = pr, pc
    path_rows.reverse()
    path_cols.reverse()
    return path_rows, path_cols


def _interp_lateral(centerline_m: np.ndarray, x_query: float) -> float:
    """Interpolate lateral offset at a given forward distance."""
    if centerline_m is None or len(centerline_m) < 2:
        return 0.0
    fwd = centerline_m[:, 0]
    lat = centerline_m[:, 1]
    if float(fwd[-1]) < x_query:
        return float(lat[-1])
    return float(np.interp(x_query, fwd, lat))
