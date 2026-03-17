"""
safe_corridor.py
================
Research impl: Distance Transform Safe Corridor Extraction (Idea 2).

Replaces/supplements the fragile row-wise corridor_from_mask() approach with a
Dijkstra-on-cost-grid algorithm that finds the globally maximum-clearance path
through the BEV mask.

Key idea:
  dt = scipy.ndimage.distance_transform_edt(bev_mask)
  cost = 1 / (dt + eps) ** alpha        # low cost = high clearance = preferred
  Path = Dijkstra from ego (bottom-center) upward through BEV

This mirrors the ESDF-based corridor planning used in real-time robotics
and the Dual-BEV navigation approach from arXiv:2501.18351.

Papers:
  Dual-BEV Navigation (arXiv:2501.18351)
  ESDF corridor planning (robotics literature)
  Road Segmentation for ADAS (arXiv:2505.12206)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from config import (
    DT_CORRIDOR_ENABLED,
    DT_CORRIDOR_COST_EXPONENT,
    DT_CORRIDOR_LATERAL_DRIFT_PX,
    DT_CORRIDOR_SG_WINDOW,
    DT_CORRIDOR_CONFIDENCE_NORM_M,
    BEV_PIXELS_PER_METER_FORWARD,
    BEV_PIXELS_PER_METER_LATERAL,
    NAV_BEV_FORWARD_M,
    NAV_BEV_LATERAL_M,
    bev_ego_x_px,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DtCorridorResult:
    """
    Research impl: Result of DT-safe-corridor extraction.

    Fields
    ------
    centerline_px : np.ndarray, shape (N, 2)
        Centerline in pixel coordinates as (row, col).
    width_m_per_point : np.ndarray, shape (N,)
        Clearance-to-boundary in meters at each centerline point (= dt * m_per_px * 2).
    confidence : float, range [0, 1]
        Quality estimate. Higher means wider corridor and longer valid path.
    dt_map : np.ndarray, shape (H, W), float32
        Full Euclidean distance transform of the BEV mask (pixels).
    centerline_m : np.ndarray, shape (N, 2)
        Centerline in metric ego frame: column 0 = forward_m, column 1 = lateral_m.
    """
    centerline_px: np.ndarray       # (N, 2) in (row, col)
    width_m_per_point: np.ndarray   # (N,) clearance in meters at each point
    confidence: float               # 0–1
    dt_map: np.ndarray              # full EDT map (for debug/visualization)
    centerline_m: np.ndarray        # (N, 2) in (forward_m, lateral_m)


# ---------------------------------------------------------------------------
# DtSafeCorridor class
# ---------------------------------------------------------------------------

class DtSafeCorridor:
    """
    Research impl: Dijkstra-based maximum-clearance centerline extractor.

    Usage
    -----
    corridor = DtSafeCorridor()
    result = corridor.extract(bev_mask_uint8)

    The `extract()` method accepts any BEV mask (uint8, >0 = drivable) and
    returns a DtCorridorResult with the globally safest centerline.

    Algorithm
    ---------
    1. Compute EDT on the BEV mask.
    2. Build a cost grid: cost = 1 / (dt + eps)^alpha.
       Pixels outside the mask get infinite cost.
    3. Run Dijkstra from ego (bottom-center of BEV) forward through the mask.
       At each step, move one row upward and allow ±lateral_drift_px lateral shift.
    4. Backtrack the minimum-cost path to obtain centerline pixel coordinates.
    5. Smooth with Savitzky-Golay filter to remove grid quantization jitter.
    6. Convert to metric ego frame and compute per-point width and confidence.
    """

    def __init__(
        self,
        bev_forward_m: float = NAV_BEV_FORWARD_M,
        bev_lateral_m: float = NAV_BEV_LATERAL_M,
        cost_exponent: float = DT_CORRIDOR_COST_EXPONENT,
        lateral_drift_px: int = DT_CORRIDOR_LATERAL_DRIFT_PX,
        sg_window: int = DT_CORRIDOR_SG_WINDOW,
        sg_poly: int = 2,
        eps: float = 0.5,
    ) -> None:
        self.bev_forward_m = float(bev_forward_m)
        self.bev_lateral_m = float(bev_lateral_m)
        self.cost_exponent = float(cost_exponent)
        self.lateral_drift_px = int(max(1, lateral_drift_px))
        # Savitzky-Golay params — window must be odd and >= poly+2
        self.sg_window = int(sg_window) if int(sg_window) % 2 == 1 else int(sg_window) + 1
        self.sg_poly = int(sg_poly)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        bev_mask_uint8: np.ndarray,
        prev_centerline_px: Optional[np.ndarray] = None,
    ) -> DtCorridorResult:
        """
        Extract the maximum-clearance centerline from a BEV mask.

        Parameters
        ----------
        bev_mask_uint8 : np.ndarray, shape (H, W), uint8
            BEV road mask; >0 pixels are drivable.
        prev_centerline_px : optional np.ndarray
            Previous frame's centerline (row, col) for continuity seeding.
            Currently reserved for future use (warm-start Dijkstra).

        Returns
        -------
        DtCorridorResult
        """
        if not _HAS_SCIPY:
            return self._fallback_result(bev_mask_uint8)

        mask = (bev_mask_uint8 > 0).astype(np.uint8)
        H, W = mask.shape

        # --- Step 1: EDT ---
        dt = distance_transform_edt(mask).astype(np.float32)

        # --- Step 2: Cost grid ---
        # Pixels outside mask get infinite cost so Dijkstra never routes through them
        cost = np.full((H, W), fill_value=1e9, dtype=np.float32)
        road = mask > 0
        cost[road] = (1.0 / (dt[road] + self.eps) ** self.cost_exponent).astype(np.float32)

        # --- Step 3: Dijkstra forward from ego ---
        start_row = H - 1
        start_col = int(round(bev_ego_x_px(W)))

        # If ego position is not on road, find nearest road pixel in bottom band.
        # Search upward row by row (up to 30px) to handle cases where the mask
        # doesn't reach the very bottom of the BEV (common in real BEV images).
        if not bool(road[start_row, start_col]):
            found = False
            for search_row in range(start_row, max(0, start_row - 30), -1):
                if np.any(road[search_row]):
                    start_row = search_row
                    start_col = self._find_nearest_road_col(road, search_row, W)
                    found = True
                    break
            if not found:
                start_col = self._find_nearest_road_col(road, start_row, W)

        centerline_rows, centerline_cols = self._dijkstra_forward(
            cost, road, start_row, start_col, H, W
        )

        if len(centerline_rows) < 2:
            return self._fallback_result(bev_mask_uint8, dt)

        # --- Step 4+5: Savitzky-Golay smoothing of column coordinates ---
        rows_arr = np.array(centerline_rows, dtype=np.float32)
        cols_arr = np.array(centerline_cols, dtype=np.float32)

        if len(cols_arr) >= self.sg_window:
            try:
                cols_smooth = savgol_filter(
                    cols_arr, window_length=self.sg_window, polyorder=self.sg_poly
                ).astype(np.float32)
                cols_smooth = np.clip(cols_smooth, 0.0, float(W - 1))
            except Exception:
                cols_smooth = cols_arr
        else:
            cols_smooth = cols_arr

        centerline_px = np.stack([rows_arr, cols_smooth], axis=1).astype(np.float32)

        # --- Step 6: Convert to metric and compute width/confidence ---
        m_per_px_fwd = self.bev_forward_m / max(1.0, float(H - 1))
        m_per_px_lat = self.bev_lateral_m / max(1.0, float(W - 1))
        m_per_px_mean = 0.5 * (m_per_px_fwd + m_per_px_lat)

        # Metric: forward = distance from bottom (ego), lateral = offset from center
        forward_m = (float(H - 1) - rows_arr) * m_per_px_fwd
        ego_x = bev_ego_x_px(W)
        lateral_m = ((cols_smooth - ego_x) / max(1.0, float(W - 1))) * self.bev_lateral_m
        centerline_m = np.stack([forward_m, lateral_m], axis=1).astype(np.float32)

        # Per-point width estimate: 2 * DT_at_centerline * m_per_px
        row_int = np.clip(rows_arr.astype(np.int32), 0, H - 1)
        col_int = np.clip(cols_smooth.astype(np.int32), 0, W - 1)
        dt_at_center = dt[row_int, col_int].astype(np.float32)
        width_m = (dt_at_center * 2.0 * m_per_px_mean).astype(np.float32)

        # Confidence: mean clearance normalized by CONFIDENCE_NORM_M, times valid fraction
        mean_dt_m = float(np.mean(dt_at_center)) * m_per_px_mean
        valid_fraction = float(len(centerline_rows)) / float(H)
        confidence = float(
            np.clip(mean_dt_m / max(1e-6, float(DT_CORRIDOR_CONFIDENCE_NORM_M)), 0.0, 1.0)
            * np.clip(valid_fraction, 0.0, 1.0)
        )

        return DtCorridorResult(
            centerline_px=centerline_px,
            width_m_per_point=width_m,
            confidence=confidence,
            dt_map=dt,
            centerline_m=centerline_m,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_nearest_road_col(self, road: np.ndarray, row: int, W: int) -> int:
        """Find the nearest road column to center in the given row."""
        center = int(round(bev_ego_x_px(W)))
        road_cols = np.where(road[row])[0]
        if len(road_cols) == 0:
            # Search up a few rows
            for r in range(row - 1, max(0, row - 20), -1):
                road_cols = np.where(road[r])[0]
                if len(road_cols) > 0:
                    break
        if len(road_cols) == 0:
            return center
        return int(road_cols[np.argmin(np.abs(road_cols - center))])

    def _dijkstra_forward(
        self,
        cost: np.ndarray,
        road: np.ndarray,
        start_row: int,
        start_col: int,
        H: int,
        W: int,
    ) -> tuple[list[int], list[int]]:
        """
        Dijkstra from (start_row, start_col) moving upward (decreasing row).

        Movement model:
          - Each step moves one row upward (row - 1)
          - Lateral drift: any column within ±lateral_drift_px of current column
          - Only road pixels are traversable (cost < 1e8)

        Returns lists of (rows, cols) for the optimal path from start to top.
        """
        drift = self.lateral_drift_px
        INF = float("inf")

        # dist[row, col] = accumulated cost from start
        dist = np.full((H, W), fill_value=INF, dtype=np.float64)
        parent = np.full((H, W, 2), fill_value=-1, dtype=np.int32)

        dist[start_row, start_col] = float(cost[start_row, start_col])
        heap = [(float(dist[start_row, start_col]), start_row, start_col)]

        # Forward-only: we only relax upward rows, so this is effectively
        # a shortest-path through a DAG (no back-edges).
        best_row_reached = start_row

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c] + 1e-9:
                continue

            next_row = r - 1
            if next_row < 0:
                break

            col_lo = max(0, c - drift)
            col_hi = min(W - 1, c + drift)

            for nc in range(col_lo, col_hi + 1):
                if not bool(road[next_row, nc]):
                    continue
                step_cost = float(cost[next_row, nc])
                # Add small lateral-movement penalty to prefer straighter paths
                lateral_step = abs(nc - c)
                step_cost += 0.01 * float(lateral_step)
                new_dist = float(dist[r, c]) + step_cost
                if new_dist < float(dist[next_row, nc]):
                    dist[next_row, nc] = new_dist
                    parent[next_row, nc, 0] = r
                    parent[next_row, nc, 1] = c
                    heapq.heappush(heap, (new_dist, next_row, nc))
                    if next_row < best_row_reached:
                        best_row_reached = next_row

        if best_row_reached >= start_row:
            # No forward progress — return empty
            return [], []

        # Find best column at the furthest reached row
        dist_top = dist[best_row_reached, :]
        valid_cols = np.where(dist_top < INF)[0]
        if len(valid_cols) == 0:
            return [], []

        end_col = int(valid_cols[np.argmin(dist_top[valid_cols])])

        # Backtrack
        path_rows: list[int] = []
        path_cols: list[int] = []
        r, c = best_row_reached, end_col
        while r != -1 and c != -1:
            path_rows.append(r)
            path_cols.append(c)
            pr, pc = int(parent[r, c, 0]), int(parent[r, c, 1])
            if pr == -1 or pc == -1:
                break
            r, c = pr, pc

        # Reverse so path goes from ego (start) to furthest reached row
        path_rows.reverse()
        path_cols.reverse()
        return path_rows, path_cols

    def _fallback_result(
        self,
        bev_mask_uint8: np.ndarray,
        dt: Optional[np.ndarray] = None,
    ) -> DtCorridorResult:
        """Return a zero-confidence result when extraction fails or scipy unavailable."""
        H, W = bev_mask_uint8.shape
        if dt is None:
            dt = np.zeros((H, W), dtype=np.float32)
        return DtCorridorResult(
            centerline_px=np.zeros((0, 2), dtype=np.float32),
            width_m_per_point=np.zeros(0, dtype=np.float32),
            confidence=0.0,
            dt_map=dt,
            centerline_m=np.zeros((0, 2), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_dt_corridor: Optional[DtSafeCorridor] = None


def get_default_dt_corridor() -> DtSafeCorridor:
    """Return a module-level DtSafeCorridor instance (lazy init)."""
    global _default_dt_corridor
    if _default_dt_corridor is None:
        _default_dt_corridor = DtSafeCorridor()
    return _default_dt_corridor
