"""
Camera-space path planners for monocular sidewalk masks.

These planners intentionally avoid any BEV/IPM warp so they can serve as
comparators or fallbacks when homography-based planning becomes brittle.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Optional

import cv2
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except ImportError:
    _HAS_SAVGOL = False


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _compute_heading_deg(points_xy: np.ndarray) -> float:
    if points_xy is None or len(points_xy) < 2:
        return 0.0
    start = points_xy[0].astype(np.float32)
    idx = min(len(points_xy) - 1, max(1, len(points_xy) * 2 // 5))
    end = points_xy[idx].astype(np.float32)
    dx = float(end[0] - start[0])
    dy = float(start[1] - end[1])
    if dy <= 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(dx, dy)))


def _polyline_length_px(points_xy: np.ndarray) -> float:
    if points_xy is None or len(points_xy) < 2:
        return 0.0
    d = np.diff(points_xy.astype(np.float32), axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _connected_center_component(mask: np.ndarray, center_x: int) -> np.ndarray:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask.copy()
    h, w = mask.shape
    band_y0 = max(0, h - max(8, h // 10))
    band = labels[band_y0:, :]
    best_idx = 0
    best_score = -1.0
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        touches_bottom = bool(np.any(band == idx))
        cx = float(centroids[idx][0])
        center_bonus = 1.0 - min(1.0, abs(cx - float(center_x)) / max(1.0, float(w - 1)))
        score = area + (0.5 * area if touches_bottom else 0.0) + 0.35 * area * max(0.0, center_bonus)
        if score > best_score:
            best_score = score
            best_idx = idx
    out = np.zeros_like(mask, dtype=np.uint8)
    if best_idx > 0:
        out[labels == best_idx] = 1
    return out


def _preprocess_mask(mask_255: np.ndarray, work_size: tuple[int, int], min_component_area_px: int) -> np.ndarray:
    work_w, work_h = int(work_size[0]), int(work_size[1])
    mask = cv2.resize((mask_255 > 0).astype(np.uint8), (work_w, work_h), interpolation=cv2.INTER_NEAREST)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(max(3, work_w // 40)), _odd(max(3, work_h // 40))))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(max(3, work_w // 60)), _odd(max(3, work_h // 60))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    center_x = work_w // 2
    mask = _connected_center_component(mask, center_x)
    if int(np.count_nonzero(mask)) == 0:
        return mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(max(1, min_component_area_px)):
            keep[labels == idx] = 1
    if int(np.count_nonzero(keep)) > 0:
        mask = _connected_center_component(keep, center_x)
    return mask.astype(np.uint8)


def _smooth_cols(cols: np.ndarray, window: int) -> np.ndarray:
    if cols is None or len(cols) < 5:
        return cols.astype(np.float32, copy=True)
    if not _HAS_SAVGOL:
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        out = np.convolve(cols.astype(np.float32), kernel, mode="same")
        out[0] = cols[0]
        out[-1] = cols[-1]
        return out.astype(np.float32)
    win = min(len(cols) if len(cols) % 2 == 1 else len(cols) - 1, _odd(window))
    if win < 5:
        return cols.astype(np.float32, copy=True)
    return savgol_filter(cols.astype(np.float32), window_length=win, polyorder=2).astype(np.float32)


def _row_run_midpoint(mask_row: np.ndarray, center_x: int, min_width_px: int) -> Optional[int]:
    xs = np.flatnonzero(mask_row > 0)
    if xs.size == 0:
        return None
    runs = []
    start = int(xs[0])
    prev = int(xs[0])
    for value in xs[1:]:
        cur = int(value)
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
        prev = cur
    runs.append((start, prev))
    valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= int(max(1, min_width_px))]
    if not valid_runs:
        valid_runs = runs
    best_run = min(valid_runs, key=lambda r: abs(0.5 * float(r[0] + r[1]) - float(center_x)))
    return int(round(0.5 * float(best_run[0] + best_run[1])))


@dataclass(frozen=True)
class ImagePathPlannerConfig:
    work_size: tuple[int, int] = (160, 96)  # (W, H)
    row_step_px: int = 2
    top_crop_frac: float = 0.22
    min_width_px: int = 6
    min_component_area_px: int = 120
    min_valid_rows: int = 12
    min_forward_span_px: int = 24
    smooth_window: int = 9
    lateral_drift_px: int = 18
    cost_exponent: float = 1.5
    cost_eps: float = 0.75


@dataclass(frozen=True)
class ImagePathResult:
    has_path: bool
    path_px: np.ndarray
    work_path_px: np.ndarray
    clean_mask: np.ndarray
    confidence: float
    mean_clearance_px: float
    forward_span_px: float
    heading_deg: float
    path_source: str


class CameraMidpointPlanner:
    def __init__(self, config: Optional[ImagePathPlannerConfig] = None) -> None:
        self.cfg = config or ImagePathPlannerConfig()

    def plan(self, mask_255: np.ndarray) -> ImagePathResult:
        orig_h, orig_w = mask_255.shape[:2]
        mask = _preprocess_mask(mask_255, self.cfg.work_size, self.cfg.min_component_area_px)
        work_h, work_w = mask.shape
        if int(np.count_nonzero(mask)) < 25:
            return self._empty(mask, orig_w, orig_h)

        center_x = work_w * 0.5
        prev_center = float(center_x)
        top_y = max(0, int(round(work_h * float(self.cfg.top_crop_frac))))
        rows = list(range(work_h - 1, top_y - 1, -max(1, int(self.cfg.row_step_px))))
        points: list[tuple[float, float]] = []
        widths: list[float] = []

        for y in rows:
            xs = np.flatnonzero(mask[y] > 0)
            if xs.size == 0:
                continue
            runs = []
            start = int(xs[0])
            prev = int(xs[0])
            for val in xs[1:]:
                cur = int(val)
                if cur != prev + 1:
                    runs.append((start, prev))
                    start = cur
                prev = cur
            runs.append((start, prev))
            valid_runs = [r for r in runs if (r[1] - r[0] + 1) >= int(self.cfg.min_width_px)]
            if not valid_runs:
                continue

            def _run_score(run: tuple[int, int]) -> tuple[float, float]:
                mid = 0.5 * float(run[0] + run[1])
                width = float(run[1] - run[0] + 1)
                return (abs(mid - prev_center) - 0.08 * width, abs(mid - center_x))

            best_run = min(valid_runs, key=_run_score)
            mid = 0.5 * float(best_run[0] + best_run[1])
            prev_center = mid
            points.append((mid, float(y)))
            widths.append(float(best_run[1] - best_run[0] + 1))

        if len(points) < int(self.cfg.min_valid_rows):
            return self._empty(mask, orig_w, orig_h)

        pts = np.asarray(points, dtype=np.float32)
        cols = _smooth_cols(pts[:, 0], self.cfg.smooth_window)
        pts[:, 0] = np.clip(cols, 0.0, float(work_w - 1))
        forward_span = float(pts[0, 1] - pts[-1, 1])
        if forward_span < float(self.cfg.min_forward_span_px):
            return self._empty(mask, orig_w, orig_h)

        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        row_idx = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, work_h - 1)
        col_idx = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, work_w - 1)
        clearance = dt[row_idx, col_idx]
        conf = _clip((len(pts) / max(1.0, len(rows))) * (float(np.mean(clearance)) / max(1.0, float(np.mean(widths) * 0.5))), 0.0, 1.0)

        path_orig = self._to_original(pts, orig_w, orig_h, work_w, work_h)
        return ImagePathResult(
            has_path=True,
            path_px=path_orig,
            work_path_px=np.round(pts).astype(np.int32),
            clean_mask=(mask * 255).astype(np.uint8),
            confidence=float(conf),
            mean_clearance_px=float(np.mean(clearance)) if len(clearance) else 0.0,
            forward_span_px=float((orig_h / max(1.0, work_h)) * forward_span),
            heading_deg=_compute_heading_deg(path_orig.astype(np.float32)),
            path_source="img_midpoint",
        )

    @staticmethod
    def _to_original(pts: np.ndarray, orig_w: int, orig_h: int, work_w: int, work_h: int) -> np.ndarray:
        out = pts.astype(np.float32).copy()
        out[:, 0] = out[:, 0] * max(1.0, float(orig_w - 1)) / max(1.0, float(work_w - 1))
        out[:, 1] = out[:, 1] * max(1.0, float(orig_h - 1)) / max(1.0, float(work_h - 1))
        return np.round(out).astype(np.int32)

    def _empty(self, mask: np.ndarray, orig_w: int, orig_h: int) -> ImagePathResult:
        return ImagePathResult(
            has_path=False,
            path_px=np.zeros((0, 2), dtype=np.int32),
            work_path_px=np.zeros((0, 2), dtype=np.int32),
            clean_mask=(mask * 255).astype(np.uint8),
            confidence=0.0,
            mean_clearance_px=0.0,
            forward_span_px=0.0,
            heading_deg=0.0,
            path_source="img_midpoint",
        )


class CameraDtPlanner:
    def __init__(self, config: Optional[ImagePathPlannerConfig] = None) -> None:
        self.cfg = config or ImagePathPlannerConfig()

    def plan(self, mask_255: np.ndarray) -> ImagePathResult:
        orig_h, orig_w = mask_255.shape[:2]
        mask = _preprocess_mask(mask_255, self.cfg.work_size, self.cfg.min_component_area_px)
        work_h, work_w = mask.shape
        if int(np.count_nonzero(mask)) < 25:
            return self._empty(mask)

        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        road = mask > 0
        cost = np.full((work_h, work_w), fill_value=1e9, dtype=np.float32)
        cost[road] = (1.0 / (dt[road] + float(self.cfg.cost_eps)) ** float(self.cfg.cost_exponent)).astype(np.float32)

        start_row = work_h - 1
        start_col = work_w // 2
        if not bool(road[start_row, start_col]):
            found = False
            for search_row in range(start_row, max(-1, start_row - 20), -1):
                if np.any(road[search_row]):
                    start_row = search_row
                    midpoint = _row_run_midpoint(mask[search_row], work_w // 2, self.cfg.min_width_px)
                    start_col = int(midpoint if midpoint is not None else (work_w // 2))
                    found = True
                    break
            if not found:
                return self._empty(mask)

        dist = np.full((work_h, work_w), fill_value=np.inf, dtype=np.float64)
        parent = np.full((work_h, work_w, 2), fill_value=-1, dtype=np.int32)
        dist[start_row, start_col] = float(cost[start_row, start_col])
        heap = [(float(dist[start_row, start_col]), start_row, start_col)]
        best_row = start_row

        while heap:
            cur_dist, row, col = heapq.heappop(heap)
            if cur_dist > float(dist[row, col]) + 1e-9:
                continue
            next_row = row - 1
            if next_row < max(0, int(round(work_h * float(self.cfg.top_crop_frac)))):
                best_row = next_row + 1 if next_row + 1 < row else best_row
                break
            col_lo = max(0, col - int(self.cfg.lateral_drift_px))
            col_hi = min(work_w - 1, col + int(self.cfg.lateral_drift_px))
            advanced = False
            for next_col in range(col_lo, col_hi + 1):
                if not bool(road[next_row, next_col]):
                    continue
                step = float(cost[next_row, next_col]) + 0.01 * abs(float(next_col - col))
                cand = float(dist[row, col]) + step
                if cand < float(dist[next_row, next_col]):
                    dist[next_row, next_col] = cand
                    parent[next_row, next_col, 0] = row
                    parent[next_row, next_col, 1] = col
                    heapq.heappush(heap, (cand, next_row, next_col))
                    advanced = True
                    if next_row < best_row:
                        best_row = next_row
            if not advanced and row < best_row:
                best_row = row

        row_cost = dist[best_row]
        valid_cols = np.flatnonzero(np.isfinite(row_cost))
        if valid_cols.size == 0:
            return self._empty(mask)
        end_col = int(valid_cols[np.argmin(row_cost[valid_cols])])
        path = []
        row = int(best_row)
        col = int(end_col)
        while row >= 0 and col >= 0:
            path.append((float(col), float(row)))
            pr = int(parent[row, col, 0])
            pc = int(parent[row, col, 1])
            if pr < 0 or pc < 0:
                break
            row, col = pr, pc
        path.reverse()
        if len(path) < int(self.cfg.min_valid_rows):
            return self._empty(mask)

        pts = np.asarray(path, dtype=np.float32)
        pts[:, 0] = np.clip(_smooth_cols(pts[:, 0], self.cfg.smooth_window), 0.0, float(work_w - 1))
        forward_span = float(pts[0, 1] - pts[-1, 1])
        if forward_span < float(self.cfg.min_forward_span_px):
            return self._empty(mask)

        row_idx = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, work_h - 1)
        col_idx = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, work_w - 1)
        clearance = dt[row_idx, col_idx]
        conf = _clip((len(pts) / max(1.0, float(work_h))) * (float(np.mean(clearance)) / 8.0), 0.0, 1.0)
        path_orig = CameraMidpointPlanner._to_original(pts, orig_w, orig_h, work_w, work_h)
        return ImagePathResult(
            has_path=True,
            path_px=path_orig,
            work_path_px=np.round(pts).astype(np.int32),
            clean_mask=(mask * 255).astype(np.uint8),
            confidence=float(conf),
            mean_clearance_px=float(np.mean(clearance)) if len(clearance) else 0.0,
            forward_span_px=float((orig_h / max(1.0, work_h)) * forward_span),
            heading_deg=_compute_heading_deg(path_orig.astype(np.float32)),
            path_source="img_dt",
        )

    def _empty(self, mask: np.ndarray) -> ImagePathResult:
        return ImagePathResult(
            has_path=False,
            path_px=np.zeros((0, 2), dtype=np.int32),
            work_path_px=np.zeros((0, 2), dtype=np.int32),
            clean_mask=(mask * 255).astype(np.uint8),
            confidence=0.0,
            mean_clearance_px=0.0,
            forward_span_px=0.0,
            heading_deg=0.0,
            path_source="img_dt",
        )
