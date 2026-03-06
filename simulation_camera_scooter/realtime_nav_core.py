#!/usr/bin/env python3
"""
realtime_nav_core.py
====================
Lightweight BEV path extraction + adaptive pure pursuit controller.

Design goals:
- 8 Hz real-time on small CPU boards (Rock 5B class)
- no heavy optimization / no extra sensors
- robust branch handling (T/Y) with hysteresis
"""

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _odd(v: int) -> int:
    v = max(1, int(v))
    return v if (v % 2 == 1) else v + 1


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-9)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _polyline_length_m(pts_m: np.ndarray) -> float:
    if pts_m is None or len(pts_m) < 2:
        return 0.0
    d = np.diff(pts_m, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _resample_polyline(pts_m: np.ndarray, ds_m: float, max_len_m: Optional[float] = None) -> np.ndarray:
    """Uniform arc-length resampling. Keeps first and last point."""
    if pts_m is None or len(pts_m) < 2:
        return pts_m.copy() if pts_m is not None else np.zeros((0, 2), dtype=np.float32)

    seg = np.diff(pts_m, axis=0)
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total < 1e-6:
        return pts_m[:1].copy()
    if max_len_m is not None:
        total = min(total, float(max_len_m))
    samples = np.arange(0.0, total + 1e-6, float(ds_m), dtype=np.float32)
    if samples[-1] < total:
        samples = np.append(samples, np.float32(total))
    x = np.interp(samples, cum, pts_m[:, 0]).astype(np.float32)
    y = np.interp(samples, cum, pts_m[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def _polyline_curvature_mean(pts_m: np.ndarray) -> float:
    """Mean absolute curvature from discrete 3-point estimate."""
    if pts_m is None or len(pts_m) < 3:
        return 0.0
    p0 = pts_m[:-2]
    p1 = pts_m[1:-1]
    p2 = pts_m[2:]
    a = p1 - p0
    b = p2 - p1
    c = p2 - p0
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1) + 1e-6
    kappa = 2.0 * cross / denom
    return float(np.mean(np.abs(kappa)))


def _polyline_end_heading(pts_m: np.ndarray) -> float:
    """Heading angle in radians in metric frame (x=forward, y=lateral)."""
    if pts_m is None or len(pts_m) < 2:
        return 0.0
    n = len(pts_m)
    i0 = max(0, n - 6)
    vec = pts_m[-1] - pts_m[i0]
    if _safe_norm(vec) < 1e-5:
        vec = pts_m[-1] - pts_m[0]
    return float(math.atan2(vec[1], max(1e-6, vec[0])))


@dataclass
class PathExtractorConfig:
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0
    work_size: Tuple[int, int] = (220, 220)  # (W, H)

    close_kernel_m: float = 0.15
    open_kernel_m: float = 0.10
    min_sidewalk_width_m: float = 0.50  # was 0.80 â€” tier1 tuning: less aggressive erosion
    hole_fill_max_area_m2: float = 2.0   # was 0.20 â€” [TIER2] fill larger internal gaps
    min_component_area_m2: float = 0.20

    radius_floor_m: float = 0.12
    branch_min_len_m: float = 0.30  # was 0.55 â€” tier1 tuning: keep shorter skeleton branches
    branch_min_radius_m: float = 0.15
    junction_prune_angle_deg: float = 75.0
    junction_short_branch_len_m: float = 1.20

    path_horizon_m: float = 4.5
    min_path_len_m: float = 0.80  # was 1.50 â€” tier1 tuning: shorter minimum path accepted
    min_forward_span_m: float = 0.80
    search_max_depth: int = 6
    search_top_k: int = 6
    path_sample_ds_m: float = 0.10
    switch_margin: float = 0.15
    branch_hold_frames: int = 4

    spline_lambda: float = 0.08
    spline_kappa_max_m_inv: float = 0.90

    fallback_enabled: bool = True
    fallback_hold_frames: int = 14
    fallback_row_step_px: int = 3
    fallback_min_row_pixels: int = 6
    fallback_min_width_m: float = 0.22
    fallback_min_forward_span_m: float = 0.45
    fallback_target_span_m: float = 1.30
    fallback_prev_blend: float = 0.55
    fallback_center_pull: float = 0.05
    fallback_low_conf_occ_ratio: float = 0.10
    fallback_low_conf_span_m: float = 0.90
    fallback_low_conf_prev_blend: float = 0.55
    fallback_recenter_gain: float = 0.0
    fallback_low_conf_max_abs_lateral_m: float = 5.0
    fallback_output_lateral_clip_m: float = 0.25


@dataclass
class PurePursuitConfig:
    dt_s: float = 1.0 / 8.0
    wheelbase_m: float = 0.72
    delta_max_deg: float = 30.0

    lookahead_min_m: float = 1.20
    lookahead_max_m: float = 4.00
    lookahead_epsilon: float = 0.08
    lookahead_alpha_v0: float = 0.20
    lookahead_alpha_v1: float = 0.12

    steer_tau_s: float = 0.25
    steer_rate_max_deg_s: float = 75.0

    kappa_step_max_m_inv: float = 0.35
    path_discont_lat_m: float = 0.45
    path_discont_head_deg: float = 25.0

    fallback_decay: float = 0.85
    fallback_hold_frames: int = 5


@dataclass
class AxisEdge:
    edge_id: int
    n0: int
    n1: int
    pix_path: np.ndarray     # [N,2] in working-grid pixels (x,y)
    pts_m: np.ndarray        # [N,2] in metric (forward x, lateral y)
    length_m: float
    mean_radius_m: float
    mean_curvature_m_inv: float
    signature: Tuple[int, int, int, int]


@dataclass
class PathCandidate:
    points_m: np.ndarray
    length_m: float
    first_edge_sig: Tuple[int, int, int, int]
    progress_m: float
    curvature_m_inv: float
    heading_end_rad: float
    cost: float = 0.0


@dataclass
class PathPlanResult:
    has_path: bool
    path_model: Optional["CubicPathModel"]
    best_path_m: np.ndarray
    best_path_px: np.ndarray
    candidate_paths_m: List[np.ndarray]
    candidate_paths_px: List[np.ndarray]
    candidate_lengths_m: List[float]
    best_idx: int
    skeleton_px: np.ndarray
    graph_nodes: int
    graph_edges: int
    t_skeleton_ms: float
    t_path_ms: float


@dataclass
class ControlOutput:
    steer_deg: float
    lookahead_m: float
    kappa_cmd_m_inv: float
    heading_deg: float
    target_x_m: float
    target_y_m: float
    valid_path: bool


class CubicPathModel:
    """Regularized cubic y(x) path in ego metric frame."""

    def __init__(self, coeff: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, s_grid: np.ndarray, kappa_grid: np.ndarray):
        self.coeff = coeff.astype(np.float64)
        self.x_grid = x_grid.astype(np.float64)
        self.y_grid = y_grid.astype(np.float64)
        self.s_grid = s_grid.astype(np.float64)
        self.kappa_grid = kappa_grid.astype(np.float64)

    @property
    def length_m(self) -> float:
        return float(self.s_grid[-1]) if len(self.s_grid) else 0.0

    @property
    def x_max_m(self) -> float:
        return float(self.x_grid[-1]) if len(self.x_grid) else 0.0

    def y_of_x(self, x_m: float) -> float:
        x = float(np.clip(x_m, 0.0, self.x_max_m))
        a0, a1, a2, a3 = self.coeff
        return float(a0 + a1 * x + a2 * x * x + a3 * x * x * x)

    def heading_of_x(self, x_m: float) -> float:
        x = float(np.clip(x_m, 0.0, self.x_max_m))
        _, a1, a2, a3 = self.coeff
        dy_dx = a1 + 2.0 * a2 * x + 3.0 * a3 * x * x
        return float(math.atan2(dy_dx, 1.0))

    def curvature_of_x(self, x_m: float) -> float:
        x = float(np.clip(x_m, 0.0, self.x_max_m))
        _, a1, a2, a3 = self.coeff
        dy_dx = a1 + 2.0 * a2 * x + 3.0 * a3 * x * x
        d2y_dx2 = 2.0 * a2 + 6.0 * a3 * x
        return float(d2y_dx2 / max(1e-6, (1.0 + dy_dx * dy_dx) ** 1.5))

    def query_s(self, s_m: float) -> Tuple[float, float, float]:
        if len(self.s_grid) == 0:
            return 0.0, 0.0, 0.0
        s = float(np.clip(s_m, 0.0, self.length_m))
        x = float(np.interp(s, self.s_grid, self.x_grid))
        y = float(np.interp(s, self.s_grid, self.y_grid))
        kappa = float(np.interp(s, self.s_grid, self.kappa_grid))
        return x, y, kappa

    def sample_xy(self, ds_m: float = 0.10) -> np.ndarray:
        if self.length_m <= 1e-6:
            return np.zeros((0, 2), dtype=np.float32)
        s = np.arange(0.0, self.length_m + 1e-6, ds_m)
        x = np.interp(s, self.s_grid, self.x_grid)
        y = np.interp(s, self.s_grid, self.y_grid)
        return np.stack([x, y], axis=1).astype(np.float32)


class BEVPathExtractor:
    """
    Medial-axis path extraction:
    BEV mask -> preprocess -> DT+thinning -> branch pruning -> graph search -> cubic fit.
    """

    def __init__(self, cfg: Optional[PathExtractorConfig] = None):
        self.cfg = cfg or PathExtractorConfig()
        self.prev_first_edge_sig: Optional[Tuple[int, int, int, int]] = None
        self.prev_end_heading: float = 0.0
        self.branch_hold_counter: int = 0
        self.prev_best_path_m: Optional[np.ndarray] = None
        self.no_path_counter: int = 0

    # -------------------------------------------------------------------------
    # Coordinate conversion
    # -------------------------------------------------------------------------
    def _metric_from_pixel(self, pts_px: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        """pixel (x,y) -> metric (forward x, lateral y)."""
        h, w = shape_hw
        if len(pts_px) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        u = pts_px[:, 0].astype(np.float32)
        v = pts_px[:, 1].astype(np.float32)
        forward = (float(h - 1) - v) / max(1.0, float(h - 1)) * self.cfg.bev_forward_m
        lateral = (u / max(1.0, float(w - 1)) - 0.5) * self.cfg.bev_lateral_m
        return np.stack([forward, lateral], axis=1).astype(np.float32)

    def _pixel_from_metric(self, pts_m: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        """metric (forward x, lateral y) -> pixel (x,y)."""
        h, w = shape_hw
        if pts_m is None or len(pts_m) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        f = np.clip(pts_m[:, 0], 0.0, self.cfg.bev_forward_m)
        l = np.clip(pts_m[:, 1], -self.cfg.bev_lateral_m / 2.0, self.cfg.bev_lateral_m / 2.0)
        u = ((l / self.cfg.bev_lateral_m) + 0.5) * max(1.0, float(w - 1))
        v = (1.0 - f / self.cfg.bev_forward_m) * max(1.0, float(h - 1))
        out = np.stack([u, v], axis=1)
        out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))
        return np.round(out).astype(np.int32)

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------
    def _preprocess(self, bev_mask_255: np.ndarray) -> np.ndarray:
        work_w, work_h = self.cfg.work_size
        mask = (bev_mask_255 > 0).astype(np.uint8)
        mask = cv2.resize(mask, (work_w, work_h), interpolation=cv2.INTER_NEAREST)

        m_per_px = 0.5 * (
            self.cfg.bev_forward_m / max(1.0, float(work_h - 1))
            + self.cfg.bev_lateral_m / max(1.0, float(work_w - 1))
        )
        close_k = _odd(int(round(self.cfg.close_kernel_m / max(1e-6, m_per_px))))
        open_k = _odd(int(round(self.cfg.open_kernel_m / max(1e-6, m_per_px))))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

        # Fill small holes only
        mask_255 = (mask * 255).astype(np.uint8)
        inv = cv2.bitwise_not(mask_255)
        ff = inv.copy()
        ff_mask = np.zeros((work_h + 2, work_w + 2), np.uint8)
        cv2.floodFill(ff, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        holes_bin = (holes > 0).astype(np.uint8)

        hole_max_px = int(round(self.cfg.hole_fill_max_area_m2 / max(1e-9, m_per_px * m_per_px)))
        num_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes_bin, connectivity=8)
        for idx in range(1, num_h):
            if int(stats_h[idx, cv2.CC_STAT_AREA]) <= hole_max_px:
                mask_255[labels_h == idx] = 255
        mask = (mask_255 > 0).astype(np.uint8)

        # Minimum width enforcement; relax slightly when occupancy is already very sparse.
        occ_ratio = float(np.count_nonzero(mask)) / max(1.0, float(mask.size))
        width_floor_m = float(self.cfg.min_sidewalk_width_m)
        if occ_ratio < 0.08:
            width_floor_m = min(width_floor_m, 0.35)
        min_w_k = _odd(int(round((width_floor_m * 0.5) / max(1e-6, m_per_px))))
        k_min = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, min_w_k), max(3, min_w_k)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_min, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_open, iterations=1)

        # Keep components near ego band or largest valid component
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num > 1:
            keep = np.zeros_like(mask, dtype=np.uint8)
            min_area_px = int(round(self.cfg.min_component_area_m2 / max(1e-9, m_per_px * m_per_px)))
            band_y0 = max(0, work_h - max(8, work_h // 8))
            band_x0 = max(0, work_w // 2 - max(8, work_w // 8))
            band_x1 = min(work_w, work_w // 2 + max(8, work_w // 8))
            ego_ids = set(np.unique(labels[band_y0:work_h, band_x0:band_x1]).tolist())
            ego_ids.discard(0)
            for idx in range(1, num):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area_px:
                    continue
                if idx in ego_ids or not ego_ids:
                    keep[labels == idx] = 1
            if not np.any(keep):
                # fallback: largest component
                idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                keep[labels == idx] = 1
            mask = keep
        return (mask * 255).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Skeleton extraction
    # -------------------------------------------------------------------------
    def _thin(self, mask_255: np.ndarray) -> np.ndarray:
        try:
            from cv2.ximgproc import thinning, THINNING_GUOHALL
            sk = thinning(mask_255, THINNING_GUOHALL)
            return ((sk > 0).astype(np.uint8) * 255)
        except Exception as e:
            print(f"[Skeleton] Guo-Hall thinning failed ({e}), using fallback skeletonization")
            # lightweight fallback skeletonization
            img = (mask_255 > 0).astype(np.uint8)
            skel = np.zeros_like(img)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    break
            return ((skel > 0).astype(np.uint8) * 255)

    def _extract_medial_axis(self, mask_255: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        work_h, work_w = mask_255.shape
        m_per_px = 0.5 * (
            self.cfg.bev_forward_m / max(1.0, float(work_h - 1))
            + self.cfg.bev_lateral_m / max(1.0, float(work_w - 1))
        )
        dist_px = cv2.distanceTransform((mask_255 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist_m = dist_px * m_per_px
        skel = self._thin(mask_255)
        skel[(dist_m < self.cfg.radius_floor_m)] = 0
        return (skel > 0).astype(np.uint8), dist_m.astype(np.float32)

    # -------------------------------------------------------------------------
    # Graph construction from skeleton pixels
    # -------------------------------------------------------------------------
    @staticmethod
    def _neighbor_pixels(x: int, y: int, w: int, h: int) -> List[Tuple[int, int]]:
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < w and 0 <= ny_ < h:
                    out.append((nx_, ny_))
        return out

    def _extract_edges(self, skel_bin: np.ndarray, dist_m: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        ys, xs = np.where(skel_bin > 0)
        pixels = [(int(x), int(y)) for x, y in zip(xs, ys)]
        if not pixels:
            return [], []
        pix_set = set(pixels)
        h, w = skel_bin.shape

        deg: Dict[Tuple[int, int], int] = {}
        for p in pixels:
            x, y = p
            d = 0
            for nx_, ny_ in self._neighbor_pixels(x, y, w, h):
                if (nx_, ny_) in pix_set:
                    d += 1
            deg[p] = d
        nodes = [p for p in pixels if deg[p] != 2]
        node_set = set(nodes)
        if not nodes:
            # rare loop-only shape: fallback endpoints by min/max forward pixels
            p_bot = max(pixels, key=lambda p: p[1])
            p_top = min(pixels, key=lambda p: p[1])
            nodes = [p_bot, p_top]
            node_set = set(nodes)

        visited_steps = set()
        edges_pix: List[np.ndarray] = []
        for node in nodes:
            nx_list = [nb for nb in self._neighbor_pixels(node[0], node[1], w, h) if nb in pix_set]
            for nb in nx_list:
                step_key = tuple(sorted((node, nb)))
                if step_key in visited_steps:
                    continue
                path = [node, nb]
                visited_steps.add(step_key)
                prev = node
                cur = nb
                while cur not in node_set:
                    nbs = [q for q in self._neighbor_pixels(cur[0], cur[1], w, h) if q in pix_set and q != prev]
                    if not nbs:
                        break
                    nxt = nbs[0]
                    step_key = tuple(sorted((cur, nxt)))
                    if step_key in visited_steps:
                        break
                    path.append(nxt)
                    visited_steps.add(step_key)
                    prev, cur = cur, nxt
                if len(path) >= 2 and cur in node_set:
                    edges_pix.append(np.asarray(path, dtype=np.int32))
        return edges_pix, nodes

    def _build_graph(self, skel_bin: np.ndarray, dist_m: np.ndarray) -> Tuple[np.ndarray, List[AxisEdge], List[List[int]], np.ndarray]:
        edges_pix, nodes_pix = self._extract_edges(skel_bin, dist_m)
        if not edges_pix:
            return np.zeros((0, 2), dtype=np.float32), [], [], skel_bin

        # node indexing
        node_unique = sorted({tuple(p.tolist()) for e in edges_pix for p in (e[0], e[-1])})
        node_id = {p: i for i, p in enumerate(node_unique)}
        node_xy_m = self._metric_from_pixel(np.asarray(node_unique, dtype=np.float32), skel_bin.shape).astype(np.float32)

        edges: List[AxisEdge] = []
        for i, epx in enumerate(edges_pix):
            p0 = tuple(epx[0].tolist())
            p1 = tuple(epx[-1].tolist())
            n0 = node_id[p0]
            n1 = node_id[p1]
            pts_m = self._metric_from_pixel(epx.astype(np.float32), skel_bin.shape)
            length_m = _polyline_length_m(pts_m)
            r_vals = [float(dist_m[int(p[1]), int(p[0])]) for p in epx]
            mean_radius = float(np.mean(r_vals)) if r_vals else 0.0
            curv = _polyline_curvature_mean(pts_m)
            a = (int(round(pts_m[0, 0] * 4.0)), int(round(pts_m[0, 1] * 4.0)))
            b = (int(round(pts_m[-1, 0] * 4.0)), int(round(pts_m[-1, 1] * 4.0)))
            sig = (a[0], a[1], b[0], b[1]) if a <= b else (b[0], b[1], a[0], a[1])
            edges.append(
                AxisEdge(
                    edge_id=i,
                    n0=n0,
                    n1=n1,
                    pix_path=epx,
                    pts_m=pts_m,
                    length_m=length_m,
                    mean_radius_m=mean_radius,
                    mean_curvature_m_inv=curv,
                    signature=sig,
                )
            )

        # basic pruning
        kept = [
            e for e in edges
            if e.length_m >= self.cfg.branch_min_len_m and e.mean_radius_m >= self.cfg.branch_min_radius_m
        ]
        if not kept:
            return np.zeros((0, 2), dtype=np.float32), [], [], skel_bin

        # build adjacency for angle-based pruning
        adj_tmp = [[] for _ in range(len(node_xy_m))]
        for e in kept:
            adj_tmp[e.n0].append(e)
            adj_tmp[e.n1].append(e)

        prune_ids = set()
        ang_thresh = math.radians(self.cfg.junction_prune_angle_deg)
        for nid, inc in enumerate(adj_tmp):
            if len(inc) < 3:
                continue

            # choose "forward dominant" edge at the junction
            tan_ref = None
            best_gain = -1e9
            for e in inc:
                pts = e.pts_m if e.n0 == nid else e.pts_m[::-1]
                k = min(5, len(pts) - 1)
                vec = pts[k] - pts[0]
                vec_n = vec / _safe_norm(vec)
                forward_gain = float(pts[-1, 0] - pts[0, 0])
                if forward_gain > best_gain:
                    best_gain = forward_gain
                    tan_ref = vec_n
            if tan_ref is None:
                continue

            for e in inc:
                if e.length_m >= self.cfg.junction_short_branch_len_m:
                    continue
                pts = e.pts_m if e.n0 == nid else e.pts_m[::-1]
                k = min(5, len(pts) - 1)
                vec = pts[k] - pts[0]
                vec_n = vec / _safe_norm(vec)
                forward_gain = float(pts[-1, 0] - pts[0, 0])
                dotv = float(np.clip(np.dot(tan_ref, vec_n), -1.0, 1.0))
                ang = math.acos(dotv)
                if ang > ang_thresh and forward_gain < 0.5:
                    prune_ids.add(e.edge_id)

        edges_final = [e for e in kept if e.edge_id not in prune_ids]
        if not edges_final:
            return np.zeros((0, 2), dtype=np.float32), [], [], skel_bin

        # compact edge IDs + adjacency
        for i, e in enumerate(edges_final):
            e.edge_id = i
        adj = [[] for _ in range(len(node_xy_m))]
        for e in edges_final:
            adj[e.n0].append(e.edge_id)
            adj[e.n1].append(e.edge_id)
        return node_xy_m, edges_final, adj, skel_bin

    # -------------------------------------------------------------------------
    # Candidate path search
    # -------------------------------------------------------------------------
    def _edge_points_from_node(self, edge: AxisEdge, from_node: int) -> np.ndarray:
        if edge.n0 == from_node:
            return edge.pts_m
        return edge.pts_m[::-1]

    def _other_node(self, edge: AxisEdge, node_id: int) -> int:
        return edge.n1 if edge.n0 == node_id else edge.n0

    def _search_candidates(
        self,
        node_xy_m: np.ndarray,
        edges: List[AxisEdge],
        adj: List[List[int]],
    ) -> List[PathCandidate]:
        if len(node_xy_m) == 0 or not edges:
            return []

        # Choose start near ego origin (bottom-center in BEV metric).
        # This avoids selecting upper/side branches as the seed when topology is noisy.
        active_nodes = [i for i in range(len(node_xy_m)) if adj[i]]
        if not active_nodes:
            return []

        ego_band_max_forward_m = min(2.5, 0.35 * self.cfg.path_horizon_m)
        near_ego_nodes = [
            i for i in active_nodes
            if float(node_xy_m[i][0]) <= ego_band_max_forward_m
        ]
        start_pool = near_ego_nodes if near_ego_nodes else active_nodes

        ranked_starts = sorted(
            start_pool,
            key=lambda i: (
                math.hypot(float(node_xy_m[i][0]), float(node_xy_m[i][1])),
                abs(float(node_xy_m[i][1])),
                float(node_xy_m[i][0]),
            ),
        )

        max_depth = int(self.cfg.search_max_depth)
        max_len = float(self.cfg.path_horizon_m)
        branch_limit = 3  # per node expansion limit (keeps compute bounded)

        # Pick the first start node that yields at least one forward-growing edge.
        stack = []
        for start in ranked_starts:
            trial_stack = []
            for eid in adj[start]:
                e = edges[eid]
                pts = self._edge_points_from_node(e, start)
                f_gain = float(pts[-1, 0] - pts[0, 0])
                if f_gain < -0.25:
                    continue
                trial_stack.append({
                    "node": self._other_node(e, start),
                    "pts": pts.copy(),
                    "len": float(e.length_m),
                    "depth": 1,
                    "used": {eid},
                    "first_sig": e.signature,
                })
            if trial_stack:
                stack = trial_stack
                break

        if not stack:
            return []

        candidates: List[PathCandidate] = []
        while stack:
            st = stack.pop()
            cur_pts = st["pts"]
            cur_len = st["len"]
            cur_node = st["node"]
            cur_depth = st["depth"]
            used = st["used"]
            first_sig = st["first_sig"]

            if cur_len >= max_len or cur_depth >= max_depth:
                pass
            else:
                next_states = []
                for eid in adj[cur_node]:
                    if eid in used:
                        continue
                    e = edges[eid]
                    seg = self._edge_points_from_node(e, cur_node)
                    f_gain = float(seg[-1, 0] - seg[0, 0])
                    if f_gain < -0.25:
                        continue
                    new_pts = np.vstack([cur_pts, seg[1:]]) if len(cur_pts) else seg.copy()
                    new_len = cur_len + float(e.length_m)
                    next_states.append({
                        "node": self._other_node(e, cur_node),
                        "pts": new_pts,
                        "len": new_len,
                        "depth": cur_depth + 1,
                        "used": used | {eid},
                        "first_sig": first_sig,
                        "progress": float(np.max(new_pts[:, 0])) if len(new_pts) else 0.0,
                    })
                if next_states:
                    next_states.sort(key=lambda d: d["progress"], reverse=True)
                    stack.extend(next_states[:branch_limit])
                    continue

            # terminal candidate
            if cur_len >= self.cfg.min_path_len_m and len(cur_pts) >= 2:
                pts = _resample_polyline(cur_pts, self.cfg.path_sample_ds_m, max_len_m=max_len)
                # Reject lateral-only traces before scoring/fitting; these look like
                # "possible paths" in pixels but have no forward progress in metric x.
                x_vals = pts[:, 0]
                if float(np.max(x_vals) - np.min(x_vals)) < self.cfg.min_forward_span_m:
                    continue
                progress = float(np.max(pts[:, 0])) if len(pts) else 0.0
                curv = _polyline_curvature_mean(pts)
                h_end = _polyline_end_heading(pts)
                candidates.append(
                    PathCandidate(
                        points_m=pts,
                        length_m=_polyline_length_m(pts),
                        first_edge_sig=first_sig,
                        progress_m=progress,
                        curvature_m_inv=curv,
                        heading_end_rad=h_end,
                    )
                )

        return candidates

    def _score_candidates(self, cands: List[PathCandidate]) -> List[PathCandidate]:
        if not cands:
            return cands
        H = max(1.0, self.cfg.path_horizon_m)
        prev_heading = self.prev_end_heading
        for c in cands:
            j_prog = (H - min(H, c.progress_m)) / H
            j_curv = c.curvature_m_inv / 0.6
            j_head = abs(c.heading_end_rad - prev_heading) / math.pi
            j_hys = 1.0 if (self.prev_first_edge_sig is not None and c.first_edge_sig != self.prev_first_edge_sig) else 0.0
            # weighted sum
            c.cost = 2.0 * j_prog + 1.5 * j_curv + 0.8 * j_head + 1.2 * j_hys
        cands.sort(key=lambda c: c.cost)
        return cands

    def _apply_hysteresis(self, cands: List[PathCandidate]) -> int:
        if not cands:
            return -1
        best_idx = 0
        best = cands[best_idx]
        prev_sig = self.prev_first_edge_sig
        if prev_sig is not None and best.first_edge_sig != prev_sig:
            prev_idx = next((i for i, c in enumerate(cands) if c.first_edge_sig == prev_sig), -1)
            if prev_idx >= 0 and self.branch_hold_counter < self.cfg.branch_hold_frames:
                if cands[best_idx].cost > cands[prev_idx].cost - self.cfg.switch_margin:
                    best_idx = prev_idx

        chosen = cands[best_idx]
        if self.prev_first_edge_sig == chosen.first_edge_sig:
            self.branch_hold_counter = min(50, self.branch_hold_counter + 1)
        else:
            self.branch_hold_counter = 0
            self.prev_first_edge_sig = chosen.first_edge_sig
        self.prev_end_heading = chosen.heading_end_rad
        return best_idx

    # -------------------------------------------------------------------------
    # Spline fit
    # -------------------------------------------------------------------------
    def _fit_regularized_cubic(self, pts_m: np.ndarray) -> Optional[CubicPathModel]:
        if pts_m is None or len(pts_m) < 4:
            return None
        # Keep forward section and enforce non-decreasing x for stable y(x) fit
        x = pts_m[:, 0].astype(np.float64)
        y = pts_m[:, 1].astype(np.float64)
        valid = x >= 0.0
        x, y = x[valid], y[valid]
        if len(x) < 4:
            return None
        x = np.maximum.accumulate(x)  # avoid small backward oscillations
        x = np.clip(x, 0.0, self.cfg.path_horizon_m)

        # collapse duplicate x by averaging y
        xu, inv = np.unique(np.round(x, 3), return_inverse=True)
        yu = np.zeros_like(xu)
        cnt = np.zeros_like(xu)
        for i, idx in enumerate(inv):
            yu[idx] += y[i]
            cnt[idx] += 1.0
        yu /= np.maximum(cnt, 1.0)
        if len(xu) < 4 or float(xu[-1] - xu[0]) < 1.0:
            return None

        # regularized cubic fit: y = a0 + a1 x + a2 x^2 + a3 x^3
        def solve_once(x_fit: np.ndarray, y_fit: np.ndarray, lam: float) -> np.ndarray:
            A = np.column_stack([
                np.ones_like(x_fit),
                x_fit,
                x_fit * x_fit,
                x_fit * x_fit * x_fit,
            ])
            R = np.diag([0.0, 0.0, 1.0, 1.0])
            M = A.T @ A + lam * R
            b = A.T @ y_fit
            return np.linalg.solve(M, b)

        lam = float(self.cfg.spline_lambda)
        x_fit = xu.copy()
        y_fit = yu.copy()
        coeff = solve_once(x_fit, y_fit, lam)

        # evaluate and clamp curvature by one retry
        def evaluate(coeff_in: np.ndarray, x_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            xg = np.linspace(0.0, x_max, 120, dtype=np.float64)
            a0, a1, a2, a3 = coeff_in
            yg = a0 + a1 * xg + a2 * xg * xg + a3 * xg * xg * xg
            dy = a1 + 2.0 * a2 * xg + 3.0 * a3 * xg * xg
            ddy = 2.0 * a2 + 6.0 * a3 * xg
            kappa = ddy / np.maximum(1e-6, (1.0 + dy * dy) ** 1.5)
            ds = np.sqrt(np.diff(xg) ** 2 + np.diff(yg) ** 2)
            sg = np.concatenate(([0.0], np.cumsum(ds)))
            return xg, yg, sg, kappa

        x_max = float(min(self.cfg.path_horizon_m, x_fit[-1]))
        xg, yg, sg, kappa = evaluate(coeff, x_max)
        if float(np.max(np.abs(kappa))) > self.cfg.spline_kappa_max_m_inv and x_max > 2.0:
            x_max = x_max * 0.8
            use = x_fit <= x_max
            if np.count_nonzero(use) >= 4:
                coeff = solve_once(x_fit[use], y_fit[use], lam * 2.0)
                xg, yg, sg, kappa = evaluate(coeff, x_max)
        if len(sg) < 2:
            return None
        return CubicPathModel(coeff=coeff, x_grid=xg, y_grid=yg, s_grid=sg, kappa_grid=kappa)

    def _path_lateral_at_x(self, pts_m: np.ndarray, x_q: float) -> float:
        if pts_m is None or len(pts_m) < 2:
            return 0.0
        x = np.maximum.accumulate(pts_m[:, 0].astype(np.float32))
        y = pts_m[:, 1].astype(np.float32)
        if float(x[-1] - x[0]) < 1e-5:
            return float(y[-1])
        return float(np.interp(float(x_q), x, y))

    def _extend_path_forward(self, pts_m: np.ndarray, target_span_m: float) -> np.ndarray:
        if pts_m is None or len(pts_m) < 2:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pts_m.astype(np.float32).copy()
        pts[:, 0] = np.maximum.accumulate(pts[:, 0])
        span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        if span >= target_span_m:
            return pts

        if len(pts) >= 4:
            vec = pts[-1] - pts[-4]
        else:
            vec = pts[-1] - pts[0]
        if float(vec[0]) < 0.08:
            vec = np.array([0.35, 0.0], dtype=np.float32)
        vec = vec / max(1e-6, _safe_norm(vec))
        if float(vec[0]) < 0.15:
            vec[0] = 0.15
            vec = vec / max(1e-6, _safe_norm(vec))

        cur = pts[-1].copy()
        x_goal = min(float(self.cfg.path_horizon_m), float(np.min(pts[:, 0]) + target_span_m))
        max_lat = float(self.cfg.bev_lateral_m * 0.5)
        ext = []
        step = max(0.05, float(self.cfg.path_sample_ds_m))
        max_steps = int(self.cfg.path_horizon_m / step) + 8
        for _ in range(max_steps):
            if float(cur[0]) >= x_goal - 1e-4:
                break
            cur = cur + vec * step
            cur[0] = min(cur[0], float(self.cfg.path_horizon_m))
            cur[1] = float(np.clip(cur[1], -max_lat, max_lat))
            ext.append(cur.copy())
        if ext:
            pts = np.vstack([pts, np.asarray(ext, dtype=np.float32)])
        return pts

    def _fallback_centerline(self, mask_255: np.ndarray) -> np.ndarray:
        if not self.cfg.fallback_enabled:
            return np.zeros((0, 2), dtype=np.float32)

        mask = (mask_255 > 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        h, w = mask.shape
        occ_ratio = float(cv2.countNonZero(mask)) / max(1.0, float(mask.size))
        m_per_px = 0.5 * (
            self.cfg.bev_forward_m / max(1.0, float(h - 1))
            + self.cfg.bev_lateral_m / max(1.0, float(w - 1))
        )
        min_width_px = max(
            int(self.cfg.fallback_min_row_pixels),
            int(round(self.cfg.fallback_min_width_m / max(1e-6, m_per_px))),
        )
        row_step = max(1, int(self.cfg.fallback_row_step_px))
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        center_x = 0.5 * float(w - 1)
        points_px: List[Tuple[float, float]] = []

        for y in range(h - 1, -1, -row_step):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) < int(self.cfg.fallback_min_row_pixels):
                continue
            left = int(np.percentile(xs, 5))
            right = int(np.percentile(xs, 95))
            if right <= left:
                continue
            if (right - left + 1) < min_width_px:
                continue

            span = np.arange(left, right + 1, dtype=np.int32)
            if span.size == 0:
                continue
            row_dist = dist[y, span]
            if row_dist.size == 0:
                continue
            # Prefer medial pixels while softly pulling toward BEV centerline.
            score = row_dist - float(self.cfg.fallback_center_pull) * np.abs(span.astype(np.float32) - center_x)
            x_pick = float(span[int(np.argmax(score))])
            points_px.append((x_pick, float(y)))

        if len(points_px) < 2:
            return np.zeros((0, 2), dtype=np.float32)

        pts_px = np.asarray(points_px, dtype=np.float32)
        if len(pts_px) >= 5:
            kern = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            x_smooth = np.convolve(pts_px[:, 0], kern, mode="same")
            x_smooth[0] = pts_px[0, 0]
            x_smooth[-1] = pts_px[-1, 0]
            pts_px[:, 0] = x_smooth

        pts_m = self._metric_from_pixel(pts_px, mask.shape)
        order = np.argsort(pts_m[:, 0], kind="stable")
        pts_m = pts_m[order]
        pts_m[:, 0] = np.maximum.accumulate(pts_m[:, 0])

        provisional_low_conf = occ_ratio < float(self.cfg.fallback_low_conf_occ_ratio)

        # Blend with previous path for continuity when available.
        if self.prev_best_path_m is not None and len(self.prev_best_path_m) >= 4:
            xq = np.clip(pts_m[:, 0], 0.0, float(self.cfg.path_horizon_m))
            y_prev = np.interp(
                xq,
                np.maximum.accumulate(self.prev_best_path_m[:, 0]),
                self.prev_best_path_m[:, 1],
                left=float(self.prev_best_path_m[0, 1]),
                right=float(self.prev_best_path_m[-1, 1]),
            ).astype(np.float32)
            b = float(np.clip(self.cfg.fallback_prev_blend, 0.0, 0.9))
            if provisional_low_conf:
                b = min(b, float(np.clip(self.cfg.fallback_low_conf_prev_blend, 0.0, 0.9)))
            pts_m[:, 1] = (b * y_prev + (1.0 - b) * pts_m[:, 1]).astype(np.float32)

        # Anchor fallback path through ego center to suppress persistent side bias.
        ego_shift = float(np.clip(pts_m[0, 1], -1.2, 1.2))
        pts_m[:, 1] = (pts_m[:, 1] - ego_shift).astype(np.float32)

        pts_m = _resample_polyline(pts_m, self.cfg.path_sample_ds_m, max_len_m=self.cfg.path_horizon_m)
        if len(pts_m) < 2:
            return np.zeros((0, 2), dtype=np.float32)

        raw_span = float(np.max(pts_m[:, 0]) - np.min(pts_m[:, 0]))
        low_conf = provisional_low_conf or (raw_span < float(self.cfg.fallback_low_conf_span_m))
        if low_conf:
            gain = float(np.clip(self.cfg.fallback_recenter_gain, 0.0, 0.95))
            pts_m[:, 1] = ((1.0 - gain) * pts_m[:, 1]).astype(np.float32)
            max_lat = float(max(0.15, self.cfg.fallback_low_conf_max_abs_lateral_m))
            pts_m[:, 1] = np.clip(pts_m[:, 1], -max_lat, max_lat)

        pts_m = self._extend_path_forward(pts_m, float(self.cfg.fallback_target_span_m))
        if len(pts_m) < 2:
            return np.zeros((0, 2), dtype=np.float32)

        lat_clip = float(max(0.05, self.cfg.fallback_output_lateral_clip_m))
        pts_m[:, 1] = np.clip(pts_m[:, 1], -lat_clip, lat_clip)
        span = float(np.max(pts_m[:, 0]) - np.min(pts_m[:, 0]))
        if span < float(self.cfg.fallback_min_forward_span_m):
            return np.zeros((0, 2), dtype=np.float32)
        return _resample_polyline(pts_m, self.cfg.path_sample_ds_m, max_len_m=self.cfg.path_horizon_m)

    def _hold_previous_path(self) -> np.ndarray:
        if self.prev_best_path_m is None or len(self.prev_best_path_m) < 2:
            return np.zeros((0, 2), dtype=np.float32)
        held = self.prev_best_path_m.astype(np.float32).copy()
        held[:, 0] = np.maximum.accumulate(held[:, 0])
        decay = float(max(0.55, 1.0 - 0.05 * self.no_path_counter))
        held[:, 1] *= decay
        lat_clip = float(max(0.05, self.cfg.fallback_output_lateral_clip_m))
        held[:, 1] = np.clip(held[:, 1], -lat_clip, lat_clip)
        return _resample_polyline(held, self.cfg.path_sample_ds_m, max_len_m=self.cfg.path_horizon_m)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process(self, bev_mask_255: np.ndarray) -> PathPlanResult:
        t0 = time.time()
        orig_h, orig_w = bev_mask_255.shape

        mask = self._preprocess(bev_mask_255)
        t1 = time.time()
        skel_bin, dist_m = self._extract_medial_axis(mask)
        node_xy_m, edges, adj, skel_work = self._build_graph(skel_bin, dist_m)
        t2 = time.time()

        candidates = self._search_candidates(node_xy_m, edges, adj)
        candidates = self._score_candidates(candidates)
        best_idx = self._apply_hysteresis(candidates)
        t3 = time.time()

        graph_has_choice = best_idx >= 0 and best_idx < len(candidates)
        best_path_m = candidates[best_idx].points_m if graph_has_choice else np.zeros((0, 2), dtype=np.float32)
        if graph_has_choice:
            best_path_m = _resample_polyline(best_path_m, self.cfg.path_sample_ds_m, self.cfg.path_horizon_m)
        path_model = self._fit_regularized_cubic(best_path_m) if graph_has_choice else None

        # Fallback 1: centerline recovery from sparse/noisy mask.
        fallback_selected = False
        if path_model is None:
            fb_path = self._fallback_centerline(mask)
            if len(fb_path) >= 2:
                fb_model = self._fit_regularized_cubic(fb_path)
                if fb_model is not None:
                    accept_fb = True
                    if self.prev_best_path_m is not None and len(self.prev_best_path_m) >= 4:
                        x_probe = min(1.0, fb_model.x_max_m, float(np.max(self.prev_best_path_m[:, 0])))
                        y_prev = self._path_lateral_at_x(self.prev_best_path_m, x_probe)
                        y_fb = fb_model.y_of_x(x_probe)
                        max_jump = min(0.90, 0.45 * self.cfg.bev_lateral_m)
                        if abs(y_fb - y_prev) > max_jump and self.no_path_counter < 2:
                            accept_fb = False
                    if accept_fb:
                        best_path_m = fb_path
                        path_model = fb_model
                        graph_has_choice = False
                        fallback_selected = True

        # Fallback 2: short hold of previous valid path if no current path can be fit.
        if path_model is None and self.no_path_counter < int(self.cfg.fallback_hold_frames):
            hold_path = self._hold_previous_path()
            if len(hold_path) >= 2:
                hold_model = self._fit_regularized_cubic(hold_path)
                if hold_model is not None:
                    best_path_m = hold_path
                    path_model = hold_model
                    graph_has_choice = False
                    fallback_selected = True

        has_path = path_model is not None and len(best_path_m) >= 2
        if has_path:
            self.prev_best_path_m = best_path_m.astype(np.float32).copy()
            self.prev_end_heading = _polyline_end_heading(best_path_m)
            self.no_path_counter = 0
        else:
            self.no_path_counter = min(1000, self.no_path_counter + 1)

        # Ensure returned candidate list always contains selected path.
        if has_path and graph_has_choice and not fallback_selected:
            order = [best_idx] + [
                i for i in range(len(candidates))
                if i != best_idx
            ][: max(0, self.cfg.search_top_k - 1)]
            best_idx_return = 0
        elif has_path:
            order = []
            best_idx_return = 0
        else:
            order = list(range(min(self.cfg.search_top_k, len(candidates))))
            best_idx_return = -1

        if has_path and not order:
            cand_paths_m = [best_path_m]
            cand_lens_m = [float(_polyline_length_m(best_path_m))]
        else:
            cand_paths_m = [candidates[i].points_m for i in order]
            cand_lens_m = [float(candidates[i].length_m) for i in order]
        cand_paths_px = [self._pixel_from_metric(p, (orig_h, orig_w)) for p in cand_paths_m]
        best_path_px = self._pixel_from_metric(best_path_m, (orig_h, orig_w)) if len(best_path_m) else np.zeros((0, 2), dtype=np.int32)
        skel_px = cv2.resize((skel_work > 0).astype(np.uint8) * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        t4 = time.time()
        return PathPlanResult(
            has_path=bool(has_path),
            path_model=path_model,
            best_path_m=best_path_m.astype(np.float32),
            best_path_px=best_path_px.astype(np.int32),
            candidate_paths_m=[p.astype(np.float32) for p in cand_paths_m],
            candidate_paths_px=[p.astype(np.int32) for p in cand_paths_px],
            candidate_lengths_m=cand_lens_m,
            best_idx=best_idx_return,
            skeleton_px=skel_px.astype(np.uint8),
            graph_nodes=int(len(node_xy_m)),
            graph_edges=int(len(edges)),
            t_skeleton_ms=(t2 - t1) * 1000.0,
            t_path_ms=(t4 - t2) * 1000.0,
        )


class AdaptivePurePursuitController:
    """Improved pure pursuit with curvature-adaptive lookahead and steering filtering."""

    def __init__(self, cfg: Optional[PurePursuitConfig] = None):
        self.cfg = cfg or PurePursuitConfig()
        self.prev_delta_rad = 0.0
        self.prev_kappa_ref = 0.0
        self.prev_path: Optional[CubicPathModel] = None
        self.fail_count = 0

    def _is_discontinuous(self, new_path: CubicPathModel) -> bool:
        if self.prev_path is None:
            return False
        x_probe = min(1.0, new_path.x_max_m, self.prev_path.x_max_m)
        y_new = new_path.y_of_x(x_probe)
        y_old = self.prev_path.y_of_x(x_probe)
        h_new = math.degrees(new_path.heading_of_x(x_probe))
        h_old = math.degrees(self.prev_path.heading_of_x(x_probe))
        if abs(y_new - y_old) > self.cfg.path_discont_lat_m:
            return True
        if abs(h_new - h_old) > self.cfg.path_discont_head_deg:
            return True
        return False

    def update(self, path: Optional[CubicPathModel], speed_mps: float) -> ControlOutput:
        dt = self.cfg.dt_s
        delta_max = math.radians(self.cfg.delta_max_deg)
        rate_max = math.radians(self.cfg.steer_rate_max_deg_s)

        active_path = path
        valid_path = active_path is not None and active_path.length_m > 0.3
        if valid_path and self._is_discontinuous(active_path):
            # reject discontinuous update for this cycle
            active_path = self.prev_path
            valid_path = active_path is not None and active_path.length_m > 0.3

        if valid_path:
            x_ref = min(1.2, active_path.x_max_m)
            kappa_ref = active_path.curvature_of_x(x_ref)
            dk = np.clip(kappa_ref - self.prev_kappa_ref, -self.cfg.kappa_step_max_m_inv, self.cfg.kappa_step_max_m_inv)
            kappa_ref = float(self.prev_kappa_ref + dk)
            self.prev_kappa_ref = kappa_ref

            alpha = self.cfg.lookahead_alpha_v0 + self.cfg.lookahead_alpha_v1 * float(np.clip(speed_mps, 1.0, 3.0))
            L = self.cfg.lookahead_min_m + alpha / (abs(kappa_ref) + self.cfg.lookahead_epsilon)
            L = _clip(L, self.cfg.lookahead_min_m, self.cfg.lookahead_max_m)

            x_t, y_t, _ = active_path.query_s(min(L, active_path.length_m))
            kappa_cmd = (2.0 * y_t) / max(1e-6, (x_t * x_t + y_t * y_t))
            delta_raw = math.atan(self.cfg.wheelbase_m * kappa_cmd)
            delta_raw = _clip(delta_raw, -delta_max, delta_max)
            heading_deg = math.degrees(math.atan2(y_t, max(1e-4, x_t)))

            self.prev_path = active_path
            self.fail_count = 0
        else:
            self.fail_count += 1
            decay = self.cfg.fallback_decay if self.fail_count <= self.cfg.fallback_hold_frames else self.cfg.fallback_decay * 0.85
            delta_raw = self.prev_delta_rad * decay
            L = self.cfg.lookahead_min_m
            x_t, y_t = L, 0.0
            kappa_cmd = 0.0
            heading_deg = math.degrees(delta_raw)

        # Exponential smoothing
        beta = dt / (self.cfg.steer_tau_s + dt)
        delta_f = self.prev_delta_rad + beta * (delta_raw - self.prev_delta_rad)

        # Rate limiting
        max_step = rate_max * dt
        delta_cmd = _clip(delta_f, self.prev_delta_rad - max_step, self.prev_delta_rad + max_step)

        self.prev_delta_rad = delta_cmd
        return ControlOutput(
            steer_deg=math.degrees(delta_cmd),
            lookahead_m=float(L),
            kappa_cmd_m_inv=float(kappa_cmd),
            heading_deg=float(heading_deg),
            target_x_m=float(x_t),
            target_y_m=float(y_t),
            valid_path=bool(valid_path),
        )


