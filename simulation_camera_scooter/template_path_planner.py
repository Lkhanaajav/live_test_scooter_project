"""
template_path_planner.py
========================
Lightweight corridor extraction and ego-anchored template approval for BEV masks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v) + 1e-9)


def _polyline_length_m(pts_m: np.ndarray) -> float:
    if pts_m is None or len(pts_m) < 2:
        return 0.0
    d = np.diff(pts_m, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def _resample_polyline(pts_m: np.ndarray, ds_m: float) -> np.ndarray:
    if pts_m is None or len(pts_m) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    seg = np.diff(pts_m, axis=0)
    seg_len = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= 1e-6:
        return np.zeros((0, 2), dtype=np.float32)
    ds_m = max(1e-3, float(ds_m))
    samples = np.arange(0.0, total + 1e-6, ds_m, dtype=np.float32)
    if samples[-1] < total:
        samples = np.append(samples, np.float32(total))
    x = np.interp(samples, cum, pts_m[:, 0]).astype(np.float32)
    y = np.interp(samples, cum, pts_m[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=1)


def _path_lateral_at_x(pts_m: np.ndarray, x_q: float) -> float:
    if pts_m is None or len(pts_m) < 2:
        return 0.0
    x = np.maximum.accumulate(pts_m[:, 0].astype(np.float32))
    y = pts_m[:, 1].astype(np.float32)
    if float(x[-1] - x[0]) < 1e-5:
        return float(y[-1])
    return float(np.interp(float(x_q), x, y))


def _cubic_coefficients(horizon_m: float, end_lateral_m: float, end_heading_rad: float) -> np.ndarray:
    L = max(0.5, float(horizon_m))
    dy1 = math.tan(float(end_heading_rad))
    a0 = 0.0
    a1 = 0.0
    a2 = 3.0 * float(end_lateral_m) / (L * L) - dy1 / L
    a3 = dy1 / (L * L) - 2.0 * float(end_lateral_m) / (L * L * L)
    return np.array([a0, a1, a2, a3], dtype=np.float64)


def _eval_cubic(coeff: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a0, a1, a2, a3 = coeff
    y = a0 + a1 * x_grid + a2 * x_grid * x_grid + a3 * x_grid * x_grid * x_grid
    dy = a1 + 2.0 * a2 * x_grid + 3.0 * a3 * x_grid * x_grid
    ddy = 2.0 * a2 + 6.0 * a3 * x_grid
    return y.astype(np.float32), dy.astype(np.float32), ddy.astype(np.float32)


def _curvature_from_cubic(coeff: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    _, dy, ddy = _eval_cubic(coeff, x_grid)
    return (ddy / np.maximum(1e-6, (1.0 + dy * dy) ** 1.5)).astype(np.float32)


def _build_arc_template_path(
    horizon_m: float,
    ds_m: float,
    turn_start_m: float,
    end_heading_rad: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Straight section [0, turn_start] then a constant-curvature circular arc.

    The arc radius is chosen so the arc terminates exactly at x = horizon_m.
    Curvature is constant = 1/R in the arc zone (zero in straight zone).
    This produces a clean 'straight then bend' shape matching real road geometry.

      end_heading_rad > 0  →  right turn (positive lateral)
      end_heading_rad < 0  →  left turn  (negative lateral)
      end_heading_rad = 0  →  straight   (radius → ∞)
    """
    horizon = max(1.5, float(horizon_m))
    ds = max(0.05, float(ds_m))
    ts = float(np.clip(turn_start_m, 0.0, horizon - 0.3))
    theta = abs(float(end_heading_rad))
    sign = 1.0 if float(end_heading_rad) >= 0.0 else -1.0

    if theta < 1e-4:
        # Straight template: no arc
        n = max(3, int(horizon / ds) + 2)
        x_grid = np.linspace(0.0, horizon, n, dtype=np.float32)
        pts_m = np.stack([x_grid, np.zeros_like(x_grid)], axis=1).astype(np.float32)
        return np.zeros(4, dtype=np.float32), pts_m, 0.0

    # Radius: arc forward span = R * sin(theta) must equal (horizon - turn_start)
    arc_forward_span = max(0.1, horizon - ts)
    R = arc_forward_span / math.sin(theta)
    R = max(0.5, R)

    # Straight section
    n_straight = max(2, int(ts / ds) + 1)
    x_straight = np.linspace(0.0, ts, n_straight, dtype=np.float32)
    y_straight = np.zeros(n_straight, dtype=np.float32)

    # Arc section: sweep angle phi in [0, theta]
    arc_len = R * theta
    n_arc = max(3, int(arc_len / ds) + 2)
    phi = np.linspace(0.0, theta, n_arc, dtype=np.float32)
    x_arc = (ts + R * np.sin(phi)).astype(np.float32)
    y_arc = (sign * R * (1.0 - np.cos(phi))).astype(np.float32)

    # Combine, skip duplicate at joint
    x_all = np.concatenate([x_straight, x_arc[1:]])
    y_all = np.concatenate([y_straight, y_arc[1:]])

    # Clip to horizon
    mask = x_all <= horizon + ds * 0.5
    pts_m = np.stack([x_all[mask], y_all[mask]], axis=1).astype(np.float32)
    pts_m = _resample_polyline(pts_m, ds)

    return np.zeros(4, dtype=np.float32), pts_m, 1.0 / R


def _build_turn_template_path(
    horizon_m: float,
    ds_m: float,
    turn_start_m: float,
    end_lateral_m: float,
    end_heading_rad: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    horizon = max(1.5, float(horizon_m))
    ds = max(0.05, float(ds_m))
    x_grid = np.arange(0.0, horizon + 1e-6, ds, dtype=np.float32)
    if x_grid[-1] < horizon:
        x_grid = np.append(x_grid, np.float32(horizon))

    turn_start = float(np.clip(turn_start_m, 0.0, horizon - 0.75))
    if turn_start <= 1e-3:
        coeff = _cubic_coefficients(horizon, end_lateral_m, end_heading_rad)
        y_grid, _, _ = _eval_cubic(coeff, x_grid)
        curv = _curvature_from_cubic(coeff, x_grid)
        pts_m = np.stack([x_grid, y_grid], axis=1).astype(np.float32)
        max_curv = float(np.max(np.abs(curv))) if len(curv) else 0.0
        return coeff.astype(np.float32), _resample_polyline(pts_m, ds), max_curv

    local_len = max(0.75, horizon - turn_start)
    coeff = _cubic_coefficients(local_len, end_lateral_m, end_heading_rad)
    y_grid = np.zeros_like(x_grid, dtype=np.float32)
    curv = np.zeros_like(x_grid, dtype=np.float32)
    turn_mask = x_grid >= turn_start
    x_local = (x_grid[turn_mask] - turn_start).astype(np.float32)
    y_turn, _, _ = _eval_cubic(coeff, x_local)
    y_grid[turn_mask] = y_turn.astype(np.float32)
    curv[turn_mask] = _curvature_from_cubic(coeff, x_local)
    pts_m = np.stack([x_grid, y_grid], axis=1).astype(np.float32)
    max_curv = float(np.max(np.abs(curv))) if len(curv) else 0.0
    return coeff.astype(np.float32), _resample_polyline(pts_m, ds), max_curv


def _pixel_to_forward_m(row_px: np.ndarray, height: int, bev_forward_m: float) -> np.ndarray:
    return ((float(height - 1) - row_px.astype(np.float32)) / max(1.0, float(height - 1)) * float(bev_forward_m)).astype(
        np.float32
    )


def _pixel_to_lateral_m(col_px: np.ndarray, width: int, bev_lateral_m: float) -> np.ndarray:
    return ((col_px.astype(np.float32) / max(1.0, float(width - 1)) - 0.5) * float(bev_lateral_m)).astype(np.float32)


def _split_runs(xs: np.ndarray) -> list[tuple[int, int]]:
    if len(xs) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for val in xs[1:]:
        cur = int(val)
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
        prev = cur
    runs.append((start, prev))
    return runs


@dataclass(frozen=True)
class CorridorConfig:
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0
    row_step_px: int = 4
    min_row_pixels: int = 6
    min_width_m: float = 0.60
    max_width_m: float = 9.50
    near_field_m: float = 1.20
    min_valid_ratio: float = 0.15
    min_near_field_valid_ratio: float = 0.35
    max_width_cv: float = 0.65
    max_row_gap_m: float = 0.70
    center_bias_px: float = 18.0
    image_center_bias_px: float = 10.0
    occupancy_norm: float = 0.08
    min_forward_span_m: float = 1.50


@dataclass(frozen=True)
class Corridor:
    rows_px: np.ndarray
    rows_forward_m: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    center_px: np.ndarray
    left_lateral_m: np.ndarray
    right_lateral_m: np.ndarray
    center_lateral_m: np.ndarray
    width_m: np.ndarray
    valid_mask: np.ndarray
    valid_row_count: int
    valid_ratio: float
    near_field_valid_count: int
    near_field_row_count: int
    near_field_valid_ratio: float
    mean_width_m: float
    width_cv: float
    forward_span_m: float
    mask_occ_ratio: float
    confidence: float


@dataclass(frozen=True)
class TemplatePlannerConfig:
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0
    path_horizon_m: float = 8.0
    path_sample_ds_m: float = 0.25
    near_field_m: float = 1.20
    max_curvature_m_inv: float = 0.90
    min_containment_ratio: float = 0.60
    min_near_containment_ratio: float = 0.72
    min_supported_ratio: float = 0.45
    approval_threshold: float = 0.52
    approval_margin: float = 0.03
    low_confidence_threshold: float = 0.55
    continuity_norm_m: float = 0.45
    obstacle_penalty_weight: float = 0.30
    continuity_weight: float = 0.14
    family_reuse_bonus: float = 0.06
    family_switch_margin: float = 0.05
    straight_preference_margin: float = 0.03
    candidate_output_top_k: int = 5
    reuse_score_floor: float = 0.42
    reuse_confidence_floor: float = 0.60
    reuse_near_containment_ratio: float = 0.58
    corridor_cfg: CorridorConfig = CorridorConfig()


@dataclass(frozen=True)
class PathTemplate:
    template_id: str
    family: str
    coeff: np.ndarray
    points_m: np.ndarray
    length_m: float
    max_curvature_m_inv: float
    end_heading_rad: float
    end_lateral_m: float
    turn_start_m: float


@dataclass(frozen=True)
class ScoredTemplate:
    template: PathTemplate
    total_score: float
    containment_ratio: float
    near_containment_ratio: float
    supported_ratio: float
    clearance_score: float
    center_score: float
    continuity_score: float
    curvature_score: float
    progress_score: float
    evidence_score: float
    obstacle_penalty: float
    approved: bool


@dataclass(frozen=True)
class TemplateApprovalResult:
    corridor: Corridor
    candidates: tuple[ScoredTemplate, ...]
    selected_template_id: str
    selected_template_family: str
    selected_path_m: np.ndarray
    approved: bool
    reuse_selected_path: bool
    confidence: float
    approval_margin: float
    is_low_confidence: bool
    suggested_slowdown: float
    recommend_hold: bool


def _invalid_corridor(mask_255: np.ndarray, cfg: CorridorConfig) -> Corridor:
    mask = (mask_255 > 0).astype(np.uint8)
    h, w = mask.shape
    rows = np.arange(h - 1, -1, -max(1, int(cfg.row_step_px)), dtype=np.int32)
    forward = _pixel_to_forward_m(rows.astype(np.float32), h, cfg.bev_forward_m)
    zeros = np.zeros(len(rows), dtype=np.float32)
    return Corridor(
        rows_px=rows,
        rows_forward_m=forward,
        left_px=zeros.copy(),
        right_px=zeros.copy(),
        center_px=zeros.copy(),
        left_lateral_m=zeros.copy(),
        right_lateral_m=zeros.copy(),
        center_lateral_m=zeros.copy(),
        width_m=zeros.copy(),
        valid_mask=np.zeros(len(rows), dtype=bool),
        valid_row_count=0,
        valid_ratio=0.0,
        near_field_valid_count=0,
        near_field_row_count=int(np.count_nonzero(forward <= float(cfg.near_field_m))),
        near_field_valid_ratio=0.0,
        mean_width_m=0.0,
        width_cv=1.0,
        forward_span_m=0.0,
        mask_occ_ratio=float(np.count_nonzero(mask)) / max(1.0, float(mask.size)),
        confidence=0.0,
    )


def corridor_from_mask(mask_255: np.ndarray, cfg: CorridorConfig | None = None) -> Corridor:
    cfg = cfg or CorridorConfig()
    mask = (mask_255 > 0).astype(np.uint8)
    if mask.ndim != 2 or mask.size == 0 or int(mask.shape[0]) < 2 or int(mask.shape[1]) < 2:
        return _invalid_corridor(np.zeros((2, 2), dtype=np.uint8), cfg)
    if int(np.count_nonzero(mask)) == 0:
        return _invalid_corridor(mask_255, cfg)

    h, w = mask.shape
    rows = np.arange(h - 1, -1, -max(1, int(cfg.row_step_px)), dtype=np.int32)
    n = len(rows)
    left_px = np.zeros(n, dtype=np.float32)
    right_px = np.zeros(n, dtype=np.float32)
    center_px = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    image_center_px = 0.5 * float(w - 1)
    ref_center_px = image_center_px
    min_width_px = max(
        int(cfg.min_row_pixels),
        int(round(float(cfg.min_width_m) / max(1e-6, float(cfg.bev_lateral_m) / max(1.0, float(w - 1))))),
    )

    for i, y in enumerate(rows):
        xs = np.where(mask[int(y)] > 0)[0]
        if len(xs) < int(cfg.min_row_pixels):
            continue
        runs = _split_runs(xs)
        best_score = -1e9
        best_run: tuple[int, int] | None = None
        for x0, x1 in runs:
            width_px = int(x1 - x0 + 1)
            if width_px < min_width_px:
                continue
            mid = 0.5 * float(x0 + x1)
            score = (
                float(width_px)
                - abs(mid - ref_center_px) / max(1e-6, float(cfg.center_bias_px))
                - abs(mid - image_center_px) / max(1e-6, float(cfg.image_center_bias_px))
            )
            if score > best_score:
                best_score = score
                best_run = (x0, x1)
        if best_run is None:
            continue

        x0, x1 = best_run
        left_px[i] = float(x0)
        right_px[i] = float(x1)
        center_px[i] = 0.5 * float(x0 + x1)
        ref_center_px = 0.65 * ref_center_px + 0.35 * center_px[i]
        valid[i] = True

    rows_forward_m = _pixel_to_forward_m(rows.astype(np.float32), h, cfg.bev_forward_m)
    left_lateral_m = _pixel_to_lateral_m(left_px, w, cfg.bev_lateral_m)
    right_lateral_m = _pixel_to_lateral_m(right_px, w, cfg.bev_lateral_m)
    center_lateral_m = _pixel_to_lateral_m(center_px, w, cfg.bev_lateral_m)
    width_m = np.maximum(0.0, right_lateral_m - left_lateral_m).astype(np.float32)

    valid &= width_m >= float(cfg.min_width_m)
    valid &= width_m <= float(cfg.max_width_m)

    valid_row_count = int(np.count_nonzero(valid))
    valid_ratio = float(valid_row_count / max(1, n))
    near_mask = rows_forward_m <= float(cfg.near_field_m) + 1e-6
    near_field_row_count = int(np.count_nonzero(near_mask))
    near_field_valid_count = int(np.count_nonzero(valid & near_mask))
    near_field_valid_ratio = float(near_field_valid_count / max(1, near_field_row_count))

    if valid_row_count >= 2:
        width_valid = width_m[valid]
        mean_width = float(np.mean(width_valid))
        width_cv = float(np.std(width_valid) / max(1e-6, mean_width))
        forward_valid = rows_forward_m[valid]
        forward_span = float(np.max(forward_valid) - np.min(forward_valid))
    else:
        mean_width = 0.0
        width_cv = 1.0
        forward_span = 0.0

    mask_occ_ratio = float(np.count_nonzero(mask)) / max(1.0, float(mask.size))
    coverage_score = _clip(valid_ratio / max(1e-6, float(cfg.min_valid_ratio)), 0.0, 1.0)
    near_score = _clip(near_field_valid_ratio / max(1e-6, float(cfg.min_near_field_valid_ratio)), 0.0, 1.0)
    span_score = _clip(forward_span / max(1e-6, float(cfg.min_forward_span_m)), 0.0, 1.0)
    width_consistency = 1.0 - _clip(width_cv / max(1e-6, float(cfg.max_width_cv)), 0.0, 1.0)
    occ_score = _clip(mask_occ_ratio / max(1e-6, float(cfg.occupancy_norm)), 0.0, 1.0)
    confidence = float(
        0.30 * coverage_score
        + 0.25 * near_score
        + 0.20 * span_score
        + 0.15 * width_consistency
        + 0.10 * occ_score
    )
    if valid_row_count < 2:
        confidence *= 0.2

    return Corridor(
        rows_px=rows,
        rows_forward_m=rows_forward_m,
        left_px=left_px.astype(np.float32),
        right_px=right_px.astype(np.float32),
        center_px=center_px.astype(np.float32),
        left_lateral_m=left_lateral_m.astype(np.float32),
        right_lateral_m=right_lateral_m.astype(np.float32),
        center_lateral_m=center_lateral_m.astype(np.float32),
        width_m=width_m.astype(np.float32),
        valid_mask=valid,
        valid_row_count=valid_row_count,
        valid_ratio=valid_ratio,
        near_field_valid_count=near_field_valid_count,
        near_field_row_count=near_field_row_count,
        near_field_valid_ratio=near_field_valid_ratio,
        mean_width_m=mean_width,
        width_cv=width_cv,
        forward_span_m=forward_span,
        mask_occ_ratio=mask_occ_ratio,
        confidence=_clip(confidence, 0.0, 1.0),
    )


def generate_template_bank(cfg: TemplatePlannerConfig | None = None) -> list[PathTemplate]:
    cfg = cfg or TemplatePlannerConfig()
    horizon = max(1.5, float(cfg.path_horizon_m))
    ds = max(0.05, float(cfg.path_sample_ds_m))
    x_grid = np.arange(0.0, horizon + 1e-6, ds, dtype=np.float32)
    if x_grid[-1] < horizon:
        x_grid = np.append(x_grid, np.float32(horizon))

    # Circular arc bank: straight approach then constant-curvature bend.
    # Radius is auto-computed per arc so the path terminates at x = horizon.
    # Convention: positive end_heading_deg → right, negative → left.
    # turn_start controls where the arc begins (shorter = bend starts closer to ego).
    # end_heading_deg controls arc sweep angle = final heading off-axis.
    base_specs = [
        # (template_id, family, turn_start_m, end_heading_deg)
        ("straight_center",  "straight", 0.00,   0.0),
        ("left_gentle",      "left",     0.80, -40.0),
        ("left_medium",      "left",     0.50, -60.0),
        ("left_sharp",       "left",     0.30, -75.0),
        ("right_gentle",     "right",    0.80,  40.0),
        ("right_medium",     "right",    0.50,  60.0),
        ("right_sharp",      "right",    0.30,  75.0),
    ]

    templates: list[PathTemplate] = []
    for template_id, family, turn_start_m, end_heading_deg in base_specs:
        end_heading = math.radians(float(end_heading_deg))
        coeff, pts_m, max_curv = _build_arc_template_path(
            horizon,
            ds,
            turn_start_m,
            end_heading,
        )
        if max_curv > float(cfg.max_curvature_m_inv) + 0.05:
            continue
        end_lat = float(pts_m[-1, 1]) if len(pts_m) > 0 else 0.0
        templates.append(
            PathTemplate(
                template_id=template_id,
                family=family,
                coeff=coeff,
                points_m=pts_m.astype(np.float32),
                length_m=_polyline_length_m(pts_m),
                max_curvature_m_inv=max_curv,
                end_heading_rad=float(end_heading),
                end_lateral_m=end_lat,
                turn_start_m=float(turn_start_m),
            )
        )

    templates.sort(key=lambda t: (0 if t.family == "straight" else 1, t.family, t.turn_start_m, t.template_id))
    return templates


def _sample_corridor(corridor: Corridor, x_query: np.ndarray, cfg: CorridorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_rows = corridor.valid_mask
    xs = corridor.rows_forward_m[valid_rows]
    if len(xs) < 2:
        n = len(x_query)
        return (
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    left = corridor.left_lateral_m[valid_rows]
    right = corridor.right_lateral_m[valid_rows]
    center = corridor.center_lateral_m[valid_rows]

    q = x_query.astype(np.float32)
    valid = (q >= float(xs[0])) & (q <= float(xs[-1]))
    idx = np.searchsorted(xs, q, side="left")
    lo_idx = np.clip(idx - 1, 0, len(xs) - 1)
    hi_idx = np.clip(idx, 0, len(xs) - 1)
    x_lo = xs[lo_idx]
    x_hi = xs[hi_idx]
    gap = np.abs(x_hi - x_lo)
    valid &= gap <= float(cfg.max_row_gap_m)

    left_q = np.interp(q, xs, left).astype(np.float32)
    right_q = np.interp(q, xs, right).astype(np.float32)
    center_q = np.interp(q, xs, center).astype(np.float32)
    return valid.astype(bool), left_q, right_q, center_q


def _continuity_score(path_m: np.ndarray, prev_path_m: np.ndarray | None, norm_m: float) -> float:
    if prev_path_m is None or len(prev_path_m) < 2:
        return 0.5
    x_probes = [0.8, 1.5, 2.4]
    diffs = []
    for xq in x_probes:
        diffs.append(abs(_path_lateral_at_x(path_m, xq) - _path_lateral_at_x(prev_path_m, xq)))
    mean_diff = float(np.mean(diffs))
    return 1.0 - _clip(mean_diff / max(1e-6, float(norm_m)), 0.0, 1.0)


def _obstacle_overlap(path_m: np.ndarray, obstacle_zones_m: Sequence[Sequence[float]] | None) -> float:
    if not obstacle_zones_m:
        return 0.0
    penalty = 0.0
    for fwd_m, lat_m, rad_m in obstacle_zones_m:
        d = np.sqrt((path_m[:, 0] - float(fwd_m)) ** 2 + (path_m[:, 1] - float(lat_m)) ** 2)
        radius = max(1e-3, float(rad_m))
        hard = float(np.mean(d < radius))
        soft = float(np.mean(np.clip(1.0 - d / (1.25 * radius), 0.0, 1.0)))
        peak = float(np.max(np.clip(1.0 - d / (1.25 * radius), 0.0, 1.0)))
        penalty += 0.50 * hard + 0.30 * soft + 0.20 * peak
    return penalty / max(1.0, float(len(obstacle_zones_m)))


def _family_group(family: str) -> str:
    if family == "straight":
        return "straight"
    return family.split("_", 1)[0]


def _prioritize_candidates(
    scored: list[ScoredTemplate],
    prev_template_family: str | None,
    cfg: TemplatePlannerConfig,
) -> list[ScoredTemplate]:
    if not scored:
        return []

    prev_family = str(prev_template_family or "").strip().lower()
    prioritized = list(scored)

    if prev_family:
        decorated: list[tuple[float, ScoredTemplate]] = []
        for cand in scored:
            adjusted = float(cand.total_score)
            if cand.template.family == prev_family:
                adjusted += float(cfg.family_reuse_bonus)
            decorated.append((adjusted, cand))
        decorated.sort(key=lambda item: item[0], reverse=True)
        prioritized = [cand for _, cand in decorated]

        keep_prev = next((cand for cand in prioritized if cand.template.family == prev_family), None)
        if keep_prev is not None:
            best = prioritized[0]
            if best.template.family != prev_family:
                if keep_prev.total_score >= float(best.total_score) - float(cfg.family_switch_margin):
                    prioritized = [keep_prev] + [cand for cand in prioritized if cand is not keep_prev]

    straight_cand = next((cand for cand in prioritized if cand.template.family == "straight"), None)
    best = prioritized[0]
    if straight_cand is not None and best.template.family != "straight" and prev_family in {"", "straight"}:
        if (
            straight_cand.total_score >= float(best.total_score) - float(cfg.straight_preference_margin)
            and straight_cand.obstacle_penalty <= float(best.obstacle_penalty) + 0.05
        ):
            prioritized = [straight_cand] + [cand for cand in prioritized if cand is not straight_cand]
    return prioritized


def score_template_against_corridor(
    template: PathTemplate,
    corridor: Corridor,
    cfg: TemplatePlannerConfig | None = None,
    *,
    prev_path_m: np.ndarray | None = None,
    obstacle_zones_m: Sequence[Sequence[float]] | None = None,
) -> ScoredTemplate:
    cfg = cfg or TemplatePlannerConfig()
    path_m = template.points_m.astype(np.float32)
    support, left_q, right_q, center_q = _sample_corridor(corridor, path_m[:, 0], cfg.corridor_cfg)
    width_q = np.maximum(1e-3, right_q - left_q)
    inside = support & (path_m[:, 1] >= left_q) & (path_m[:, 1] <= right_q)
    near_mask = path_m[:, 0] <= float(cfg.near_field_m) + 1e-6

    supported_ratio = float(np.mean(support)) if len(support) else 0.0
    containment_ratio = float(np.mean(inside[support])) if np.any(support) else 0.0
    near_supported = support & near_mask
    if np.any(near_supported):
        near_containment_ratio = float(np.mean(inside[near_supported]))
    else:
        near_containment_ratio = containment_ratio

    clearance = np.minimum(path_m[:, 1] - left_q, right_q - path_m[:, 1])
    # Absolute clearance: score 1.0 at >=0.20m from wall, 0.0 on wall edge.
    # Avoids the normalized version rewarding paths that merely exist inside a wide corridor.
    _min_clearance_m = 0.20
    clearance_norm = np.clip(clearance / _min_clearance_m, 0.0, 1.0)
    clearance_score = float(np.mean(clearance_norm[support])) if np.any(support) else 0.0

    center_err = np.abs(path_m[:, 1] - center_q) / np.maximum(1e-3, 0.5 * width_q)
    center_score = 1.0 - float(np.mean(np.clip(center_err[support], 0.0, 1.0))) if np.any(support) else 0.0
    continuity = _continuity_score(path_m, prev_path_m, cfg.continuity_norm_m)
    curvature_score = 1.0 - _clip(template.max_curvature_m_inv / max(1e-6, float(cfg.max_curvature_m_inv)), 0.0, 1.0)
    progress_score = float(np.max(path_m[inside, 0]) / max(1e-6, float(cfg.path_horizon_m))) if np.any(inside) else 0.0
    evidence_score = 0.55 * float(corridor.confidence) + 0.45 * supported_ratio
    obstacle_penalty = _obstacle_overlap(path_m, obstacle_zones_m)

    total = (
        0.22 * containment_ratio
        + 0.18 * near_containment_ratio
        + 0.16 * clearance_score
        + 0.10 * center_score  # decoupled from corridor.confidence — low evidence != low centering need
        + float(cfg.continuity_weight) * continuity
        + 0.08 * curvature_score
        + 0.06 * progress_score
        + 0.06 * evidence_score
    )
    total -= float(cfg.obstacle_penalty_weight) * obstacle_penalty
    total = _clip(total, 0.0, 1.0)

    approved = bool(
        supported_ratio >= float(cfg.min_supported_ratio)
        and containment_ratio >= float(cfg.min_containment_ratio)
        and near_containment_ratio >= float(cfg.min_near_containment_ratio)
        and template.max_curvature_m_inv <= float(cfg.max_curvature_m_inv) + 0.05
    )
    return ScoredTemplate(
        template=template,
        total_score=total,
        containment_ratio=containment_ratio,
        near_containment_ratio=near_containment_ratio,
        supported_ratio=supported_ratio,
        clearance_score=clearance_score,
        center_score=center_score,
        continuity_score=continuity,
        curvature_score=curvature_score,
        progress_score=progress_score,
        evidence_score=evidence_score,
        obstacle_penalty=obstacle_penalty,
        approved=approved,
    )


def approve_template_bank(
    corridor: Corridor,
    cfg: TemplatePlannerConfig | None = None,
    *,
    prev_path_m: np.ndarray | None = None,
    prev_template_family: str | None = None,
    obstacle_zones_m: Sequence[Sequence[float]] | None = None,
    templates: Iterable[PathTemplate] | None = None,
    gps_intent_family: str | None = None,
) -> TemplateApprovalResult:
    cfg = cfg or TemplatePlannerConfig()
    templates = list(templates) if templates is not None else generate_template_bank(cfg)

    # GPS intent conditioning: filter to only intent-consistent templates.
    # "straight" intent → only straight; "left"/"right" → only that direction.
    # Per design spec: do not fall back to a different direction if the intent fails.
    _intent = str(gps_intent_family or "").strip().lower()
    _intent_filtered = bool(_intent in {"straight", "left", "right"})
    if _intent_filtered:
        intent_candidates = [t for t in templates if t.family == _intent]
        if intent_candidates:
            templates = intent_candidates
        # If no templates match the intent family (e.g. bank only has 'straight'),
        # fall through with the full bank so the approval gate can still reject.

    if not templates:
        empty = _invalid_corridor(np.zeros((2, 2), dtype=np.uint8), cfg.corridor_cfg)
        return TemplateApprovalResult(
            corridor=empty,
            candidates=tuple(),
            selected_template_id="",
            selected_template_family="",
            selected_path_m=np.zeros((0, 2), dtype=np.float32),
            approved=False,
            reuse_selected_path=False,
            confidence=0.0,
            approval_margin=0.0,
            is_low_confidence=True,
            suggested_slowdown=1.0,
            recommend_hold=True,
        )

    scored = [
        score_template_against_corridor(
            tmpl,
            corridor,
            cfg,
            prev_path_m=prev_path_m,
            obstacle_zones_m=obstacle_zones_m,
        )
        for tmpl in templates
    ]
    scored.sort(key=lambda s: s.total_score, reverse=True)
    prioritized = _prioritize_candidates(scored, prev_template_family, cfg)

    # GPS intent: when a direction is given, commit to the sharpest template
    # in that family (most lateral displacement) regardless of mask fit score.
    # The intent overrides corridor evidence — the turn may not be visible yet.
    if _intent_filtered and _intent in {"left", "right"}:
        prioritized = sorted(
            prioritized,
            key=lambda s: abs(s.template.end_lateral_m),
            reverse=True,
        )

    best = prioritized[0]
    best_group = _family_group(best.template.family)
    runner_up = next((cand for cand in prioritized[1:] if _family_group(cand.template.family) != best_group), None)
    margin = float(best.total_score - runner_up.total_score) if runner_up is not None else float(best.total_score)
    # An off-intent selection (shouldn't happen after filtering, but guards template=None fallback)
    _intent_violated = bool(
        _intent_filtered and _intent and best.template.family != _intent
    )
    approved = bool(
        best.approved
        and best.total_score >= float(cfg.approval_threshold)
        and margin >= float(cfg.approval_margin)
        and not _intent_violated
    )
    reuse_selected_path = bool(
        not approved
        and not _intent_violated
        and str(prev_template_family or "").strip().lower() == best.template.family
        and corridor.confidence >= float(cfg.reuse_confidence_floor)
        and best.total_score >= float(cfg.reuse_score_floor)
        and best.near_containment_ratio >= float(cfg.reuse_near_containment_ratio)
        and best.supported_ratio >= max(0.30, 0.7 * float(cfg.min_supported_ratio))
    )

    confidence = float(0.55 * best.total_score + 0.45 * corridor.confidence)
    if not approved:
        confidence *= 0.65
    if not best.approved:
        confidence *= 0.5
    if reuse_selected_path:
        confidence = max(confidence, min(0.72, 0.55 * best.total_score + 0.45 * corridor.confidence))
    confidence = _clip(confidence, 0.0, 1.0)

    is_low_confidence = bool(
        (not approved and not reuse_selected_path)
        or confidence < float(cfg.low_confidence_threshold)
        or (
            corridor.near_field_valid_ratio < float(cfg.corridor_cfg.min_near_field_valid_ratio)
            and corridor.valid_ratio < max(0.25, float(cfg.corridor_cfg.min_valid_ratio))
        )
    )
    suggested_slowdown = _clip(1.0 - confidence, 0.0, 1.0)
    if is_low_confidence:
        suggested_slowdown = max(0.35, suggested_slowdown)
    if corridor.confidence < 0.20 or best.near_containment_ratio < 0.50:
        suggested_slowdown = max(suggested_slowdown, 0.80)
    recommend_hold = bool(
        (not approved and not reuse_selected_path)
        and (
            # GPS intent was given but no intent-consistent template cleared the gate.
            # Per design spec: do not fall back to a different maneuver direction.
            _intent_violated
            or corridor.confidence < 0.35
            or (
                corridor.near_field_valid_ratio < float(cfg.corridor_cfg.min_near_field_valid_ratio)
                and corridor.valid_ratio < max(0.25, float(cfg.corridor_cfg.min_valid_ratio))
            )
            or (
                best.supported_ratio < max(0.25, 0.6 * float(cfg.min_supported_ratio))
                and best.near_containment_ratio < 0.50
            )
        )
    )

    return TemplateApprovalResult(
        corridor=corridor,
        candidates=tuple(prioritized),
        selected_template_id=best.template.template_id,
        selected_template_family=best.template.family,
        selected_path_m=best.template.points_m.astype(np.float32),
        approved=approved,
        reuse_selected_path=reuse_selected_path,
        confidence=confidence,
        approval_margin=margin,
        is_low_confidence=is_low_confidence,
        suggested_slowdown=suggested_slowdown,
        recommend_hold=recommend_hold,
    )
