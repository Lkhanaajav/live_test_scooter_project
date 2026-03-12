"""
Decode row-wise boundary predictions into a centerline path and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _smooth_1d(values: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    if len(values) < 3 or len(kernel) < 3:
        return values.astype(np.float32, copy=True)
    kern = np.asarray(kernel, dtype=np.float32)
    kern = kern / max(1e-6, float(kern.sum()))
    out = np.convolve(values.astype(np.float32), kern, mode="same")
    out[0] = values[0]
    out[-1] = values[-1]
    return out.astype(np.float32)


def _resample_polyline(pts_m: np.ndarray, ds_m: float) -> np.ndarray:
    if pts_m is None or len(pts_m) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    seg = np.diff(pts_m, axis=0)
    seglen = np.sqrt(np.sum(seg * seg, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])
    if total <= 1e-6:
        return np.zeros((0, 2), dtype=np.float32)
    ds_m = max(1e-3, float(ds_m))
    sq = np.arange(0.0, total + 1e-6, ds_m, dtype=np.float32)
    if sq[-1] < total:
        sq = np.append(sq, np.float32(total))
    xq = np.interp(sq, s, pts_m[:, 0])
    yq = np.interp(sq, s, pts_m[:, 1])
    return np.stack([xq, yq], axis=1).astype(np.float32)


@dataclass(frozen=True)
class BoundaryPathConfig:
    bev_forward_m: float = 10.0
    bev_lateral_m: float = 10.0
    path_sample_ds_m: float = 0.25
    valid_threshold: float = 0.5
    min_valid_ratio: float = 0.25
    near_field_row_count: int = 3
    min_near_field_valid_ratio: float = 0.5
    min_forward_span_m: float = 1.5
    min_width_m: float = 0.6
    max_width_m: float = 4.0
    max_width_cv: float = 0.45
    previous_path_blend: float = 0.2
    low_confidence_threshold: float = 0.6
    smoothing_kernel: tuple[float, ...] = (0.2, 0.6, 0.2)


@dataclass(frozen=True)
class BoundaryPathResult:
    has_path: bool
    is_low_confidence: bool
    confidence: float
    suggested_slowdown: float
    valid_row_count: int
    valid_ratio: float
    near_field_valid_count: int
    mean_width_m: float
    width_cv: float
    forward_span_m: float
    path_m: np.ndarray
    path_px: np.ndarray
    left_boundary_px: np.ndarray
    right_boundary_px: np.ndarray


def _metric_from_normalized(rows_y: np.ndarray, x_norm: np.ndarray, cfg: BoundaryPathConfig) -> np.ndarray:
    forward = (1.0 - rows_y) * float(cfg.bev_forward_m)
    lateral = (x_norm - 0.5) * float(cfg.bev_lateral_m)
    return np.stack([forward, lateral], axis=1).astype(np.float32)


def _pixel_from_metric(pts_m: np.ndarray, image_size: tuple[int, int], cfg: BoundaryPathConfig) -> np.ndarray:
    if pts_m is None or len(pts_m) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    width, height = int(image_size[0]), int(image_size[1])
    f = np.clip(pts_m[:, 0], 0.0, float(cfg.bev_forward_m))
    l = np.clip(pts_m[:, 1], -0.5 * float(cfg.bev_lateral_m), 0.5 * float(cfg.bev_lateral_m))
    u = ((l / float(cfg.bev_lateral_m)) + 0.5) * max(1.0, float(width - 1))
    v = (1.0 - f / float(cfg.bev_forward_m)) * max(1.0, float(height - 1))
    return np.round(np.stack([u, v], axis=1)).astype(np.int32)


def _blend_previous_path(pts_m: np.ndarray, prev_path_m: np.ndarray | None, blend: float) -> np.ndarray:
    if prev_path_m is None or len(prev_path_m) < 2 or blend <= 0.0:
        return pts_m
    x_prev = np.maximum.accumulate(prev_path_m[:, 0].astype(np.float32))
    y_prev = prev_path_m[:, 1].astype(np.float32)
    y_ref = np.interp(
        pts_m[:, 0],
        x_prev,
        y_prev,
        left=float(y_prev[0]),
        right=float(y_prev[-1]),
    ).astype(np.float32)
    pts = pts_m.astype(np.float32, copy=True)
    pts[:, 1] = (blend * y_ref + (1.0 - blend) * pts[:, 1]).astype(np.float32)
    return pts


def decode_boundary_prediction(
    pred: dict[str, object],
    *,
    rows_y: object,
    image_size: tuple[int, int],
    cfg: BoundaryPathConfig | None = None,
    prev_path_m: np.ndarray | None = None,
) -> BoundaryPathResult:
    cfg = cfg or BoundaryPathConfig()
    rows = _as_numpy(rows_y).reshape(-1)
    left_x = _as_numpy(pred["left_x"]).reshape(-1)
    right_x = _as_numpy(pred["right_x"]).reshape(-1)
    if "valid_prob" in pred:
        valid_prob = _as_numpy(pred["valid_prob"]).reshape(-1)
    elif "valid_logits" in pred:
        valid_prob = _sigmoid(_as_numpy(pred["valid_logits"]).reshape(-1))
    else:
        raise KeyError("pred must contain valid_prob or valid_logits")

    if not (len(rows) == len(left_x) == len(right_x) == len(valid_prob)):
        raise ValueError("rows_y and prediction arrays must have matching length")

    center_x = 0.5 * (left_x + right_x)
    width_m = np.clip(right_x - left_x, 0.0, None) * float(cfg.bev_lateral_m)
    geometry_ok = (
        np.isfinite(left_x)
        & np.isfinite(right_x)
        & np.isfinite(valid_prob)
        & (left_x >= 0.0)
        & (right_x <= 1.0)
        & (right_x > left_x)
        & (width_m >= float(cfg.min_width_m))
        & (width_m <= float(cfg.max_width_m))
    )
    valid_mask = (valid_prob >= float(cfg.valid_threshold)) & geometry_ok
    valid_row_count = int(valid_mask.sum())
    valid_ratio = float(valid_row_count / max(1, len(rows)))
    near_count = int(max(0, min(len(rows), int(cfg.near_field_row_count))))
    near_field_valid_count = int(valid_mask[:near_count].sum()) if near_count else 0
    min_near_valid = int(np.ceil(float(cfg.min_near_field_valid_ratio) * float(near_count))) if near_count else 0

    if valid_row_count < 2:
        return BoundaryPathResult(
            has_path=False,
            is_low_confidence=True,
            confidence=0.0,
            suggested_slowdown=1.0,
            valid_row_count=valid_row_count,
            valid_ratio=valid_ratio,
            near_field_valid_count=near_field_valid_count,
            mean_width_m=0.0,
            width_cv=1.0,
            forward_span_m=0.0,
            path_m=np.zeros((0, 2), dtype=np.float32),
            path_px=np.zeros((0, 2), dtype=np.int32),
            left_boundary_px=np.zeros((0, 2), dtype=np.int32),
            right_boundary_px=np.zeros((0, 2), dtype=np.int32),
        )

    rows_sel = rows[valid_mask]
    left_sel = np.clip(left_x[valid_mask], 0.0, 1.0)
    right_sel = np.clip(right_x[valid_mask], 0.0, 1.0)
    center_sel = np.clip(center_x[valid_mask], 0.0, 1.0)
    width_sel = width_m[valid_mask]

    path_m = _metric_from_normalized(rows_sel, center_sel, cfg)
    left_m = _metric_from_normalized(rows_sel, left_sel, cfg)
    right_m = _metric_from_normalized(rows_sel, right_sel, cfg)
    order = np.argsort(path_m[:, 0], kind="stable")
    path_m = path_m[order]
    left_m = left_m[order]
    right_m = right_m[order]
    width_sel = width_sel[order]

    if len(path_m) >= 3:
        path_m[:, 1] = _smooth_1d(path_m[:, 1], cfg.smoothing_kernel)
    path_m[:, 0] = np.maximum.accumulate(path_m[:, 0])
    keep = np.ones(len(path_m), dtype=bool)
    keep[1:] = np.diff(path_m[:, 0]) > 1e-4
    path_m = path_m[keep]
    left_m = left_m[keep]
    right_m = right_m[keep]
    width_sel = width_sel[keep]

    if len(path_m) >= 2:
        path_m = _blend_previous_path(path_m, prev_path_m, float(cfg.previous_path_blend))

    forward_span_m = float(path_m[-1, 0] - path_m[0, 0]) if len(path_m) >= 2 else 0.0
    has_path = bool(len(path_m) >= 2 and forward_span_m >= float(cfg.min_forward_span_m))

    mean_width_m = float(width_sel.mean()) if len(width_sel) else 0.0
    width_cv = float(width_sel.std() / max(1e-6, width_sel.mean())) if len(width_sel) else 1.0
    coverage_score = float(np.clip(valid_ratio / max(1e-6, float(cfg.min_valid_ratio)), 0.0, 1.0))
    near_score = float(near_field_valid_count / max(1, near_count)) if near_count else 0.0
    span_score = float(np.clip(forward_span_m / max(1e-6, float(cfg.min_forward_span_m)), 0.0, 1.0))
    width_consistency = float(1.0 - np.clip(width_cv / max(1e-6, float(cfg.max_width_cv)), 0.0, 1.0))
    confidence = 0.35 * coverage_score + 0.30 * near_score + 0.20 * span_score + 0.15 * width_consistency
    if not has_path:
        confidence *= 0.25
    confidence = float(np.clip(confidence, 0.0, 1.0))
    is_low_confidence = bool(
        not has_path
        or confidence < float(cfg.low_confidence_threshold)
        or near_field_valid_count < max(1, min_near_valid)
    )
    suggested_slowdown = float(np.clip(1.0 - confidence, 0.0, 1.0))
    if is_low_confidence:
        suggested_slowdown = max(0.35, suggested_slowdown)
    if not has_path:
        suggested_slowdown = 1.0

    path_out = _resample_polyline(path_m, float(cfg.path_sample_ds_m)) if has_path else path_m.astype(np.float32)
    left_px = _pixel_from_metric(left_m, image_size, cfg)
    right_px = _pixel_from_metric(right_m, image_size, cfg)
    path_px = _pixel_from_metric(path_out, image_size, cfg)

    return BoundaryPathResult(
        has_path=has_path,
        is_low_confidence=is_low_confidence,
        confidence=confidence,
        suggested_slowdown=suggested_slowdown,
        valid_row_count=int(len(path_m)),
        valid_ratio=valid_ratio,
        near_field_valid_count=near_field_valid_count,
        mean_width_m=mean_width_m,
        width_cv=width_cv,
        forward_span_m=forward_span_m,
        path_m=path_out.astype(np.float32),
        path_px=path_px,
        left_boundary_px=left_px,
        right_boundary_px=right_px,
    )
