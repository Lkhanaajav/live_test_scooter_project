"""
Boundary target extraction for a lightweight sidewalk-boundary network.

The output format is row-based so a tiny model can predict left/right edge
positions instead of a dense segmentation mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryTargets:
    rows_y: np.ndarray
    valid: np.ndarray
    left_x: np.ndarray
    right_x: np.ndarray
    center_x: np.ndarray
    width_px: np.ndarray

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "rows_y": self.rows_y.tolist(),
            "valid": self.valid.astype(np.uint8).tolist(),
            "left_x": self.left_x.tolist(),
            "right_x": self.right_x.tolist(),
            "center_x": self.center_x.tolist(),
            "width_px": self.width_px.tolist(),
        }


def _normalize_x(x_px: float, width: int) -> float:
    if width <= 1:
        return 0.0
    return float(x_px) / float(width - 1)


def extract_boundary_targets(
    mask: np.ndarray,
    *,
    row_step: int = 4,
    min_width_px: int = 4,
    bottom_crop_px: int = 0,
) -> BoundaryTargets:
    """
    Extract row-wise left/right sidewalk boundaries from a binary mask.

    Invalid rows are marked with valid=0 and x targets = -1.0. The rows are
    ordered from bottom to top, matching the control importance of near field
    geometry.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if row_step <= 0:
        raise ValueError("row_step must be > 0")
    if min_width_px < 1:
        raise ValueError("min_width_px must be >= 1")

    mask_bin = mask > 0
    h, w = mask_bin.shape
    row_start = max(0, h - 1 - int(bottom_crop_px))
    rows = np.arange(row_start, -1, -row_step, dtype=np.int32)

    valid = np.zeros(len(rows), dtype=bool)
    left_x = np.full(len(rows), -1.0, dtype=np.float32)
    right_x = np.full(len(rows), -1.0, dtype=np.float32)
    center_x = np.full(len(rows), -1.0, dtype=np.float32)
    width_px = np.zeros(len(rows), dtype=np.float32)

    for i, y in enumerate(rows):
        xs = np.flatnonzero(mask_bin[y])
        if xs.size == 0:
            continue
        width = int(xs[-1] - xs[0] + 1)
        if width < min_width_px:
            continue
        valid[i] = True
        width_px[i] = float(width)
        left_x[i] = _normalize_x(float(xs[0]), w)
        right_x[i] = _normalize_x(float(xs[-1]), w)
        center_x[i] = _normalize_x(0.5 * float(xs[0] + xs[-1]), w)

    rows_y = rows.astype(np.float32)
    if h > 1:
        rows_y /= float(h - 1)

    return BoundaryTargets(
        rows_y=rows_y,
        valid=valid,
        left_x=left_x,
        right_x=right_x,
        center_x=center_x,
        width_px=width_px,
    )


def build_boundary_record(
    mask: np.ndarray,
    *,
    name: str = "",
    image_path: str = "",
    mask_path: str = "",
    source: str = "",
    row_step: int = 4,
    min_width_px: int = 4,
    bottom_crop_px: int = 0,
) -> dict[str, object]:
    targets = extract_boundary_targets(
        mask,
        row_step=row_step,
        min_width_px=min_width_px,
        bottom_crop_px=bottom_crop_px,
    )
    valid_count = int(targets.valid.sum())
    mean_width_px = float(targets.width_px[targets.valid].mean()) if valid_count else 0.0
    near_field_valid = bool(targets.valid[0]) if len(targets.valid) else False
    return {
        "name": name,
        "image_path": image_path,
        "mask_path": mask_path,
        "source": source,
        "row_step": int(row_step),
        "min_width_px": int(min_width_px),
        "bottom_crop_px": int(bottom_crop_px),
        "num_rows": int(len(targets.rows_y)),
        "num_valid_rows": valid_count,
        "valid_ratio": float(valid_count / max(1, len(targets.rows_y))),
        "near_field_valid": near_field_valid,
        "mean_width_px": mean_width_px,
        "targets": targets.to_dict(),
    }
