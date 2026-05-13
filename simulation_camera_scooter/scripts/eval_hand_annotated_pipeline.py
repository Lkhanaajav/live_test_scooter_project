#!/usr/bin/env python3
"""
Research evaluation on hand-annotated frames.

Outputs:
  - segmentation metrics for baseline vs candidate masks
  - planner metrics for BEV and no-BEV planners against GT masks
  - representative failure/contrast images
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_DIR = PROJECT_ROOT / "simulation_camera_scooter"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from bev_calibration import load_bev_params  # noqa: E402
from config import BEV_SIZE, NAV_BEV_FORWARD_M, NAV_BEV_LATERAL_M  # noqa: E402
try:
    from dt_path_planner import DtPathPlanner, DtPlannerConfig  # noqa: E402
    _HAS_DT_PLANNER = True
except ImportError:
    _HAS_DT_PLANNER = False
from image_path_planner import CameraDtPlanner, CameraMidpointPlanner  # noqa: E402
from masks import clean_bev_mask_enhanced  # noqa: E402
from realtime_nav_core import BEVPathExtractor, PathExtractorConfig  # noqa: E402


DEFAULT_IMAGES_ROOT = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "images"
DEFAULT_MASKS_ROOT = PROJECT_ROOT / "outputs" / "hand_annotations" / "v1" / "masks"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "research" / "artifacts"
BASELINE_MODEL = SIM_DIR / "models" / "my-segformer-road"
CANDIDATE_MODEL = PROJECT_ROOT / "outputs" / "training" / "binary_segformer_oneformer_teacher" / "best_checkpoint"
FINETUNED_MODEL = PROJECT_ROOT / "outputs" / "training" / "binary_segformer_hand_annotated" / "best_checkpoint"


@dataclass(frozen=True)
class FrameRecord:
    key: str
    video: str
    frame_name: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class SegmentationCase:
    name: str
    model_dir: Path
    threshold: float
    variant: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks-root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--per-video-limit", type=int, default=0)
    parser.add_argument("--videos", nargs="*", default=[])
    return parser.parse_args()


def _ensure_dirs(output_root: Path) -> dict[str, Path]:
    images_dir = output_root / "images"
    tables_dir = output_root / "tables"
    videos_dir = output_root / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return {"images": images_dir, "tables": tables_dir, "videos": videos_dir}


def _sample_evenly(items: list[FrameRecord], limit: int) -> list[FrameRecord]:
    if limit <= 0 or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, num=limit, dtype=np.int32)
    return [items[int(i)] for i in idx]


def _frame_records(
    images_root: Path,
    masks_root: Path,
    videos_filter: set[str],
    max_frames: int,
    per_video_limit: int,
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for video_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        if videos_filter and video_dir.name not in videos_filter:
            continue
        mask_dir = masks_root / video_dir.name
        if not mask_dir.is_dir():
            continue
        video_records: list[FrameRecord] = []
        for image_path in sorted(video_dir.glob("*.jpg")):
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                continue
            video_records.append(
                FrameRecord(
                    key=f"{video_dir.name}/{image_path.stem}",
                    video=video_dir.name,
                    frame_name=image_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
        records.extend(_sample_evenly(video_records, per_video_limit))
    if max_frames > 0:
        records = records[: int(max_frames)]
    if not records:
        raise FileNotFoundError("No hand-annotated frame pairs found.")
    return records


class SegmentationPredictor:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = str(model_dir)
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def predict_prob(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(**inputs).logits
            up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)[0]
            probs = torch.softmax(up, dim=0)[1].detach().cpu().numpy().astype(np.float32)
        return probs, (time.perf_counter() - t0) * 1000.0


class ConfidenceHoldCleaner:
    def __init__(self, threshold: float, band: float = 0.08) -> None:
        self.threshold = float(threshold)
        self.band = float(max(0.01, band))
        self.prev_mask: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_mask = None

    def apply(self, prob: np.ndarray) -> np.ndarray:
        high = prob >= (self.threshold + self.band)
        low = prob <= (self.threshold - self.band)
        if self.prev_mask is None or self.prev_mask.shape != prob.shape:
            mask = prob >= self.threshold
        else:
            mask = self.prev_mask.copy()
            mask[high] = True
            mask[low] = False
            uncertain = ~(high | low)
            mask[uncertain] = self.prev_mask[uncertain]
        mask_u8 = _camera_cleanup(mask.astype(np.uint8) * 255)
        self.prev_mask = mask_u8 > 0
        return mask_u8


def _camera_cleanup(mask_255: np.ndarray) -> np.ndarray:
    mask = (mask_255 > 0).astype(np.uint8)
    h, w = mask.shape
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (w // 60) | 1), max(3, (h // 60) | 1)))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (w // 90) | 1), max(3, (h // 90) | 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask * 255

    center_x = 0.5 * float(w - 1)
    best_idx = 0
    best_score = -1.0
    band = labels[max(0, h - max(10, h // 8)) :, :]
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx = float(centroids[idx][0])
        center_bonus = 1.0 - min(1.0, abs(cx - center_x) / max(1.0, center_x))
        touches_bottom = bool(np.any(band == idx))
        score = area + (0.5 * area if touches_bottom else 0.0) + 0.35 * area * max(0.0, center_bonus)
        if score > best_score:
            best_score = score
            best_idx = idx
    out = np.zeros_like(mask, dtype=np.uint8)
    if best_idx > 0:
        out[labels == best_idx] = 1
    return out * 255


def _mask_metrics(pred_255: np.ndarray, gt_255: np.ndarray) -> dict[str, float]:
    pred = pred_255 > 0
    gt = gt_255 > 0
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    union = tp + fp + fn
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
    iou = tp / max(1.0, union)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_pixels": float(np.count_nonzero(pred)),
        "gt_pixels": float(np.count_nonzero(gt)),
    }


def _stability_iou(mask_a_255: np.ndarray | None, mask_b_255: np.ndarray) -> float:
    if mask_a_255 is None or mask_a_255.shape != mask_b_255.shape:
        return 1.0
    a = mask_a_255 > 0
    b = mask_b_255 > 0
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / max(1.0, union)


def evaluate_segmentation(
    records: list[FrameRecord],
    cases: list[SegmentationCase],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    df_rows: list[dict[str, object]] = []
    case_masks: dict[str, dict[str, np.ndarray]] = {case.name: {} for case in cases}
    for case in cases:
        print(f"[seg] evaluating {case.name} from {case.model_dir}")
        predictor = SegmentationPredictor(case.model_dir, device=args.device)
        cleaner = ConfidenceHoldCleaner(case.threshold) if case.variant == "confhold" else None
        prev_video = None
        prev_mask = None
        for rec in records:
            if prev_video != rec.video:
                prev_video = rec.video
                prev_mask = None
                if cleaner is not None:
                    cleaner.reset()
            frame = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            gt = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or gt is None:
                raise FileNotFoundError(f"Could not read {rec.image_path} or {rec.mask_path}")
            prob, infer_ms = predictor.predict_prob(frame)
            if case.variant == "raw":
                pred_255 = ((prob >= case.threshold).astype(np.uint8) * 255)
            elif case.variant == "confhold":
                pred_255 = cleaner.apply(prob)
            else:
                raise ValueError(f"Unknown segmentation variant: {case.variant}")
            metrics = _mask_metrics(pred_255, gt)
            df_rows.append(
                {
                    "case": case.name,
                    "video": rec.video,
                    "frame": rec.frame_name,
                    "key": rec.key,
                    "infer_ms": infer_ms,
                    "stability_iou_prev": _stability_iou(prev_mask, pred_255),
                    **metrics,
                }
            )
            case_masks[case.name][rec.key] = pred_255
            prev_mask = pred_255
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(df_rows), case_masks


def _row_midpoint(mask_255: np.ndarray, y: int, ref_x: float) -> float | None:
    row = mask_255[int(np.clip(y, 0, mask_255.shape[0] - 1))]
    xs = np.flatnonzero(row > 0)
    if xs.size == 0:
        return None
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
    best = min(runs, key=lambda r: abs(0.5 * float(r[0] + r[1]) - float(ref_x)))
    return 0.5 * float(best[0] + best[1])


def _path_metrics(path_px: np.ndarray, gt_mask_255: np.ndarray, gt_dt: np.ndarray) -> dict[str, float]:
    if path_px is None or len(path_px) < 2:
        return {
            "has_path": 0.0,
            "inside_gt_ratio": 0.0,
            "mean_gt_clearance_px": 0.0,
            "mean_center_error_px": 999.0,
            "forward_span_px": 0.0,
            "heading_deg": 0.0,
            "path_points": 0.0,
        }
    h, w = gt_mask_255.shape[:2]
    pts = np.round(path_px.astype(np.float32)).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    inside = (gt_mask_255[pts[:, 1], pts[:, 0]] > 0).astype(np.float32)
    clear = gt_dt[pts[:, 1], pts[:, 0]].astype(np.float32)
    center_errors = []
    for x, y in pts:
        mid = _row_midpoint(gt_mask_255, int(y), float(x))
        if mid is not None:
            center_errors.append(abs(float(x) - float(mid)))
    return {
        "has_path": 1.0,
        "inside_gt_ratio": float(np.mean(inside)) if len(inside) else 0.0,
        "mean_gt_clearance_px": float(np.mean(clear)) if len(clear) else 0.0,
        "mean_center_error_px": float(np.mean(center_errors)) if center_errors else 999.0,
        "forward_span_px": float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
        "heading_deg": _heading_deg(pts.astype(np.float32)),
        "path_points": float(len(pts)),
    }


def _heading_deg(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    start = points_xy[0]
    idx = min(len(points_xy) - 1, max(1, len(points_xy) * 2 // 5))
    end = points_xy[idx]
    dx = float(end[0] - start[0])
    dy = float(start[1] - end[1])
    if dy <= 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)))


def _warp_mask_to_bev(mask_255: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(mask_255, h_matrix, BEV_SIZE)


def _project_bev_path_to_camera(path_bev_px: np.ndarray, h_inv: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if path_bev_px is None or len(path_bev_px) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pts = path_bev_px.astype(np.float32).reshape(-1, 1, 2)
    cam = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
    h, w = shape_hw
    cam[:, 0] = np.clip(cam[:, 0], 0.0, float(w - 1))
    cam[:, 1] = np.clip(cam[:, 1], 0.0, float(h - 1))
    return np.round(cam).astype(np.int32)


def _bev_clean(mask_bev_255: np.ndarray, nearfield_frac: float) -> np.ndarray:
    m_per_px = 0.5 * (
        float(NAV_BEV_FORWARD_M) / max(1.0, float(BEV_SIZE[1] - 1))
        + float(NAV_BEV_LATERAL_M) / max(1.0, float(BEV_SIZE[0] - 1))
    )
    clean = clean_bev_mask_enhanced(mask_bev_255, m_per_px, enhanced=True)
    if nearfield_frac < 0.999:
        top_rows = int(round(clean.shape[0] * (1.0 - float(nearfield_frac))))
        clean[: max(0, top_rows), :] = 0
    return clean


def _plan_bev_dt(mask_255: np.ndarray, h_matrix: np.ndarray, h_inv: np.ndarray, nearfield_frac: float = 1.0) -> tuple[np.ndarray, str, float]:
    if not _HAS_DT_PLANNER:
        return np.empty((0, 2), dtype=np.float32), "unavailable", 0.0
    bev = _warp_mask_to_bev(mask_255, h_matrix)
    bev = _bev_clean(bev, nearfield_frac=nearfield_frac)
    planner = DtPathPlanner(
        DtPlannerConfig(
            bev_forward_m=NAV_BEV_FORWARD_M,
            bev_lateral_m=NAV_BEV_LATERAL_M,
            work_size=(220, 220),
            min_path_len_m=0.60,
            hold_frames=0,
            path_horizon_m=8.0,
            path_sample_ds_m=0.10,
        )
    )
    planner.reset()
    t0 = time.perf_counter()
    result = planner.plan(bev)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return _project_bev_path_to_camera(result.best_path_px, h_inv, mask_255.shape[:2]), result.path_source, runtime_ms


def _plan_bev_graph(mask_255: np.ndarray, h_matrix: np.ndarray, h_inv: np.ndarray) -> tuple[np.ndarray, str, float]:
    bev = _warp_mask_to_bev(mask_255, h_matrix)
    bev = _bev_clean(bev, nearfield_frac=1.0)
    extractor = BEVPathExtractor(
        PathExtractorConfig(
            planner_mode="dijkstra",
            template_planner_enabled=False,
            use_dt_planner=False,
            work_size=(220, 220),
            min_path_len_m=0.60,
        )
    )
    t0 = time.perf_counter()
    result = extractor.process(bev)
    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return _project_bev_path_to_camera(result.best_path_px, h_inv, mask_255.shape[:2]), str(result.path_source), runtime_ms


def evaluate_planners(
    records: list[FrameRecord],
    mask_cases: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    planner_rows: list[dict[str, object]] = []
    path_cache: dict[tuple[str, str, str], np.ndarray] = {}
    img_dt = CameraDtPlanner()
    img_mid = CameraMidpointPlanner()
    shape_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    for rec in records:
        gt_mask = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        if gt_mask is None or frame is None:
            raise FileNotFoundError(f"Could not read {rec.mask_path} or {rec.image_path}")
        gt_dt = cv2.distanceTransform((gt_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        shape_key = (frame.shape[1], frame.shape[0])
        if shape_key not in shape_cache:
            h_matrix, h_inv, _ = load_bev_params(shape_key)
            shape_cache[shape_key] = (h_matrix, h_inv)
        h_matrix, h_inv = shape_cache[shape_key]
        for mask_case, masks in mask_cases.items():
            mask_255 = masks[rec.key]

            img_cases = []
            t0 = time.perf_counter()
            img_dt_res = img_dt.plan(mask_255)
            img_cases.append(("img_dt", img_dt_res.path_px, img_dt_res.path_source, (time.perf_counter() - t0) * 1000.0))
            t0 = time.perf_counter()
            img_mid_res = img_mid.plan(mask_255)
            img_cases.append(("img_midpoint", img_mid_res.path_px, img_mid_res.path_source, (time.perf_counter() - t0) * 1000.0))

            bev_dt_path, bev_dt_source, bev_dt_ms = _plan_bev_dt(mask_255, h_matrix, h_inv, nearfield_frac=1.0)
            bev_nf_path, bev_nf_source, bev_nf_ms = _plan_bev_dt(mask_255, h_matrix, h_inv, nearfield_frac=0.72)
            bev_graph_path, bev_graph_source, bev_graph_ms = _plan_bev_graph(mask_255, h_matrix, h_inv)

            planner_cases = [
                ("bev_dt_full", bev_dt_path, bev_dt_source, bev_dt_ms),
                ("bev_dt_nearfield", bev_nf_path, bev_nf_source, bev_nf_ms),
                ("bev_graph", bev_graph_path, bev_graph_source, bev_graph_ms),
                *img_cases,
            ]

            for planner_name, path_px, path_source, runtime_ms in planner_cases:
                metrics = _path_metrics(path_px, gt_mask, gt_dt)
                planner_rows.append(
                    {
                        "mask_case": mask_case,
                        "planner": planner_name,
                        "video": rec.video,
                        "frame": rec.frame_name,
                        "key": rec.key,
                        "runtime_ms": runtime_ms,
                        "planner_path_source": path_source,
                        **metrics,
                    }
                )
                path_cache[(mask_case, planner_name, rec.key)] = path_px
    return pd.DataFrame(planner_rows), path_cache


def _aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for keys, group in df.groupby(by):
        row = {}
        if not isinstance(keys, tuple):
            keys = (keys,)
        for name, value in zip(by, keys):
            row[name] = value
        for col in group.columns:
            if col in row or col in {"frame", "key", "video", "planner_path_source"}:
                continue
            if np.issubdtype(group[col].dtype, np.number):
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_p10"] = float(group[col].quantile(0.10))
                row[f"{col}_p90"] = float(group[col].quantile(0.90))
        row["count"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_path(frame_bgr: np.ndarray, path_px: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    out = frame_bgr.copy()
    if path_px is not None and len(path_px) >= 2:
        pts = np.round(path_px.astype(np.float32)).astype(np.int32)
        for idx in range(len(pts) - 1):
            cv2.line(out, tuple(pts[idx]), tuple(pts[idx + 1]), color, 2, cv2.LINE_AA)
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


def save_contact_sheet(
    records: list[FrameRecord],
    planner_df: pd.DataFrame,
    path_cache: dict[tuple[str, str, str], np.ndarray],
    output_images_dir: Path,
) -> None:
    subset = planner_df[(planner_df["mask_case"] == "baseline_raw")]
    bev = subset[subset["planner"] == "bev_dt_full"].set_index("key")
    img = subset[subset["planner"] == "img_dt"].set_index("key")
    interesting = []
    for key in bev.index.intersection(img.index):
        bev_inside = float(bev.loc[key, "inside_gt_ratio"])
        img_inside = float(img.loc[key, "inside_gt_ratio"])
        score = img_inside - bev_inside
        if score > 0.20:
            interesting.append((score, key))
    interesting.sort(reverse=True)
    selected = [key for _, key in interesting[:6]]
    if not selected:
        return

    tiles = []
    rec_map = {rec.key: rec for rec in records}
    for key in selected:
        rec = rec_map[key]
        frame = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or gt_mask is None:
            continue
        overlay = frame.copy()
        overlay[gt_mask > 0] = cv2.addWeighted(overlay[gt_mask > 0], 0.35, np.full_like(overlay[gt_mask > 0], (0, 180, 0)), 0.65, 0)
        bev_path = path_cache.get(("baseline_raw", "bev_dt_full", key), np.zeros((0, 2), dtype=np.int32))
        img_path = path_cache.get(("baseline_raw", "img_dt", key), np.zeros((0, 2), dtype=np.int32))
        tile = _draw_path(overlay, bev_path, (0, 0, 255), "BEV DT")
        tile = _draw_path(tile, img_path, (255, 255, 0), "IMG DT")
        cv2.putText(tile, key.replace("/", " "), (8, tile.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tile = cv2.resize(tile, (480, 270), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
        cv2.imwrite(str(output_images_dir / f"planner_compare_{key.replace('/', '_')}.png"), tile)

    rows = []
    for idx in range(0, len(tiles), 2):
        pair = tiles[idx : idx + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.hstack(pair))
    sheet = np.vstack(rows)
    cv2.imwrite(str(output_images_dir / "planner_bev_vs_img_dt_contact_sheet.png"), sheet)


def main() -> int:
    output_root = Path(args.output_root)
    out_dirs = _ensure_dirs(output_root)
    records = _frame_records(
        images_root=Path(args.images_root),
        masks_root=Path(args.masks_root),
        videos_filter=set(args.videos),
        max_frames=int(args.max_frames),
        per_video_limit=int(args.per_video_limit),
    )
    print(f"[setup] frames: {len(records)}")

    seg_cases = [
        SegmentationCase("baseline_raw", BASELINE_MODEL, 0.50, "raw"),
        SegmentationCase("candidate_raw", CANDIDATE_MODEL, 0.60, "raw"),
        SegmentationCase("candidate_confhold", CANDIDATE_MODEL, 0.60, "confhold"),
    ]
    if FINETUNED_MODEL.exists():
        seg_cases.extend([
            SegmentationCase("finetuned_raw", FINETUNED_MODEL, 0.60, "raw"),
            SegmentationCase("finetuned_confhold", FINETUNED_MODEL, 0.60, "confhold"),
        ])

    seg_df, seg_masks = evaluate_segmentation(records, seg_cases)
    seg_summary = _aggregate(seg_df, by=["case"])
    seg_by_video = _aggregate(seg_df, by=["case", "video"])
    seg_df.to_csv(out_dirs["tables"] / "segmentation_hand_annotations_per_frame.csv", index=False)
    seg_summary.to_csv(out_dirs["tables"] / "segmentation_hand_annotations_summary.csv", index=False)
    seg_by_video.to_csv(out_dirs["tables"] / "segmentation_hand_annotations_by_video.csv", index=False)
    (out_dirs["tables"] / "segmentation_hand_annotations_summary.json").write_text(
        seg_summary.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    gt_masks = {rec.key: cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE) for rec in records}
    mask_cases = {
        "oracle_gt": gt_masks,
        "baseline_raw": seg_masks["baseline_raw"],
        "candidate_confhold": seg_masks["candidate_confhold"],
    }
    if "finetuned_confhold" in seg_masks:
        mask_cases["finetuned_confhold"] = seg_masks["finetuned_confhold"]
    planner_df, path_cache = evaluate_planners(records, mask_cases)
    planner_summary = _aggregate(planner_df, by=["mask_case", "planner"])
    planner_by_video = _aggregate(planner_df, by=["mask_case", "planner", "video"])
    planner_df.to_csv(out_dirs["tables"] / "planner_hand_annotations_per_frame.csv", index=False)
    planner_summary.to_csv(out_dirs["tables"] / "planner_hand_annotations_summary.csv", index=False)
    planner_by_video.to_csv(out_dirs["tables"] / "planner_hand_annotations_by_video.csv", index=False)
    (out_dirs["tables"] / "planner_hand_annotations_summary.json").write_text(
        planner_summary.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    save_contact_sheet(records, planner_df, path_cache, out_dirs["images"])
    print(f"[done] wrote tables to {out_dirs['tables']}")
    print(f"[done] wrote images to {out_dirs['images']}")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main())
