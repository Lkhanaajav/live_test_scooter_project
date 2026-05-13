"""
Dataset helpers for lightweight row-wise sidewalk-boundary prediction.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


@dataclass(frozen=True)
class BoundarySample:
    image: torch.Tensor
    valid: torch.Tensor
    left_x: torch.Tensor
    right_x: torch.Tensor
    center_x: torch.Tensor
    rows_y: torch.Tensor
    width_px: torch.Tensor
    name: str
    image_path: str
    mask_path: str
    source: str


class BoundaryRecordDataset(Dataset):
    """
    Dataset over exported boundary-target JSONL records.

    Images are loaded at access time and resized to a fixed training size.
    Boundary targets are assumed to be normalized to [0, 1] by the exporter.
    """

    def __init__(
        self,
        records_path: str,
        *,
        image_size: tuple[int, int] = (160, 96),
    ) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must be positive")
        self.records_path = os.path.abspath(records_path)
        self.base_dir = os.path.dirname(self.records_path)
        self.image_size = tuple(int(v) for v in image_size)
        self.records = self._load_records(self.records_path)
        if not self.records:
            raise ValueError(f"No records found in {records_path}")
        self.num_rows = len(self.records[0]["targets"]["rows_y"])

    @staticmethod
    def _load_records(records_path: str) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def __getitem__(self, idx: int) -> BoundarySample:
        rec = self.records[idx]
        image_path = _resolve_path(self.base_dir, str(rec["image_path"]))
        mask_path = _resolve_path(self.base_dir, str(rec.get("mask_path", "")))
        image = self._load_image(image_path)
        targets = rec["targets"]
        return BoundarySample(
            image=image,
            valid=torch.tensor(targets["valid"], dtype=torch.float32),
            left_x=torch.tensor(targets["left_x"], dtype=torch.float32),
            right_x=torch.tensor(targets["right_x"], dtype=torch.float32),
            center_x=torch.tensor(targets["center_x"], dtype=torch.float32),
            rows_y=torch.tensor(targets["rows_y"], dtype=torch.float32),
            width_px=torch.tensor(targets["width_px"], dtype=torch.float32),
            name=str(rec.get("name", "")),
            image_path=image_path,
            mask_path=mask_path,
            source=str(rec.get("source", "")),
        )


def boundary_collate(batch: list[BoundarySample]) -> dict[str, object]:
    return {
        "image": torch.stack([sample.image for sample in batch], dim=0),
        "valid": torch.stack([sample.valid for sample in batch], dim=0),
        "left_x": torch.stack([sample.left_x for sample in batch], dim=0),
        "right_x": torch.stack([sample.right_x for sample in batch], dim=0),
        "center_x": torch.stack([sample.center_x for sample in batch], dim=0),
        "rows_y": torch.stack([sample.rows_y for sample in batch], dim=0),
        "width_px": torch.stack([sample.width_px for sample in batch], dim=0),
        "name": [sample.name for sample in batch],
        "image_path": [sample.image_path for sample in batch],
        "mask_path": [sample.mask_path for sample in batch],
        "source": [sample.source for sample in batch],
    }
