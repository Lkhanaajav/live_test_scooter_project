import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boundary_dataset import BoundaryRecordDataset, boundary_collate
from boundary_model import TinyBoundaryNet, boundary_loss, boundary_metrics


def _write_sample(tmp_path, idx: int, left: int, right: int):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 120 + idx
    image_path = tmp_path / f"img_{idx}.png"
    cv2.imwrite(str(image_path), image)

    record = {
        "name": f"sample_{idx}",
        "image_path": str(image_path),
        "mask_path": str(tmp_path / f"mask_{idx}.png"),
        "source": "pseudo",
        "targets": {
            "rows_y": [1.0, 0.5, 0.0],
            "valid": [1, 1, 0],
            "left_x": [left / 31.0, left / 31.0, -1.0],
            "right_x": [right / 31.0, right / 31.0, -1.0],
            "center_x": [((left + right) * 0.5) / 31.0, ((left + right) * 0.5) / 31.0, -1.0],
            "width_px": [right - left + 1, right - left + 1, 0.0],
        },
    }
    return record


def test_boundary_dataset_loads_records(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records = [_write_sample(tmp_path, 0, 6, 18), _write_sample(tmp_path, 1, 7, 19)]
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ds = BoundaryRecordDataset(str(records_path), image_size=(32, 24))
    sample = ds[0]

    assert len(ds) == 2
    assert ds.num_rows == 3
    assert tuple(sample.image.shape) == (3, 24, 32)
    assert sample.valid.tolist() == [1.0, 1.0, 0.0]


def test_boundary_collate_and_model_forward(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records = [_write_sample(tmp_path, 0, 6, 18), _write_sample(tmp_path, 1, 8, 20)]
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ds = BoundaryRecordDataset(str(records_path), image_size=(32, 24))
    batch = boundary_collate([ds[0], ds[1]])

    model = TinyBoundaryNet(num_rows=ds.num_rows)
    pred = model(batch["image"])

    assert tuple(pred["left_x"].shape) == (2, 3)
    assert tuple(pred["right_x"].shape) == (2, 3)
    assert tuple(pred["valid_logits"].shape) == (2, 3)


def test_boundary_loss_and_metrics_are_finite(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records = [_write_sample(tmp_path, 0, 6, 18), _write_sample(tmp_path, 1, 8, 20)]
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ds = BoundaryRecordDataset(str(records_path), image_size=(32, 24))
    batch = boundary_collate([ds[0], ds[1]])
    model = TinyBoundaryNet(num_rows=ds.num_rows)
    pred = model(batch["image"])

    loss, parts = boundary_loss(pred, batch)
    metrics = boundary_metrics(pred, batch)

    assert torch.isfinite(loss)
    assert parts["loss_total"] >= 0.0
    assert 0.0 <= metrics["valid_acc"] <= 1.0
