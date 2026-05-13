"""
Tiny row-wise boundary network, loss, and evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBoundaryNet(nn.Module):
    """
    Small CNN that predicts left/right boundary positions plus a valid-row score.
    """

    def __init__(self, num_rows: int, pooled_width: int = 8) -> None:
        super().__init__()
        if num_rows <= 0:
            raise ValueError("num_rows must be > 0")
        self.num_rows = int(num_rows)
        self.pooled_width = int(pooled_width)

        self.encoder = nn.Sequential(
            ConvBlock(3, 16, stride=2),
            ConvBlock(16, 24, stride=2),
            ConvBlock(24, 32, stride=2),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, 64, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.num_rows, self.pooled_width))
        self.head = nn.Sequential(
            nn.Linear(64 * self.pooled_width, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(image)
        feat = self.pool(feat)
        feat = feat.permute(0, 2, 1, 3).contiguous()
        feat = feat.view(feat.shape[0], feat.shape[1], -1)
        out = self.head(feat)
        left_x = torch.sigmoid(out[..., 0])
        right_x = torch.sigmoid(out[..., 1])
        valid_logits = out[..., 2]
        center_x = 0.5 * (left_x + right_x)
        return {
            "left_x": left_x,
            "right_x": right_x,
            "center_x": center_x,
            "valid_logits": valid_logits,
            "valid_prob": torch.sigmoid(valid_logits),
        }


@dataclass(frozen=True)
class BoundaryLossWeights:
    valid_bce: float = 1.0
    edge_reg: float = 2.0
    center_reg: float = 1.0
    ordering_penalty: float = 0.5


def boundary_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    weights: BoundaryLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or BoundaryLossWeights()
    valid = batch["valid"]
    valid_mask = valid > 0.5

    valid_loss = F.binary_cross_entropy_with_logits(pred["valid_logits"], valid)

    if valid_mask.any():
        left_loss = F.smooth_l1_loss(pred["left_x"][valid_mask], batch["left_x"][valid_mask])
        right_loss = F.smooth_l1_loss(pred["right_x"][valid_mask], batch["right_x"][valid_mask])
        center_loss = F.smooth_l1_loss(pred["center_x"][valid_mask], batch["center_x"][valid_mask])
    else:
        zero = pred["left_x"].sum() * 0.0
        left_loss = zero
        right_loss = zero
        center_loss = zero

    ordering_loss = torch.relu(pred["left_x"] - pred["right_x"]).mean()

    total = (
        weights.valid_bce * valid_loss
        + weights.edge_reg * (left_loss + right_loss)
        + weights.center_reg * center_loss
        + weights.ordering_penalty * ordering_loss
    )
    parts = {
        "loss_total": float(total.detach().item()),
        "loss_valid_bce": float(valid_loss.detach().item()),
        "loss_left": float(left_loss.detach().item()),
        "loss_right": float(right_loss.detach().item()),
        "loss_center": float(center_loss.detach().item()),
        "loss_ordering": float(ordering_loss.detach().item()),
    }
    return total, parts


def boundary_metrics(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    valid_thresh: float = 0.5,
) -> dict[str, float]:
    valid_true = batch["valid"] > 0.5
    valid_pred = pred["valid_prob"] >= valid_thresh
    valid_acc = float((valid_true == valid_pred).float().mean().item())

    metrics = {
        "valid_acc": valid_acc,
        "valid_true_ratio": float(valid_true.float().mean().item()),
        "valid_pred_ratio": float(valid_pred.float().mean().item()),
    }

    overlap = valid_true & valid_pred
    if overlap.any():
        left_mae = torch.abs(pred["left_x"][overlap] - batch["left_x"][overlap]).mean()
        right_mae = torch.abs(pred["right_x"][overlap] - batch["right_x"][overlap]).mean()
        center_mae = torch.abs(pred["center_x"][overlap] - batch["center_x"][overlap]).mean()
        metrics.update(
            left_mae=float(left_mae.item()),
            right_mae=float(right_mae.item()),
            center_mae=float(center_mae.item()),
        )
    else:
        metrics.update(left_mae=0.0, right_mae=0.0, center_mae=0.0)

    return metrics
