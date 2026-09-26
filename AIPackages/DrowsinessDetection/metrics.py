#!/usr/bin/env python3
"""Multiclass classification metrics for DMD drowsiness evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


def _to_1d_long(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(tensor.shape)}")
    return tensor.detach().long().cpu()


def multiclass_accuracy(logits: Tensor, targets: Tensor) -> float:
    """Top-1 accuracy over a batch of logits ``[N, C]`` and labels ``[N]``."""
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be [N, C], got shape {tuple(logits.shape)}"
        )
    targets = _to_1d_long(targets, "targets")
    preds = logits.detach().argmax(dim=-1).cpu()
    if preds.shape[0] != targets.shape[0]:
        raise ValueError("logits/targets batch size mismatch")
    if preds.numel() == 0:
        return float("nan")
    return float((preds == targets).float().mean().item())


def confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
) -> Tensor:
    """Return an integer confusion matrix ``[C, C]`` (rows=true, cols=pred)."""
    preds = _to_1d_long(preds, "preds")
    targets = _to_1d_long(targets, "targets")
    if preds.shape != targets.shape:
        raise ValueError("preds/targets shape mismatch")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    valid = (
        (targets >= 0)
        & (targets < num_classes)
        & (preds >= 0)
        & (preds < num_classes)
    )
    targets = targets[valid]
    preds = preds[valid]
    if targets.numel() == 0:
        return matrix
    indices = targets * num_classes + preds
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)


def per_class_precision_recall_f1(
    confmat: Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, Tensor]:
    """Compute per-class precision / recall / F1 from a confusion matrix."""
    if confmat.ndim != 2 or confmat.shape[0] != confmat.shape[1]:
        raise ValueError(
            f"confmat must be square [C, C], got {tuple(confmat.shape)}"
        )
    confmat = confmat.float()
    tp = torch.diag(confmat)
    fp = confmat.sum(dim=0) - tp
    fn = confmat.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": confmat.sum(dim=1).long(),
    }


def macro_f1_from_confmat(confmat: Tensor) -> float:
    """Unweighted mean of per-class F1 scores."""
    stats = per_class_precision_recall_f1(confmat)
    return float(stats["f1"].mean().item())


class MulticlassMetricMeter:
    """Accumulate predictions across batches for epoch-level metrics."""

    def __init__(self, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self._preds: List[Tensor] = []
        self._targets: List[Tensor] = []
        self.total_loss = 0.0
        self.total_examples = 0

    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        *,
        loss: Optional[Tensor] = None,
    ) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must be [N, {self.num_classes}], "
                f"got {tuple(logits.shape)}"
            )
        targets = _to_1d_long(targets, "targets")
        preds = logits.detach().argmax(dim=-1).cpu()
        self._preds.append(preds)
        self._targets.append(targets)
        batch_size = int(targets.numel())
        self.total_examples += batch_size
        if loss is not None:
            self.total_loss += float(loss.detach().item()) * batch_size

    def compute(self) -> Dict[str, float]:
        if self.total_examples == 0:
            return {
                "loss": float("nan"),
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
            }
        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)
        confmat = confusion_matrix(preds, targets, self.num_classes)
        accuracy = float((preds == targets).float().mean().item())
        return {
            "loss": self.total_loss / self.total_examples,
            "accuracy": accuracy,
            "macro_f1": macro_f1_from_confmat(confmat),
            "num_examples": float(self.total_examples),
        }

    def compute_detailed(
        self,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        summary = self.compute()
        if self.total_examples == 0:
            summary["confusion_matrix"] = torch.zeros(
                self.num_classes, self.num_classes, dtype=torch.int64
            )
            return summary
        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)
        confmat = confusion_matrix(preds, targets, self.num_classes)
        per_class = per_class_precision_recall_f1(confmat)
        summary["confusion_matrix"] = confmat
        summary["per_class"] = per_class
        if class_names is not None:
            summary["class_names"] = list(class_names)
        return summary
