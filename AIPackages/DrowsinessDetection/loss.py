#!/usr/bin/env python3
"""Multiclass classification loss for DMD drowsiness training."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class GazeZoneClassificationLoss(nn.Module):
    """Cross-entropy over raw class logits (no softmax in the model).

    Parameters
    ----------
    class_weights:
        Optional per-class weights for imbalanced drowsiness classes
        (e.g. dominant ``open`` eyes-state). Shape ``[num_classes]``.
    label_smoothing:
        Optional label smoothing in ``[0, 1)``.
    """

    def __init__(
        self,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        self.label_smoothing = float(label_smoothing)
        if class_weights is not None:
            if class_weights.ndim != 1:
                raise ValueError(
                    "class_weights must be 1-D, "
                    f"got shape {tuple(class_weights.shape)}"
                )
            self.register_buffer(
                "class_weights", class_weights.detach().float().clone()
            )
        else:
            self.class_weights = None  # type: ignore[assignment]

        self._criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute mean cross-entropy.

        Parameters
        ----------
        logits:
            ``[batch, num_classes]`` unnormalized scores.
        targets:
            ``[batch]`` integer class indices (``torch.long``).
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be [batch, num_classes], got {tuple(logits.shape)}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"targets must be [batch], got {tuple(targets.shape)}"
            )
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"batch mismatch: logits={logits.shape[0]} "
                f"targets={targets.shape[0]}"
            )
        if targets.dtype != torch.long:
            targets = targets.long()
        return self._criterion(logits, targets)


def build_loss(
    class_weights: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
) -> GazeZoneClassificationLoss:
    """Factory matching the multiclass drowsiness training objective."""
    return GazeZoneClassificationLoss(
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
