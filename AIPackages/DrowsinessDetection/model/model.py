#!/usr/bin/env python3
"""Causal TCN classifier for DMD short-window drowsiness features.

Pipeline::

    [batch, T, F]
        → Conv1d F→64
        → residual temporal blocks (dilations 1, 2, 4)
        → last-timestep representation
        → Linear 64→num_classes
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_INPUT_DIM = 12
DEFAULT_TCN_CHANNELS = 64
DEFAULT_TCN_KERNEL_SIZE = 3
DEFAULT_TCN_DILATIONS = (1, 2, 4)


class CausalConv1d(nn.Module):
    """1-D convolution with left-only padding so outputs depend on past/present."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be positive, got {dilation}")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, channels, time]
        if self.left_padding:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Residual causal temporal block (two dilated convolutions)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.dropout(F.relu(self.conv1(x)))
        out = self.dropout(F.relu(self.conv2(out)))
        return F.relu(out + residual)


class GazeZoneTCN(nn.Module):
    """Small causal TCN for short drowsiness-feature windows."""

    def __init__(
        self,
        num_classes: int,
        input_dim: int = DEFAULT_INPUT_DIM,
        channels: int = DEFAULT_TCN_CHANNELS,
        kernel_size: int = DEFAULT_TCN_KERNEL_SIZE,
        dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if not dilations or any(int(d) < 1 for d in dilations):
            raise ValueError(
                f"dilations must contain positive integers, got {dilations}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)

        self.input_proj = nn.Conv1d(self.input_dim, self.channels, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalBlock(
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.classifier = nn.Linear(self.channels, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolutions for ReLU and the classifier conservatively."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Return raw class logits for ``[batch, T, F]`` (or ``[batch, F]``)."""
        if not features.is_floating_point():
            raise TypeError(
                f"features must be floating point, got dtype {features.dtype}"
            )
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected features shaped [batch, time, "
                f"{self.input_dim}], got {tuple(features.shape)}"
            )
        if features.shape[1] < 1:
            raise ValueError("time dimension must be at least 1")

        # Conv1d expects [batch, channels, time].
        x = features.transpose(1, 2).contiguous()
        x = F.relu(self.input_proj(x))
        for block in self.temporal_blocks:
            x = block(x)
        last = x[:, :, -1]
        return self.classifier(last)

    @torch.no_grad()
    def predict_proba(self, features: Tensor) -> Tensor:
        """Return class probabilities for inference."""
        return torch.softmax(self(features), dim=-1)

    @torch.no_grad()
    def predict(self, features: Tensor) -> Tensor:
        """Return the most likely class index for each input sample."""
        return self(features).argmax(dim=-1)

    @property
    def num_trainable_parameters(self) -> int:
        """Number of parameters updated by the optimizer."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_model(
    num_classes: int,
    input_dim: int = DEFAULT_INPUT_DIM,
    channels: int = DEFAULT_TCN_CHANNELS,
    kernel_size: int = DEFAULT_TCN_KERNEL_SIZE,
    dilations: Sequence[int] = DEFAULT_TCN_DILATIONS,
    dropout: float = 0.15,
) -> GazeZoneTCN:
    """Construct the small causal drowsiness TCN."""
    return GazeZoneTCN(
        num_classes=num_classes,
        input_dim=input_dim,
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    )


def main() -> None:
    """Run a small forward/backward smoke test on CUDA GPU 1 when available."""
    from device import DEFAULT_GPU_ID, configure_cuda, resolve_device
    from label_contract import NUM_CLASSES

    batch_size = 8
    window_size = 20
    num_classes = NUM_CLASSES
    input_dim = DEFAULT_INPUT_DIM
    device = resolve_device(DEFAULT_GPU_ID, require_cuda=False)
    configure_cuda(device)

    model = build_model(num_classes=num_classes, input_dim=input_dim).to(device)

    features = torch.randn(
        batch_size,
        window_size,
        input_dim,
        dtype=torch.float32,
        device=device,
    )
    labels = torch.randint(
        0, num_classes, (batch_size,), dtype=torch.long, device=device
    )

    logits = model(features)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    print(model)
    print(f"trainable parameters: {model.num_trainable_parameters:,}")
    print(f"input shape:  {tuple(features.shape)} on {features.device}")
    print(f"logits shape: {tuple(logits.shape)} on {logits.device}")
    print(f"loss: {loss.item():.6f}")
    print("forward/backward smoke test: OK")


if __name__ == "__main__":
    # Allow ``python -m model.model`` / ``python model/model.py`` from package root.
    import sys
    from pathlib import Path

    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    main()
