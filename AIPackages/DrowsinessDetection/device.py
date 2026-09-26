#!/usr/bin/env python3
"""Shared CUDA / device helpers for the DMD drowsiness pipeline.

Default physical GPU index is ``1`` (``cuda:1``).
"""

from __future__ import annotations

from typing import Optional, Union

import torch

DEFAULT_GPU_ID = 1
DeviceLike = Union[torch.device, str, int]


def resolve_device(
    gpu_id: int = DEFAULT_GPU_ID,
    *,
    require_cuda: bool = True,
) -> torch.device:
    """Return ``cuda:{gpu_id}`` or CPU.

    Parameters
    ----------
    gpu_id:
        Physical CUDA device index. Default is ``1``.
    require_cuda:
        If True and CUDA is unavailable (or ``gpu_id`` is out of range),
        raise ``RuntimeError`` instead of silently falling back to CPU.
    """
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "CUDA is required but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch or pass require_cuda=False."
            )
        return torch.device("cpu")

    if gpu_id < 0:
        if require_cuda:
            raise ValueError(f"gpu_id must be >= 0 when using CUDA, got {gpu_id}")
        return torch.device("cpu")

    if gpu_id >= torch.cuda.device_count():
        message = (
            f"gpu_id={gpu_id} is unavailable; "
            f"found {torch.cuda.device_count()} CUDA device(s)."
        )
        if require_cuda:
            raise RuntimeError(message)
        raise ValueError(message)

    return torch.device(f"cuda:{gpu_id}")


def configure_cuda(device: torch.device) -> None:
    """Enable cudnn benchmark and print the active CUDA device."""
    if device.type != "cuda":
        print(f"Using device: {device}")
        return
    torch.backends.cudnn.benchmark = True
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    print(
        f"Using device: {device} | {name} | "
        f"capability={capability[0]}.{capability[1]} | "
        f"memory={torch.cuda.get_device_properties(index).total_memory / 1e9:.1f} GB"
    )


def to_device(
    tensor: torch.Tensor,
    device: DeviceLike,
    *,
    dtype: Optional[torch.dtype] = None,
    non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    """Move a tensor to ``device`` with CUDA-friendly ``non_blocking`` defaults."""
    if isinstance(device, int):
        device = resolve_device(device)
    elif isinstance(device, str):
        device = torch.device(device)
    if non_blocking is None:
        non_blocking = device.type == "cuda"
    return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)


def dataloader_kwargs_for_device(
    device: torch.device,
    *,
    num_workers: int = 0,
) -> dict:
    """DataLoader kwargs tuned for the active device."""
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
