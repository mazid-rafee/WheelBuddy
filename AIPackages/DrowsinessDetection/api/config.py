"""Runtime configuration for the drowsiness inference API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_RELATIVE = "saved_weights/best_loss_v3.pt"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_DEVICE = "auto"

# Sampling rate is stored in v3 checkpoints. When DROWSINESS_SAMPLING_RATE_HZ is
# set, the API enforces an exact match. When unset, the checkpoint value (or 15)
# is used.
ENV_SAMPLING_RATE = "DROWSINESS_SAMPLING_RATE_HZ"


@dataclass(frozen=True)
class Settings:
    package_root: Path
    checkpoint_path: Path
    api_key: str | None
    device_preference: str
    host: str
    port: int
    expected_sampling_rate_hz: float | None


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def resolve_checkpoint_path(raw: str | None = None) -> Path:
    """Resolve checkpoint path relative to the DrowsinessDetection package root."""
    value = (raw if raw is not None else _env("DROWSINESS_CHECKPOINT_PATH")).strip()
    if not value:
        value = DEFAULT_CHECKPOINT_RELATIVE
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PACKAGE_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def resolve_device(preference: str | None = None) -> torch.device:
    pref = (preference if preference is not None else _env("DROWSINESS_DEVICE", DEFAULT_DEVICE))
    pref = pref.strip().lower() or DEFAULT_DEVICE
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def load_settings() -> Settings:
    api_key = _env("DROWSINESS_API_KEY")
    sampling_raw = _env(ENV_SAMPLING_RATE)
    expected_rate: float | None
    if sampling_raw:
        expected_rate = float(sampling_raw)
        if not (expected_rate > 0.0) or expected_rate != expected_rate:
            raise ValueError(
                f"{ENV_SAMPLING_RATE} must be a finite positive float, got {sampling_raw!r}"
            )
    else:
        expected_rate = None

    host = _env("DROWSINESS_HOST", DEFAULT_HOST) or DEFAULT_HOST
    port_raw = _env("DROWSINESS_PORT", str(DEFAULT_PORT)) or str(DEFAULT_PORT)
    port = int(port_raw)

    return Settings(
        package_root=PACKAGE_ROOT,
        checkpoint_path=resolve_checkpoint_path(),
        api_key=api_key or None,
        device_preference=_env("DROWSINESS_DEVICE", DEFAULT_DEVICE) or DEFAULT_DEVICE,
        host=host,
        port=port,
        expected_sampling_rate_hz=expected_rate,
    )
