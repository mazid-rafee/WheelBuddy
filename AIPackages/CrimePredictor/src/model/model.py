"""Categorical embedding MLP for per-hour crime-rate prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from time_bins import HOUR_BIN_STARTS, NUM_TIME_BINS, TIME_BIN_HOURS


NUM_RATE_OUTPUTS = 4
RATE_EPS = 1e-8
VALID_HOUR_BINS: tuple[int, ...] = HOUR_BIN_STARTS

DEFAULT_SEVERITY_WEIGHTS: tuple[float, float, float, float] = (
    4.0,
    1.5,
    2.0,
    1.0,
)

RATE_NAMES: tuple[str, ...] = (
    "person_rate",
    "property_rate",
    "society_rate",
    "other_rate",
)


def resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device {resolved}, but torch.cuda.is_available() is False"
        )
    return resolved


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class CrimeRateMLP(nn.Module):
    """Predict nonnegative hourly rates from five categorical inputs."""

    def __init__(
        self,
        *,
        num_h3_embeddings: int,
        num_city_embeddings: int,
        h3_embedding_dim: int = 24,
        city_embedding_dim: int = 8,
        month_embedding_dim: int = 4,
        weekday_embedding_dim: int = 3,
        time_bin_embedding_dim: int = 3,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.15,
        num_outputs: int = NUM_RATE_OUTPUTS,
        severity_weights: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        if num_h3_embeddings < 2:
            raise ValueError("num_h3_embeddings must be at least 2 (UNK + one cell)")
        if num_city_embeddings < 2:
            raise ValueError("num_city_embeddings must be at least 2 (UNK + one city)")
        if num_outputs != NUM_RATE_OUTPUTS:
            raise ValueError(f"num_outputs must be {NUM_RATE_OUTPUTS}")

        self.num_h3_embeddings = int(num_h3_embeddings)
        self.num_city_embeddings = int(num_city_embeddings)
        self.num_outputs = int(num_outputs)
        self.h3_embedding_dim = int(h3_embedding_dim)
        self.city_embedding_dim = int(city_embedding_dim)
        self.month_embedding_dim = int(month_embedding_dim)
        self.weekday_embedding_dim = int(weekday_embedding_dim)
        self.time_bin_embedding_dim = int(time_bin_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)

        self.h3_embedding = nn.Embedding(
            num_h3_embeddings, h3_embedding_dim, padding_idx=0
        )
        self.city_embedding = nn.Embedding(
            num_city_embeddings, city_embedding_dim, padding_idx=0
        )
        self.month_embedding = nn.Embedding(12, month_embedding_dim)
        self.weekday_embedding = nn.Embedding(7, weekday_embedding_dim)
        self.time_bin_embedding = nn.Embedding(NUM_TIME_BINS, time_bin_embedding_dim)

        combined_dim = (
            h3_embedding_dim
            + city_embedding_dim
            + month_embedding_dim
            + weekday_embedding_dim
            + time_bin_embedding_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.rate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_outputs),
        )

        weights = (
            severity_weights
            if severity_weights is not None
            else DEFAULT_SEVERITY_WEIGHTS
        )
        self.register_buffer(
            "severity_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=True,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_batch(
        self,
        h3_cell_index: torch.Tensor,
        city_index: torch.Tensor,
        month: torch.Tensor,
        day_of_week: torch.Tensor,
        hour_bin_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (
            h3_cell_index,
            city_index,
            month,
            day_of_week,
            hour_bin_start,
        )
        names = (
            "h3_cell_index",
            "city_index",
            "month",
            "day_of_week",
            "hour_bin_start",
        )
        for name, tensor in zip(names, tensors):
            if tensor.ndim != 1:
                raise ValueError(
                    f"{name} must be 1-D [batch], got shape {tuple(tensor.shape)}"
                )
        batch = h3_cell_index.shape[0]
        for name, tensor in zip(names, tensors):
            if tensor.shape[0] != batch:
                raise ValueError(f"{name} batch size mismatch")
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, but model is on {self.device}"
                )

        if (h3_cell_index < 0).any() or (h3_cell_index >= self.num_h3_embeddings).any():
            raise ValueError("h3_cell_index out of range")
        if (city_index < 0).any() or (city_index >= self.num_city_embeddings).any():
            raise ValueError("city_index out of range")
        if (month < 1).any() or (month > 12).any():
            raise ValueError("month must be in 1..12")
        if (day_of_week < 0).any() or (day_of_week > 6).any():
            raise ValueError("day_of_week must be in 0..6")
        if not torch.isin(
            hour_bin_start,
            torch.tensor(VALID_HOUR_BINS, device=hour_bin_start.device),
        ).all():
            raise ValueError(f"hour_bin_start must be one of {VALID_HOUR_BINS}")

        month_index = month - 1
        weekday_index = day_of_week
        time_bin_index = hour_bin_start // TIME_BIN_HOURS
        return h3_cell_index, city_index, month_index, weekday_index, time_bin_index

    def forward(
        self,
        h3_cell_index: torch.Tensor,
        city_index: torch.Tensor,
        month: torch.Tensor,
        day_of_week: torch.Tensor,
        hour_bin_start: torch.Tensor,
    ) -> torch.Tensor:
        (
            h3_cell_index,
            city_index,
            month_index,
            weekday_index,
            time_bin_index,
        ) = self._validate_batch(
            h3_cell_index, city_index, month, day_of_week, hour_bin_start
        )

        combined = torch.cat(
            (
                self.h3_embedding(h3_cell_index),
                self.city_embedding(city_index),
                self.month_embedding(month_index),
                self.weekday_embedding(weekday_index),
                self.time_bin_embedding(time_bin_index),
            ),
            dim=1,
        )
        hidden = self.input_projection(combined)
        hidden = self.blocks(hidden)
        raw_outputs = self.rate_head(hidden)
        return F.softplus(raw_outputs) + RATE_EPS

    def predict_total_rate(
        self,
        h3_cell_index: torch.Tensor,
        city_index: torch.Tensor,
        month: torch.Tensor,
        day_of_week: torch.Tensor,
        hour_bin_start: torch.Tensor,
    ) -> torch.Tensor:
        rates = self.forward(
            h3_cell_index, city_index, month, day_of_week, hour_bin_start
        )
        return rates.sum(dim=1)

    def predict_severity_weighted_rate(
        self,
        h3_cell_index: torch.Tensor,
        city_index: torch.Tensor,
        month: torch.Tensor,
        day_of_week: torch.Tensor,
        hour_bin_start: torch.Tensor,
        severity_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rates = self.forward(
            h3_cell_index, city_index, month, day_of_week, hour_bin_start
        )
        weights = (
            self.severity_weights
            if severity_weights is None
            else severity_weights.to(device=rates.device, dtype=rates.dtype)
        )
        return (rates * weights.unsqueeze(0)).sum(dim=1)


CrimeRiskMLP = CrimeRateMLP


def build_model(
    *,
    num_h3_embeddings: int,
    num_city_embeddings: int,
    h3_embedding_dim: int = 24,
    city_embedding_dim: int = 8,
    month_embedding_dim: int = 4,
    weekday_embedding_dim: int = 3,
    time_bin_embedding_dim: int = 3,
    hidden_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.15,
    device: str | torch.device | None = None,
) -> CrimeRateMLP:
    model = CrimeRateMLP(
        num_h3_embeddings=num_h3_embeddings,
        num_city_embeddings=num_city_embeddings,
        h3_embedding_dim=h3_embedding_dim,
        city_embedding_dim=city_embedding_dim,
        month_embedding_dim=month_embedding_dim,
        weekday_embedding_dim=weekday_embedding_dim,
        time_bin_embedding_dim=time_bin_embedding_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    if device is not None:
        model = model.to(resolve_torch_device(device))
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    device = resolve_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_h3_embeddings=50,
        num_city_embeddings=10,
        device=device,
    )
    batch = 16
    h3 = torch.randint(0, 50, (batch,), device=device)  # includes UNK=0
    city = torch.randint(0, 10, (batch,), device=device)
    month = torch.randint(1, 13, (batch,), device=device)
    dow = torch.randint(0, 7, (batch,), device=device)
    hour = torch.tensor(
        VALID_HOUR_BINS, device=device
    )[torch.randint(0, 6, (batch,), device=device)]

    model.eval()
    with torch.no_grad():
        rates = model(h3, city, month, dow, hour)
    assert rates.shape == (batch, 4)
    assert torch.isfinite(rates).all()
    assert (rates > 0).all()

    model.train()
    rates_t = model(h3, city, month, dow, hour)
    loss = rates_t.sum()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("rates shape:", tuple(rates.shape))
    print("trainable params:", n_params)
    print("device:", device)
    print("self-test passed")
