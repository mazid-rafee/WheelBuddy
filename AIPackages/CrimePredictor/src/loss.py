"""Exposure-aware Poisson loss for spatiotemporal crime-rate prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_RATE_CATEGORIES = 4
EXPECTED_COUNT_EPS = 1e-8


class ExposurePoissonLoss(nn.Module):
    """Poisson NLL between observed counts and rate * exposure expected counts.

    The four crime-category rates are treated as independent Poisson processes.
    Category losses are averaged equally (no class weights / softmax).
    """

    def __init__(self, eps: float = EXPECTED_COUNT_EPS) -> None:
        super().__init__()
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = float(eps)

    def forward(
        self,
        rates_per_hour: torch.Tensor,
        observed_counts: torch.Tensor,
        exposure_hours: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the mean Poisson NLL over batch and four categories.

        Args:
            rates_per_hour: Nonnegative predicted rates ``[batch_size, 4]``.
            observed_counts: Nonnegative observed counts ``[batch_size, 4]``.
            exposure_hours: Positive exposure ``[batch_size]`` or ``[batch_size, 1]``.
            return_details: When True, return a dict with loss, expected counts,
                and per-category mean losses.

        Returns:
            Scalar loss tensor, or a details dictionary when requested.
        """
        if rates_per_hour.ndim != 2 or rates_per_hour.shape[1] != NUM_RATE_CATEGORIES:
            raise ValueError(
                f"rates_per_hour must have shape [batch, {NUM_RATE_CATEGORIES}], "
                f"got {tuple(rates_per_hour.shape)}"
            )
        if observed_counts.shape != rates_per_hour.shape:
            raise ValueError(
                "observed_counts shape must match rates_per_hour: "
                f"{tuple(observed_counts.shape)} vs {tuple(rates_per_hour.shape)}"
            )

        exposure = exposure_hours
        if exposure.ndim == 2:
            if exposure.shape[1] != 1:
                raise ValueError(
                    "exposure_hours must have shape [batch] or [batch, 1], "
                    f"got {tuple(exposure.shape)}"
                )
            exposure = exposure.squeeze(-1)
        if exposure.ndim != 1:
            raise ValueError(
                f"exposure_hours must be 1-D after squeeze, got {tuple(exposure.shape)}"
            )
        if exposure.shape[0] != rates_per_hour.shape[0]:
            raise ValueError(
                "exposure_hours batch size mismatch: "
                f"{exposure.shape[0]} vs {rates_per_hour.shape[0]}"
            )

        if not torch.isfinite(rates_per_hour).all():
            raise ValueError("rates_per_hour contain NaN or Inf")
        if (rates_per_hour < 0).any():
            raise ValueError("rates_per_hour must be nonnegative")
        if not torch.isfinite(observed_counts).all():
            raise ValueError("observed_counts contain NaN or Inf")
        if (observed_counts < 0).any():
            raise ValueError("observed_counts must be nonnegative")
        if not torch.isfinite(exposure).all():
            raise ValueError("exposure_hours contain NaN or Inf")
        if (exposure <= 0).any():
            raise ValueError("exposure_hours must be strictly positive")

        expected_counts = rates_per_hour * exposure.unsqueeze(-1)
        expected_counts = expected_counts.clamp_min(self.eps)

        # Mean over all batch x category elements (equivalent to per-element mean).
        loss = F.poisson_nll_loss(
            expected_counts,
            observed_counts,
            log_input=False,
            full=False,
            reduction="mean",
        )

        if not torch.isfinite(loss):
            raise ValueError(f"computed Poisson loss is not finite: {float(loss)}")

        if not return_details:
            return loss

        per_element = F.poisson_nll_loss(
            expected_counts,
            observed_counts,
            log_input=False,
            full=False,
            reduction="none",
        )
        return {
            "loss": loss,
            "expected_counts": expected_counts,
            "per_category_loss": per_element.mean(dim=0),
        }


def build_loss() -> ExposurePoissonLoss:
    """Construct the default exposure-aware Poisson loss module."""
    return ExposurePoissonLoss()


if __name__ == "__main__":
    torch.manual_seed(0)
    loss_fn = build_loss()
    assert isinstance(loss_fn, ExposurePoissonLoss)

    batch_size = 8
    # Mix of positive and zero observed counts.
    observed = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 3.0],
            [4.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    exposure = torch.tensor(
        [4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
        dtype=torch.float32,
    )

    # Well-matched rates: expected ≈ observed.
    good_rates = (observed / exposure.unsqueeze(-1)).clamp_min(1e-6)
    good_rates = good_rates.detach().requires_grad_(True)
    good_loss = loss_fn(good_rates, observed, exposure)
    assert good_loss.ndim == 0
    assert torch.isfinite(good_loss)
    good_loss.backward()
    assert good_rates.grad is not None
    assert torch.isfinite(good_rates.grad).all()

    # Poorly matched rates: constant over-prediction.
    bad_rates = torch.full(
        (batch_size, 4), 2.0, dtype=torch.float32, requires_grad=True
    )
    bad_loss = loss_fn(bad_rates, observed, exposure)
    assert torch.isfinite(bad_loss)
    assert float(good_loss.detach()) < float(bad_loss.detach()), (
        f"expected closer predictions to have lower loss: "
        f"good={float(good_loss.detach())} bad={float(bad_loss.detach())}"
    )
    bad_loss.backward()
    assert bad_rates.grad is not None
    assert torch.isfinite(bad_rates.grad).all()

    # Exposure multiplication: rate * exposure == expected_counts.
    rates = torch.ones(batch_size, 4, dtype=torch.float32)
    details = loss_fn(
        rates,
        observed,
        exposure,
        return_details=True,
    )
    assert isinstance(details, dict)
    expected = details["expected_counts"]
    manual = rates * exposure.unsqueeze(-1)
    assert torch.allclose(expected, manual.clamp_min(EXPECTED_COUNT_EPS))
    assert details["per_category_loss"].shape == (4,)
    assert torch.isfinite(details["loss"])

    # Also accept exposure shaped [batch, 1].
    loss_2d = loss_fn(rates, observed, exposure.unsqueeze(-1))
    assert torch.allclose(loss_2d, details["loss"])

    print("good_loss:", float(good_loss.detach()))
    print("bad_loss:", float(bad_loss.detach()))
    print("per_category_loss:", details["per_category_loss"].detach().tolist())
    print("self-test passed")
