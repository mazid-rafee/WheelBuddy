"""Streaming regression metrics for exposure-aware crime-rate prediction.

Accumulates MAE/RMSE, Poisson deviance, and calibration statistics from batched
predictions without storing individual examples. Classification metrics are not
computed.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import torch

CATEGORY_NAMES: tuple[str, ...] = (
    "person",
    "property",
    "society",
    "other",
)
NUM_CATEGORIES = len(CATEGORY_NAMES)
MU_EPS = 1e-8


def _as_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_rmse(sum_sq: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float((sum_sq / float(n)) ** 0.5)


def _safe_mae(sum_abs: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float(sum_abs / float(n))


def _calibration_ratio(sum_expected: float, sum_observed: float) -> float | None:
    if sum_observed == 0.0:
        return None
    return float(sum_expected / sum_observed)


def poisson_deviance_elements(
    observed: torch.Tensor,
    expected: torch.Tensor,
    *,
    eps: float = MU_EPS,
) -> torch.Tensor:
    """Elementwise Poisson deviance between counts ``y`` and means ``mu``."""
    y = observed.to(dtype=torch.float64)
    mu = expected.to(dtype=torch.float64).clamp_min(0.0)
    mu_safe = mu.clamp_min(eps)
    # y > 0: 2 * (y * log(y / mu) - (y - mu)), with mu clamped for log stability.
    # y == 0: 2 * mu (unclamped so exact zeros remain exact).
    return torch.where(
        y > 0,
        2.0 * (y * torch.log(y / mu_safe) - (y - mu_safe)),
        2.0 * mu,
    )


class CrimeRateMetrics:
    """Accumulate streaming crime-rate regression metrics on CPU."""

    def __init__(self, category_names: tuple[str, ...] = CATEGORY_NAMES) -> None:
        if len(category_names) != NUM_CATEGORIES:
            raise ValueError(
                f"category_names must contain {NUM_CATEGORIES} names, "
                f"got {len(category_names)}"
            )
        self.category_names = tuple(category_names)
        self.reset()

    def reset(self) -> None:
        """Zero all running totals."""
        self.num_records = 0
        self._sum_abs_rate = 0.0
        self._sum_sq_rate = 0.0
        self._sum_abs_count = 0.0
        self._sum_sq_count = 0.0
        self._sum_poisson_dev = 0.0
        self._n_category_elements = 0

        self._sum_abs_total_rate = 0.0
        self._sum_sq_total_rate = 0.0
        self._sum_abs_total_count = 0.0
        self._sum_sq_total_count = 0.0

        self._sum_expected = 0.0
        self._sum_observed = 0.0

        self._sum_abs_rate_cat = [0.0] * NUM_CATEGORIES
        self._sum_sq_rate_cat = [0.0] * NUM_CATEGORIES
        self._sum_abs_count_cat = [0.0] * NUM_CATEGORIES
        self._sum_sq_count_cat = [0.0] * NUM_CATEGORIES
        self._sum_expected_cat = [0.0] * NUM_CATEGORIES
        self._sum_observed_cat = [0.0] * NUM_CATEGORIES

    @torch.no_grad()
    def update(
        self,
        rates_per_hour: torch.Tensor,
        observed_counts: torch.Tensor,
        exposure_hours: torch.Tensor,
    ) -> None:
        """Update running metrics from one batch.

        Args:
            rates_per_hour: Predicted hourly rates ``[batch, 4]``.
            observed_counts: Observed counts ``[batch, 4]``.
            exposure_hours: Positive exposure hours ``[batch]``.
        """
        if rates_per_hour.ndim != 2 or rates_per_hour.shape[1] != NUM_CATEGORIES:
            raise ValueError(
                f"rates_per_hour must have shape [batch, {NUM_CATEGORIES}], "
                f"got {tuple(rates_per_hour.shape)}"
            )
        if observed_counts.shape != rates_per_hour.shape:
            raise ValueError(
                "observed_counts shape must match rates_per_hour: "
                f"{tuple(observed_counts.shape)} vs {tuple(rates_per_hour.shape)}"
            )
        if exposure_hours.ndim == 2 and exposure_hours.shape[1] == 1:
            exposure_hours = exposure_hours.squeeze(-1)
        if exposure_hours.ndim != 1:
            raise ValueError(
                f"exposure_hours must be 1-D [batch], got {tuple(exposure_hours.shape)}"
            )
        if exposure_hours.shape[0] != rates_per_hour.shape[0]:
            raise ValueError(
                "batch size mismatch between rates and exposure_hours: "
                f"{rates_per_hour.shape[0]} vs {exposure_hours.shape[0]}"
            )
        if rates_per_hour.shape[0] == 0:
            return

        rates = rates_per_hour.detach().to(device="cpu", dtype=torch.float64)
        counts = observed_counts.detach().to(device="cpu", dtype=torch.float64)
        exposure = exposure_hours.detach().to(device="cpu", dtype=torch.float64)

        if not torch.isfinite(rates).all():
            raise ValueError("rates_per_hour contain NaN or Inf")
        if (rates < 0).any():
            raise ValueError("rates_per_hour must be nonnegative")
        if not torch.isfinite(counts).all():
            raise ValueError("observed_counts contain NaN or Inf")
        if (counts < 0).any():
            raise ValueError("observed_counts must be nonnegative")
        if not torch.isfinite(exposure).all():
            raise ValueError("exposure_hours contain NaN or Inf")
        if (exposure <= 0).any():
            raise ValueError("exposure_hours must be strictly positive")

        expected_counts = rates * exposure.unsqueeze(1)
        observed_rates = counts / exposure.unsqueeze(1)

        rate_err = rates - observed_rates
        count_err = expected_counts - counts

        batch_size = int(rates.shape[0])
        self.num_records += batch_size
        self._n_category_elements += batch_size * NUM_CATEGORIES

        self._sum_abs_rate += float(rate_err.abs().sum().item())
        self._sum_sq_rate += float((rate_err * rate_err).sum().item())
        self._sum_abs_count += float(count_err.abs().sum().item())
        self._sum_sq_count += float((count_err * count_err).sum().item())

        deviance = poisson_deviance_elements(counts, expected_counts)
        self._sum_poisson_dev += float(deviance.sum().item())

        total_rate_pred = rates.sum(dim=1)
        total_rate_obs = observed_rates.sum(dim=1)
        total_count_pred = expected_counts.sum(dim=1)
        total_count_obs = counts.sum(dim=1)
        total_rate_err = total_rate_pred - total_rate_obs
        total_count_err = total_count_pred - total_count_obs

        self._sum_abs_total_rate += float(total_rate_err.abs().sum().item())
        self._sum_sq_total_rate += float((total_rate_err * total_rate_err).sum().item())
        self._sum_abs_total_count += float(total_count_err.abs().sum().item())
        self._sum_sq_total_count += float(
            (total_count_err * total_count_err).sum().item()
        )

        self._sum_expected += float(expected_counts.sum().item())
        self._sum_observed += float(counts.sum().item())

        for category_index in range(NUM_CATEGORIES):
            r_err = rate_err[:, category_index]
            c_err = count_err[:, category_index]
            self._sum_abs_rate_cat[category_index] += float(r_err.abs().sum().item())
            self._sum_sq_rate_cat[category_index] += float((r_err * r_err).sum().item())
            self._sum_abs_count_cat[category_index] += float(c_err.abs().sum().item())
            self._sum_sq_count_cat[category_index] += float(
                (c_err * c_err).sum().item()
            )
            self._sum_expected_cat[category_index] += float(
                expected_counts[:, category_index].sum().item()
            )
            self._sum_observed_cat[category_index] += float(
                counts[:, category_index].sum().item()
            )

    def compute(self) -> dict[str, Any]:
        """Return JSON-serializable overall and per-category metrics."""
        n_rec = self.num_records
        n_elem = self._n_category_elements

        per_category: dict[str, dict[str, float | None]] = {}
        for index, name in enumerate(self.category_names):
            per_category[name] = {
                "rate_mae": _safe_mae(self._sum_abs_rate_cat[index], n_rec),
                "rate_rmse": _safe_rmse(self._sum_sq_rate_cat[index], n_rec),
                "count_mae": _safe_mae(self._sum_abs_count_cat[index], n_rec),
                "count_rmse": _safe_rmse(self._sum_sq_count_cat[index], n_rec),
                "calibration_ratio": _calibration_ratio(
                    self._sum_expected_cat[index],
                    self._sum_observed_cat[index],
                ),
            }

        return {
            "num_records": int(n_rec),
            "rate_mae": _safe_mae(self._sum_abs_rate, n_elem),
            "rate_rmse": _safe_rmse(self._sum_sq_rate, n_elem),
            "count_mae": _safe_mae(self._sum_abs_count, n_elem),
            "count_rmse": _safe_rmse(self._sum_sq_count, n_elem),
            "poisson_deviance": _safe_mae(self._sum_poisson_dev, n_elem),
            "total_rate_mae": _safe_mae(self._sum_abs_total_rate, n_rec),
            "total_rate_rmse": _safe_rmse(self._sum_sq_total_rate, n_rec),
            "total_count_mae": _safe_mae(self._sum_abs_total_count, n_rec),
            "total_count_rmse": _safe_rmse(self._sum_sq_total_count, n_rec),
            "calibration_ratio": _calibration_ratio(
                self._sum_expected, self._sum_observed
            ),
            "per_category": per_category,
        }


def format_metrics(metrics: Mapping[str, Any]) -> str:
    """Return a concise printable summary of overall rate metrics."""
    calibration = metrics["calibration_ratio"]
    calibration_text = (
        "None" if calibration is None else f"{float(calibration):.4f}"
    )
    return (
        f"records={metrics['num_records']} | "
        f"rate_mae={float(metrics['rate_mae']):.6f} | "
        f"rate_rmse={float(metrics['rate_rmse']):.6f} | "
        f"count_mae={float(metrics['count_mae']):.4f} | "
        f"count_rmse={float(metrics['count_rmse']):.4f} | "
        f"pois_dev={float(metrics['poisson_deviance']):.4f} | "
        f"total_rate_mae={float(metrics['total_rate_mae']):.6f} | "
        f"calib={calibration_text}"
    )


def format_per_category_metrics(metrics: Mapping[str, Any]) -> str:
    """Return a readable table of per-category rate/count metrics."""
    per_category = metrics["per_category"]
    if not isinstance(per_category, MutableMapping) and not isinstance(
        per_category, dict
    ):
        raise TypeError("metrics['per_category'] must be a mapping")

    header = (
        f"{'category':<12} {'rate_mae':>12} {'rate_rmse':>12} "
        f"{'count_mae':>12} {'count_rmse':>12} {'calib':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, stats in per_category.items():
        calib = stats["calibration_ratio"]
        calib_text = "None" if calib is None else f"{float(calib):.4f}"
        lines.append(
            f"{name:<12} "
            f"{float(stats['rate_mae']):>12.6f} "
            f"{float(stats['rate_rmse']):>12.6f} "
            f"{float(stats['count_mae']):>12.4f} "
            f"{float(stats['count_rmse']):>12.4f} "
            f"{calib_text:>10}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    torch.manual_seed(0)

    # Batch 1: perfect predictions with mixed zero/positive counts and exposures.
    exposure_1 = torch.tensor([4.0, 8.0, 12.0, 24.0], dtype=torch.float32)
    observed_1 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 2.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    rates_1 = observed_1 / exposure_1.unsqueeze(1)

    metrics = CrimeRateMetrics()
    metrics.update(rates_1, observed_1, exposure_1)
    perfect = metrics.compute()
    assert perfect["num_records"] == 4
    assert perfect["rate_mae"] < 1e-7
    assert perfect["rate_rmse"] < 1e-7
    assert perfect["count_mae"] < 1e-6
    assert perfect["count_rmse"] < 1e-6
    assert perfect["total_rate_mae"] < 1e-7
    assert perfect["total_rate_rmse"] < 1e-7
    assert perfect["total_count_mae"] < 1e-6
    assert perfect["total_count_rmse"] < 1e-6
    assert perfect["poisson_deviance"] < 1e-6
    assert perfect["calibration_ratio"] is not None
    assert abs(float(perfect["calibration_ratio"]) - 1.0) < 1e-6
    for name, stats in perfect["per_category"].items():
        assert float(stats["rate_mae"]) < 1e-7
        assert float(stats["count_mae"]) < 1e-6
        if stats["calibration_ratio"] is not None:
            assert abs(float(stats["calibration_ratio"]) - 1.0) < 1e-6

    # Batch 2: deliberately inaccurate predictions.
    metrics_bad = CrimeRateMetrics()
    bad_rates = rates_1 + 0.25
    metrics_bad.update(bad_rates, observed_1, exposure_1)
    bad = metrics_bad.compute()
    assert bad["rate_mae"] > perfect["rate_mae"]
    assert bad["rate_rmse"] > perfect["rate_rmse"]
    assert bad["count_mae"] > perfect["count_mae"]
    assert bad["count_rmse"] > perfect["count_rmse"]
    assert bad["poisson_deviance"] > perfect["poisson_deviance"]
    assert bad["total_rate_mae"] > perfect["total_rate_mae"]
    assert bad["calibration_ratio"] is not None
    assert bad["calibration_ratio"] > 1.0

    # Streaming accumulation equals a single combined update.
    metrics_stream = CrimeRateMetrics()
    metrics_stream.update(rates_1[:2], observed_1[:2], exposure_1[:2])
    metrics_stream.update(rates_1[2:], observed_1[2:], exposure_1[2:])
    streamed = metrics_stream.compute()
    assert streamed["num_records"] == perfect["num_records"]
    assert abs(streamed["rate_mae"] - perfect["rate_mae"]) < 1e-12
    assert abs(streamed["count_mae"] - perfect["count_mae"]) < 1e-12

    # All-zero observed counts -> calibration_ratio is None.
    metrics_zero = CrimeRateMetrics()
    zero_counts = torch.zeros(3, 4)
    zero_rates = torch.full((3, 4), 0.01)
    zero_exposure = torch.tensor([4.0, 8.0, 16.0])
    metrics_zero.update(zero_rates, zero_counts, zero_exposure)
    zero_result = metrics_zero.compute()
    assert zero_result["calibration_ratio"] is None
    assert zero_result["count_mae"] > 0.0

    # JSON serializable.
    encoded = json.dumps(perfect)
    assert isinstance(encoded, str)
    encoded_bad = json.dumps(bad)
    assert "calibration_ratio" in encoded_bad

    # Reset clears state.
    metrics_bad.reset()
    cleared = metrics_bad.compute()
    assert cleared["num_records"] == 0
    assert cleared["rate_mae"] == 0.0
    assert cleared["calibration_ratio"] is None

    print(format_metrics(perfect))
    print(format_metrics(bad))
    print(format_per_category_metrics(bad))
    print("self-test passed")
