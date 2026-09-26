"""CrimePredictor neural models."""

from time_bins import NUM_TIME_BINS, TIME_BIN_HOURS

from .model import (
    DEFAULT_SEVERITY_WEIGHTS,
    NUM_RATE_OUTPUTS,
    RATE_NAMES,
    VALID_HOUR_BINS,
    CrimeRateMLP,
    CrimeRiskMLP,
    ResidualMLPBlock,
    build_model,
    resolve_torch_device,
)

__all__ = [
    "DEFAULT_SEVERITY_WEIGHTS",
    "NUM_RATE_OUTPUTS",
    "NUM_TIME_BINS",
    "RATE_NAMES",
    "TIME_BIN_HOURS",
    "VALID_HOUR_BINS",
    "CrimeRateMLP",
    "CrimeRiskMLP",
    "ResidualMLPBlock",
    "build_model",
    "resolve_torch_device",
]
