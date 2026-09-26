"""Shared spatiotemporal time-bin configuration for the crime-rate pipeline."""

from __future__ import annotations

TIME_BIN_HOURS = 3
NUM_TIME_BINS = 24 // TIME_BIN_HOURS
HOUR_BIN_STARTS: tuple[int, ...] = tuple(
    bin_index * TIME_BIN_HOURS for bin_index in range(NUM_TIME_BINS)
)
TEMPORAL_COMBINATIONS_PER_CELL = 12 * 7 * NUM_TIME_BINS

if 24 % TIME_BIN_HOURS != 0:
    raise ValueError(f"TIME_BIN_HOURS={TIME_BIN_HOURS} must divide 24")
