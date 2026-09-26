"""Unit tests for adjusted route-risk aggregation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from api.preprocess import RouteCellSample  # noqa: E402
from api.service import (  # noqa: E402
    HIGH_HOUR_GAP_WEIGHT,
    _adjusted_severity_weighted_sum,
)
from time_bins import HOUR_BIN_STARTS  # noqa: E402


def _sample(h3: str, *, sequence_index: int = 0) -> RouteCellSample:
    return RouteCellSample(
        sequence_index=sequence_index,
        h3_cell=h3,
        entry_time_utc=datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc),
        local_hour=12,
        day_of_week_name="Friday",
        month_name="September",
        city_name="miami",
        month_index=9,
        day_of_week_index=4,
        hour_bin_start=12,
    )


def test_adjusted_cell_formula_matches_ios_weight() -> None:
    current = 0.01
    high = 0.10
    expected = current + HIGH_HOUR_GAP_WEIGHT * (high - current)
    result = _adjusted_severity_weighted_sum(
        [_sample("a")],
        [current],
        {"a": high},
    )
    assert result == pytest.approx(expected)


def test_unique_cells_counted_once() -> None:
    cells = [_sample("a", sequence_index=0), _sample("a", sequence_index=1), _sample("b", sequence_index=2)]
    severities = [0.01, 0.99, 0.05]
    high = {"a": 0.10, "b": 0.05}
    expected = (0.01 + HIGH_HOUR_GAP_WEIGHT * (0.10 - 0.01)) + 0.05
    assert _adjusted_severity_weighted_sum(cells, severities, high) == pytest.approx(expected)


def test_time_bins_are_three_hour_starts() -> None:
    assert HOUR_BIN_STARTS == (0, 3, 6, 9, 12, 15, 18, 21)
