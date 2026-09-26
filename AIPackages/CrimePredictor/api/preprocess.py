"""Route geometry → ordered H3 cells with estimated local entry times."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from api.schemas import RouteCandidate

# Training used H3 resolution 9 (pre_process.H3_RESOLUTION).
H3_RESOLUTION = 9

# Approximate H3 res-9 edge length (~174 m). Densify below that so crossed
# cells between Google polyline vertices are not skipped.
DEFAULT_MAX_SEGMENT_METERS = 80.0

EARTH_RADIUS_M = 6_371_000.0

WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

# Deterministic IANA zones for training cities (no reverse-geocoding API).
CITY_TIMEZONES: Mapping[str, str] = {
    "austin": "America/Chicago",
    "boston": "America/New_York",
    "charlotte": "America/New_York",
    "detroit": "America/Detroit",
    "fort worth": "America/Chicago",
    "houston": "America/Chicago",
    "louisville": "America/Kentucky/Louisville",
    "memphis": "America/Chicago",
    "nashville": "America/Chicago",
    "new york": "America/New_York",
    "virginia beach": "America/New_York",
}


class PreprocessError(Exception):
    """Structured route preprocessing failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        route_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.route_id = route_id
        self.details = details or {}


@dataclass(frozen=True)
class RouteCellSample:
    sequence_index: int
    h3_cell: str
    entry_time_utc: datetime
    local_hour: int
    day_of_week_index: int  # Mon=0 .. Sun=6 (pandas/training convention)
    day_of_week_name: str
    month_index: int  # 1..12
    month_name: str
    city_name: str
    hour_bin_start: int


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into (lat, lng) points."""
    coordinates: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    def next_value() -> int:
        nonlocal index
        result = 0
        shift = 0
        while index < length:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        else:
            raise PreprocessError(
                "malformed_polyline",
                "encoded polyline ended before a coordinate was complete",
            )
        return ~(result >> 1) if result & 1 else (result >> 1)

    try:
        while index < length:
            lat += next_value()
            lng += next_value()
            coordinates.append((lat / 1e5, lng / 1e5))
    except PreprocessError:
        raise
    except Exception as exc:  # pragma: no cover
        raise PreprocessError(
            "malformed_polyline",
            f"failed to decode polyline: {exc}",
        ) from exc

    if len(coordinates) < 2:
        raise PreprocessError(
            "malformed_polyline",
            "decoded polyline must contain at least two points",
        )
    for lat_v, lng_v in coordinates:
        if not (-90.0 <= lat_v <= 90.0 and -180.0 <= lng_v <= 180.0):
            raise PreprocessError(
                "malformed_polyline",
                f"decoded coordinate out of range: ({lat_v}, {lng_v})",
            )
        if not math.isfinite(lat_v) or not math.isfinite(lng_v):
            raise PreprocessError(
                "malformed_polyline",
                "decoded coordinate is non-finite",
            )
    return coordinates


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def densify_coordinates(
    points: Sequence[tuple[float, float]],
    *,
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> list[tuple[float, float]]:
    """Insert interpolated points so consecutive vertices are <= max spacing."""
    if max_segment_meters <= 0:
        raise ValueError("max_segment_meters must be positive")
    if len(points) < 2:
        return list(points)

    densified: list[tuple[float, float]] = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        distance = haversine_meters(start[0], start[1], end[0], end[1])
        if distance <= max_segment_meters:
            densified.append(end)
            continue
        steps = int(math.ceil(distance / max_segment_meters))
        for step_index in range(1, steps):
            t = step_index / steps
            densified.append(
                (
                    start[0] + (end[0] - start[0]) * t,
                    start[1] + (end[1] - start[1]) * t,
                )
            )
        densified.append(end)
    return densified


def latlng_to_h3_cell(latitude: float, longitude: float, resolution: int = H3_RESOLUTION) -> str:
    try:
        import h3
    except ImportError as exc:  # pragma: no cover
        raise PreprocessError(
            "missing_preprocessing_artifact",
            "h3 package is required for route preprocessing",
        ) from exc

    if hasattr(h3, "latlng_to_cell"):
        return str(h3.latlng_to_cell(float(latitude), float(longitude), int(resolution)))
    return str(h3.geo_to_h3(float(latitude), float(longitude), int(resolution)))


def collapse_consecutive_duplicates(cells: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    for cell in cells:
        if not collapsed or collapsed[-1] != cell:
            collapsed.append(cell)
    return collapsed


def path_to_ordered_h3_cells(
    points: Sequence[tuple[float, float]],
    *,
    resolution: int = H3_RESOLUTION,
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> list[str]:
    densified = densify_coordinates(points, max_segment_meters=max_segment_meters)
    cells = [latlng_to_h3_cell(lat, lng, resolution) for lat, lng in densified]
    return collapse_consecutive_duplicates(cells)


def cumulative_distances(points: Sequence[tuple[float, float]]) -> list[float]:
    cum = [0.0]
    for start, end in zip(points[:-1], points[1:]):
        cum.append(cum[-1] + haversine_meters(start[0], start[1], end[0], end[1]))
    return cum


def map_cells_to_entry_times(
    cells: Sequence[str],
    points: Sequence[tuple[float, float]],
    *,
    departure_time_utc: datetime,
    duration_seconds: float,
) -> list[tuple[str, datetime]]:
    """Assign each ordered unique cell its first-entry UTC timestamp along the path."""
    if not cells:
        raise PreprocessError("malformed_polyline", "route produced no H3 cells")
    densified = densify_coordinates(points)
    densified_cells = [
        latlng_to_h3_cell(lat, lng, H3_RESOLUTION) for lat, lng in densified
    ]
    distances = cumulative_distances(densified)
    total_distance = distances[-1]
    if total_distance <= 0.0:
        # Degenerate geometry: every cell enters at departure.
        return [(cell, departure_time_utc) for cell in cells]

    first_index: dict[str, int] = {}
    for index, cell in enumerate(densified_cells):
        if cell not in first_index:
            first_index[cell] = index

    timed: list[tuple[str, datetime]] = []
    for cell in cells:
        point_index = first_index.get(cell, 0)
        fraction = distances[point_index] / total_distance
        offset_seconds = duration_seconds * fraction
        entry = departure_time_utc + timedelta(seconds=offset_seconds)
        timed.append((cell, entry))
    return timed


def assign_cell_times_from_scaled_steps(
    route: RouteCandidate,
    *,
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> tuple[list[tuple[float, float]], list[tuple[str, datetime]]]:
    """Decode/densify a route and estimate per-cell entry times with traffic scaling."""
    static_sum = sum(step.static_duration_seconds for step in route.steps)
    if static_sum <= 0.0:
        raise PreprocessError(
            "malformed_polyline",
            "sum of step.static_duration_seconds must be positive",
            route_id=route.route_id,
        )
    scale = float(route.duration_seconds) / static_sum

    all_points: list[tuple[float, float]] = []
    cell_entries: list[tuple[str, datetime]] = []
    elapsed = 0.0
    last_cell: str | None = None

    for step in route.steps:
        try:
            raw_points = decode_polyline(step.encoded_polyline)
        except PreprocessError as exc:
            raise PreprocessError(exc.code, exc.message, route_id=route.route_id) from exc

        densified = densify_coordinates(
            raw_points, max_segment_meters=max_segment_meters
        )
        if all_points and densified and all_points[-1] == densified[0]:
            densified = densified[1:]
        if not densified:
            continue

        distances = cumulative_distances(densified)
        step_distance = distances[-1]
        adjusted_duration = float(step.static_duration_seconds) * scale

        for index, (lat, lng) in enumerate(densified):
            if step_distance > 0.0:
                local_offset = adjusted_duration * (distances[index] / step_distance)
            else:
                local_offset = 0.0
            entry_time = route.departure_time_utc + timedelta(
                seconds=elapsed + local_offset
            )
            cell = latlng_to_h3_cell(lat, lng, H3_RESOLUTION)
            if cell != last_cell:
                cell_entries.append((cell, entry_time))
                last_cell = cell
            all_points.append((lat, lng))

        elapsed += adjusted_duration

    if not cell_entries:
        raise PreprocessError(
            "malformed_polyline",
            "route produced no H3 cells after decoding",
            route_id=route.route_id,
        )
    return all_points, cell_entries


def local_calendar_features(
    entry_time_utc: datetime,
    city_name: str,
) -> tuple[int, int, int, str, str]:
    """Return (hour, day_of_week Mon=0, month 1..12, weekday name, month name)."""
    zone_name = CITY_TIMEZONES.get(city_name)
    if zone_name is None:
        raise PreprocessError(
            "unknown_city",
            f"no geographic timezone mapping for city {city_name!r}",
        )
    local = entry_time_utc.astimezone(ZoneInfo(zone_name))
    hour = int(local.hour)
    # Matching pre_process.py: pandas dayofweek Monday=0 .. Sunday=6.
    day_of_week = int(local.weekday())
    month = int(local.month)
    return (
        hour,
        day_of_week,
        month,
        WEEKDAY_NAMES[day_of_week],
        MONTH_NAMES[month],
    )


def hour_to_bin_start(hour: int) -> int:
    # Matches time_bins.TIME_BIN_HOURS used during training.
    time_bin_hours = 3
    if hour < 0 or hour > 23:
        raise PreprocessError(
            "internal_inference_failure",
            f"local hour out of range: {hour}",
        )
    return (hour // time_bin_hours) * time_bin_hours


def build_route_cell_samples(
    route: RouteCandidate,
    *,
    h3_to_city: Mapping[str, str],
    known_h3_cells: set[str],
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> list[RouteCellSample]:
    """Full route → ordered cell samples with training-compatible features."""
    samples, _stats = build_route_cell_samples_with_stats(
        route,
        h3_to_city=h3_to_city,
        known_h3_cells=known_h3_cells,
        max_segment_meters=max_segment_meters,
    )
    return samples


def summarize_route_h3_vocab(
    route: RouteCandidate,
    *,
    known_h3_cells: set[str],
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> dict[str, object]:
    """Count total / out-of-vocabulary H3 cells for one route (order preserved)."""
    _, cell_entries = assign_cell_times_from_scaled_steps(
        route, max_segment_meters=max_segment_meters
    )
    cells = [cell for cell, _ in cell_entries]
    oov_cells = [cell for cell in cells if cell not in known_h3_cells]
    # Unique OOV examples for the error payload (stable order, capped).
    seen: set[str] = set()
    example_oov: list[str] = []
    for cell in oov_cells:
        if cell in seen:
            continue
        seen.add(cell)
        example_oov.append(cell)
        if len(example_oov) >= 5:
            break
    return {
        "route_id": route.route_id,
        "cell_count": len(cells),
        "scored_cell_count": len(cells) - len(oov_cells),
        "out_of_vocabulary_count": len(oov_cells),
        "example_oov_cells": example_oov,
    }


def build_route_cell_samples_with_stats(
    route: RouteCandidate,
    *,
    h3_to_city: Mapping[str, str],
    known_h3_cells: set[str],
    max_segment_meters: float = DEFAULT_MAX_SEGMENT_METERS,
) -> tuple[list[RouteCellSample], dict[str, object]]:
    """Build in-vocabulary samples; OOV H3 cells are skipped (not an error)."""
    _, cell_entries = assign_cell_times_from_scaled_steps(
        route, max_segment_meters=max_segment_meters
    )
    traversed_cells = [cell for cell, _ in cell_entries]
    oov_cells = [cell for cell in traversed_cells if cell not in known_h3_cells]
    seen: set[str] = set()
    example_oov: list[str] = []
    for cell in oov_cells:
        if cell in seen:
            continue
        seen.add(cell)
        example_oov.append(cell)
        if len(example_oov) >= 5:
            break

    samples: list[RouteCellSample] = []
    for sequence_index, (h3_cell, entry_time_utc) in enumerate(cell_entries):
        if h3_cell not in known_h3_cells:
            continue
        city_name = h3_to_city.get(h3_cell)
        if city_name is None:
            # In-vocab cell without a city mapping is a data integrity problem.
            raise PreprocessError(
                "unknown_city",
                f"could not resolve city_name for H3 cell {h3_cell}",
                route_id=route.route_id,
                details={
                    "routes": [
                        {
                            "route_id": route.route_id,
                            "cell_count": len(traversed_cells),
                            "scored_cell_count": len(samples),
                            "out_of_vocabulary_count": len(oov_cells),
                            "example_oov_cells": example_oov,
                        }
                    ]
                },
            )
        hour, dow, month, dow_name, month_name = local_calendar_features(
            entry_time_utc, city_name
        )
        samples.append(
            RouteCellSample(
                sequence_index=sequence_index,
                h3_cell=h3_cell,
                entry_time_utc=entry_time_utc.astimezone(timezone.utc),
                local_hour=hour,
                day_of_week_index=dow,
                day_of_week_name=dow_name,
                month_index=month,
                month_name=month_name,
                city_name=city_name,
                hour_bin_start=hour_to_bin_start(hour),
            )
        )

    stats: dict[str, object] = {
        "route_id": route.route_id,
        "cell_count": len(traversed_cells),
        "scored_cell_count": len(samples),
        "out_of_vocabulary_count": len(oov_cells),
        "example_oov_cells": example_oov,
    }
    return samples, stats
