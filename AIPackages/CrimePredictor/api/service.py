"""Model loading and batched route scoring for the FastAPI server."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

from api.preprocess import (
    CITY_TIMEZONES,
    RouteCellSample,
    build_route_cell_samples_with_stats,
    hour_to_bin_start,
)
from api.schemas import (
    CellPrediction,
    PredictRoutesRequest,
    PredictRoutesResponse,
    PredictionSummary,
    RouteCandidate,
    RoutePrediction,
    TimeBinSafetyScore,
)

logger = logging.getLogger("crime_predictor.api")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_ROOT / "src"
DEFAULT_CHECKPOINT = SRC_DIR / "saved_weights" / "best.pt"
DEFAULT_METADATA_DIR = PACKAGE_ROOT / "data" / "crime_rate_metadata"
DEFAULT_PARQUET = PACKAGE_ROOT / "data" / "crime_rate_dataset.parquet"
DEFAULT_H3_CITY_CACHE = PACKAGE_ROOT / "data" / "crime_rate_metadata" / "h3_cell_to_city.json"
DEFAULT_BATCH_SIZE = 2048

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import CrimeRateInference  # noqa: E402
from time_bins import HOUR_BIN_STARTS  # noqa: E402

# Matches RouteRiskScorer.highHourGapWeight on iOS.
HIGH_HOUR_GAP_WEIGHT = 0.30


def _adjusted_severity_weighted_sum(
    cells: list[RouteCellSample],
    severities: list[float],
    high_hour_by_h3: dict[str, float],
    *,
    gap_weight: float = HIGH_HOUR_GAP_WEIGHT,
) -> float:
    """Sum adjusted risk once per unique H3 cell for one time bin."""
    seen: set[str] = set()
    total = 0.0
    for sample, current in zip(cells, severities):
        if sample.h3_cell in seen:
            continue
        seen.add(sample.h3_cell)
        high = high_hour_by_h3.get(sample.h3_cell, current)
        total += current + gap_weight * max(0.0, high - current)
    return total


class ModelUnavailableError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "model_unavailable"


class InferenceFailureError(RuntimeError):
    def __init__(self, message: str, *, code: str = "internal_inference_failure") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class RuntimeArtifacts:
    engine: CrimeRateInference
    checkpoint_name: str
    device: str
    h3_to_city: dict[str, str]
    known_h3_cells: set[str]
    batch_size: int = DEFAULT_BATCH_SIZE


def resolve_checkpoint_path() -> Path:
    override = os.environ.get("CRIME_MODEL_PATH", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = DEFAULT_CHECKPOINT.resolve()
    if not path.is_file():
        raise ModelUnavailableError(f"checkpoint not found: {path.name}")
    return path


def resolve_metadata_dir() -> Path:
    override = os.environ.get("CRIME_METADATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = DEFAULT_METADATA_DIR.resolve()
    if not path.is_dir():
        raise ModelUnavailableError(f"metadata directory missing: {path}")
    return path


def resolve_parquet_path() -> Path:
    override = os.environ.get("CRIME_PARQUET_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_PARQUET.resolve()


def _load_h3_city_mapping(
    *,
    parquet_path: Path,
    cache_path: Path,
    expected_cells: int,
) -> dict[str, str]:
    import json

    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        mapping = {str(k): str(v) for k, v in payload.items()}
        if len(mapping) >= expected_cells:
            return mapping

    if not parquet_path.is_file():
        raise ModelUnavailableError(
            "missing preprocessing artifact: crime_rate_dataset.parquet "
            "(required to map H3 cells to city_name)"
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ModelUnavailableError(
            "pyarrow is required to build H3→city mapping from the training parquet"
        ) from exc

    mapping: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(
        batch_size=65_536,
        columns=["h3_cell", "city_name"],
    ):
        data = batch.to_pydict()
        for cell, city in zip(data["h3_cell"], data["city_name"]):
            cell_key = str(cell)
            city_key = str(city).strip().lower()
            existing = mapping.get(cell_key)
            if existing is None:
                mapping[cell_key] = city_key
            elif existing != city_key:
                conflicts.setdefault(cell_key, {existing}).add(city_key)
        if len(mapping) >= expected_cells and not conflicts:
            # Still finish? Prefer full scan for conflict detection on first build.
            pass

    if conflicts:
        sample = next(iter(conflicts.items()))
        raise ModelUnavailableError(
            f"H3→city mapping is inconsistent in training data for cell {sample[0]}: {sorted(sample[1])}"
        )
    if len(mapping) < expected_cells:
        raise ModelUnavailableError(
            f"H3→city mapping incomplete: found {len(mapping)} cells, expected {expected_cells}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote H3→city cache with %s cells to %s", f"{len(mapping):,}", cache_path.name)
    return mapping


def load_runtime_artifacts(*, device: str = "cpu") -> RuntimeArtifacts:
    checkpoint_path = resolve_checkpoint_path()
    metadata_dir = resolve_metadata_dir()
    parquet_path = resolve_parquet_path()

    required = [
        metadata_dir / "city_to_index.json",
        metadata_dir / "h3_cell_to_index.json",
        metadata_dir / "index_to_city.json",
        metadata_dir / "index_to_h3_cell.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ModelUnavailableError(
            f"missing preprocessing artifact(s): {', '.join(missing)}"
        )

    try:
        engine = CrimeRateInference(
            checkpoint_path=checkpoint_path,
            metadata_dir=metadata_dir,
            device=device,
            verify_checksums=True,
        )
    except FileNotFoundError as exc:
        raise ModelUnavailableError(str(exc)) from exc
    except ValueError as exc:
        raise ModelUnavailableError(str(exc)) from exc

    known_h3 = {cell for cell in engine.h3_cell_to_index if cell != "<UNK>"}
    h3_to_city = _load_h3_city_mapping(
        parquet_path=parquet_path,
        cache_path=DEFAULT_H3_CITY_CACHE,
        expected_cells=len(known_h3),
    )
    # Keep only cells present in the training vocabulary.
    h3_to_city = {
        cell: city
        for cell, city in h3_to_city.items()
        if cell in known_h3
    }
    if len(h3_to_city) != len(known_h3):
        raise ModelUnavailableError(
            f"H3→city coverage {len(h3_to_city)} does not match vocab size {len(known_h3)}"
        )

    return RuntimeArtifacts(
        engine=engine,
        checkpoint_name=checkpoint_path.name,
        device=str(engine.device),
        h3_to_city=h3_to_city,
        known_h3_cells=known_h3,
    )


def _unique_route_cells(samples: list[RouteCellSample]) -> list[RouteCellSample]:
    """Preserve first-seen order of H3 cells along the route."""
    unique: list[RouteCellSample] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.h3_cell in seen:
            continue
        seen.add(sample.h3_cell)
        unique.append(sample)
    return unique


def _departure_local_calendar(
    route: RouteCandidate,
    samples: list[RouteCellSample],
) -> tuple[int, int, int, int]:
    """Return (month 1..12, day_of_week Mon=0, local_hour, hour_bin_start)."""
    zone_name: str | None = None
    if samples:
        zone_name = CITY_TIMEZONES.get(samples[0].city_name)

    if zone_name is None:
        if samples:
            return (
                samples[0].month_index,
                samples[0].day_of_week_index,
                samples[0].local_hour,
                samples[0].hour_bin_start,
            )
        local = route.departure_time_utc.astimezone(timezone.utc)
        hour = int(local.hour)
        return int(local.month), int(local.weekday()), hour, hour_to_bin_start(hour)

    local = route.departure_time_utc.astimezone(ZoneInfo(zone_name))
    hour = int(local.hour)
    return int(local.month), int(local.weekday()), hour, hour_to_bin_start(hour)


def _score_route_time_bins(
    artifacts: RuntimeArtifacts,
    cells: list[RouteCellSample],
    *,
    month: int,
    day_of_week: int,
) -> tuple[list[TimeBinSafetyScore], dict[str, float]]:
    """Score each TIME_BIN bin; return per-bin scores and each cell's P90 severity."""
    if not cells:
        empty_scores = [
            TimeBinSafetyScore(
                hour_bin_start=bin_start,
                severity_weighted_sum=0.0,
                adjusted_severity_weighted_sum=0.0,
                max_severity_weighted_rate=0.0,
                mean_person_rate=0.0,
                mean_property_rate=0.0,
                mean_society_rate=0.0,
                mean_other_rate=0.0,
                cell_count=0,
            )
            for bin_start in HOUR_BIN_STARTS
        ]
        return empty_scores, {}

    scores: list[TimeBinSafetyScore] = []
    cell_severity_by_bin: list[torch.Tensor] = []
    n_cells = len(cells)
    h3_idx = [artifacts.engine.h3_cell_to_index[sample.h3_cell] for sample in cells]
    city_idx = [artifacts.engine.city_to_index[sample.city_name] for sample in cells]
    months = [month] * n_cells
    dows = [day_of_week] * n_cells

    for bin_start in HOUR_BIN_STARTS:
        hours = [bin_start] * n_cells
        rates = artifacts.engine.predict_encoded(h3_idx, city_idx, months, dows, hours)
        if not torch.isfinite(rates).all():
            raise InferenceFailureError(
                "model produced non-finite outputs while scoring time bins",
                code="nonfinite_model_output",
            )
        weights = artifacts.engine.model.severity_weights.to(
            device=rates.device, dtype=rates.dtype
        )
        severity = (rates * weights.unsqueeze(0)).sum(dim=1)
        cell_severity_by_bin.append(severity)
        scores.append(
            TimeBinSafetyScore(
                hour_bin_start=int(bin_start),
                severity_weighted_sum=float(severity.sum().item()),
                max_severity_weighted_rate=float(severity.max().item()),
                mean_person_rate=float(rates[:, 0].mean().item()),
                mean_property_rate=float(rates[:, 1].mean().item()),
                mean_society_rate=float(rates[:, 2].mean().item()),
                mean_other_rate=float(rates[:, 3].mean().item()),
                cell_count=n_cells,
            )
        )

    high_hour_rates = torch.quantile(
        torch.stack(cell_severity_by_bin), 0.9, dim=0
    ).tolist()
    high_hour_by_h3 = {
        sample.h3_cell: float(rate)
        for sample, rate in zip(cells, high_hour_rates)
    }

    adjusted_scores: list[TimeBinSafetyScore] = []
    for index, score in enumerate(scores):
        severities = [float(value) for value in cell_severity_by_bin[index].tolist()]
        adjusted = _adjusted_severity_weighted_sum(cells, severities, high_hour_by_h3)
        adjusted_scores.append(
            score.model_copy(update={"adjusted_severity_weighted_sum": adjusted})
        )

    return adjusted_scores, high_hour_by_h3


def _batched_forward(
    artifacts: RuntimeArtifacts,
    samples: list[RouteCellSample],
) -> list[dict[str, float]]:
    engine = artifacts.engine
    outputs: list[dict[str, float]] = []
    batch_size = max(1, int(artifacts.batch_size))

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        h3_idx = [engine.h3_cell_to_index[sample.h3_cell] for sample in chunk]
        city_idx = [engine.city_to_index[sample.city_name] for sample in chunk]
        months = [sample.month_index for sample in chunk]
        dows = [sample.day_of_week_index for sample in chunk]
        hours = [sample.hour_bin_start for sample in chunk]

        rates = engine.predict_encoded(h3_idx, city_idx, months, dows, hours)
        if not torch.isfinite(rates).all():
            raise InferenceFailureError(
                "model produced non-finite outputs",
                code="nonfinite_model_output",
            )

        weights = engine.model.severity_weights.to(device=rates.device, dtype=rates.dtype)
        totals = rates.sum(dim=1)
        severity = (rates * weights.unsqueeze(0)).sum(dim=1)
        for row_index in range(rates.shape[0]):
            row = rates[row_index]
            outputs.append(
                {
                    "person_rate": float(row[0].item()),
                    "property_rate": float(row[1].item()),
                    "society_rate": float(row[2].item()),
                    "other_rate": float(row[3].item()),
                    "total_rate": float(totals[row_index].item()),
                    "severity_weighted_rate": float(severity[row_index].item()),
                }
            )
    return outputs


def predict_routes(
    artifacts: RuntimeArtifacts,
    request: PredictRoutesRequest,
) -> PredictRoutesResponse:
    preprocess_t0 = time.perf_counter()
    route_samples: list[
        tuple[str, list[RouteCellSample], dict[str, object], RouteCandidate]
    ] = []
    flat_samples: list[RouteCellSample] = []

    for route in request.routes:
        samples, stats = build_route_cell_samples_with_stats(
            route,
            h3_to_city=artifacts.h3_to_city,
            known_h3_cells=artifacts.known_h3_cells,
        )
        route_samples.append((route.route_id, samples, stats, route))
        flat_samples.extend(samples)

    preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0
    inference_t0 = time.perf_counter()
    flat_outputs = _batched_forward(artifacts, flat_samples) if flat_samples else []

    # Rebuild per-route responses in input order, then score all 3-hour bins.
    cursor = 0
    route_payloads: list[RoutePrediction] = []
    for route_id, samples, stats, route in route_samples:
        unique_cells = _unique_route_cells(samples)
        month, dow, _local_hour, active_bin = _departure_local_calendar(route, samples)
        time_bin_scores, high_hour_by_h3 = _score_route_time_bins(
            artifacts,
            unique_cells,
            month=month,
            day_of_week=dow,
        )

        cells: list[CellPrediction] = []
        for sample in samples:
            rates = flat_outputs[cursor]
            cursor += 1
            cells.append(
                CellPrediction(
                    sequence_index=sample.sequence_index,
                    h3_cell=sample.h3_cell,
                    entry_time_utc=sample.entry_time_utc,
                    local_hour=sample.local_hour,
                    day_of_week=sample.day_of_week_name,
                    month=sample.month_name,
                    city_name=sample.city_name,
                    severity_weighted_rate=rates["severity_weighted_rate"],
                    high_hour_severity_weighted_rate=high_hour_by_h3.get(sample.h3_cell),
                    total_rate=rates["total_rate"],
                    person_rate=rates["person_rate"],
                    property_rate=rates["property_rate"],
                    society_rate=rates["society_rate"],
                    other_rate=rates["other_rate"],
                    hour_bin_start=sample.hour_bin_start,
                )
            )

        active = next(
            (item for item in time_bin_scores if item.hour_bin_start == active_bin),
            time_bin_scores[0] if time_bin_scores else None,
        )
        if active is None:
            summary = PredictionSummary(mean=0.0, maximum=0.0, sum=0.0)
            active_bin = 0
        else:
            active_risk = (
                active.adjusted_severity_weighted_sum
                if active.adjusted_severity_weighted_sum is not None
                else active.severity_weighted_sum
            )
            n = max(active.cell_count, 1)
            summary = PredictionSummary(
                mean=active_risk / n,
                maximum=active.max_severity_weighted_rate,
                sum=active_risk,
            )

        route_payloads.append(
            RoutePrediction(
                route_id=route_id,
                cell_count=int(stats["cell_count"]),
                scored_cell_count=len(cells),
                out_of_vocabulary_count=int(stats["out_of_vocabulary_count"]),
                active_hour_bin_start=int(active_bin),
                prediction_summary=summary,
                time_bin_scores=time_bin_scores,
                cells=cells,
            )
        )

    inference_ms = (time.perf_counter() - inference_t0) * 1000.0
    total_oov = sum(int(stats["out_of_vocabulary_count"]) for _, _, stats, _ in route_samples)
    logger.info(
        "predict-routes request_id=%s route_count=%s scored_cells=%s oov_cells=%s "
        "time_bins=%s preprocess_ms=%.1f inference_ms=%.1f device=%s",
        request.request_id,
        len(request.routes),
        len(flat_samples),
        total_oov,
        len(HOUR_BIN_STARTS),
        preprocess_ms,
        inference_ms,
        artifacts.device,
    )

    return PredictRoutesResponse(
        request_id=request.request_id,
        model_version=artifacts.checkpoint_name,
        routes=route_payloads,
    )


def smoke_validation_example(artifacts: RuntimeArtifacts) -> dict[str, Any]:
    """Run one known training-row-shaped example through the shared loader."""
    # Known Austin cell from crime_rate_dataset.parquet (audit smoke sample).
    prediction = artifacts.engine.predict(
        city_name="austin",
        h3_cell="8948985a4d3ffff",
        month=1,
        day_of_week=0,
        hour_bin_start=0,
    )
    values = [
        prediction.person_rate,
        prediction.property_rate,
        prediction.society_rate,
        prediction.other_rate,
        prediction.total_rate,
        prediction.severity_weighted_rate,
    ]
    if any(not math_isfinite(v) for v in values):
        raise InferenceFailureError(
            "smoke validation produced non-finite rates",
            code="nonfinite_model_output",
        )
    return prediction.as_dict()


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
