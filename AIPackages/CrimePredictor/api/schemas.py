"""Strict Pydantic request/response models for CrimePredictor inference API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

MAX_ROUTES_PER_REQUEST = 8
MAX_STEPS_PER_ROUTE = 256
MAX_POLYLINE_CHARS = 100_000
MAX_TOTAL_POLYLINE_CHARS = 500_000


class RouteStep(BaseModel):
    encoded_polyline: str = Field(..., min_length=1)
    distance_meters: float = Field(..., ge=0.0)
    static_duration_seconds: float = Field(..., ge=0.0)

    @field_validator("encoded_polyline")
    @classmethod
    def polyline_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("encoded_polyline must be nonempty")
        if len(stripped) > MAX_POLYLINE_CHARS:
            raise ValueError(
                f"encoded_polyline exceeds {MAX_POLYLINE_CHARS} characters"
            )
        return stripped

    @field_validator("distance_meters", "static_duration_seconds")
    @classmethod
    def finite_nonnegative(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("numeric fields must be finite")
        return float(value)


class RouteCandidate(BaseModel):
    route_id: str = Field(..., min_length=1)
    departure_time_utc: datetime
    duration_seconds: float = Field(..., gt=0.0)
    steps: list[RouteStep] = Field(..., min_length=1)

    @field_validator("route_id")
    @classmethod
    def route_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("route_id must be nonempty")
        return stripped

    @field_validator("departure_time_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("departure_time_utc must include a timezone (UTC)")
        return value.astimezone(timezone.utc)

    @field_validator("duration_seconds")
    @classmethod
    def finite_positive_duration(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("duration_seconds must be finite")
        return float(value)

    @model_validator(mode="after")
    def validate_steps_budget(self) -> RouteCandidate:
        if len(self.steps) > MAX_STEPS_PER_ROUTE:
            raise ValueError(
                f"route {self.route_id!r} exceeds {MAX_STEPS_PER_ROUTE} steps"
            )
        static_sum = sum(step.static_duration_seconds for step in self.steps)
        if static_sum <= 0.0:
            raise ValueError(
                f"route {self.route_id!r} must have positive total static_duration_seconds"
            )
        return self


class PredictRoutesRequest(BaseModel):
    request_id: Annotated[str, Field(min_length=1)]
    routes: list[RouteCandidate] = Field(..., min_length=1)

    @field_validator("request_id")
    @classmethod
    def request_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request_id must be nonempty")
        # Accept UUID strings or other opaque IDs; prefer UUID when provided.
        try:
            UUID(stripped)
        except ValueError:
            pass
        return stripped

    @model_validator(mode="after")
    def validate_routes(self) -> PredictRoutesRequest:
        if len(self.routes) > MAX_ROUTES_PER_REQUEST:
            raise ValueError(
                f"at most {MAX_ROUTES_PER_REQUEST} routes are allowed per request"
            )
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_id values must be unique within a request")
        total_polyline = sum(
            len(step.encoded_polyline)
            for route in self.routes
            for step in route.steps
        )
        if total_polyline > MAX_TOTAL_POLYLINE_CHARS:
            raise ValueError(
                f"total encoded polyline payload exceeds {MAX_TOTAL_POLYLINE_CHARS} characters"
            )
        return self


class PredictionSummary(BaseModel):
    mean: float
    maximum: float
    sum: float


class TimeBinSafetyScore(BaseModel):
    """Route-level safety aggregates for one training 3-hour time bin."""

    hour_bin_start: int
    # Sum of severity_weighted_rate over unique in-vocab H3 cells on the route.
    severity_weighted_sum: float
    # Unique-cell adjusted risk sum for this bin (30% high-hour gap); omitted on legacy servers.
    adjusted_severity_weighted_sum: float | None = None
    # Max per-cell severity_weighted_rate under this bin.
    max_severity_weighted_rate: float
    mean_person_rate: float
    mean_property_rate: float
    mean_society_rate: float
    mean_other_rate: float
    cell_count: int


class CellPrediction(BaseModel):
    sequence_index: int
    h3_cell: str
    entry_time_utc: datetime
    local_hour: int
    day_of_week: str
    month: str
    city_name: str
    # Model outputs nonnegative per-hour rates (softplus), not probabilities.
    severity_weighted_rate: float
    # 90th percentile of this cell's severity across departure weekday/month time bins.
    high_hour_severity_weighted_rate: float | None = None
    total_rate: float
    person_rate: float
    property_rate: float
    society_rate: float
    other_rate: float
    hour_bin_start: int


class RoutePrediction(BaseModel):
    route_id: str
    # Total H3 cells along the densified route (including OOV skips).
    cell_count: int
    # Cells that were scored (in training vocabulary).
    scored_cell_count: int
    # Cells skipped because they were outside the training H3 vocabulary.
    out_of_vocabulary_count: int
    # 3-hour bin used for route ranking (from departure local time).
    active_hour_bin_start: int
    # prediction_summary.sum is the active bin's severity_weighted_sum.
    prediction_summary: PredictionSummary
    # Safety score for every 3-hour training bin over the same route geometry.
    time_bin_scores: list[TimeBinSafetyScore]
    cells: list[CellPrediction]


class PredictRoutesResponse(BaseModel):
    request_id: str
    model_version: str
    routes: list[RoutePrediction]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    checkpoint: str
    device: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    route_id: str | None = None
