"""Strict Pydantic request/response models for drowsiness inference."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FeatureSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_ms: int = Field(..., ge=0)
    values: list[float] = Field(..., min_length=1)

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, value: list[float]) -> list[float]:
        for cell in value:
            if cell != cell or cell in (float("inf"), float("-inf")):
                raise ValueError("sample values must be finite (no NaN/Inf)")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Accept both keys; prefer feature_schema_version when both are sent.
    feature_schema_version: str | None = Field(default=None, min_length=1)
    schema_version: str | None = Field(default=None, min_length=1)
    session_id: str = Field(..., min_length=1)
    sequence_id: int = Field(..., ge=0)
    sent_at_utc: datetime
    sampling_rate_hz: float
    feature_names: list[str] = Field(..., min_length=1)
    samples: list[FeatureSample] = Field(..., min_length=1)

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id must be nonempty")
        return stripped

    @field_validator("sampling_rate_hz")
    @classmethod
    def sampling_rate_positive_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")) or value <= 0.0:
            raise ValueError("sampling_rate_hz must be a finite positive number")
        return float(value)

    @model_validator(mode="after")
    def validate_timestamps_and_schema(self) -> PredictRequest:
        if not self.feature_schema_version and not self.schema_version:
            raise ValueError(
                "feature_schema_version (or legacy schema_version) is required"
            )
        if not self.samples:
            raise ValueError("samples must be nonempty")
        width = len(self.samples[0].values)
        prev_ts: int | None = None
        for sample in self.samples:
            if len(sample.values) != width:
                raise ValueError("all samples must have the same feature count")
            if prev_ts is not None and sample.timestamp_ms <= prev_ts:
                raise ValueError("sample timestamps must be strictly increasing")
            prev_ts = sample.timestamp_ms
        return self

    @property
    def resolved_schema_version(self) -> str:
        return str(self.feature_schema_version or self.schema_version)


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    sequence_id: int
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    feature_schema_version: str
    inference_latency_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_version: str
    device: str
    feature_schema_version: str | None = None
    # Backward-compatible alias of feature_schema_version.
    schema_version: str | None = None
    feature_count: int
    feature_names: list[str]
    window_frames: int
    sampling_rate_hz: float | None
    class_names: list[str]
    auth_required: bool = False


class RootResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    docs: str
    health: str
    predict: str


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
