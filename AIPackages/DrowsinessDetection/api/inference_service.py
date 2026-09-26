"""Checkpoint loading and single-window prediction for the API."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from api.config import Settings, resolve_device
from api.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("drowsiness.api")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from inference import (  # noqa: E402
    SchemaContractError,
    load_checkpoint,
    normalize_windows,
    predict_proba,
)
from feature_contract import (  # noqa: E402
    DROWSINESS_FEATURE_NAMES,
    FEATURE_COUNT,
    FEATURE_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSIONS,
)
from label_contract import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES  # noqa: E402


class ModelUnavailableError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "model_unavailable"


class InferenceFailureError(RuntimeError):
    def __init__(self, message: str, *, code: str = "internal_inference_failure") -> None:
        super().__init__(message)
        self.code = code


class ContractError(ValueError):
    """Request is syntactically valid JSON but incompatible with the model contract."""

    def __init__(self, message: str, *, code: str = "contract_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    model_version: str
    feature_names: list[str]
    class_to_idx: dict[str, int]
    class_names: list[str]
    window_frames: int
    sampling_rate_hz: float | None
    standardizer: Any
    load_id: int


_LOAD_COUNTER = 0


class InferenceService:
    """Owns the single loaded TCN and enforces the exact training contract."""

    def __init__(self, settings: Settings) -> None:
        global _LOAD_COUNTER
        device = resolve_device(settings.device_preference)
        try:
            model, checkpoint, window_size = load_checkpoint(
                settings.checkpoint_path, device
            )
        except FileNotFoundError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except SchemaContractError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise ModelUnavailableError(
                f"failed to load checkpoint: {exc}"
            ) from exc

        feature_names = list(checkpoint["feature_names"])
        class_to_idx = dict(CLASS_TO_IDX)
        class_names = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
        standardizer = checkpoint["_standardizer"]
        sampling_rate = checkpoint.get("sampling_rate_hz")
        if settings.expected_sampling_rate_hz is not None:
            sampling_rate = settings.expected_sampling_rate_hz
        elif sampling_rate is None:
            sampling_rate = 15.0

        _LOAD_COUNTER += 1
        self._loaded = LoadedModel(
            model=model,
            device=device,
            checkpoint_path=settings.checkpoint_path,
            model_version=settings.checkpoint_path.stem,
            feature_names=feature_names,
            class_to_idx=class_to_idx,
            class_names=class_names,
            window_frames=int(window_size),
            sampling_rate_hz=float(sampling_rate) if sampling_rate is not None else None,
            standardizer=standardizer,
            load_id=_LOAD_COUNTER,
        )
        self._settings = settings
        self._predict_call_count = 0

    @property
    def loaded(self) -> LoadedModel:
        return self._loaded

    @property
    def predict_call_count(self) -> int:
        return self._predict_call_count

    def metadata(self) -> dict[str, Any]:
        loaded = self._loaded
        return {
            "model_version": loaded.model_version,
            "device": str(loaded.device),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(loaded.feature_names),
            "feature_count": FEATURE_COUNT,
            "class_names": list(loaded.class_names),
            "window_frames": loaded.window_frames,
            "sampling_rate_hz": loaded.sampling_rate_hz,
            "load_id": loaded.load_id,
        }

    def _validate_contract(self, request: PredictRequest) -> None:
        loaded = self._loaded
        schema = request.resolved_schema_version
        if schema in LEGACY_SCHEMA_VERSIONS or str(schema) != FEATURE_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported feature_schema_version={schema!r}; "
                f"expected {FEATURE_SCHEMA_VERSION!r}",
                code="unsupported_schema_version",
            )
        if len(request.samples) != loaded.window_frames:
            raise ContractError(
                f"expected exactly {loaded.window_frames} samples "
                f"(model window_size), got {len(request.samples)}",
                code="incorrect_window_length",
            )
        if len(request.feature_names) != FEATURE_COUNT:
            raise ContractError(
                f"expected {FEATURE_COUNT} feature_names, "
                f"got {len(request.feature_names)}",
                code="incorrect_feature_count",
            )
        if request.feature_names != list(DROWSINESS_FEATURE_NAMES):
            if set(request.feature_names) == set(DROWSINESS_FEATURE_NAMES):
                raise ContractError(
                    "feature_names order does not match the model contract",
                    code="incorrect_feature_order",
                )
            raise ContractError(
                "feature_names do not match the model contract",
                code="incorrect_feature_names",
            )
        for sample in request.samples:
            if len(sample.values) != FEATURE_COUNT:
                raise ContractError(
                    f"each sample must contain {FEATURE_COUNT} values, "
                    f"got {len(sample.values)}",
                    code="incorrect_feature_count",
                )
        if loaded.sampling_rate_hz is not None:
            if abs(float(request.sampling_rate_hz) - float(loaded.sampling_rate_hz)) > 1e-6:
                raise ContractError(
                    f"sampling_rate_hz must be {loaded.sampling_rate_hz}, "
                    f"got {request.sampling_rate_hz}",
                    code="incorrect_sampling_rate",
                )

    def predict(self, request: PredictRequest) -> PredictResponse:
        self._validate_contract(request)
        loaded = self._loaded
        self._predict_call_count += 1
        load_id_before = loaded.load_id

        matrix = [list(sample.values) for sample in request.samples]
        try:
            normalized = normalize_windows(matrix, loaded.standardizer).unsqueeze(0)
        except SchemaContractError as exc:
            raise ContractError(str(exc), code="invalid_feature_packet") from exc

        if tuple(normalized.shape) != (1, loaded.window_frames, FEATURE_COUNT):
            raise ContractError(
                f"internal tensor shape {tuple(normalized.shape)} does not match "
                f"(1, {loaded.window_frames}, {FEATURE_COUNT})",
                code="invalid_tensor_shape",
            )

        t0 = time.perf_counter()
        try:
            probs = predict_proba(loaded.model, normalized, loaded.device)
            if not torch.isfinite(probs).all():
                raise InferenceFailureError(
                    "model produced non-finite probabilities",
                    code="nonfinite_model_output",
                )
            row = probs[0]
            if row.numel() != NUM_CLASSES:
                raise InferenceFailureError(
                    f"expected {NUM_CLASSES} class probs, got {row.numel()}",
                    code="incorrect_class_count",
                )
            pred_index = int(row.argmax().item())
            confidence = float(row[pred_index].item())
            label = loaded.class_names[pred_index]
            probability_map = {
                name: float(row[index].item())
                for index, name in enumerate(loaded.class_names)
            }
            prob_sum = float(sum(probability_map.values()))
            if abs(prob_sum - 1.0) > 1e-3:
                raise InferenceFailureError(
                    f"softmax probabilities sum to {prob_sum}, expected ~1.0",
                    code="invalid_probability_mass",
                )
        except InferenceFailureError:
            raise
        except Exception as exc:
            raise InferenceFailureError(str(exc)) from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if loaded.load_id != load_id_before:
            raise InferenceFailureError(
                "model was reloaded during prediction",
                code="model_reloaded",
            )

        return PredictResponse(
            session_id=request.session_id,
            sequence_id=request.sequence_id,
            label=label,
            label_index=pred_index,
            confidence=confidence,
            probabilities=probability_map,
            model_version=loaded.model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            inference_latency_ms=round(latency_ms, 3),
        )
