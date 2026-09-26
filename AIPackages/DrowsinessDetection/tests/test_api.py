"""API contract tests for DrowsinessDetection FastAPI server (schema v3, 3-class)."""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Force deterministic local settings before importing the app module.
os.environ["DROWSINESS_DEVICE"] = "cpu"
os.environ["DROWSINESS_CHECKPOINT_PATH"] = "saved_weights/schema_v3_smoke.pt"
os.environ["DROWSINESS_API_KEY"] = "test-api-key-please-change"
os.environ["DROWSINESS_SAMPLING_RATE_HZ"] = "15.0"

from api.app import create_app  # noqa: E402
from api.inference_service import InferenceService  # noqa: E402
from feature_contract import (  # noqa: E402
    DROWSINESS_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
)
from label_contract import CLASS_TO_IDX, IDX_TO_CLASS  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    application = create_app()
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def model_contract(client: TestClient) -> dict[str, Any]:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    return payload


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-api-key-please-change"}


def _valid_request(contract: dict[str, Any], *, mutate: dict[str, Any] | None = None) -> dict[str, Any]:
    feature_names = list(contract["feature_names"])
    window = int(contract["window_frames"])
    feature_count = int(contract["feature_count"])
    samples = []
    base_ts = 1_790_093_823_000
    for index in range(window):
        values = [0.0] * feature_count
        values[0] = 1.0  # face_detected
        values[1] = 0.01 * index  # yaw
        values[4] = 1.0  # left_eye_valid
        values[5] = 1.0  # right_eye_valid
        values[6] = 0.25  # left_eye_aspect_ratio
        values[7] = 0.25  # right_eye_aspect_ratio
        values[8] = 0.45  # left_pupil_rel_x
        values[9] = 0.50  # left_pupil_rel_y
        values[10] = 0.55  # right_pupil_rel_x
        values[11] = 0.50  # right_pupil_rel_y
        samples.append(
            {
                "timestamp_ms": base_ts + index * 67,
                "values": values,
            }
        )
    payload: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "session_id": "test-session-1",
        "sequence_id": 42,
        "sent_at_utc": "2026-09-22T18:00:00Z",
        "sampling_rate_hz": float(contract["sampling_rate_hz"]),
        "feature_names": feature_names,
        "samples": samples,
    }
    if mutate:
        payload.update(mutate)
    return payload


def test_health_returns_model_contract(client: TestClient, model_contract: dict[str, Any]) -> None:
    assert model_contract["status"] == "ok"
    assert model_contract["model_version"] == "schema_v3_smoke"
    assert model_contract["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert model_contract["schema_version"] == FEATURE_SCHEMA_VERSION
    assert model_contract["feature_count"] == 12
    assert model_contract["window_frames"] >= 1
    assert model_contract["sampling_rate_hz"] == 15.0
    assert model_contract["class_names"] == [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]
    assert model_contract["feature_names"] == list(DROWSINESS_FEATURE_NAMES)
    assert model_contract["auth_required"] is True


def test_valid_request_produces_prediction(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=_valid_request(model_contract),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == "test-session-1"
    assert body["sequence_id"] == 42
    assert body["label"] in model_contract["class_names"]
    assert body["label_index"] == model_contract["class_names"].index(body["label"])
    assert body["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert math.isfinite(body["confidence"])
    assert set(body["probabilities"]) == {"closed", "open", "undefined"}
    assert all(math.isfinite(v) for v in body["probabilities"].values())
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["inference_latency_ms"] >= 0.0


def test_legacy_schema_version_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    payload = _valid_request(
        model_contract, mutate={"feature_schema_version": "1"}
    )
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 422
    body = response.json()
    code = body.get("error", {}).get("code") or ""
    assert code in {"unsupported_schema_version", "validation_error"}


def test_session_and_sequence_preserved(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    payload = _valid_request(
        model_contract,
        mutate={"session_id": "abc-123", "sequence_id": 7},
    )
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "abc-123"
    assert body["sequence_id"] == 7


def test_incorrect_feature_order_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    names = list(model_contract["feature_names"])
    names[0], names[1] = names[1], names[0]
    payload = _valid_request(model_contract, mutate={"feature_names": names})
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "incorrect_feature_order"


def test_incorrect_feature_count_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    names = list(model_contract["feature_names"])[:-1]
    payload = _valid_request(model_contract, mutate={"feature_names": names})
    for sample in payload["samples"]:
        sample["values"] = sample["values"][: len(names)]
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "incorrect_feature_count"


def test_incorrect_window_length_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    payload = _valid_request(model_contract)
    payload["samples"] = payload["samples"][:1]
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "incorrect_window_length"


def test_incorrect_sampling_rate_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    payload = _valid_request(model_contract, mutate={"sampling_rate_hz": 30.0})
    response = client.post(
        "/v1/drowsiness/predict",
        headers=_auth_headers(),
        json=payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "incorrect_sampling_rate"


def test_nonfinite_input_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    import json

    payload = _valid_request(model_contract)
    payload["samples"][0]["values"][0] = float("nan")
    body = json.dumps(payload, allow_nan=True)
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    response = client.post(
        "/v1/drowsiness/predict",
        headers=headers,
        content=body,
    )
    assert response.status_code == 422


def test_missing_api_key_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    response = client.post(
        "/v1/drowsiness/predict",
        json=_valid_request(model_contract),
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_incorrect_api_key_rejected(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    response = client.post(
        "/v1/drowsiness/predict",
        headers={"X-API-Key": "wrong-key"},
        json=_valid_request(model_contract),
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_deterministic_repeated_inputs(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    payload = _valid_request(model_contract)
    first = client.post(
        "/v1/drowsiness/predict", headers=_auth_headers(), json=payload
    )
    second = client.post(
        "/v1/drowsiness/predict", headers=_auth_headers(), json=payload
    )
    assert first.status_code == 200
    assert second.status_code == 200
    a = first.json()
    b = second.json()
    assert a["label"] == b["label"]
    assert a["label_index"] == b["label_index"]
    assert a["probabilities"] == b["probabilities"]


def test_model_not_reloaded_per_request(
    client: TestClient, model_contract: dict[str, Any]
) -> None:
    service: InferenceService = client.app.state.service
    load_id = service.loaded.load_id
    before = service.predict_call_count
    for _ in range(3):
        response = client.post(
            "/v1/drowsiness/predict",
            headers=_auth_headers(),
            json=_valid_request(model_contract),
        )
        assert response.status_code == 200
    assert service.loaded.load_id == load_id
    assert service.predict_call_count == before + 3


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["predict"] == "/v1/drowsiness/predict"
