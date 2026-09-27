"""FastAPI application for local CrimeRateMLP route inference.

Endpoints:
- ``GET /health``: reports whether model artifacts are loaded.
- ``POST /predict-routes``: scores candidate routes (used by the iOS app).

Errors are returned as ``{"error": {code, message, request_id, route_id[, details]}}``
with status 422 (request validation), 400 (route preprocessing), 503 (model
unavailable) or 500 (inference failure).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.preprocess import PreprocessError
from api.schemas import HealthResponse, PredictRoutesRequest, PredictRoutesResponse
from api.service import (
    InferenceFailureError,
    ModelUnavailableError,
    RuntimeArtifacts,
    load_runtime_artifacts,
    predict_routes,
)

logger = logging.getLogger("crime_predictor.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
    route_id: str | None = None,
    details: dict | None = None,
) -> JSONResponse:
    """Build the uniform JSON error envelope; ``details`` is omitted when empty."""
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "route_id": route_id,
        }
    }
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts on CPU at startup and store them on ``app.state``.

    A ``ModelUnavailableError`` is logged and re-raised, which aborts startup.
    Artifacts are cleared on shutdown.
    """
    try:
        artifacts = load_runtime_artifacts(device="cpu")
    except ModelUnavailableError as exc:
        logger.exception("Failed to load CrimePredictor artifacts: %s", exc)
        raise
    app.state.artifacts = artifacts
    logger.info(
        "CrimePredictor ready checkpoint=%s device=%s h3_cells=%s",
        artifacts.checkpoint_name,
        artifacts.device,
        f"{len(artifacts.known_h3_cells):,}",
    )
    yield
    app.state.artifacts = None


def create_app() -> FastAPI:
    """Construct the FastAPI app with error handlers and routes registered."""
    app = FastAPI(
        title="CrimePredictor Inference API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="validation_error",
            message=str(exc.errors()),
        )

    @app.exception_handler(PreprocessError)
    async def preprocess_exception_handler(
        request: Request, exc: PreprocessError
    ) -> JSONResponse:
        request_id = None
        # Best-effort: echo request_id from the raw body; any parse failure is ignored.
        try:
            payload = await request.json()
            request_id = payload.get("request_id")
        except Exception:
            request_id = None
        return _error_response(
            status_code=400,
            code=exc.code,
            message=exc.message,
            request_id=request_id,
            route_id=exc.route_id,
            details=exc.details or None,
        )

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return _error_response(
            status_code=503,
            code=exc.code,
            message=str(exc),
        )

    @app.exception_handler(InferenceFailureError)
    async def inference_failure_handler(
        request: Request, exc: InferenceFailureError
    ) -> JSONResponse:
        return _error_response(
            status_code=500,
            code=exc.code,
            message=str(exc),
        )

    # Liveness probe: always returns status "ok"; model_loaded reflects whether
    # startup artifacts are present on app.state.
    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        artifacts: RuntimeArtifacts | None = getattr(request.app.state, "artifacts", None)
        loaded = artifacts is not None
        return HealthResponse(
            status="ok",
            model_loaded=loaded,
            checkpoint=artifacts.checkpoint_name if artifacts else "unavailable",
            device=artifacts.device if artifacts else "cpu",
        )

    # Score every candidate route in the request. Preprocess and inference errors
    # propagate to their handlers; any other exception is logged and wrapped as
    # InferenceFailureError (HTTP 500).
    @app.post("/predict-routes", response_model=PredictRoutesResponse)
    async def predict_routes_endpoint(
        payload: PredictRoutesRequest,
        request: Request,
    ) -> PredictRoutesResponse:
        artifacts: RuntimeArtifacts | None = getattr(request.app.state, "artifacts", None)
        if artifacts is None:
            raise ModelUnavailableError("model is not loaded")
        try:
            return predict_routes(artifacts, payload)
        except PreprocessError:
            raise
        except InferenceFailureError:
            raise
        except Exception as exc:
            logger.exception(
                "internal inference failure request_id=%s", payload.request_id
            )
            raise InferenceFailureError(str(exc)) from exc

    return app


app = create_app()
