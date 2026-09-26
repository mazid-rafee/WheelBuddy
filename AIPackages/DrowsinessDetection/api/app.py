"""FastAPI application for local drowsiness TCN inference."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.config import load_settings
from api.inference_service import (
    ContractError,
    InferenceFailureError,
    InferenceService,
    ModelUnavailableError,
)
from api.schemas import (
    ErrorBody,
    ErrorEnvelope,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RootResponse,
)

logger = logging.getLogger("drowsiness.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _sanitize(value: object) -> object:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _error_json(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    payload = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            details=_sanitize(details) if details is not None else None,  # type: ignore[arg-type]
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(payload.model_dump()),
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = load_settings()
        app.state.settings = settings
        if not settings.api_key:
            logger.warning(
                "DROWSINESS_API_KEY is not set; /v1/drowsiness/predict is open "
                "for local development only"
            )
        if settings.expected_sampling_rate_hz is None:
            logger.warning(
                "DROWSINESS_SAMPLING_RATE_HZ is unset; sampling_rate_hz is accepted "
                "as any finite positive value. Training code does not define a rate."
            )
        try:
            service = InferenceService(settings)
        except ModelUnavailableError:
            logger.exception("Failed to load drowsiness checkpoint")
            raise
        app.state.service = service
        meta = service.metadata()
        logger.info(
            "DrowsinessDetection ready version=%s device=%s window_frames=%s "
            "features=%s classes=%s auth_required=%s",
            meta["model_version"],
            meta["device"],
            meta["window_frames"],
            meta["feature_count"],
            meta["class_names"],
            bool(settings.api_key),
        )
        yield
        app.state.service = None

    app = FastAPI(
        title="DrowsinessDetection Inference API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_json(
            status_code=422,
            code="validation_error",
            message="request failed schema validation",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(ContractError)
    async def contract_exception_handler(
        request: Request, exc: ContractError
    ) -> JSONResponse:
        return _error_json(
            status_code=422,
            code=exc.code,
            message=exc.message,
        )

    @app.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return _error_json(
            status_code=503,
            code=exc.code,
            message="model unavailable",
        )

    @app.exception_handler(InferenceFailureError)
    async def inference_failure_handler(
        request: Request, exc: InferenceFailureError
    ) -> JSONResponse:
        logger.exception("inference failure: %s", exc)
        return _error_json(
            status_code=500,
            code=exc.code,
            message="inference failed",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return _error_json(
                status_code=exc.status_code,
                code=str(detail.get("code")),
                message=str(detail.get("message", "error")),
            )
        return _error_json(
            status_code=exc.status_code,
            code="http_error",
            message=str(detail),
        )

    async def require_api_key(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        settings = request.app.state.settings
        expected = settings.api_key
        if not expected:
            return
        provided = (x_api_key or "").strip()
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "unauthorized",
                    "message": "missing or invalid API key",
                },
            )

    @app.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        return RootResponse(
            service="DrowsinessDetection Inference API",
            version="1.0.0",
            docs="/docs",
            health="/health",
            predict="/v1/drowsiness/predict",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        service: InferenceService | None = getattr(request.app.state, "service", None)
        settings = getattr(request.app.state, "settings", None)
        if service is None:
            return HealthResponse(
                status="degraded",
                model_loaded=False,
                model_version="unavailable",
                device="unknown",
                feature_schema_version=None,
                schema_version=None,
                feature_count=0,
                feature_names=[],
                window_frames=0,
                sampling_rate_hz=None,
                class_names=[],
                auth_required=bool(settings and settings.api_key),
            )
        meta = service.metadata()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=meta["model_version"],
            device=meta["device"],
            feature_schema_version=meta.get("feature_schema_version"),
            schema_version=meta.get("schema_version"),
            feature_count=meta["feature_count"],
            feature_names=meta["feature_names"],
            window_frames=meta["window_frames"],
            sampling_rate_hz=meta["sampling_rate_hz"],
            class_names=meta["class_names"],
            auth_required=bool(settings and settings.api_key),
        )

    @app.post("/v1/drowsiness/predict", response_model=PredictResponse)
    async def predict(
        payload: PredictRequest,
        request: Request,
        _auth: None = Depends(require_api_key),
    ) -> PredictResponse:
        service: InferenceService | None = getattr(request.app.state, "service", None)
        if service is None:
            raise ModelUnavailableError("model is not loaded")
        try:
            return service.predict(payload)
        except ContractError:
            raise
        except InferenceFailureError:
            raise
        except Exception as exc:
            logger.exception("unexpected predict failure")
            raise InferenceFailureError(str(exc)) from exc

    return app


app = create_app()
