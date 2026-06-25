"""FastAPI application factory.

Loads one pipeline+model bundle at startup and serves /health, /predict,
/predict/batch, and /metrics. A middleware records latency and request/error
counts for every request (without logging payloads, which may carry sensitive
traffic data). The decision threshold profile is operator-selectable per request.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from netsentry.config import load_settings
from netsentry.log import get_logger
from netsentry.serving import metrics as M
from netsentry.serving.inference import InferenceEngine
from netsentry.serving.schemas import (
    BatchRequest,
    BatchResponse,
    FlowRequest,
    HealthResponse,
    PredictionResponse,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from netsentry.config import Settings

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI app (loads the model bundle once)."""
    from fastapi import FastAPI, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware

    settings = settings or load_settings()
    engine = InferenceEngine(settings)

    app = FastAPI(title="NetSentry", version=engine.version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.serving.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        endpoint = request.url.path
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            M.ERROR_COUNT.labels(endpoint).inc()
            raise
        elapsed = time.perf_counter() - start
        M.REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        M.REQUEST_COUNT.labels(endpoint, request.method, response.status_code).inc()
        logger.info(
            "request",
            extra={
                "endpoint": endpoint,
                "status": response.status_code,
                "ms": round(elapsed * 1e3, 2),
            },
        )
        return response

    def _resolve_profile(profile: str | None) -> str:
        chosen = profile or engine.default_profile
        if chosen not in engine.bundle.thresholds and chosen != engine.default_profile:
            available = sorted(engine.bundle.thresholds)
            raise HTTPException(
                status_code=400,
                detail=f"unknown threshold profile {chosen!r}; available: {available}",
            )
        return chosen

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_version=engine.version, loaded_at=engine.loaded_at)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: FlowRequest,
        profile: str | None = Query(default=None, description="Threshold profile."),
    ) -> PredictionResponse:
        return engine.predict([request.flow], profile=_resolve_profile(profile))[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(
        request: BatchRequest,
        profile: str | None = Query(default=None, description="Threshold profile."),
    ) -> BatchResponse:
        if len(request.flows) > settings.serving.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"batch too large: {len(request.flows)} > {settings.serving.max_batch_size}",
            )
        predictions = engine.predict(request.flows, profile=_resolve_profile(profile))
        return BatchResponse(predictions=predictions)

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        payload, content_type = M.render_latest()
        return Response(content=payload, media_type=content_type)

    return app
