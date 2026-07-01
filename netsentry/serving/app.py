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
    from collections import defaultdict

    from fastapi import FastAPI, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    settings = settings or load_settings()
    engine = InferenceEngine(settings)

    app = FastAPI(title="NetSentry", version=engine.version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.serving.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _endpoint_label(request: Request) -> str:
        """Bounded metric label: the matched route template, not the raw URL path.

        Labelling by ``request.url.path`` would let an unauthenticated caller mint
        a fresh Prometheus time series per arbitrary (e.g. 404) path — unbounded
        label cardinality that grows server memory without limit. The matched
        route template keeps the label space to the handful of declared endpoints.
        """
        route = request.scope.get("route")
        path = getattr(route, "path", None)
        return path if isinstance(path, str) else "unmatched"

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            M.ERROR_COUNT.labels(_endpoint_label(request)).inc()
            raise
        elapsed = time.perf_counter() - start
        endpoint = _endpoint_label(request)
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

    # Guard only the prediction endpoints; /health and /metrics stay open for probes.
    guarded_paths = {"/predict", "/predict/batch"}
    rate_state: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # client -> [window, count]

    @app.middleware("http")
    async def security_guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        """API-key auth + per-client fixed-window rate limit on prediction endpoints.

        Implemented as middleware (not a dependency) because ``request: Request``
        cannot be resolved as an injected type under ``from __future__ import
        annotations`` when FastAPI is imported inside the factory.
        """
        if request.url.path in guarded_paths:
            api_key = settings.serving.api_key
            if api_key and request.headers.get("x-api-key") != api_key:
                return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)
            limit = settings.serving.rate_limit_per_minute
            if limit > 0:
                client = request.client.host if request.client else "unknown"
                window = int(time.time() // 60)
                state = rate_state[client]
                if state[0] != window:
                    state[0], state[1] = window, 0
                state[1] += 1
                if state[1] > limit:
                    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        return await call_next(request)

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
