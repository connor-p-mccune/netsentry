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
    CanaryStatus,
    FlowRequest,
    HealthResponse,
    PredictionResponse,
    ReloadRequest,
    ReloadResponse,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from netsentry.config import Settings

logger = get_logger(__name__)


class _EngineHolder:
    """Mutable reference to the live engine so /admin/reload can swap it atomically.

    Every request reads ``holder.engine`` at handler entry; a reload reassigns the
    attribute in one bytecode op (atomic under the GIL), and any in-flight request
    keeps the engine reference it already read. Good enough for a single-process,
    canary-gated swap; a multi-worker deploy would reload per worker.
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI app (loads the model bundle once)."""
    from collections import defaultdict

    from fastapi import FastAPI, HTTPException, Query, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    settings = settings or load_settings()
    engine = InferenceEngine(settings)
    holder = _EngineHolder(engine)

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

    # Guard the prediction endpoints and the admin reload; /health and /metrics stay
    # open for probes.
    guarded_paths = {"/predict", "/predict/batch", "/admin/reload"}
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
        engine = holder.engine
        chosen = profile or engine.default_profile
        if chosen not in engine.bundle.thresholds and chosen != engine.default_profile:
            available = sorted(engine.bundle.thresholds)
            raise HTTPException(
                status_code=400,
                detail=f"unknown threshold profile {chosen!r}; available: {available}",
            )
        return chosen

    def _canary_status(engine: InferenceEngine) -> CanaryStatus | None:
        canary = engine.canary
        if not canary.present:
            return None
        return CanaryStatus(
            ok=canary.ok, n=canary.n, max_delta=canary.max_delta, tolerance=canary.tolerance
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        engine = holder.engine
        return HealthResponse(
            status="ok" if engine.canary.ok else "degraded",
            model_version=engine.version,
            loaded_at=engine.loaded_at,
            canary=_canary_status(engine),
            shadow_model_version=engine.shadow_version,
        )

    explain_query = Query(
        default=True,
        description="Include SHAP top_features (the expensive step); "
        "false returns an empty list for throughput-bound callers.",
    )
    exemplars_query = Query(
        default=False,
        description="Include similar_flows: the nearest known training flows "
        "(case-based evidence), when the bundle carries an exemplar index.",
    )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        request: FlowRequest,
        profile: str | None = Query(default=None, description="Threshold profile."),
        explain: bool = explain_query,
        exemplars: bool = exemplars_query,
    ) -> PredictionResponse:
        return holder.engine.predict(
            [request.flow],
            profile=_resolve_profile(profile),
            explain=explain,
            exemplars=exemplars,
        )[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(
        request: BatchRequest,
        profile: str | None = Query(default=None, description="Threshold profile."),
        explain: bool = explain_query,
        exemplars: bool = exemplars_query,
    ) -> BatchResponse:
        if len(request.flows) > settings.serving.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"batch too large: {len(request.flows)} > {settings.serving.max_batch_size}",
            )
        predictions = holder.engine.predict(
            request.flows, profile=_resolve_profile(profile), explain=explain, exemplars=exemplars
        )
        return BatchResponse(predictions=predictions)

    @app.post("/admin/reload", response_model=ReloadResponse)
    def reload_model(request: ReloadRequest) -> ReloadResponse:
        """Swap in a candidate bundle, but only if it reproduces its canaries here.

        The candidate is loaded into a fresh engine (which replays its embedded
        canaries at construction); a canary mismatch means this runtime does not
        reproduce the model that was validated, so the swap is refused with 409 and
        the current model keeps serving. `verify` gates the bytes offline; this
        gates the *behaviour* at the moment of the swap.
        """
        if not settings.serving.reload_enabled:
            raise HTTPException(status_code=404, detail="hot reload is disabled")
        # Resolve under models_dir and refuse anything escaping it (no arbitrary loads).
        models_dir = settings.paths.models_dir.resolve()
        candidate = (models_dir / request.bundle).resolve()
        if models_dir not in candidate.parents and candidate != models_dir:
            raise HTTPException(status_code=400, detail="bundle must live under the models dir")
        if not candidate.exists():
            M.MODEL_RELOADS.labels("not_found").inc()
            raise HTTPException(status_code=404, detail=f"no bundle at {request.bundle!r}")
        # Load with canary_strict off so a failing canary surfaces as a 409 here,
        # not a construction error; the explicit gate below makes the decision.
        probe = settings.model_copy(deep=True)
        probe.serving.canary_strict = False
        try:
            candidate_engine = InferenceEngine(probe, bundle_path=candidate)
        except Exception as exc:
            M.MODEL_RELOADS.labels("load_failed").inc()
            logger.warning("Reload candidate failed to load: %s", exc)
            raise HTTPException(status_code=422, detail=f"candidate failed to load: {exc}") from exc
        canary = candidate_engine.canary
        if canary.present and not canary.ok:
            M.MODEL_RELOADS.labels("canary_failed").inc()
            logger.error("Reload rejected on canary: %s", canary.message)
            raise HTTPException(status_code=409, detail=f"candidate rejected: {canary.message}")
        holder.engine = candidate_engine  # atomic swap; in-flight requests keep the old one
        M.MODEL_RELOADS.labels("promoted").inc()
        logger.info(
            "Model hot-reloaded",
            extra={"bundle": candidate.name, "version": candidate_engine.version},
        )
        return ReloadResponse(
            reloaded=True,
            model_version=candidate_engine.version,
            canary=_canary_status(candidate_engine),
            detail=f"now serving {candidate.name}",
        )

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        payload, content_type = M.render_latest()
        return Response(content=payload, media_type=content_type)

    return app
