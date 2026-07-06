"""Prometheus instrumentation: request/error counters and a latency histogram.

Collectors are module-level singletons (registered once per process) so importing
this module repeatedly — e.g. building the app in several tests — is safe.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "netsentry_requests_total", "Total HTTP requests", ["endpoint", "method", "status"]
)
ERROR_COUNT = Counter("netsentry_errors_total", "Total request errors", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "netsentry_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

# Model-behaviour metrics, so the dashboard shows what the model is *doing*, not
# just HTTP health. Bounded cardinality (a fixed handful of series).
PREDICTIONS = Counter(
    "netsentry_predictions_total", "Scored flows by thresholded decision", ["decision"]
)
ANOMALIES = Counter("netsentry_anomalies_total", "Flows flagged anomalous by the detector")
ATTACK_PROBABILITY = Histogram(
    "netsentry_attack_probability",
    "Distribution of the (calibrated) attack probability of scored flows",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
)

# Shadow-challenger metrics: the champion answers, the shadow is only measured.
# Score-delta buckets + a decision-disagreement counter are the live paired
# evidence a promotion decision wants, collected on real traffic.
SHADOW_SCORED = Counter(
    "netsentry_shadow_scored_total", "Flows also scored by the shadow challenger"
)
SHADOW_DISAGREEMENTS = Counter(
    "netsentry_shadow_disagreements_total",
    "Flows where champion and shadow decisions differ at the active profile",
)
SHADOW_SCORE_DELTA = Histogram(
    "netsentry_shadow_score_delta",
    "Absolute champion-vs-shadow calibrated attack-probability delta",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0),
)

# Feature-drift gauges, refreshed once per rolling window of served flows (see
# netsentry.monitoring). Bounded cardinality: two series, no per-feature labels.
FEATURE_DRIFT_PSI_MAX = Gauge(
    "netsentry_feature_drift_psi_max",
    "Max per-feature PSI of the last served window vs the training reference",
)
FEATURE_DRIFT_PSI_MEAN = Gauge(
    "netsentry_feature_drift_psi_mean",
    "Mean per-feature PSI of the last served window vs the training reference",
)


def render_latest() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
