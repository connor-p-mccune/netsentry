"""Prometheus instrumentation: request/error counters and a latency histogram.

Collectors are module-level singletons (registered once per process) so importing
this module repeatedly — e.g. building the app in several tests — is safe.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "netsentry_requests_total", "Total HTTP requests", ["endpoint", "method", "status"]
)
ERROR_COUNT = Counter("netsentry_errors_total", "Total request errors", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "netsentry_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)


def render_latest() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
