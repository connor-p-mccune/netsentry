"""Prometheus instrumentation: request/error counters and a latency histogram.

Implemented in Phase 8.
"""

from __future__ import annotations


def build_metrics() -> object:
    """Construct the Prometheus collectors used by the serving middleware."""
    raise NotImplementedError("Implemented in Phase 8")
