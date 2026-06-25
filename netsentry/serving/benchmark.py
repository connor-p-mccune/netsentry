"""Load-test the inference API and report latency percentiles + throughput.

Concrete latency figures read as production awareness, so this drives /predict
with a representative flow and reports p50/p95/p99 and requests/second.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data import schema
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)


def _sample_payload(settings: Settings) -> dict[str, dict[str, float]]:
    from netsentry.data.synthetic import generate_synthetic

    row = generate_synthetic(settings, rows=1, seed=settings.seed).iloc[0]
    flow = {
        col: float(row[col])
        for col in schema.FEATURE_COLUMNS
        if col in row.index and np.isfinite(row[col])
    }
    return {"flow": flow}


def run_benchmark(settings: Settings, *, base_url: str, n_requests: int) -> dict[str, float]:
    """Send ``n_requests`` to ``base_url`` and return latency/throughput stats."""
    import logging

    import httpx

    logging.getLogger("httpx").setLevel(logging.WARNING)  # don't log every request
    payload = _sample_payload(settings)
    latencies_ms: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        client.post(f"{base_url}/predict", json=payload).raise_for_status()  # warm up
        start = time.perf_counter()
        for _ in range(n_requests):
            t0 = time.perf_counter()
            response = client.post(f"{base_url}/predict", json=payload)
            response.raise_for_status()
            latencies_ms.append((time.perf_counter() - t0) * 1e3)
        total = time.perf_counter() - start

    arr = np.array(latencies_ms)
    stats = {
        "requests": float(n_requests),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_rps": float(n_requests / total) if total > 0 else 0.0,
    }
    logger.info(
        "Benchmark complete",
        extra={
            "p50_ms": round(stats["p50_ms"], 2),
            "p95_ms": round(stats["p95_ms"], 2),
            "p99_ms": round(stats["p99_ms"], 2),
            "throughput_rps": round(stats["throughput_rps"], 1),
        },
    )
    return stats
