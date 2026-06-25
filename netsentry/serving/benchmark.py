"""Load-test the inference API and report latency percentiles + throughput.

Implemented in Phase 8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def run_benchmark(settings: Settings, *, base_url: str, n_requests: int) -> dict[str, float]:
    """Send ``n_requests`` to ``base_url`` and return latency/throughput stats."""
    raise NotImplementedError("Implemented in Phase 8")
