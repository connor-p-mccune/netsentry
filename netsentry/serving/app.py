"""FastAPI application factory.

Wires /health, /predict, /predict/batch, /metrics, request-logging + latency
middleware, and a selectable threshold profile. Implemented in Phase 8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from netsentry.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI app."""
    raise NotImplementedError("Implemented in Phase 8")
