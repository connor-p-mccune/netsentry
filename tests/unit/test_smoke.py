"""Trivial import/version smoke test so CI has a green baseline from Phase 0."""

from __future__ import annotations

import netsentry


def test_version_is_a_string() -> None:
    assert isinstance(netsentry.__version__, str)
    assert netsentry.__version__


def test_get_logger_is_usable() -> None:
    logger = netsentry.get_logger("netsentry.tests")
    logger.info("smoke")  # must not raise
