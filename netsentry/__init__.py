"""NetSentry: a leakage-safe ML network intrusion detection system.

The package pairs a supervised classifier for known attacks with an unsupervised
anomaly detector for novel ones, evaluated honestly (temporal split, operational
metrics) and served behind an explainable API. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from netsentry.log import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = ["__version__", "configure_logging", "get_logger"]
