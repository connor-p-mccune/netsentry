"""Streamlit demo dashboard (data layer in core.py, UI in dashboard.py)."""

from __future__ import annotations

from netsentry.demo.core import EDITABLE_FEATURES, SAMPLE_FLOWS, predict_flow

__all__ = ["EDITABLE_FEATURES", "SAMPLE_FLOWS", "predict_flow"]
