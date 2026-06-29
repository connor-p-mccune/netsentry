"""Data layer for the Streamlit demo: sample flows and a thin predict helper.

Kept free of Streamlit so it stays importable and unit-testable; ``dashboard.py``
owns all the UI. Sample-flow keys are CIC feature names; omitted features are
imputed by the fitted pipeline at scoring time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.serving.inference import InferenceEngine
    from netsentry.serving.schemas import PredictionResponse

SAMPLE_FLOWS: dict[str, dict[str, float]] = {
    "Benign web browsing": {
        "Flow Duration": 1_200_000.0,
        "Total Fwd Packets": 8.0,
        "Total Backward Packets": 6.0,
        "Flow Packets/s": 45.0,
        "Flow Bytes/s": 1300.0,
        "Average Packet Size": 180.0,
    },
    "DoS-like (high rate)": {
        "Flow Duration": 40_000.0,
        "Total Fwd Packets": 240.0,
        "Total Backward Packets": 180.0,
        "Flow Packets/s": 6000.0,
        "Flow Bytes/s": 90000.0,
        "Average Packet Size": 60.0,
    },
    "DDoS-like (volumetric)": {
        "Flow Duration": 25_000.0,
        "Total Fwd Packets": 120.0,
        "Total Backward Packets": 400.0,
        "Flow Packets/s": 9000.0,
        "Flow Bytes/s": 120000.0,
        "Average Packet Size": 75.0,
    },
    "Port-scan-like (short)": {
        "Flow Duration": 800.0,
        "Total Fwd Packets": 2.0,
        "Total Backward Packets": 0.0,
        "Flow Packets/s": 2500.0,
        "Flow Bytes/s": 0.0,
        "SYN Flag Count": 1.0,
    },
}

# Features surfaced as editable sliders/inputs in the dashboard sidebar.
EDITABLE_FEATURES: tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Flow Bytes/s",
    "Average Packet Size",
)


def predict_flow(engine: InferenceEngine, flow: dict[str, float]) -> PredictionResponse:
    """Score one flow with the loaded engine (the dashboard's data layer)."""
    return engine.predict([flow])[0]
