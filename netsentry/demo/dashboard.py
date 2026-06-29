"""Streamlit demo dashboard for NetSentry.

Pick a sample flow (or edit the headline features) and see the model's verdict
live: predicted class, attack probability, the thresholded decision, anomaly
score, and the SHAP features driving it. Launch with ``netsentry demo`` or
``streamlit run netsentry/demo/dashboard.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from netsentry.config import load_settings
from netsentry.demo.core import EDITABLE_FEATURES, SAMPLE_FLOWS
from netsentry.serving.inference import InferenceEngine


@st.cache_resource(show_spinner="Loading model bundle…")
def _load_engine() -> InferenceEngine:
    return InferenceEngine(load_settings())


def main() -> None:
    """Render the dashboard (runs top-to-bottom under `streamlit run`)."""
    st.set_page_config(page_title="NetSentry", page_icon="🛡️", layout="centered")
    st.title("🛡️ NetSentry — live flow risk")
    st.caption(
        "Pick or edit a network flow and see the verdict, anomaly score, and the SHAP "
        "features driving it. Numbers reflect the loaded model bundle."
    )

    try:
        engine = _load_engine()
    except FileNotFoundError as exc:
        st.error(
            "No model bundle found — train one first (`netsentry train supervised`).\n\n" f"{exc}"
        )
        return

    st.sidebar.header("Flow input")
    preset = st.sidebar.selectbox("Sample flow", list(SAMPLE_FLOWS))
    flow = dict(SAMPLE_FLOWS[preset])
    st.sidebar.caption("Adjust the headline features:")
    for feature in EDITABLE_FEATURES:
        flow[feature] = st.sidebar.number_input(feature, value=float(flow.get(feature, 0.0)))
    profiles = sorted(engine.bundle.thresholds) or [engine.default_profile]
    profile = st.sidebar.selectbox("Threshold profile", profiles)

    resp = engine.predict([flow], profile=profile)[0]

    left, mid, right = st.columns(3)
    left.metric("Decision", "ATTACK" if resp.is_attack else "benign")
    mid.metric("Attack probability", f"{resp.attack_probability:.1%}")
    right.metric("Predicted class", resp.predicted_class)
    if resp.anomaly_score is not None:
        state = "flagged" if resp.is_anomaly else "normal"
        st.metric("Anomaly", f"{state} (score {resp.anomaly_score:.2f})")

    st.subheader("Why — top SHAP contributions")
    contributions = pd.DataFrame(
        [{"feature": c.feature, "contribution": c.contribution} for c in resp.top_features]
    ).set_index("feature")
    st.bar_chart(contributions)
    st.caption(f"model {resp.model_version} · profile {resp.threshold_profile}")


main()
