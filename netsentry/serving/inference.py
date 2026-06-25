"""Inference engine: load the bundle once and score flows.

Returns the full contract per flow: predicted class, attack probability, an
is-attack decision at the selected threshold profile, an optional anomaly score,
and the SHAP top contributing features.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.evaluation.metrics import attack_probability
from netsentry.explain.shap_explainer import ShapExplainer
from netsentry.features.feature_sets import model_features
from netsentry.log import get_logger
from netsentry.models.registry import latest_bundle, load_bundle
from netsentry.serving.schemas import FeatureContribution, PredictionResponse

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)


class InferenceEngine:
    """Holds the loaded pipeline+model bundle and serves predictions."""

    def __init__(self, settings: Settings, bundle_path: Path | None = None) -> None:
        path = bundle_path or settings.serving.artifact_path or latest_bundle(settings)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                "No model bundle found. Build one with the serving bundle builder "
                "(or `netsentry train`), or set serving.artifact_path."
            )
        self.settings = settings
        self.bundle = load_bundle(Path(path))
        self.explainer = ShapExplainer(self.bundle)
        meta = self.bundle.metadata
        stored_columns = meta.get("input_columns")
        self.input_columns: list[str] = (
            list(stored_columns) if isinstance(stored_columns, list) else model_features(settings)
        )
        self.benign = str(meta.get("benign_label", "BENIGN"))
        self.version = str(meta.get("version", "0"))
        self.default_profile = str(
            meta.get("default_threshold_profile", settings.serving.default_threshold_profile)
        )
        self.loaded_at = datetime.now(UTC).isoformat()
        logger.info(
            "Loaded model bundle",
            extra={"path": str(path), "version": self.version, "classes": len(self.bundle.classes)},
        )

    def _frame(self, flows: list[dict[str, float]]) -> pd.DataFrame:
        """Build a feature frame with every expected column (missing -> NaN)."""
        rows = [{col: flow.get(col, np.nan) for col in self.input_columns} for flow in flows]
        return pd.DataFrame(rows, columns=self.input_columns)

    def predict(
        self,
        flows: list[dict[str, float]],
        *,
        profile: str | None = None,
        top_k: int | None = None,
    ) -> list[PredictionResponse]:
        profile = profile or self.default_profile
        top_k = top_k or self.settings.serving.top_k_features
        frame = self._frame(flows)

        proba = self.bundle.predict_proba(frame)
        classes = self.bundle.classes
        probs = attack_probability(proba, classes, self.benign)
        threshold = self.bundle.thresholds.get(profile, 0.5)
        argmax = classes[proba.argmax(axis=1)]

        anomaly_scores = is_anomaly = None
        if self.bundle.anomaly_detector is not None:
            transformed = self.bundle.pipeline.transform(frame)
            anomaly_scores = self.bundle.anomaly_detector.score(transformed)
            is_anomaly = anomaly_scores >= (self.bundle.anomaly_threshold or float("inf"))

        responses: list[PredictionResponse] = []
        for i in range(len(flows)):
            attack_prob = float(probs[i])
            attacking = attack_prob >= threshold
            predicted = self._predicted_class(str(argmax[i]), proba[i], classes, attacking)
            top = [
                FeatureContribution(feature=name, contribution=value)
                for name, value in self.explainer.explain_row(frame.iloc[[i]], top_k)
            ]
            responses.append(
                PredictionResponse(
                    predicted_class=predicted,
                    is_attack=attacking,
                    attack_probability=attack_prob,
                    anomaly_score=float(anomaly_scores[i]) if anomaly_scores is not None else None,
                    is_anomaly=bool(is_anomaly[i]) if is_anomaly is not None else None,
                    top_features=top,
                    model_version=self.version,
                    threshold_profile=profile,
                )
            )
        return responses

    def _predicted_class(
        self, argmax_label: str, proba_row: np.ndarray, classes: np.ndarray, attacking: bool
    ) -> str:
        """Resolve the label so it agrees with the is-attack decision.

        ``is_attack`` is the actionable thresholded decision; ``predicted_class``
        names what it is when flagged and is benign otherwise (``attack_probability``
        is reported separately for transparency).
        """
        if not attacking:
            return self.benign
        if argmax_label in {"0", "1"}:  # binary model: positive class is "attack"
            return "attack"
        if argmax_label == self.benign:
            # Decision says attack but argmax is benign: report the top attack class.
            for idx in np.argsort(proba_row)[::-1]:
                if str(classes[idx]) != self.benign:
                    return str(classes[idx])
        return argmax_label
