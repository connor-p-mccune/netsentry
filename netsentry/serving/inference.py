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

from netsentry.data.schema import DESTINATION_PORT
from netsentry.data.services import PER_SERVICE_PROFILE, service_of
from netsentry.evaluation.metrics import attack_probability
from netsentry.explain.exemplars import ExemplarIndex
from netsentry.explain.shap_explainer import ShapExplainer
from netsentry.features.feature_sets import model_features
from netsentry.intel.attack_mapping import mitre_payload
from netsentry.log import get_logger
from netsentry.models.registry import latest_bundle, load_bundle
from netsentry.monitoring.monitor import DriftMonitor
from netsentry.serving.canary import CanaryResult, run_canary
from netsentry.serving.schemas import FeatureContribution, PredictionResponse, SimilarFlow

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)


def resolve_service_thresholds(
    flows: list[dict[str, float]], config: dict[str, object], fallback: float
) -> list[float]:
    """Per-flow decision thresholds for the ``per_service`` profile.

    A flow's ``Destination Port`` — accepted as request metadata, never a model
    feature — selects its service's validation-calibrated threshold. Flows that omit
    the port, and services the bundle has no calibrated entry for (thin or one-class
    validation traffic), fall back to the profile's global threshold, so the profile
    degrades to the global cut rather than misrouting.
    """
    table = config.get("thresholds")
    lookup: dict[str, float] = dict(table) if isinstance(table, dict) else {}
    raw_default = config.get("global", fallback)
    default = float(raw_default) if isinstance(raw_default, (int, float)) else fallback
    thresholds: list[float] = []
    for flow in flows:
        port = flow.get(DESTINATION_PORT)
        if port is None:
            thresholds.append(default)
            continue
        threshold = lookup.get(service_of(port))
        thresholds.append(float(threshold) if threshold is not None else default)
    return thresholds


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
        conformal = meta.get("conformal")
        self.conformal: dict[str, float] | None = conformal if isinstance(conformal, dict) else None
        self.drift = self._build_monitor(settings)
        self.exemplar_index = self._load_exemplars()
        self.canary = self._run_canary(settings)
        self.shadow, self.shadow_version = self._load_shadow(settings)
        self.loaded_at = datetime.now(UTC).isoformat()
        logger.info(
            "Loaded model bundle",
            extra={"path": str(path), "version": self.version, "classes": len(self.bundle.classes)},
        )

    def _run_canary(self, settings: Settings) -> CanaryResult:
        """Replay the bundle's behavioral canaries in this runtime, at load time.

        A failing canary means this environment does not reproduce the scores the
        bundle produced when it was validated (env skew, partial load). Strict mode
        refuses to serve — the fail-fast a deployment wants; otherwise the failure
        is loud in logs and visible on /health so probes can pull the pod.
        """
        result = run_canary(self.bundle)
        if result.present and not result.ok:
            logger.error("Model canary FAILED: %s", result.message)
            if settings.serving.canary_strict:
                raise RuntimeError(f"refusing to serve: {result.message}")
        else:
            logger.info("Model canary: %s", result.message)
        return result

    def _load_shadow(self, settings: Settings) -> tuple[ModelBundle | None, str | None]:
        """Load the optional shadow challenger; a bad shadow must never block serving."""
        path = settings.serving.shadow_artifact_path
        if path is None:
            return None, None
        try:
            shadow = load_bundle(Path(path))
            version = str(shadow.metadata.get("version", "?"))
            logger.info("Loaded shadow challenger", extra={"path": str(path), "version": version})
            return shadow, version
        except Exception as exc:
            logger.warning("Shadow challenger disabled (%s)", exc)
            return None, None

    def _observe_shadow(
        self, frame: pd.DataFrame, champion_probs: np.ndarray, decisions: list[bool], profile: str
    ) -> None:
        """Score the same flows through the shadow and record disagreement metrics.

        The shadow never touches the response: it is measured, not consulted. Each
        model is judged at its *own* threshold for the active profile (thresholds
        live on each bundle's own calibrated scale), which is the deployment-faithful
        comparison — the same shape `netsentry promote` uses offline. Per-service
        routing is a champion-only refinement; the shadow uses its global threshold.
        A failing shadow disables itself rather than taxing every request.
        """
        if self.shadow is None:
            return
        try:
            shadow_probs = self.shadow.attack_scores(frame)
            fallback = self.shadow.thresholds.get(self.default_profile, 0.5)
            threshold = float(self.shadow.thresholds.get(profile, fallback))
            from netsentry.serving import metrics as M

            disagreements = 0
            for i, champion_decision in enumerate(decisions):
                M.SHADOW_SCORE_DELTA.observe(abs(float(champion_probs[i]) - float(shadow_probs[i])))
                if bool(shadow_probs[i] >= threshold) != champion_decision:
                    disagreements += 1
            M.SHADOW_SCORED.inc(len(decisions))
            if disagreements:
                M.SHADOW_DISAGREEMENTS.inc(disagreements)
        except Exception as exc:  # the shadow must never break or slow the champion
            logger.warning("Shadow scoring disabled after failure (%s)", exc)
            self.shadow = None

    def _load_exemplars(self) -> ExemplarIndex | None:
        """Reconstruct the bundle's case base, if it carries one (best-effort)."""
        payload = self.bundle.metadata.get("exemplars")
        if not isinstance(payload, dict):
            return None
        try:
            return ExemplarIndex.from_payload(payload)
        except Exception as exc:  # exemplars are additive; never fatal
            logger.warning("Exemplar index disabled (%s)", exc)
            return None

    def _similar_flows(self, transformed: np.ndarray) -> list[list[SimilarFlow]] | None:
        """Nearest known training flows per row — case-based evidence, best-effort."""
        if self.exemplar_index is None:
            return None
        try:
            index = self.exemplar_index
            distances, idx = index.query(np.asarray(transformed), self.settings.exemplars.k)
            return [
                [
                    SimilarFlow(
                        label=str(index.labels[j]),
                        day=str(index.days[j]),
                        distance=round(float(d), 4),
                    )
                    for d, j in zip(distances[i], idx[i], strict=True)
                ]
                for i in range(distances.shape[0])
            ]
        except Exception as exc:  # retrieval must never break a prediction
            logger.warning("Exemplar retrieval skipped (%s)", exc)
            return None

    def _build_monitor(self, settings: Settings) -> DriftMonitor | None:
        """Reconstruct the drift monitor from the bundle's reference, if it carries one."""
        summary = self.bundle.metadata.get("drift_reference")
        if not isinstance(summary, dict):
            return None
        try:
            return DriftMonitor.from_summary(
                summary,
                window=settings.monitoring.serving_window,
                moderate=settings.monitoring.psi_moderate,
                major=settings.monitoring.psi_major,
            )
        except Exception as exc:  # monitoring is best-effort, never fatal
            logger.warning("Drift monitor disabled (%s)", exc)
            return None

    def _observe_drift(self, frame: pd.DataFrame) -> None:
        """Feed served flows to the rolling drift monitor and export gauges."""
        if self.drift is None:
            return
        try:
            report = self.drift.observe(frame)
            if report is not None:
                from netsentry.serving import metrics as M

                M.FEATURE_DRIFT_PSI_MAX.set(report.max_psi)
                M.FEATURE_DRIFT_PSI_MEAN.set(report.mean_psi)
        except Exception as exc:  # drift monitoring must never break a prediction
            logger.warning("Drift observation skipped (%s)", exc)

    def _record_metrics(
        self, attack_prob: float, attacking: bool, flagged_anomaly: bool | None
    ) -> None:
        """Emit per-flow model-behaviour metrics (best-effort, never fatal)."""
        try:
            from netsentry.serving import metrics as M

            M.ATTACK_PROBABILITY.observe(attack_prob)
            M.PREDICTIONS.labels("attack" if attacking else "benign").inc()
            if flagged_anomaly:
                M.ANOMALIES.inc()
        except Exception as exc:  # metrics must never break a prediction
            logger.warning("Metric emission skipped (%s)", exc)

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
        explain: bool = True,
        exemplars: bool = False,
    ) -> list[PredictionResponse]:
        """Score flows; ``explain=False`` skips SHAP (the measured majority of latency).

        Explanations are the default because they are part of the product
        contract; the opt-out exists for throughput-bound callers (bulk scoring,
        load tests) and returns an empty ``top_features`` rather than a changed
        schema, so clients need no second response model. ``exemplars=True``
        opts *in* to case-based evidence (``similar_flows``: the nearest known
        training flows) when the bundle carries an exemplar index.
        """
        profile = profile or self.default_profile
        top_k = top_k or self.settings.serving.top_k_features
        frame = self._frame(flows)
        self._observe_drift(frame)

        proba = self.bundle.predict_proba(frame)
        classes = self.bundle.classes
        probs = attack_probability(proba, classes, self.benign)
        if self.bundle.calibrator is not None:
            # Thresholds live on the calibrated scale; calibrate before comparing.
            probs = self.bundle.calibrator.transform(probs)
        threshold = self.bundle.thresholds.get(profile, 0.5)
        service_config = self.bundle.metadata.get("service_thresholds")
        if profile == PER_SERVICE_PROFILE and isinstance(service_config, dict):
            # The parity audit's fix: each flow is judged at its own service's
            # validation-calibrated threshold (global fallback when unmapped).
            row_thresholds = resolve_service_thresholds(flows, service_config, threshold)
        else:
            row_thresholds = [threshold] * len(flows)
        decisions = [bool(float(probs[i]) >= row_thresholds[i]) for i in range(len(flows))]
        self._observe_shadow(frame, probs, decisions, profile)
        argmax = classes[proba.argmax(axis=1)]

        anomaly_scores = is_anomaly = None
        transformed = None
        if self.bundle.anomaly_detector is not None or (
            exemplars and self.exemplar_index is not None
        ):
            transformed = self.bundle.pipeline.transform(frame)
        if self.bundle.anomaly_detector is not None and transformed is not None:
            anomaly_scores = self.bundle.anomaly_detector.score(transformed)
            is_anomaly = anomaly_scores >= (self.bundle.anomaly_threshold or float("inf"))
        similar = (
            self._similar_flows(transformed) if exemplars and transformed is not None else None
        )

        responses: list[PredictionResponse] = []
        for i in range(len(flows)):
            attack_prob = float(probs[i])
            attacking = decisions[i]
            flagged_anomaly = bool(is_anomaly[i]) if is_anomaly is not None else None
            self._record_metrics(attack_prob, attacking, flagged_anomaly)
            predicted = self._predicted_class(str(argmax[i]), proba[i], classes, attacking)
            pred_set, action = self._conformal_set(attack_prob)
            mitre = mitre_payload(predicted) if attacking else None
            top = (
                [
                    FeatureContribution(feature=name, contribution=value)
                    for name, value in self.explainer.explain_row(frame.iloc[[i]], top_k)
                ]
                if explain
                else []
            )
            responses.append(
                PredictionResponse(
                    predicted_class=predicted,
                    is_attack=attacking,
                    attack_probability=attack_prob,
                    anomaly_score=float(anomaly_scores[i]) if anomaly_scores is not None else None,
                    is_anomaly=flagged_anomaly,
                    top_features=top,
                    model_version=self.version,
                    threshold_profile=profile,
                    prediction_set=pred_set,
                    recommended_action=action,
                    mitre=mitre,
                    similar_flows=similar[i] if similar is not None else None,
                )
            )
        return responses

    def _conformal_set(self, attack_prob: float) -> tuple[list[str] | None, str | None]:
        """Conformal prediction set + recommended SOC action for one calibrated score.

        Sets map to actions: a single label is auto-decided; an ambiguous (both) or
        empty (neither, i.e. novel) set is routed to a human.
        """
        if self.conformal is None:
            return None, None
        in_benign = attack_prob <= self.conformal["tau_benign"]
        in_attack = (1.0 - attack_prob) <= self.conformal["tau_attack"]
        members = [
            label for label, present in ((self.benign, in_benign), ("attack", in_attack)) if present
        ]
        if in_attack and not in_benign:
            action = "auto_alert"
        elif in_benign and not in_attack:
            action = "auto_clear"
        else:  # ambiguous (both) or novel (empty) -> escalate
            action = "review"
        return members, action

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
