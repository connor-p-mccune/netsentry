"""Build the deployable serving bundle.

The served model names attacks, so it is the **multiclass** model trained on the
stratified split (all classes appear in training). A benign-fit Isolation Forest
is attached for the anomaly score. The honest temporal-split numbers remain the
reported headline (see the evaluation report) — this bundle is the namer/scorer
the API serves.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation.conformal import class_conditional_thresholds
from netsentry.evaluation.cost import cost_optimal_threshold
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.feature_sets import model_features
from netsentry.log import get_logger
from netsentry.models.anomaly import build_anomaly_detector
from netsentry.models.registry import ModelBundle
from netsentry.monitoring.monitor import reference_summary
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

SERVING_BUNDLE_NAME = "serving_bundle.joblib"


def build_serving_bundle(settings: Settings) -> Path:
    """Train the multiclass model, attach a benign anomaly detector, and save."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "multiclass"
    variant.mlflow.enabled = False

    result = fit_supervised(variant)
    bundle: ModelBundle = result.bundle

    # Validation attack probabilities (calibrated if the bundle carries a calibrator),
    # used to add a cost-optimal threshold profile and a conformal prediction set —
    # so the live API can serve both, not just the offline reports. The stratified
    # split is exchangeable, so the conformal coverage guarantee is valid here.
    benign = variant.labels.benign_label
    p_val = attack_probability(result.proba_val, result.classes, benign)
    if bundle.calibrator is not None:
        p_val = bundle.calibrator.transform(p_val)
    y_val_bin = (result.y_val.astype(str) != benign).astype(int)
    _attach_operating_profiles(bundle, variant, p_val, y_val_bin)

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    benign_train = train[train[MULTICLASS_TARGET] == benign]
    benign_val = val[val[MULTICLASS_TARGET] == benign]
    if len(benign_train) and len(benign_val):
        detector = build_anomaly_detector(variant, "iforest")
        detector.fit(bundle.pipeline.transform(benign_train))
        detector.calibrate_threshold(
            bundle.pipeline.transform(benign_val), variant.anomaly.target_fpr
        )
        bundle.anomaly_detector = detector
        bundle.anomaly_threshold = detector.threshold

    # Carry a compact drift reference (per-feature PSI bins) so the deployed model
    # can self-monitor input drift at serve time without the processed dataset.
    stored_cols = bundle.metadata.get("input_columns")
    feature_cols = list(stored_cols) if isinstance(stored_cols, list) else model_features(variant)
    reference = train.sample(
        min(len(train), variant.monitoring.reference_rows), random_state=variant.seed
    )
    bundle.metadata["drift_reference"] = reference_summary(
        reference, feature_cols, bins=variant.monitoring.psi_bins
    )

    path = settings.paths.models_dir / SERVING_BUNDLE_NAME
    bundle.save(path)
    logger.info("Built serving bundle", extra={"path": str(path), "classes": len(result.classes)})
    return path


def _attach_operating_profiles(
    bundle: ModelBundle, settings: Settings, p_val: np.ndarray, y_val_bin: np.ndarray
) -> None:
    """Add a cost-optimal threshold profile and conformal set thresholds to the bundle.

    Both are calibrated on validation so the live API can serve a cost-optimal
    decision and a conformal prediction set, not just the offline reports.
    """
    cost = settings.cost
    try:
        threshold, _ = cost_optimal_threshold(
            y_val_bin,
            p_val,
            cost.production_attack_rate,
            cost.cost_per_alert,
            cost.cost_per_miss,
            cost.grid_points,
        )
        bundle.thresholds["cost_optimal"] = threshold
    except Exception as exc:  # an extra profile must never break the build
        logger.warning("Cost-optimal profile skipped (%s)", exc)

    try:
        tau_benign, tau_attack = class_conditional_thresholds(
            p_val, y_val_bin, settings.conformal.alpha
        )
        bundle.metadata["conformal"] = {
            "alpha": settings.conformal.alpha,
            "tau_benign": float(tau_benign),
            "tau_attack": float(tau_attack),
        }
    except Exception as exc:  # conformal is additive; never fatal
        logger.warning("Conformal thresholds skipped (%s)", exc)
