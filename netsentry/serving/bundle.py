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
import pandas as pd

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.schema import DAY_COLUMN, DESTINATION_PORT
from netsentry.data.services import PER_SERVICE_PROFILE, service_of
from netsentry.data.split import load_split
from netsentry.evaluation.conformal import class_conditional_thresholds
from netsentry.evaluation.cost import cost_optimal_threshold
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
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
    _attach_service_thresholds(bundle, variant, val, p_val, y_val_bin)
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
        # Per-feature benign median (transformed space) — the occlusion reference the
        # serving engine uses to attribute an anomaly flag (?anomaly_explain=true).
        benign_reference = np.median(np.asarray(bundle.pipeline.transform(benign_train)), axis=0)
        bundle.metadata["anomaly_reference"] = [float(v) for v in benign_reference]

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

    _attach_exemplars(bundle, variant, train)
    _attach_canary(bundle, variant, val)

    path = settings.paths.models_dir / SERVING_BUNDLE_NAME
    bundle.save(path)
    logger.info("Built serving bundle", extra={"path": str(path), "classes": len(result.classes)})
    return path


def _attach_exemplars(bundle: ModelBundle, settings: Settings, train: pd.DataFrame) -> None:
    """Embed a compact, class-balanced case base for ``?exemplars=true`` responses.

    The index lives in the fitted pipeline's standardized space (the same
    transform every prediction already runs through), so retrieval at serve time
    is a matrix product against a few hundred float32 rows per class. Additive:
    a failure to embed must never break the bundle build.
    """
    try:
        from netsentry.explain.exemplars import build_exemplar_index

        days = (
            train[DAY_COLUMN].to_numpy()
            if DAY_COLUMN in train.columns
            else np.full(len(train), "?")
        )
        index = build_exemplar_index(
            bundle.pipeline.transform(train),
            train[MULTICLASS_TARGET].to_numpy(),
            days,
            settings.exemplars.per_class,
            settings.seed,
        )
        bundle.metadata["exemplars"] = index.to_payload()
        logger.info("Attached exemplar index", extra={"exemplars": len(index.labels)})
    except Exception as exc:  # exemplars are additive; never fatal
        logger.warning("Exemplar index skipped (%s)", exc)


def _attach_canary(bundle: ModelBundle, settings: Settings, val: pd.DataFrame) -> None:
    """Embed a class-mixed sample of validation flows as behavioral canaries.

    Half benign, half attack (deterministic head-of-split selection), so the canary
    exercises both score regimes. Additive: a failure to embed must never break the
    bundle build.
    """
    try:
        from netsentry.serving.canary import embed_canary

        benign = settings.labels.benign_label
        half = max(settings.serving.canary_rows // 2, 1)
        sample = pd.concat(
            [
                val[val[MULTICLASS_TARGET] == benign].head(half),
                val[val[MULTICLASS_TARGET] != benign].head(half),
            ]
        )
        if sample.empty:
            logger.warning("Canary skipped (no validation rows to sample)")
            return
        embed_canary(bundle, sample, settings)
    except Exception as exc:  # canaries are additive; never fatal
        logger.warning("Canary skipped (%s)", exc)


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


def _attach_service_thresholds(
    bundle: ModelBundle,
    settings: Settings,
    val: pd.DataFrame,
    p_val: np.ndarray,
    y_val_bin: np.ndarray,
) -> None:
    """Per-service operating thresholds — the parity audit's finding, productised.

    The subgroups report shows a single global threshold only constrains the
    *aggregate* FPR; nothing pins any individual service's queue to the budget. This
    computes, on the same calibrated validation scores every other profile uses, a
    threshold per service (grouped by ``Destination Port``, which never enters a
    prediction) at the primary FPR target. Services with thin support or one-class
    validation traffic fall back to the global threshold. Selecting the profile is
    the operator's choice: ``?profile=per_service``.
    """
    try:
        ports = val[DESTINATION_PORT].to_numpy()
        if len(ports) != len(p_val):
            raise ValueError("validation frame and scores are misaligned")
        target = settings.thresholds.primary_fpr
        min_support = settings.subgroups.min_support
        global_threshold = threshold_at_fpr(y_val_bin, p_val, target)
        services = np.array([service_of(p) for p in ports])
        table: dict[str, float] = {}
        for service in sorted(set(services.tolist())):
            mask = services == service
            if int(mask.sum()) < min_support or len(np.unique(y_val_bin[mask])) < 2:
                continue  # thin or one-class service: the global threshold serves it
            threshold = float(threshold_at_fpr(y_val_bin[mask], p_val[mask], target))
            if not np.isfinite(threshold):
                # No finite threshold meets the budget at this support (roc_curve's
                # inf sentinel): storing it would silently disable detection for the
                # service (and break strict JSON) — the global cut serves it instead.
                continue
            table[service] = threshold
        bundle.thresholds[PER_SERVICE_PROFILE] = float(global_threshold)
        bundle.metadata["service_thresholds"] = {
            "target_fpr": target,
            "min_support": min_support,
            "global": float(global_threshold),
            "thresholds": table,
        }
        logger.info(
            "Attached per-service thresholds",
            extra={"services": len(table), "target_fpr": target},
        )
    except Exception as exc:  # an extra profile must never break the build
        logger.warning("Per-service profile skipped (%s)", exc)
