"""Train the supervised classifier and log everything to MLflow.

Fits the leakage-safe pipeline on the TRAIN split only, trains trivial baselines
plus the gradient-boosted model, evaluates on the held-out test split, saves the
deployable pipeline+model bundle, and logs params/metrics/artifacts/env/seed.
The full operational metric suite lives in Phase 5; here we record enough to
establish a baseline and a determinism check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.registry import ModelBundle, save_bundle
from netsentry.models.supervised import SupervisedClassifier, build_baselines
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)


def _target_column(task: str) -> str:
    return BINARY_TARGET if task == "binary" else MULTICLASS_TARGET


def quick_metrics(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, task: str
) -> dict[str, float]:
    """A small, honest metric set (PR-AUC leads). Full suite arrives in Phase 5."""
    preds = classes[proba.argmax(axis=1)]
    metrics = {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
    }
    if task == "binary":
        pos = int(np.where(classes == 1)[0][0])
        metrics["pr_auc"] = float(average_precision_score((y_true == 1).astype(int), proba[:, pos]))
    else:
        binarized = label_binarize(y_true, classes=list(classes))
        metrics["pr_auc"] = float(average_precision_score(binarized, proba, average="macro"))
    return metrics


def train_supervised(settings: Settings) -> dict[str, Any]:
    """Run supervised training end-to-end; return a result summary."""
    seed_everything(settings.seed)
    strategy = settings.split.strategy
    task = settings.supervised.task
    target = _target_column(task)

    train = load_split(settings, strategy, "train")
    val = load_split(settings, strategy, "val")
    test = load_split(settings, strategy, "test")
    y_train = train[target].to_numpy()
    y_val = val[target].to_numpy()
    y_test = test[target].to_numpy()

    pipeline = build_pipeline(settings)
    x_train = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_val = pipeline.transform(val)
    x_test = pipeline.transform(test)
    logger.info(
        "Prepared features", extra={"n_features": x_train.shape[1], "n_train": len(y_train)}
    )

    baseline_metrics: dict[str, dict[str, float]] = {}
    for name, estimator in build_baselines(settings).items():
        estimator.fit(x_train, y_train)
        baseline_metrics[name] = quick_metrics(
            y_test, estimator.predict_proba(x_test), np.asarray(estimator.classes_), task
        )

    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    metrics = quick_metrics(y_test, model.predict_proba(x_test), model.classes_, task)

    bundle = ModelBundle(
        pipeline=pipeline,
        model=model,
        metadata={
            "version": "0.1.0",
            "task": task,
            "split_strategy": strategy,
            "backend": model.backend,
            "classes": [str(c) for c in model.classes_],
            "n_features": int(x_train.shape[1]),
            "n_train": len(y_train),
            "created_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
        },
    )
    bundle_path = settings.paths.models_dir / f"supervised_{task}_{strategy}.joblib"
    save_bundle(bundle, bundle_path)

    with track_run(settings, f"supervised_{task}_{strategy}") as run:
        run.log_params(
            {
                "task": task,
                "split_strategy": strategy,
                "backend": model.backend,
                "seed": settings.seed,
                "n_features": int(x_train.shape[1]),
                "n_train": len(y_train),
                "class_weight": settings.supervised.class_weight,
                "n_estimators": settings.supervised.n_estimators,
                "learning_rate": settings.supervised.learning_rate,
            }
        )
        run.log_metrics({f"model_{k}": v for k, v in metrics.items()})
        for name, bm in baseline_metrics.items():
            run.log_metrics({f"baseline_{name}_{k}": v for k, v in bm.items()})
        run.log_dict(bundle.metadata, "bundle_metadata.json")
        run.log_artifact(bundle_path)

    # Honesty gate: a near-perfect score on this data almost certainly means leakage.
    if metrics["pr_auc"] > 0.999:
        logger.warning(
            "PR-AUC > 0.999 — treat as a likely leakage bug and investigate before trusting it."
        )

    logger.info(
        "Trained supervised model",
        extra={
            "task": task,
            "split": strategy,
            "pr_auc": round(metrics["pr_auc"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "majority_pr_auc": round(baseline_metrics["majority"]["pr_auc"], 4),
        },
    )
    return {"metrics": metrics, "baselines": baseline_metrics, "bundle": str(bundle_path)}
