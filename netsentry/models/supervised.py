"""Supervised classifiers: trivial baselines plus a gradient-boosted model.

The boosted model prefers **LightGBM** and falls back to scikit-learn's
``HistGradientBoostingClassifier`` when LightGBM is unavailable, so the pipeline
runs anywhere. Imbalance is handled with balanced sample weights (not default
resampling), early stopping uses the validation set, and seeding is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from netsentry.log import get_logger
from netsentry.models.base import BaseModel
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.base import EvalSet

logger = get_logger(__name__)


def resolve_backend(settings: Settings) -> str:
    """Pick the boosting backend, honouring config and availability."""
    backend = settings.supervised.backend
    if backend in {"auto", "lightgbm"} and is_available("lightgbm"):
        return "lightgbm"
    if backend == "lightgbm":
        logger.warning("LightGBM requested but not installed; using hist_gbdt fallback")
    return "hist_gbdt"


def build_baselines(settings: Settings) -> dict[str, Any]:
    """Trivial reference models every real number must beat."""
    class_weight = "balanced" if settings.supervised.class_weight == "balanced" else None
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=settings.seed,
        ),
    }


class SupervisedClassifier(BaseModel):
    """Gradient-boosted classifier with imbalance handling and early stopping."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = resolve_backend(settings)
        self.model: Any = None
        self.classes_: np.ndarray = np.empty(0)

    def _build_estimator(self) -> Any:
        cfg = self.settings.supervised
        if self.backend == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                max_depth=cfg.max_depth,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                min_child_samples=cfg.min_child_samples,
                reg_lambda=cfg.reg_lambda,
                random_state=self.settings.seed,
                n_jobs=cfg.n_jobs,
                deterministic=True,
                force_row_wise=True,
                verbosity=-1,
            )
        # scikit-learn fallback: self-validating internal early stopping.
        return HistGradientBoostingClassifier(
            max_iter=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.num_leaves,
            min_samples_leaf=cfg.min_child_samples,
            l2_regularization=cfg.reg_lambda,
            random_state=self.settings.seed,
            early_stopping=True,
            validation_fraction=self.settings.split.val_size,
            n_iter_no_change=max(10, cfg.early_stopping_rounds // 2),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: EvalSet | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> SupervisedClassifier:
        """Fit, balancing classes; optional per-row ``sample_weight`` (e.g. weak-label
        confidence) multiplies into the balanced weight rather than replacing it."""
        self.model = self._build_estimator()
        balanced = (
            compute_sample_weight("balanced", y)
            if self.settings.supervised.class_weight == "balanced"
            else None
        )
        if sample_weight is not None:
            extra = np.asarray(sample_weight, dtype=float)
            sample_weight = extra * balanced if balanced is not None else extra
        else:
            sample_weight = balanced
        if self.backend == "lightgbm" and eval_set is not None:
            import lightgbm as lgb

            self.model.fit(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=[eval_set],
                callbacks=[
                    lgb.early_stopping(
                        self.settings.supervised.early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(0),
                ],
            )
        else:
            self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))
