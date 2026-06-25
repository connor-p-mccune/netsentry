"""The deployable bundle: fitted pipeline + model + metadata in one artifact.

Bundling the fitted preprocessing pipeline with the model guarantees serve-time
preprocessing is identical to training — eliminating train/serve skew. Serving
loads exactly this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import joblib
import numpy as np

from netsentry.log import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

    from netsentry.config import Settings
    from netsentry.models.base import BaseModel

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """A fitted preprocessing pipeline + model + descriptive metadata."""

    pipeline: Pipeline
    model: BaseModel
    metadata: dict[str, object] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess a raw flow frame and return class probabilities."""
        return np.asarray(self.model.predict_proba(self.pipeline.transform(df)))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess a raw flow frame and return hard predictions."""
        return np.asarray(self.model.predict(self.pipeline.transform(df)))

    def feature_names(self) -> list[str]:
        """Names of the features emitted by the fitted pipeline (post-transform)."""
        return list(self.pipeline.named_steps["features"].get_feature_names_out())

    @property
    def classes(self) -> np.ndarray:
        return np.asarray(self.model.classes_)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved model bundle", extra={"path": str(path)})
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        return cast("ModelBundle", joblib.load(path))


def save_bundle(bundle: ModelBundle, path: Path) -> Path:
    """Persist a pipeline+model bundle to disk; return the written path."""
    return bundle.save(path)


def load_bundle(path: Path) -> ModelBundle:
    """Load a previously saved bundle."""
    if not path.exists():
        raise FileNotFoundError(f"No model bundle at {path}. Train one with `netsentry train`.")
    return ModelBundle.load(path)


def latest_bundle(settings: Settings) -> Path | None:
    """Return the most recently modified bundle under the models directory, if any."""
    models_dir = settings.paths.models_dir
    if not models_dir.exists():
        return None
    bundles = sorted(models_dir.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return bundles[-1] if bundles else None
