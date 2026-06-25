"""Unsupervised novel-attack detection (the "detect the unknown" component).

Two benign-only detectors:

- **Isolation Forest** — a fast, strong baseline (always available).
- **Autoencoder** — a PyTorch MLP trained to reconstruct benign traffic; flows
  with high reconstruction error are anomalous. (Optional ``ae`` extra; falls
  back gracefully when Torch is absent.)

Both expose a uniform interface and a decision threshold calibrated to a target
false-positive rate on a **benign** validation set, so "1% of benign traffic
fires" is an explicit, operator-set budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import IsolationForest

from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.utils.optional import require

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)


class AnomalyDetector(ABC):
    """Benign-fit anomaly detector; higher score == more anomalous."""

    threshold: float = float("inf")

    @abstractmethod
    def fit(self, x_benign: np.ndarray) -> AnomalyDetector: ...

    @abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score per row (higher == more anomalous)."""

    def calibrate_threshold(self, x_benign_val: np.ndarray, target_fpr: float) -> float:
        """Set the threshold so ~``target_fpr`` of benign validation flows fire."""
        scores = self.score(x_benign_val)
        self.threshold = float(np.quantile(scores, 1.0 - target_fpr))
        return self.threshold

    def is_anomaly(self, x: np.ndarray) -> np.ndarray:
        return self.score(x) >= self.threshold


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest fit on benign traffic only."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None

    def fit(self, x_benign: np.ndarray) -> IsolationForestDetector:
        cfg = self.settings.anomaly
        self._model = IsolationForest(
            n_estimators=cfg.iforest_n_estimators,
            max_samples=cfg.iforest_max_samples,
            contamination=cfg.iforest_contamination,
            random_state=self.settings.seed,
            n_jobs=self.settings.supervised.n_jobs,
        )
        self._model.fit(np.asarray(x_benign))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        # score_samples: higher == more normal, so negate for an anomaly score.
        return -np.asarray(self._model.score_samples(np.asarray(x)))


class AutoencoderDetector(AnomalyDetector):
    """Benign-only PyTorch autoencoder; reconstruction error is the anomaly score."""

    def __init__(self, settings: Settings) -> None:
        require("torch", purpose="The autoencoder anomaly detector")
        self.settings = settings
        self.cfg = settings.anomaly.autoencoder
        self._model: Any = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _build(self, input_dim: int) -> Any:
        import torch.nn as nn

        dims = [input_dim, *self.cfg.hidden_dims]
        encoder_layers: list[Any] = []
        for in_dim, out_dim in pairwise(dims):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        decoder_layers: list[Any] = []
        for in_dim, out_dim in pairwise(reversed(dims)):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        decoder_layers = decoder_layers[:-1]  # no activation on the reconstruction
        return nn.Sequential(*encoder_layers, *decoder_layers)

    def fit(self, x_benign: np.ndarray) -> AutoencoderDetector:
        import torch
        from torch import nn, optim

        seed_everything(self.settings.seed)
        x = np.asarray(x_benign, dtype=np.float32)
        # Standardise on benign train statistics (the AE is sensitive to scale).
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0) + 1e-8
        x = (x - self._mean) / self._std

        n_val = max(1, int(0.1 * len(x)))
        x_tr, x_val = x[:-n_val], x[-n_val:]
        model = self._build(x.shape[1])
        optimizer = optim.Adam(
            model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        loss_fn = nn.MSELoss()
        tr = torch.from_numpy(x_tr)
        val = torch.from_numpy(x_val)

        best_val = float("inf")
        patience = 0
        epochs_run = 0
        for epoch in range(self.cfg.epochs):
            epochs_run = epoch + 1
            model.train()
            perm = torch.randperm(len(tr))
            for start in range(0, len(tr), self.cfg.batch_size):
                batch = tr[perm[start : start + self.cfg.batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val), val))
            if val_loss < best_val - 1e-5:
                best_val, patience = val_loss, 0
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break
        self._model = model
        logger.info(
            "Trained autoencoder", extra={"epochs": epochs_run, "val_loss": round(best_val, 5)}
        )
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        import torch

        assert self._mean is not None and self._std is not None, "fit() must run before score()"
        arr = (np.asarray(x, dtype=np.float32) - self._mean) / self._std
        self._model.eval()
        with torch.no_grad():
            recon = self._model(torch.from_numpy(arr)).numpy()
        return np.asarray(np.mean((arr - recon) ** 2, axis=1))


def build_anomaly_detector(settings: Settings, kind: str) -> AnomalyDetector:
    """Construct an anomaly detector (``iforest`` or ``autoencoder``)."""
    if kind == "iforest":
        return IsolationForestDetector(settings)
    if kind == "autoencoder":
        return AutoencoderDetector(settings)
    raise ValueError(f"Unknown anomaly detector kind: {kind!r}")
