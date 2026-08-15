"""Deep tabular models: an MLP and an FT-Transformer, held to the pipeline's protocol.

Gradient-boosted trees have been this project's supervised model since phase 4, chosen the way
most people choose them -- because the literature says trees win on tabular data. That is a
citation, not a measurement, and the claim it stands on (Grinsztajn et al. 2022; Shwartz-Ziv &
Armon 2022) is about tabular data in general, not about ninety flow statistics with a 20% attack
rate and a temporal split that shares no attack class between train and test. This module builds
the two architectures the comparison needs so the claim can be *checked here*, on the same
features, the same split, the same seed and the same operating metric.

- ``TabularMLP`` -- the obvious baseline: batch-normalised fully connected layers with dropout.
  If a plain MLP were enough, the transformer would have nothing to justify.
- ``FTTransformer`` -- the feature tokenizer plus transformer of Gorishniy et al. (2021). Each
  numeric feature becomes its own learned token (``x_j * W_j + b_j``), a ``[CLS]`` token is
  prepended, and self-attention lets the model form feature interactions explicitly rather than
  through the axis-aligned splits a tree is limited to. That is the architectural argument for
  trying it at all: the interaction study found real second-order structure in these features,
  and attention is the mechanism designed for exactly that.

Both wrap in the same fit/predict_proba shape as ``SupervisedClassifier``, handle imbalance with
a positive class weight rather than resampling (the project's standing rule), early-stop on
validation PR-AUC rather than loss (the metric that decides deployment), and take the global
seed. PyTorch is an optional extra, so both import it lazily and fail with an actionable message.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.log import get_logger
from netsentry.models.pauc import pairwise_pauc_surrogate
from netsentry.utils.optional import require

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DeepTabularConfig

logger = get_logger(__name__)


@dataclass
class TrainingTrace:
    """What the fit actually did -- the part a training-curve plot is made of."""

    epochs_run: int
    best_epoch: int
    best_val_pr_auc: float
    seconds: float
    n_parameters: int


def _build_mlp(n_features: int, cfg: DeepTabularConfig) -> Any:
    """Batch-normalised MLP: the baseline the transformer has to beat to be worth its cost."""
    from torch import nn

    layers: list[Any] = []
    width = n_features
    for size in cfg.hidden_sizes:
        layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(cfg.dropout)]
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def _build_ft_transformer(n_features: int, cfg: DeepTabularConfig) -> Any:
    """The feature tokenizer + transformer encoder of Gorishniy et al. (2021)."""
    import torch
    from torch import nn

    class FeatureTokenizer(nn.Module):
        """One learned token per feature: ``token_j = x_j * W_j + b_j``, plus a CLS token.

        The multiplication by the scalar feature value is what makes this a *tokenizer* rather
        than a linear layer: every feature keeps its own embedding direction, so attention can
        address "the packet-length feature" specifically instead of a mixture of all of them.
        """

        def __init__(self, n_features: int, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, dim))
            self.bias = nn.Parameter(torch.empty(n_features, dim))
            self.cls = nn.Parameter(torch.empty(1, 1, dim))
            for tensor in (self.weight, self.bias, self.cls):
                nn.init.uniform_(tensor, -1.0 / np.sqrt(dim), 1.0 / np.sqrt(dim))

        def forward(self, x: Any) -> Any:
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(len(x), 1, tokens.shape[-1])
            return torch.cat([cls, tokens], dim=1)

    class FTTransformer(nn.Module):
        """Tokenize, attend, read the CLS token."""

        def __init__(self) -> None:
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_features, cfg.token_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.token_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.token_dim * 2,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm: the stable choice at this depth
            )
            # Nested tensors are a fast path for padded batches and are incompatible with
            # pre-norm; every row here is the same length, so there is nothing to gain.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=cfg.n_blocks, enable_nested_tensor=False
            )
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.token_dim), nn.ReLU(), nn.Linear(cfg.token_dim, 1)
            )

        def forward(self, x: Any) -> Any:
            encoded = self.encoder(self.tokenizer(x))
            return self.head(encoded[:, 0])  # the CLS position aggregates the sequence

    return FTTransformer()


class TorchTabularClassifier:
    """A PyTorch tabular classifier with the same contract as the boosted model.

    ``architecture`` is ``"mlp"`` or ``"ft_transformer"``. Imbalance is handled with
    ``pos_weight`` in the loss, not by resampling; early stopping watches **validation PR-AUC**
    rather than validation loss, because loss and the deployment metric disagree under 20%
    prevalence and it is the metric that decides what ships.
    """

    def __init__(
        self,
        settings: Settings,
        architecture: str = "mlp",
        *,
        objective: str = "bce",
        pauc_alpha: float | None = None,
    ) -> None:
        require("torch", purpose="The deep tabular models")
        self.settings = settings
        self.architecture = architecture
        # ``objective`` selects what the network is actually trained to do: "bce" is
        # class-weighted cross-entropy (the incumbent), "pauc" is the partial-AUC surrogate that
        # only cares about how positives rank against the highest-scoring negatives -- i.e. the
        # flows that decide where the deployed threshold lands.
        self.objective = objective
        self.pauc_alpha = pauc_alpha
        self.cfg: DeepTabularConfig = settings.deep_tabular
        self.model: Any = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.trace: TrainingTrace | None = None

    def _build(self, n_features: int) -> Any:
        if self.architecture == "mlp":
            return _build_mlp(n_features, self.cfg)
        if self.architecture == "ft_transformer":
            return _build_ft_transformer(n_features, self.cfg)
        raise ValueError(f"unknown architecture {self.architecture!r}")

    def n_parameters(self) -> int:
        """Trainable parameters -- the honest size comparison against a tree ensemble."""
        if self.model is None:
            return 0
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TorchTabularClassifier:
        """Train with AdamW, class-weighted BCE and PR-AUC early stopping."""
        import torch
        from torch import nn, optim

        torch.manual_seed(self.settings.seed)
        generator = torch.Generator().manual_seed(self.settings.seed)
        x_train = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_train = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.model = self._build(X.shape[1])
        positives = float(np.sum(y == 1))
        negatives = float(np.sum(y == 0))
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        alpha = (
            self.pauc_alpha if self.pauc_alpha is not None else self.settings.thresholds.primary_fpr
        )
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )

        best_state: dict[str, Any] | None = None
        best_score = -np.inf
        best_epoch = 0
        waited = 0
        start = time.perf_counter()
        epochs_run = 0
        for epoch in range(self.cfg.epochs):
            epochs_run = epoch + 1
            self.model.train()
            order = torch.randperm(len(x_train), generator=generator)
            for begin in range(0, len(order), self.cfg.batch_size):
                index = order[begin : begin + self.cfg.batch_size]
                if len(index) < 2:  # BatchNorm needs more than one row
                    continue
                optimizer.zero_grad()
                logits = self.model(x_train[index]).squeeze(-1)
                targets = y_train[index]
                if self.objective == "pauc":
                    loss = pairwise_pauc_surrogate(
                        logits[targets > 0.5],
                        logits[targets <= 0.5],
                        alpha=alpha,
                        temperature=self.cfg.pauc_temperature,
                    )
                else:
                    loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            if eval_set is None:
                continue
            score = float(
                average_precision_score(eval_set[1], self.predict_proba(eval_set[0])[:, 1])
            )
            if score > best_score + 1e-6:
                best_score, best_epoch, waited = score, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                waited += 1
                if waited >= self.cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.trace = TrainingTrace(
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            best_val_pr_auc=float(best_score) if np.isfinite(best_score) else float("nan"),
            seconds=time.perf_counter() - start,
            n_parameters=self.n_parameters(),
        )
        logger.info(
            "Trained deep tabular model",
            extra={
                "architecture": self.architecture,
                "epochs": epochs_run,
                "best_epoch": best_epoch,
                "params": self.trace.n_parameters,
            },
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Two-column class probabilities, matching the scikit-learn contract."""
        import torch

        assert self.model is not None
        self.model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            data = torch.from_numpy(np.asarray(X, dtype=np.float32))
            for begin in range(0, len(data), 4096):
                logits = self.model(data[begin : begin + 4096]).squeeze(-1)
                out.append(torch.sigmoid(logits).numpy())
        positive = np.concatenate(out) if out else np.zeros(0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard labels at 0.5 -- present for interface parity; the project thresholds on scores."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
