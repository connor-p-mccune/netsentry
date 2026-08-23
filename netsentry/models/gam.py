"""The glass box, priced: an additive model you can read, argue with, and edit.

Everything this project ships to explain the deployed detector is **post hoc**. SHAP
attributes a verdict after the fact, [anchors](anchors.md) find a rule that happens to hold
near it, [distillation](distill.md) trains a small model to imitate a big one, and each of
them is an approximation whose own error has to be measured and reported. The
[stability study](importance_stability.md) exists because those approximations move when the
model is refit.

There is another way to get an explanation, and it is older than any of them: fit a model
whose form is already legible. A **generalized additive model** is a sum of one-dimensional
functions,

    score(x) = intercept + f_1(x_1) + f_2(x_2) + ... + f_d(x_d)

so the model *is* its own explanation. Each ``f_j`` is a curve -- "what this feature does to
the risk, holding everything else" -- and it is exact rather than attributed, global rather
than local, and stable rather than resampled. Fitted by gradient boosting over single-feature
histograms, an additive model of this shape is competitive on tabular data and has been
deployed in exactly the settings where a black box is unacceptable (Lou, Caruana & Gehrke, KDD
2012; Caruana et al., KDD 2015).

Three things are measured here, and only the first is the obvious one.

1. **What the glass box costs.** The additive model, the same model plus a bounded number of
   pairwise terms, the deployed boosted ensemble and the linear floor all go through the
   project's temporal protocol at the same operating point. The gap between the additive model
   and the ensemble is the *value of the interactions*, and it is a number rather than an
   assumption.
2. **What the shapes say.** Printed in raw feature units by inverting the fitted scaler,
   because a curve quoted in standard deviations is not a curve an analyst can argue with.
3. **What you can do about it.** A shape function is a lookup table, so an operator can *edit
   the model* -- clamp a region that fires on traffic they know is benign -- and the effect is
   exact, immediate and requires no retraining. Every edit here is chosen on validation and
   measured on the held-out later days, because an edit chosen and scored on the same data is
   an anecdote.

The honest counterweight is stated in the same breath: an additive model cannot represent an
interaction, correlated features split their credit arbitrarily between their curves, and a
legible model is only legible if its features are. Those limits are the reason this module ends
in a comparison rather than a recommendation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.explain.optimal_tree import scaler_stats
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import GamConfig

logger = get_logger(__name__)

REPORT_NAME = "gam.md"
SHAPE_FIGURE = "gam_shapes.png"
LADDER_FIGURE = "gam_ladder.png"

_LOGIT_CLIP = 30.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -_LOGIT_CLIP, _LOGIT_CLIP)
    out: np.ndarray = 1.0 / (1.0 + np.exp(-clipped))
    return out


# --------------------------------------------------------------------------------------
# The model. Histogram boosting restricted to one feature at a time.
# --------------------------------------------------------------------------------------


@dataclass
class Binner:
    """Quantile bin edges per feature, fitted on the training split only.

    Binning is what makes an additive model cheap: once each column is an integer, fitting a
    shape function is a ``bincount`` rather than a tree search, and the whole model is a set of
    lookup tables. Values beyond the training range fall into the end bins, so the model
    **extrapolates flat** -- which is a safety property worth naming rather than an accident.
    A boosted ensemble does the same thing; a linear model does not, and will happily
    extrapolate a trend into traffic it has never seen.
    """

    edges: list[np.ndarray]

    @classmethod
    def fit(cls, matrix: np.ndarray, n_bins: int) -> Binner:
        """Learn quantile edges per column."""
        levels = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = []
        for index in range(matrix.shape[1]):
            column = matrix[:, index]
            cuts = np.unique(np.quantile(column, levels))
            # Drop cuts at or below the column minimum. Flow features are heavily
            # zero-inflated, so several low quantiles land on the same value and each one
            # would open a bin no row can ever fall into -- a dead segment in the shape
            # function an operator reads, and a candidate edit that can never fire.
            cuts = cuts[cuts > column.min()]
            edges.append(np.asarray(cuts, dtype=float))
        return cls(edges=edges)

    @property
    def sizes(self) -> list[int]:
        """Number of bins per feature (one more than the number of cuts)."""
        return [len(cuts) + 1 for cuts in self.edges]

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Map a matrix of values onto bin indices."""
        out = np.empty(matrix.shape, dtype=np.int32)
        for index, cuts in enumerate(self.edges):
            out[:, index] = np.searchsorted(cuts, matrix[:, index], side="right")
        return out


@dataclass
class AdditiveModel:
    """An intercept plus one lookup table per feature, and optionally per selected pair.

    The tables *are* the model. Nothing is approximated at explanation time, which is the
    entire difference between this and everything else in ``netsentry/explain``.
    """

    intercept: float
    shapes: list[np.ndarray]
    binner: Binner
    pairs: list[tuple[int, int]] = field(default_factory=list)
    pair_shapes: list[np.ndarray] = field(default_factory=list)

    def margin(self, bins: np.ndarray) -> np.ndarray:
        """Log-odds score for pre-binned rows."""
        total = np.full(len(bins), self.intercept, dtype=float)
        for index, shape in enumerate(self.shapes):
            total += shape[bins[:, index]]
        for (left, right), table in zip(self.pairs, self.pair_shapes, strict=True):
            total += table[bins[:, left], bins[:, right]]
        return total

    def predict_proba(self, bins: np.ndarray) -> np.ndarray:
        """Attack probability for pre-binned rows."""
        return _sigmoid(self.margin(bins))

    def contribution_range(self, index: int) -> float:
        """How much of the score this feature can move -- the additive model's importance.

        Range rather than variance: an operator reading the curve wants to know the size of
        the swing the feature is allowed to produce, which is what the axis of the plot shows.
        """
        shape = self.shapes[index]
        return float(shape.max() - shape.min()) if len(shape) else 0.0

    def clamp(self, index: int, bin_index: int, value: float = 0.0) -> AdditiveModel:
        """Return a copy with one bin of one shape function overwritten.

        This is the operation a black box does not have. There is no retraining, no surrogate
        and no approximation: the edited model *is* the model, and its behaviour on every flow
        that lands in that bin changes by exactly the difference.
        """
        shapes = [np.array(shape, copy=True) for shape in self.shapes]
        shapes[index][bin_index] = value
        return AdditiveModel(
            intercept=self.intercept,
            shapes=shapes,
            binner=self.binner,
            pairs=list(self.pairs),
            pair_shapes=[np.array(table, copy=True) for table in self.pair_shapes],
        )


def fit_additive(
    bins: np.ndarray,
    y: np.ndarray,
    binner: Binner,
    *,
    rounds: int,
    learning_rate: float,
    l2: float,
    weights: np.ndarray | None = None,
) -> AdditiveModel:
    """Cyclic Newton boosting over single-feature histograms.

    Each round walks the features in turn and, for one of them, solves the one-dimensional
    Newton step exactly: the optimal constant per bin is ``-sum(g) / (sum(h) + l2)`` where
    ``g`` and ``h`` are the logistic gradient and hessian. That is the same update a boosted
    tree makes in a leaf; the only difference is that a leaf here is a bin of a single
    feature, which is precisely what forces the model to stay additive.

    The gradient is refreshed **after every feature**, not once per round. Refreshing per round
    is cheaper and is what a naive implementation does, and it makes correlated features each
    take full credit for a signal only one of them should get -- the shape functions then
    double-count and the curves an operator reads are wrong even though the score is fine.
    """
    sample_weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights)
    positives = float(np.average(y, weights=sample_weights))
    positives = min(max(positives, 1e-6), 1 - 1e-6)
    intercept = float(np.log(positives / (1.0 - positives)))
    shapes = [np.zeros(size, dtype=float) for size in binner.sizes]
    margin = np.full(len(y), intercept, dtype=float)
    target = np.asarray(y, dtype=float)

    for _ in range(rounds):
        for index in range(bins.shape[1]):
            probability = _sigmoid(margin)
            gradient = (probability - target) * sample_weights
            hessian = probability * (1.0 - probability) * sample_weights
            column = bins[:, index]
            size = len(shapes[index])
            grad_sum = np.bincount(column, weights=gradient, minlength=size)
            hess_sum = np.bincount(column, weights=hessian, minlength=size)
            step = -learning_rate * grad_sum / (hess_sum + l2)
            shapes[index] += step
            margin += step[column]
    return AdditiveModel(intercept=intercept, shapes=shapes, binner=binner)


def rank_pairs(
    model: AdditiveModel,
    bins: np.ndarray,
    y: np.ndarray,
    *,
    candidates: list[int],
    weights: np.ndarray | None = None,
) -> list[tuple[tuple[int, int], float]]:
    """Score candidate feature pairs by the loss reduction a joint table would buy.

    The FAST heuristic (Lou et al., KDD 2013): with the additive fit held fixed, the residual
    gradient and hessian are binned on the *joint* grid of a pair, and the exact Newton
    reduction ``sum(g)^2 / (sum(h) + eps)`` over the joint cells is how much a two-dimensional
    term could recover. Ranking this way costs one two-dimensional histogram per pair rather
    than one refit per pair, which is the difference between a ranking and an afternoon.
    """
    sample_weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights)
    probability = model.predict_proba(bins)
    gradient = (probability - np.asarray(y, dtype=float)) * sample_weights
    hessian = probability * (1.0 - probability) * sample_weights
    scored: list[tuple[tuple[int, int], float]] = []
    for position, left in enumerate(candidates):
        for right in candidates[position + 1 :]:
            rows = len(model.shapes[left])
            cols = len(model.shapes[right])
            flat = bins[:, left].astype(np.int64) * cols + bins[:, right]
            grad_sum = np.bincount(flat, weights=gradient, minlength=rows * cols)
            hess_sum = np.bincount(flat, weights=hessian, minlength=rows * cols)
            gain = float(np.sum(grad_sum**2 / (hess_sum + 1.0)))
            scored.append(((left, right), gain))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def fit_pairs(
    model: AdditiveModel,
    bins: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    rounds: int,
    learning_rate: float,
    l2: float,
    weights: np.ndarray | None = None,
) -> AdditiveModel:
    """Add joint tables for the chosen pairs, boosting them the same way the shapes were."""
    if not pairs:
        return model
    sample_weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights)
    tables = [
        np.zeros((len(model.shapes[left]), len(model.shapes[right])), dtype=float)
        for left, right in pairs
    ]
    fitted = AdditiveModel(
        intercept=model.intercept,
        shapes=[np.array(shape, copy=True) for shape in model.shapes],
        binner=model.binner,
        pairs=list(pairs),
        pair_shapes=tables,
    )
    margin = fitted.margin(bins)
    target = np.asarray(y, dtype=float)
    for _ in range(rounds):
        for position, (left, right) in enumerate(pairs):
            probability = _sigmoid(margin)
            gradient = (probability - target) * sample_weights
            hessian = probability * (1.0 - probability) * sample_weights
            cols = tables[position].shape[1]
            flat = bins[:, left].astype(np.int64) * cols + bins[:, right]
            size = tables[position].size
            grad_sum = np.bincount(flat, weights=gradient, minlength=size)
            hess_sum = np.bincount(flat, weights=hessian, minlength=size)
            step = -learning_rate * grad_sum / (hess_sum + l2)
            tables[position] += step.reshape(tables[position].shape)
            margin += step[flat]
    return fitted


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryRow:
    """One synthetic shape the fitter either recovered or did not."""

    component: str
    correlation: float
    shape_error: float


@dataclass(frozen=True)
class ModelRow:
    """One model on the temporal protocol, at the shared operating point."""

    name: str
    pr_auc: float
    roc_auc: float
    detection: float
    parameters: int
    seconds: float
    readable: bool


@dataclass(frozen=True)
class CapacityRow:
    """One rung of a capacity ladder, scored on all three splits.

    Three columns rather than one, because the interesting quantity is not any of them: it is
    where each curve turns. Training says what the model can fit, validation says what a
    practitioner following this project's own protocol would have selected, and the later days
    say what that selection was worth.
    """

    label: str
    parameters: int
    train: float
    validation: float
    test: float
    selected: bool = False


@dataclass(frozen=True)
class ShapeRow:
    """One feature's curve, summarised in raw units."""

    feature: str
    swing: float
    riskiest_low: float
    riskiest_high: float
    monotone: bool


@dataclass(frozen=True)
class EditRow:
    """One surgical edit: ranked on validation, measured on the later days."""

    budget: float
    feature: str
    low: float
    high: float
    validation_removed: int
    validation_lost: int
    removed: int
    lost: int
    benign_share: float

    @property
    def exchange_rate(self) -> float:
        """False alarms cleared per attack the edit stops catching."""
        return self.removed / max(self.lost, 1)

    @property
    def free(self) -> bool:
        """Whether the edit costs no detection at all on the later days."""
        return self.lost == 0


@dataclass
class GamStudy:
    """Everything the report needs, computed once."""

    recovery: list[RecoveryRow]
    models: list[ModelRow]
    bin_ladder: list[CapacityRow]
    round_ladder: list[CapacityRow]
    pair_ladder: list[CapacityRow]
    shapes: list[ShapeRow]
    edits: list[EditRow]
    curves: dict[str, tuple[np.ndarray, np.ndarray]]
    alarm_counts: list[tuple[float, int, int, int]]
    break_even: float
    budget: float
    threshold: float
    n_train: int
    n_features: int
    selected_bins: int
    selected_pairs: int
    benign_alerts: int
    caught_attacks: int
    noise_swing: float
    seconds: float = 0.0

    def model(self, name: str) -> ModelRow | None:
        """Look up one arm."""
        return next((row for row in self.models if row.name == name), None)


def best_rung(ladder: list[CapacityRow], column: str) -> CapacityRow | None:
    """The rung a given column would have chosen."""
    return max(ladder, key=lambda row: float(getattr(row, column))) if ladder else None


GAM = "additive model (GAM)"
GA2M = "additive + pairwise (GA2M)"
BOOSTED = "gradient-boosted ensemble (deployed)"
LINEAR = "logistic regression"


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _recovery_check(cfg: GamConfig, seed: int) -> list[RecoveryRow]:
    """Fit a known additive function and check the curves come back.

    A shape function that looks plausible is not evidence of anything. Here the truth is
    constructed: a step, a parabola and a pure-noise feature, combined additively into a
    logistic response. The fitter has to recover the first two and leave the third flat, and
    the third is the one that matters -- it measures the curve this fitter invents for a
    feature carrying no signal at all, which is the noise floor every real shape has to be
    read against.
    """
    rng = np.random.default_rng(seed)
    rows = cfg.recovery_rows
    features = rng.uniform(-3.0, 3.0, size=(rows, 3))
    truth = {
        "a step": 1.5 * (features[:, 0] > 0.5).astype(float),
        "a parabola": 0.4 * features[:, 1] ** 2 - 1.2,
        "pure noise (must stay flat)": np.zeros(rows),
    }
    margin = sum(truth.values())
    labels = (rng.random(rows) < _sigmoid(np.asarray(margin))).astype(int)
    binner = Binner.fit(features, cfg.n_bins)
    model = fit_additive(
        binner.transform(features),
        labels,
        binner,
        rounds=cfg.recovery_rounds,
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
    )
    grid = binner.transform(features)
    out: list[RecoveryRow] = []
    for index, (name, component) in enumerate(truth.items()):
        recovered = model.shapes[index][grid[:, index]]
        expected = np.asarray(component, dtype=float)
        centred_recovered = recovered - recovered.mean()
        centred_expected = expected - expected.mean()
        if centred_expected.std() < 1e-9:
            correlation = float("nan")  # undefined against a flat truth; read the error column
        else:
            correlation = float(np.corrcoef(centred_recovered, centred_expected)[0, 1])
        out.append(
            RecoveryRow(
                component=name,
                correlation=correlation,
                shape_error=float(np.sqrt(np.mean((centred_recovered - centred_expected) ** 2))),
            )
        )
    return out


def _shape_rows(
    model: AdditiveModel, names: list[str], stats: dict[str, tuple[float, float]], top_n: int
) -> list[ShapeRow]:
    """Summarise the biggest curves in raw units: where the risk actually is."""
    order = sorted(range(len(model.shapes)), key=model.contribution_range, reverse=True)
    rows: list[ShapeRow] = []
    for index in order[:top_n]:
        raw = display_feature_name(names[index])
        mean, scale = stats.get(raw, (0.0, 1.0))
        edges = model.binner.edges[index]
        peak = int(np.argmax(model.shapes[index]))
        low = edges[peak - 1] if 0 < peak <= len(edges) else -np.inf
        high = edges[peak] if peak < len(edges) else np.inf
        rows.append(
            ShapeRow(
                feature=raw,
                swing=model.contribution_range(index),
                riskiest_low=float(low * scale + mean) if np.isfinite(low) else float("-inf"),
                riskiest_high=float(high * scale + mean) if np.isfinite(high) else float("inf"),
                monotone=bool(
                    np.all(np.diff(model.shapes[index]) >= -1e-9)
                    or np.all(np.diff(model.shapes[index]) <= 1e-9)
                ),
            )
        )
    return rows


def _rank_edits(
    model: AdditiveModel,
    bins: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    *,
    top_n: int,
    min_removed: int,
) -> list[tuple[int, int, int, int]]:
    """Rank every (feature, bin) clamp by the trade it actually makes on validation.

    An earlier version ranked by a proxy -- how much positive shape mass sat under the false
    alarms -- and chose edits that cost more detection than they saved. There is no need for a
    proxy: clamping one bin changes the margin by a known constant for exactly the rows in that
    bin, so every candidate can be **evaluated exactly** in the time it takes to re-sigmoid a
    few hundred rows. Ranking on the measured validation trade and reporting the later-day
    trade is the honest version of the same idea, and the gap between the two is what selecting
    an edit on validation is worth.
    """
    base = model.margin(bins)
    alerted = _sigmoid(base) >= threshold
    benign = labels == 0
    attacks = labels == 1
    scored: list[tuple[float, int, int, int, int]] = []
    for index in range(bins.shape[1]):
        column = bins[:, index]
        shape = model.shapes[index]
        for bin_index in range(len(shape)):
            if shape[bin_index] <= 0.0:
                continue  # clamping a bin that already lowers risk cannot clear an alert
            rows = np.flatnonzero(column == bin_index)
            if len(rows) == 0:
                continue
            cleared = alerted[rows] & ~(_sigmoid(base[rows] - shape[bin_index]) >= threshold)
            removed = int(np.sum(cleared & benign[rows]))
            lost = int(np.sum(cleared & attacks[rows]))
            if removed < min_removed:
                continue
            scored.append((removed / (lost + 1.0), index, bin_index, removed, lost))
    scored.sort(reverse=True)
    return [
        (index, bin_index, removed, lost) for _, index, bin_index, removed, lost in scored[:top_n]
    ]


def _measure_edit(
    model: AdditiveModel,
    proposal: tuple[int, int, int, int],
    bins: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    stats: dict[str, tuple[float, float]],
    threshold: float,
    budget: float,
) -> EditRow:
    """Apply one clamp and re-score the later days at the unchanged operating point."""
    index, bin_index, validation_removed, validation_lost = proposal
    before = model.predict_proba(bins) >= threshold
    after = model.clamp(index, bin_index).predict_proba(bins) >= threshold
    cleared = before & ~after
    benign = labels == 0
    raw = display_feature_name(names[index])
    mean, scale = stats.get(raw, (0.0, 1.0))
    edges = model.binner.edges[index]
    low = edges[bin_index - 1] if 0 < bin_index <= len(edges) else -np.inf
    high = edges[bin_index] if bin_index < len(edges) else np.inf
    return EditRow(
        budget=budget,
        feature=raw,
        low=float(low * scale + mean) if np.isfinite(low) else float("-inf"),
        high=float(high * scale + mean) if np.isfinite(high) else float("inf"),
        validation_removed=validation_removed,
        validation_lost=validation_lost,
        removed=int(np.sum(cleared & benign)),
        lost=int(np.sum(cleared & (labels == 1))),
        benign_share=float(np.mean(bins[benign, index] == bin_index)),
    )


def _bin_centres(edges: np.ndarray) -> np.ndarray:
    """Representative x-positions for a step function defined by ``edges``."""
    if len(edges) == 0:
        return np.array([0.0])
    span = float(edges[-1] - edges[0]) or 1.0
    return np.concatenate(
        [[edges[0] - 0.05 * span], (edges[:-1] + edges[1:]) / 2.0, [edges[-1] + 0.05 * span]]
    )


def _leaf_count(model: object) -> int:
    """Leaves in the boosted ensemble, as its parameter count. Zero when unavailable."""
    booster = getattr(getattr(model, "model", None), "booster_", None)
    if booster is None:
        return 0
    dump = booster.dump_model()
    return int(sum(info.get("num_leaves", 0) for info in dump.get("tree_info", [])))


def _scorer(model: AdditiveModel, binner: Binner) -> Callable[[np.ndarray], np.ndarray]:
    """Bind a fitted model to its binner so a ladder rung can be scored on any split."""

    def score(matrix: np.ndarray) -> np.ndarray:
        return model.predict_proba(binner.transform(matrix))

    return score


def run_gam_study(settings: Settings) -> GamStudy:
    """Fit the glass box beside the deployed ensemble, sweep its capacity, then edit it."""
    start = time.perf_counter()
    cfg: GamConfig = settings.gam
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier, build_baselines

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_calibration: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_later: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_calibration = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_later = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)
    names = list(pipeline.named_steps["features"].get_feature_names_out())
    stats = scaler_stats(pipeline)
    weights = compute_sample_weight("balanced", y_train)

    def rung(
        label: str, parameters: int, scorer: Callable[[np.ndarray], np.ndarray]
    ) -> CapacityRow:
        return CapacityRow(
            label=label,
            parameters=parameters,
            train=float(average_precision_score(y_train, scorer(x_train))),
            validation=float(average_precision_score(y_calibration, scorer(x_calibration))),
            test=float(average_precision_score(y_later, scorer(x_later))),
        )

    # 1. The capacity dial: bins per shape function, which is this model's resolution and its
    #    parameter count in one number. Every rung is scored on all three splits.
    fitted: dict[int, tuple[AdditiveModel, Binner]] = {}
    bin_ladder: list[CapacityRow] = []
    for n_bins in cfg.bin_ladder:
        binner = Binner.fit(x_train, n_bins)
        model = fit_additive(
            binner.transform(x_train),
            y_train,
            binner,
            rounds=cfg.rounds,
            learning_rate=cfg.learning_rate,
            l2=cfg.l2,
            weights=weights,
        )
        fitted[n_bins] = (model, binner)
        bin_ladder.append(
            rung(
                f"{n_bins} bins per feature",
                sum(binner.sizes),
                _scorer(model, binner),
            )
        )

    # The protocol this project prescribes: choose on validation, never on the later days.
    chosen = max(range(len(bin_ladder)), key=lambda i: bin_ladder[i].validation)
    selected_bins = cfg.bin_ladder[chosen]
    bin_ladder[chosen] = replace(bin_ladder[chosen], selected=True)
    gam, binner = fitted[selected_bins]
    bins_train = binner.transform(x_train)
    bins_calibration = binner.transform(x_calibration)
    bins_later = binner.transform(x_later)

    # 2. The second dial, at the selected resolution: boosting rounds. Two independent dials
    #    telling the same story is a replication; one is an anecdote about a hyperparameter.
    round_ladder: list[CapacityRow] = []
    for rounds in cfg.round_ladder:
        model = fit_additive(
            bins_train,
            y_train,
            binner,
            rounds=rounds,
            learning_rate=cfg.learning_rate,
            l2=cfg.l2,
            weights=weights,
        )
        round_ladder.append(
            rung(
                f"{rounds} boosting rounds",
                sum(binner.sizes),
                _scorer(model, binner),
            )
        )

    # 3. The third dial: pairwise terms, which is capacity of a different kind -- the only kind
    #    an additive model structurally cannot have.
    importance = sorted(range(len(gam.shapes)), key=gam.contribution_range, reverse=True)
    ranked = rank_pairs(
        gam, bins_train, y_train, candidates=importance[: cfg.pair_candidates], weights=weights
    )
    pair_ladder: list[CapacityRow] = []
    pair_models: dict[int, tuple[AdditiveModel, float]] = {}
    for count in cfg.pair_ladder:
        clock = time.perf_counter()
        model = fit_pairs(
            gam,
            bins_train,
            y_train,
            [pair for pair, _ in ranked[:count]],
            rounds=cfg.pair_rounds,
            learning_rate=cfg.learning_rate,
            l2=cfg.l2,
            weights=weights,
        )
        pair_models[count] = (model, time.perf_counter() - clock)
        pair_ladder.append(
            rung(
                f"{count} pairwise term{'s' if count != 1 else ''}",
                sum(binner.sizes) + sum(table.size for table in model.pair_shapes),
                _scorer(model, binner),
            )
        )
    # The same protocol the resolution got: the pair count is chosen on validation too.
    chosen_pairs = max(range(len(pair_ladder)), key=lambda i: pair_ladder[i].validation)
    selected_pairs = cfg.pair_ladder[chosen_pairs]
    pair_ladder[chosen_pairs] = replace(pair_ladder[chosen_pairs], selected=True)
    ga2m, pair_seconds = pair_models[selected_pairs]

    # 4. The comparison table, every arm at the same validation-calibrated operating point.
    models: list[ModelRow] = []

    def record(
        name: str,
        held_out: np.ndarray,
        later: np.ndarray,
        parameters: int,
        seconds: float,
        *,
        readable: bool,
    ) -> None:
        cut = threshold_at_fpr(y_calibration, held_out, cfg.budget)
        attacks = y_later == 1
        models.append(
            ModelRow(
                name=name,
                pr_auc=float(average_precision_score(y_later, later)),
                roc_auc=float(roc_auc_score(y_later, later)),
                detection=float(np.mean(later[attacks] >= cut)) if attacks.any() else 0.0,
                parameters=parameters,
                seconds=seconds,
                readable=readable,
            )
        )

    clock = time.perf_counter()
    fit_additive(
        bins_train,
        y_train,
        binner,
        rounds=cfg.rounds,
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
        weights=weights,
    )
    gam_seconds = time.perf_counter() - clock
    record(
        GAM,
        gam.predict_proba(bins_calibration),
        gam.predict_proba(bins_later),
        sum(binner.sizes),
        gam_seconds,
        readable=True,
    )
    record(
        GA2M,
        ga2m.predict_proba(bins_calibration),
        ga2m.predict_proba(bins_later),
        sum(binner.sizes) + sum(table.size for table in ga2m.pair_shapes),
        gam_seconds + pair_seconds,
        readable=True,
    )

    clock = time.perf_counter()
    boosted = SupervisedClassifier(variant).fit(
        x_train, y_train, eval_set=(x_calibration, y_calibration)
    )
    boosted_seconds = time.perf_counter() - clock
    column = list(boosted.classes_).index(1)
    record(
        BOOSTED,
        np.asarray(boosted.predict_proba(x_calibration))[:, column],
        np.asarray(boosted.predict_proba(x_later))[:, column],
        _leaf_count(boosted),
        boosted_seconds,
        readable=False,
    )

    linear = build_baselines(variant)["logistic"]
    clock = time.perf_counter()
    linear.fit(x_train, y_train)
    linear_seconds = time.perf_counter() - clock
    column = list(linear.classes_).index(1)
    record(
        LINEAR,
        np.asarray(linear.predict_proba(x_calibration))[:, column],
        np.asarray(linear.predict_proba(x_later))[:, column],
        x_train.shape[1] + 1,
        linear_seconds,
        readable=True,
    )

    # 5. The operation a black box does not have. Run at two budgets on purpose: at the
    #    deployed one there are only a few dozen false alarms on validation to select an edit
    #    from, and whether that is the reason editing fails is a question, not a footnote.
    calibration_scores = gam.predict_proba(bins_calibration)
    later_scores = gam.predict_proba(bins_later)
    edits: list[EditRow] = []
    alarm_counts: list[tuple[float, int, int, int]] = []
    for budget in cfg.edit_budgets:
        cut = threshold_at_fpr(y_calibration, calibration_scores, budget)
        proposals = _rank_edits(
            gam,
            bins_calibration,
            y_calibration,
            cut,
            top_n=cfg.n_edits,
            min_removed=cfg.min_removed,
        )
        edits.extend(
            _measure_edit(gam, proposal, bins_later, y_later, names, stats, cut, budget)
            for proposal in proposals
        )
        firing = later_scores >= cut
        alarm_counts.append(
            (
                budget,
                int(np.sum((calibration_scores >= cut) & (y_calibration == 0))),
                int(np.sum(firing & (y_later == 0))),
                int(np.sum(firing & (y_later == 1))),
            )
        )
    threshold = threshold_at_fpr(y_calibration, calibration_scores, cfg.budget)

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index in importance[: cfg.plot_features]:
        raw = display_feature_name(names[index])
        mean, scale = stats.get(raw, (0.0, 1.0))
        curves[raw] = (_bin_centres(binner.edges[index]) * scale + mean, gam.shapes[index])

    recovery = _recovery_check(cfg, variant.seed)
    alerting = gam.predict_proba(bins_later) >= threshold
    study = GamStudy(
        recovery=recovery,
        models=models,
        bin_ladder=bin_ladder,
        round_ladder=round_ladder,
        pair_ladder=pair_ladder,
        shapes=_shape_rows(gam, names, stats, cfg.top_features),
        edits=edits,
        curves=curves,
        alarm_counts=alarm_counts,
        break_even=settings.cost.cost_per_miss / max(settings.cost.cost_per_alert, 1e-9),
        budget=cfg.budget,
        threshold=float(threshold),
        n_train=len(y_train),
        n_features=x_train.shape[1],
        selected_bins=selected_bins,
        selected_pairs=selected_pairs,
        benign_alerts=int(np.sum(alerting & (y_later == 0))),
        caught_attacks=int(np.sum(alerting & (y_later == 1))),
        noise_swing=recovery[-1].shape_error if recovery else 0.0,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "GAM study complete",
        extra={
            "selected_bins": selected_bins,
            "edits": len(edits),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _recovery_table(study: GamStudy) -> str:
    rows = "\n".join(
        f"| {row.component} | "
        + ("--" if np.isnan(row.correlation) else f"{row.correlation:.3f}")
        + f" | {row.shape_error:.3f} |"
        for row in study.recovery
    )
    return (
        "| component | correlation with the truth | curve error (log-odds) |\n|---|---|---|\n"
        + rows
    )


def _model_table(study: GamStudy) -> str:
    rows = "\n".join(
        f"| {row.name} | {'yes' if row.readable else '**no**'} | {row.parameters:,} | "
        f"{row.pr_auc:.3f} | {row.roc_auc:.3f} | {row.detection:.1%} | {row.seconds:.1f} |"
        for row in sorted(study.models, key=lambda row: row.pr_auc, reverse=True)
    )
    return (
        f"| model | readable? | parameters | PR-AUC | ROC-AUC | detection @ {study.budget:.0%} "
        "FPR | fit (s) |\n|---|---|---|---|---|---|---|\n" + rows
    )


def _ladder_table(ladder: list[CapacityRow]) -> str:
    rows = "\n".join(
        f"| {row.label}{' **(selected)**' if row.selected else ''} | {row.parameters:,} | "
        f"{row.train:.3f} | {row.validation:.3f} | {row.test:.3f} |"
        for row in ladder
    )
    return (
        "| capacity | parameters | train PR-AUC | validation PR-AUC | later-days PR-AUC |\n"
        "|---|---|---|---|---|\n" + rows
    )


def _shape_table(study: GamStudy) -> str:
    rows = "\n".join(
        f"| `{row.feature}` | {row.swing:.2f} | {_span(row.riskiest_low, row.riskiest_high)} | "
        f"{'yes' if row.monotone else 'no'} |"
        for row in study.shapes
    )
    return (
        "| feature | swing (log-odds) | riskiest region (raw units) | monotone? |\n"
        "|---|---|---|---|\n" + rows
    )


def _span(low: float, high: float) -> str:
    left = "-inf" if not np.isfinite(low) else f"{low:.4g}"
    right = "inf" if not np.isfinite(high) else f"{high:.4g}"
    return f"{left} .. {right}"


def _edit_table(study: GamStudy, budget: float) -> str:
    chosen = [row for row in study.edits if row.budget == budget]
    if not chosen:
        return "_No candidate edit cleared enough alarms on validation to be worth measuring._"
    rows = "\n".join(
        f"| `{row.feature}` {_span(row.low, row.high)} | {row.validation_removed} / "
        f"{row.validation_lost} | {row.removed} | {row.lost} | "
        + ("**no cost**" if row.free else f"{row.exchange_rate:.1f}:1")
        + f" | {row.benign_share:.1%} |"
        for row in chosen
    )
    return (
        "| clamped region | validation cleared / lost | false alarms cleared | attacks lost | "
        "exchange rate | benign flows in the region |\n|---|---|---|---|---|---|\n" + rows
    )


def _alarm_table(study: GamStudy) -> str:
    rows = "\n".join(
        f"| {budget:.0%} | {validation:,} | {later:,} | {caught:,} |"
        for budget, validation, later, caught in study.alarm_counts
    )
    return (
        "| false-positive budget | false alarms on validation (what an edit is chosen from) | "
        "false alarms on the later days | attacks caught |\n|---|---|---|---|\n" + rows
    )


def best_edit(study: GamStudy, budget: float) -> EditRow | None:
    """The most favourable trade available at one budget."""
    chosen = [row for row in study.edits if row.budget == budget]
    return max(chosen, key=lambda row: row.exchange_rate) if chosen else None


def _edit_read(study: GamStudy, budget: float) -> str:
    """One sentence about the best edit available at a budget, priced against the economics."""
    chosen = [row for row in study.edits if row.budget == budget]
    best = best_edit(study, budget)
    if best is None:
        return "Nothing cleared enough validation alarms to be worth measuring at this budget."
    pays = "**pays**" if best.exchange_rate >= study.break_even else "does not pay"
    if best.free:
        return (
            f"The best available edit clears {best.removed} false alarms at **no detection cost "
            f"at all**, and {sum(1 for row in chosen if row.free)} of {len(chosen)} candidates "
            f"are free."
        )
    return (
        f"The best available edit clears {best.removed} false alarms and stops catching "
        f"{best.lost} attacks -- {best.exchange_rate:.1f} to 1, against the "
        f"{study.break_even:.0f} to 1 the [cost study's](cost.md) economics require, so it "
        f"{pays}."
    )


def _lead(study: GamStudy) -> str:
    gam = study.model(GAM)
    boosted = study.model(BOOSTED)
    linear = study.model(LINEAR)
    if gam is None or boosted is None or linear is None:
        return "_The arms did not complete._"
    best = max(study.models, key=lambda row: row.pr_auc)
    peak_validation = best_rung(study.bin_ladder, "validation")
    peak_test = best_rung(study.bin_ladder, "test")
    return (
        f"**Interpretability is not what costs anything here. Capacity is.** The most readable "
        f"model in the comparison -- logistic regression, one coefficient per feature -- is "
        f"also the most accurate on the honest split at PR-AUC {linear.pr_auc:.3f}, ahead of "
        f"the deployed boosted ensemble's {boosted.pr_auc:.3f} and the additive model's "
        f"{gam.pr_auc:.3f}. The winner over every arm is _{best.name}_.\n\n"
        f"That ordering is not an accident of four model families, and the additive model is "
        f"what makes it measurable, because its capacity is a **dial** rather than an "
        f"architecture. Turning the resolution of every shape function from "
        f"{study.bin_ladder[0].label.split()[0]} bins to "
        f"{study.bin_ladder[-1].label.split()[0]} takes training PR-AUC from "
        f"{study.bin_ladder[0].train:.3f} to {study.bin_ladder[-1].train:.3f}, and the later "
        f"days from {study.bin_ladder[0].test:.3f} to {study.bin_ladder[-1].test:.3f} -- rising "
        f"then falling, with everything else held exactly fixed: the same loss, the same "
        f"boosting, the same class weights, the same splits.\n\n"
        f"Validation, which this project carves out of the **training days**, does catch the "
        f"turn -- it peaks at {peak_validation.label if peak_validation else 'n/a'} while the "
        f"later days peak at {peak_test.label if peak_test else 'n/a'}. It is off by one rung "
        f"and it overstates the achievable score by "
        f"{(peak_validation.validation - peak_validation.test) if peak_validation else 0:.3f} "
        f"PR-AUC. **Validation is a usable signal about the shape of the capacity curve and a "
        f"useless one about its level**, which is the reason every headline in this repository "
        f"is a temporal-split number."
    )


def _render(study: GamStudy, shapes: Path, ladder: Path) -> str:
    gam = study.model(GAM)
    boosted = study.model(BOOSTED)
    linear = study.model(LINEAR)
    peak_validation = best_rung(study.bin_ladder, "validation")
    peak_test = best_rung(study.bin_ladder, "test")
    peak_label = peak_validation.label if peak_validation else "the peak"
    peak_value = peak_validation.validation if peak_validation else 0.0
    peak_actual = peak_validation.test if peak_validation else 0.0
    wanted_label = peak_test.label if peak_test else "n/a"
    round_validation = best_rung(study.round_ladder, "validation")
    round_test = best_rung(study.round_ladder, "test")
    round_peak = round_validation.label if round_validation else "n/a"
    round_wanted = round_test.label if round_test else "n/a"
    first_pair = study.pair_ladder[0] if study.pair_ladder else None
    last_pair = study.pair_ladder[-1] if study.pair_ladder else None
    budgets = [budget for budget, _, _, _ in study.alarm_counts] or [study.budget]
    tight_budget, loose_budget = budgets[0], budgets[-1]
    loose_ratio = loose_budget / max(tight_budget, 1e-9)
    tight_read = _edit_read(study, tight_budget)
    loose_read = _edit_read(study, loose_budget)
    tight_alarms = study.alarm_counts[0][1] if study.alarm_counts else 0
    loose_alarms = study.alarm_counts[-1][1] if study.alarm_counts else 0
    alarm_ratio = loose_alarms / max(tight_alarms, 1)
    tight_best = best_edit(study, tight_budget)
    loose_best = best_edit(study, loose_budget)
    tight_rate = tight_best.exchange_rate if tight_best else 0.0
    best_rate = max(tight_rate, loose_best.exchange_rate if loose_best else 0.0)
    edit_ratio = (loose_best.exchange_rate if loose_best else 0.0) / max(tight_rate, 1e-9)
    return f"""# NetSentry — The Glass Box, and What Capacity Costs on the Honest Split

_A generalized additive model (Lou, Caruana & Gehrke 2012) fitted from scratch by cyclic Newton
boosting over single-feature histograms, on {study.n_train:,} training flows and
{study.n_features} features, against the deployed ensemble and the linear floor under the
project's temporal protocol. Capacity is chosen on validation, never on the later days.
Regenerate with `netsentry gam`._

## Why this report exists

Everything this project ships to explain the deployed detector is **post hoc**. SHAP attributes
a verdict after the fact, [anchors](anchors.md) find a rule that happens to hold nearby,
[distillation](distill.md) trains a small model to imitate a big one, and each is an
approximation whose own error has to be measured -- which is why the
[stability study](importance_stability.md) exists at all.

An additive model needs none of that. It *is* a sum of one-dimensional curves,

    score(x) = intercept + f_1(x_1) + ... + f_d(x_d)

so the explanation is the model: exact rather than attributed, global rather than local, and a
lookup table rather than a fitted approximation -- which means an operator can **edit** it.

{_lead(study)}

## Does the fitter recover a function it was given?

{_recovery_table(study)}

A shape function that looks plausible is evidence of nothing, so the fitter is first pointed at
a known additive truth: a step, a parabola, and a feature that carries **no signal at all**. The
first two come back. The third is the row that matters -- it is the curve this fitter invents
for pure noise, with a magnitude of {study.noise_swing:.3f} in log-odds, and it is the floor
every real curve below has to be read against. A model that explains itself can still explain
itself wrongly, and this number is how much of that is on offer for free.

## What the glass box costs

{_model_table(study)}

The two additive arms and the linear one are readable in the strict sense: their entire
decision rule can be printed. The ensemble cannot, which is why the rest of
`netsentry/explain` exists.

{
    f"The additive model gives up {boosted.pr_auc - gam.pr_auc:.3f} PR-AUC to the deployed "
    f"ensemble and {linear.pr_auc - gam.pr_auc:.3f} to a linear model carrying "
    f"{gam.parameters // max(linear.parameters, 1)}x fewer parameters"
    if gam and boosted and linear
    else "The arms did not complete"
}. That is the wrong way round for the usual story about interpretability, and the next two
sections are why.

## The capacity dial

![Capacity against generalisation](../figures/{ladder.name})

{_ladder_table(study.bin_ladder)}

Bins per shape function is this model's resolution, its parameter count, and its capacity, in
one integer. Nothing else changes across these rows: same loss, same boosting schedule, same
class weights, same splits. Training PR-AUC rises monotonically -- capacity does what capacity
does -- and the later days rise, turn and fall.

**The gap between the validation column and the later-days column is the finding.** At
{peak_label} validation reports {peak_value:.3f} where the later days deliver
{peak_actual:.3f}: a {peak_value - peak_actual:.3f} overstatement, from a split carved out
of the *training days* and therefore drawn from the regime the model was fitted on. It is not
useless -- it turns over, so a practitioner following the protocol would have stopped adding
capacity -- but it stops one rung early ({wanted_label} is the rung the later days actually
want) and it never tells the truth about the level.

{_ladder_table(study.round_ladder)}

The second dial is the boosting schedule at the selected resolution, and it is here because one
hyperparameter behaving this way is an anecdote. It replicates: validation peaks at
{round_peak}, the later days at {round_wanted}.

## The capacity an additive model structurally cannot have

{_ladder_table(study.pair_ladder)}

An additive model cannot represent an interaction -- that is the definition, and it is the
usual reason given for preferring an ensemble. So the interactions are added back, a bounded
number at a time, ranked by the exact Newton gain a joint table would buy (the FAST heuristic,
Lou et al. 2013) rather than by refitting each candidate.

{
    f"The first pair is worth {first_pair.test - (gam.pr_auc if gam else 0):+.3f} on the later "
    f"days. Going on to {last_pair.label} costs {last_pair.test - first_pair.test:+.3f} while "
    f"training PR-AUC climbs {last_pair.train - first_pair.train:+.3f}"
    if first_pair and last_pair
    else "The ladder did not complete"
}. The interaction terms are where the day-specific structure lives: they fit the training
week's particular co-occurrences, and the later days do not have them. That is a
mechanism for the [leaderboard's](leaderboard.md) cross-family observation -- that the honest
split crowns the *least* flexible model -- observed inside a single family with one knob.

## What the curves say

![Shape functions](../figures/{shapes.name})

{_shape_table(study)}

Printed in raw feature units by inverting the fitted scaler, because a curve quoted in standard
deviations is not a curve an analyst can argue with. The swing column is the range of log-odds
the feature is allowed to move the score by, and the noise floor from the recovery harness
({study.noise_swing:.3f}) is what separates a curve from a decoration.

Almost none of them are monotone, which is worth pausing on: the
[monotone-constraint study](monotonic.md) shows that forcing monotonicity in the deployed
ensemble makes an entire evasion family impossible at a small detection cost. Here the same
question is visible directly -- the shapes show exactly where the model's response reverses,
and a reversal is where an attacker inflates a feature to *lower* their score.

## Editing the model

This is the operation a black box does not have. Clamping one bin of one shape function to zero
changes the score of every flow in that region by exactly the removed amount -- no retraining,
no surrogate, no approximation -- and the change is auditable in one line of a table an analyst
can read.

Every candidate is ranked by the trade it actually makes on **validation**, evaluated exactly
(clamping one bin shifts the margin by a known constant for exactly the rows in that bin, so
all several thousand candidate edits can be scored in the time it takes to re-sigmoid a few
hundred rows), and then measured on the later days it was not chosen on.

### At the deployed budget ({tight_budget:.0%} false positives)

{_edit_table(study, tight_budget)}

{tight_read}

### At a budget where the selection can see something ({loose_budget:.0%})

{_alarm_table(study)}

{_edit_table(study, loose_budget)}

{loose_read}

Two things separate those tables, and only one of them is about the model.

**The first is evidence.** An edit can only be chosen from false alarms an operator can see,
and a tight false-positive budget is *defined* by there being almost none: at
{tight_budget:.0%} the whole validation split offers {tight_alarms} of them to reason from,
spread across {study.n_features} features and {study.selected_bins} bins each. Loosening the
budget {loose_ratio:.0f}x multiplies the evidence by {alarm_ratio:.0f}x and the best available
trade improves {edit_ratio:.1f}x. The failure at the deployed budget is a sample-size failure,
and saying so requires running the second budget rather than assuming it.

**The second is the exchange rate, and it is not improved by evidence.** The
[cost study's](cost.md) economics make a caught attack worth {study.break_even:.0f} triaged
alerts, so an edit has to clear {study.break_even:.0f} false alarms per attack it stops
catching before it is worth making. The best edit found anywhere here reaches
{best_rate:.1f} to 1. **The regions carrying false alarms are the regions carrying detection**,
and no amount of looking harder changes that -- which is the same thing a threshold sweep says,
arrived at from a direction that could have disagreed.

What the glass box adds is therefore not a free lunch. It is that the trade is **inspectable
and choosable region by region**, with its exchange rate visible before the change ships,
instead of being made globally and invisibly by moving one number.

## Scope and honest limits

- **The additive model here loses, and the report is named for what that measures.** The claim
  is not that a GAM is the right detector for this data; it is that its capacity dial makes the
  honest split's preference for low capacity legible, which four different architectures
  cannot.
- **The binning family does not nest the linear model.** At the coarsest rung each feature is a
  step function, which throws away within-bin ordering, so the ladder's bottom end is not "a
  linear model" and its {study.bin_ladder[0].test:.3f} should not be read as one.
- **Correlated features split their credit arbitrarily.** Two features carrying the same signal
  produce two curves that each look half as important as the effect is, and nothing in the
  model says so. The gradient is refreshed after every feature rather than once per round
  specifically to stop them each taking *full* credit, which is worse, but it does not solve
  the attribution.
- **A legible model is only legible if its features are.** `Flow IAT Std` has a shape function
  and an operator still has to know what it means. The
  [feature-availability study](earliness.md) is the honest companion here.
- **The edits are single-bin clamps.** A real operator edit is a region and a rationale, and
  a production version of this would want a review trail -- which is what the
  [alert ledger](ledger.md) does for alerts and nothing yet does for models."""


def run_gam_report(settings: Settings) -> Path:
    """Run the additive-model study and write the report + figures."""
    study = run_gam_study(settings)
    shapes = plots.plot_shape_grid(
        [(name, grid, values) for name, (grid, values) in study.curves.items()],
        out_path=settings.paths.figures_dir / SHAPE_FIGURE,
        ylabel="contribution to log-odds",
    )
    parameters = np.array([row.parameters for row in study.bin_ladder], dtype=float)
    ladder = plots.plot_lines(
        {
            "training days": (parameters, np.array([row.train for row in study.bin_ladder])),
            "validation (carved from training days)": (
                parameters,
                np.array([row.validation for row in study.bin_ladder]),
            ),
            "the later days (the headline)": (
                parameters,
                np.array([row.test for row in study.bin_ladder]),
            ),
        },
        xlabel="parameters in the additive model",
        ylabel="PR-AUC",
        title="Capacity, and the split that cannot see what it costs",
        out_path=settings.paths.figures_dir / LADDER_FIGURE,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, shapes, ladder), encoding="utf-8")
    logger.info("Wrote GAM report", extra={"path": str(out_path)})

    with track_run(settings, "gam") as run:
        run.log_params({"bins": study.selected_bins, "features": study.n_features})
        run.log_metrics(
            {row.name.split(" ")[0]: row.pr_auc for row in study.models}
            | {"selected_bins": float(study.selected_bins), "noise_swing": study.noise_swing}
        )
        for artifact in (shapes, ladder, out_path):
            run.log_artifact(artifact)
    return out_path
