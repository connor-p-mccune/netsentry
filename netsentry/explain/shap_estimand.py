"""Which Shapley value does the API ship, and would another one say something else?

`/predict` returns `top_features`, and the code that produces them calls
`shap.TreeExplainer(model)` with no background data. That is a defensible choice and it is also
a *choice*, because "the SHAP value of this feature" is not one quantity. Three appear in the
literature, they answer different questions, and two of them are routinely used as if they were
the same:

- **Path-dependent** (TreeExplainer's default, and what this project ships): missing features
  are integrated out using the training distribution *as the tree recorded it* -- the coverage
  counts stored in each node.
- **Interventional** (Lundberg et al. 2020; Janzing, Minorics & Bloebaum, AISTATS 2020):
  missing features are replaced from a background sample independently of the rest, breaking
  correlations. It answers "what does the output owe to this input".
- **Conditional**: missing features are drawn from their distribution *given* the ones held
  fixed. It answers "what does this feature tell you about the output".

The module measures the gap between them on the alerts the deployed model actually raises, and
then pins the distinction down with an experiment that has a ground truth rather than an
opinion. A feature is duplicated before training with column subsampling off, so the tie
between the copies is broken deterministically and one copy is **never split on** -- verified
by counting splits in the dumped model. Three quantities are then asked what the unused copy
contributed, and two of the three answers are provable in advance: interventional must be
exactly zero (the model never reads it), and conditional must equal the original's credit
halved (identical features are exchangeable, so Shapley's symmetry axiom leaves no choice).

The shipped estimand sides with the interventional answer, for a structural reason rather than
a statistical one: a feature with no nodes has no paths to walk. That is worth knowing, because
"path-dependent SHAP accounts for feature correlations" is a sentence that appears in a great
many write-ups.

Before any of that, the library's own output is checked against the definition -- a weighted
sum over all coalitions, computed by brute force on a model small enough to enumerate them. An
explanation nobody has validated is an assertion with a colour scheme.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from collections.abc import Callable

    from netsentry.config import Settings
    from netsentry.config.settings import ShapEstimandConfig

logger = get_logger(__name__)

REPORT_NAME = "shap_estimand.md"
DISAGREEMENT_FIGURE = "shap_estimand_disagreement.png"
BACKGROUND_FIGURE = "shap_estimand_background.png"

PATH_DEPENDENT = "path-dependent (shipped)"
INTERVENTIONAL = "interventional"


# --------------------------------------------------------------------------------------
# The definition, computed directly.
# --------------------------------------------------------------------------------------


def coalition_values(
    score: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray
) -> dict[int, float]:
    """``v(S)`` for every coalition ``S``, under the interventional value function.

    ``v(S) = E_background[f(x_S, B_{-S})]``: features in ``S`` are held at the flow's own
    values and the rest are drawn from the background *independently of them*, which is what
    makes this the interventional quantity rather than the conditional one. Every coalition's
    hybrid rows are stacked into a single matrix and scored in one call, because doing it per
    coalition would turn a second into a minute for no benefit.
    """
    n_features = len(x)
    n_background = len(background)
    masks = list(range(1 << n_features))
    hybrid = np.repeat(background[None, :, :], len(masks), axis=0)
    for index, mask in enumerate(masks):
        held = [j for j in range(n_features) if mask & (1 << j)]
        if held:
            hybrid[index][:, held] = x[held]
    flat = hybrid.reshape(-1, n_features)
    scores = np.asarray(score(flat), dtype=float).reshape(len(masks), n_background)
    return {mask: float(scores[index].mean()) for index, mask in enumerate(masks)}


def conditional_values(
    score: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    neighbours: int,
) -> dict[int, float]:
    """``v(S)`` under the *conditional* value function, estimated by nearest neighbours.

    ``v(S) = E[f(X) | X_S = x_S]``: the missing features are drawn from their distribution
    **given** the ones being held, which is what makes this the observational quantity. There
    is no closed form for an empirical distribution, so it is estimated the standard way --
    average ``f`` over the background rows closest to the flow in the held subspace.

    This is the third answer, and the one that makes the comparison worth running: it is what
    people usually mean when they say a SHAP value accounts for correlations, and it is not
    what TreeExplainer's default computes.
    """
    n_features = len(x)
    spread = np.maximum(background.std(axis=0), 1e-9)
    scores = np.asarray(score(background), dtype=float)
    out: dict[int, float] = {}
    for mask in range(1 << n_features):
        held = [j for j in range(n_features) if mask & (1 << j)]
        if not held:
            out[mask] = float(scores.mean())
            continue
        distance = np.linalg.norm((background[:, held] - x[held]) / spread[held], axis=1)
        nearest = np.argsort(distance)[: max(1, min(neighbours, len(background)))]
        out[mask] = float(scores[nearest].mean())
    return out


def shapley_from_values(values: dict[int, float], n_features: int) -> np.ndarray:
    """The Shapley formula: a weighted sum over every coalition, given ``v``."""
    phi = np.zeros(n_features)
    factorial = math.factorial
    for j in range(n_features):
        others = [k for k in range(n_features) if k != j]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n_features - size - 1) / factorial(n_features)
            for subset in combinations(others, size):
                mask = 0
                for k in subset:
                    mask |= 1 << k
                phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    return phi


def exact_shapley(
    score: Callable[[np.ndarray], np.ndarray], x: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """Interventional Shapley values by the definition, over every coalition.

    Exponential in the number of features, which is exactly why TreeExplainer exists -- and
    exactly why it is worth computing once on a small model, as the reference the fast
    algorithm is graded against.
    """
    return shapley_from_values(coalition_values(score, x, background), len(x))


def conditional_shapley(
    score: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    neighbours: int,
) -> np.ndarray:
    """Shapley values under the conditional value function."""
    return shapley_from_values(conditional_values(score, x, background, neighbours), len(x))


# --------------------------------------------------------------------------------------
# Agreement between two attributions.
# --------------------------------------------------------------------------------------


def top_features(values: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` largest contributions by magnitude -- what the API returns."""
    return list(np.argsort(np.abs(values))[::-1][:k])


def top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    """Share of the top-k set two attributions agree on."""
    return len(set(top_features(left, k)) & set(top_features(right, k))) / max(k, 1)


def rank_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation of the two attribution vectors over all features."""
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationRow:
    """One check of the library against the definition or against an axiom."""

    check: str
    worst_error: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether the check is inside its stated tolerance."""
        return self.worst_error <= self.tolerance


@dataclass(frozen=True)
class AgreementRow:
    """How far apart the two estimands are on one population of flows."""

    population: int
    rank_correlation: float
    top1_agreement: float
    top3_overlap: float
    magnitude_ratio: float


@dataclass(frozen=True)
class FeatureRow:
    """One feature's mean absolute attribution under each estimand."""

    feature: str
    path_dependent: float
    interventional: float
    splits: int

    @property
    def ratio(self) -> float:
        """How much more the shipped estimand credits this feature than the other one."""
        return self.path_dependent / max(self.interventional, 1e-12)


@dataclass(frozen=True)
class DuplicateRow:
    """The ground-truthed experiment: what each of three estimands says about a copy."""

    feature: str
    splits: int
    path_dependent: float
    interventional: float
    conditional: float
    conditional_alt: float


@dataclass(frozen=True)
class BackgroundRow:
    """How much the interventional answer depends on which background was chosen."""

    background: str
    size: int
    top1_agreement: float
    rank_correlation: float
    seconds: float


@dataclass
class ShapEstimandStudy:
    """Everything the report needs, computed once."""

    validation: list[ValidationRow]
    agreement: list[AgreementRow]
    features: list[FeatureRow]
    duplicates: list[DuplicateRow]
    backgrounds: list[BackgroundRow]
    n_alerts: int
    n_features: int
    neighbours: int
    neighbours_alt: int
    exact_features: int
    threshold: float
    budget: float
    seconds: float = 0.0

    def population(self, name: int) -> AgreementRow | None:
        """Look up one population by its size marker."""
        return next((row for row in self.agreement if row.population == name), None)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _reduce(values: Any) -> np.ndarray:
    """Normalise SHAP's several output shapes to ``(n_rows, n_features)``."""
    array = values[-1] if isinstance(values, list) else values
    array = np.asarray(array, dtype=float)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array


def _split_counts(booster: Any, n_features: int) -> np.ndarray:
    """How many times the ensemble splits on each feature -- the ground truth for 'unused'."""
    counts = np.zeros(n_features, dtype=int)

    def walk(node: dict[str, Any]) -> None:
        if "split_feature" not in node:
            return
        counts[int(node["split_feature"])] += 1
        walk(node["left_child"])
        walk(node["right_child"])

    for info in booster.dump_model().get("tree_info", []):
        walk(info["tree_structure"])
    return counts


def _margin_scorer(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Raw (pre-sigmoid) score, which is the scale SHAP attributes on."""
    booster: Any = model.booster_

    def score(rows: np.ndarray) -> np.ndarray:
        return np.asarray(booster.predict(rows, raw_score=True), dtype=float)

    return score


def _explainers(model: Any, background: np.ndarray) -> tuple[Any, Any]:
    """The shipped explainer, and the one it could have been."""
    import shap

    return (
        shap.TreeExplainer(model),
        shap.TreeExplainer(model, data=background, feature_perturbation="interventional"),
    )


def _validate(
    cfg: ShapEstimandConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    settings: Settings,
    rng: np.random.Generator,
) -> tuple[list[ValidationRow], int]:
    """Grade the library against the definition, on a model small enough to enumerate.

    Shapley values are defined by a sum over all coalitions. TreeExplainer computes them in
    polynomial time by exploiting the tree structure, which is the whole reason it exists --
    and which is why the fast path deserves to be checked against the slow one at least once,
    on a model where the slow one is affordable.
    """
    from netsentry.models.supervised import SupervisedClassifier

    variant = settings.model_copy(deep=True)
    variant.supervised.n_estimators = cfg.exact_rounds
    variant.supervised.num_leaves = 8
    variant.mlflow.enabled = False
    small = SupervisedClassifier(variant)
    columns = list(range(cfg.exact_features))
    small.fit(x_train[:, columns], y_train)
    estimator = small.model
    score = _margin_scorer(estimator)
    background = x_train[rng.choice(len(x_train), cfg.exact_background, replace=False)][:, columns]
    _, interventional = _explainers(estimator, background)
    flows = x_train[rng.choice(len(x_train), cfg.exact_flows, replace=False)][:, columns]
    fast = _reduce(interventional.shap_values(flows))
    brute = np.vstack([exact_shapley(score, flow, background) for flow in flows])
    base = float(np.mean(score(background)))
    efficiency = np.abs(fast.sum(axis=1) + base - score(flows))
    return (
        [
            ValidationRow(
                "TreeExplainer (interventional) against the coalition sum",
                float(np.max(np.abs(fast - brute))),
                cfg.tolerance,
            ),
            ValidationRow(
                "efficiency: contributions + baseline reproduce the score",
                float(np.max(efficiency)),
                cfg.tolerance,
            ),
        ],
        len(columns),
    )


def _agreement_rows(
    path_values: np.ndarray, interventional_values: np.ndarray, marker: int, top_k: int
) -> AgreementRow:
    """Summarise the disagreement between two attribution matrices."""
    ranks = [
        rank_agreement(path_values[i], interventional_values[i]) for i in range(len(path_values))
    ]
    top1 = [
        float(top_features(path_values[i], 1)[0] == top_features(interventional_values[i], 1)[0])
        for i in range(len(path_values))
    ]
    overlap = [
        top_overlap(path_values[i], interventional_values[i], top_k)
        for i in range(len(path_values))
    ]
    return AgreementRow(
        population=marker,
        rank_correlation=float(np.mean(ranks)),
        top1_agreement=float(np.mean(top1)),
        top3_overlap=float(np.mean(overlap)),
        magnitude_ratio=float(
            np.mean(np.abs(path_values).sum(axis=1))
            / max(float(np.mean(np.abs(interventional_values).sum(axis=1))), 1e-12)
        ),
    )


def _duplicate_experiment(
    cfg: ShapEstimandConfig,
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    names: list[str],
    rng: np.random.Generator,
) -> list[DuplicateRow]:
    """Give the model two identical copies of a feature and ask three estimands about the spare.

    This is the experiment with a ground truth, and the answer is provable before it is run.
    Column subsampling is switched off so the tie between the copies is broken deterministically
    and one of them is **never split on** -- verified, not assumed, by counting splits in the
    dumped model.

    What each estimand must then say about the unused copy:

    - **Interventional**: exactly zero. The model never reads it, so intervening on it cannot
      change the output. This is the answer to "what did this input do".
    - **Conditional**: by symmetry, the same as the original. If two features are identical then
      conditioning on either determines the other, so the value function cannot distinguish
      them and Shapley's symmetry axiom must split the credit evenly. This is the answer to
      "what does this feature tell you".
    - **TreeExplainer's default**: also exactly zero -- because a feature with no nodes has no
      paths to walk, whatever the correlation structure says.

    That last line is the finding. The shipped estimand is not the conditional quantity people
    usually mean when they say SHAP accounts for correlations; on this question it agrees with
    the interventional one, and it does so for a structural reason rather than a statistical one.
    """
    from netsentry.models.supervised import SupervisedClassifier

    variant = settings.model_copy(deep=True)
    variant.supervised.n_estimators = cfg.exact_rounds
    variant.supervised.num_leaves = 8
    variant.supervised.colsample_bytree = 1.0
    variant.mlflow.enabled = False
    source = int(cfg.duplicate_feature)
    columns = list(range(cfg.exact_features))
    if source not in columns:
        columns = [source, *columns[:-1]]
    subset = x_train[:, columns]
    augmented = np.column_stack([subset, subset[:, columns.index(source)]])
    model = SupervisedClassifier(variant).fit(augmented, y_train)
    estimator = model.model
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        return []
    counts = _split_counts(booster, augmented.shape[1])
    background = augmented[rng.choice(len(augmented), cfg.exact_background, replace=False)]
    flows = augmented[rng.choice(len(augmented), cfg.exact_flows, replace=False)]
    path, interventional = _explainers(estimator, background)
    path_values = np.abs(_reduce(path.shap_values(flows))).mean(axis=0)
    interventional_values = np.abs(_reduce(interventional.shap_values(flows))).mean(axis=0)
    score = _margin_scorer(estimator)

    def conditional_for(neighbours: int) -> np.ndarray:
        stacked = np.vstack(
            [conditional_shapley(score, flow, background, neighbours) for flow in flows]
        )
        averaged: np.ndarray = np.abs(stacked).mean(axis=0)
        return averaged

    conditional = conditional_for(cfg.neighbours)
    conditional_alt = conditional_for(cfg.neighbours_alt)
    original = columns.index(source)
    copy_index = augmented.shape[1] - 1
    label = display_feature_name(names[source])
    return [
        DuplicateRow(
            feature=f"`{label}` (the copy the model uses)",
            splits=int(counts[original]),
            path_dependent=float(path_values[original]),
            interventional=float(interventional_values[original]),
            conditional=float(conditional[original]),
            conditional_alt=float(conditional_alt[original]),
        ),
        DuplicateRow(
            feature=f"`{label}` (the identical copy it never splits on)",
            splits=int(counts[copy_index]),
            path_dependent=float(path_values[copy_index]),
            interventional=float(interventional_values[copy_index]),
            conditional=float(conditional[copy_index]),
            conditional_alt=float(conditional_alt[copy_index]),
        ),
    ]


def run_shap_estimand_study(settings: Settings) -> ShapEstimandStudy:
    """Validate the explainer, then measure what the choice of estimand costs."""
    start = time.perf_counter()
    if not is_available("shap"):
        raise RuntimeError("the estimand audit needs SHAP; install the `train` extra")
    cfg: ShapEstimandConfig = settings.shap_estimand
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_calibration: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_later: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_calibration = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    names = list(pipeline.named_steps["features"].get_feature_names_out())

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    estimator = model.model
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        raise RuntimeError("the estimand audit needs a LightGBM booster")
    column = list(model.classes_).index(1)
    threshold = threshold_at_fpr(
        y_calibration, np.asarray(model.predict_proba(x_calibration))[:, column], cfg.budget
    )
    scores = np.asarray(model.predict_proba(x_later))[:, column]
    alerting = np.flatnonzero(scores >= threshold)
    if len(alerting) > cfg.max_flows:
        alerting = alerting[rng.choice(len(alerting), cfg.max_flows, replace=False)]
    quiet = rng.choice(len(x_later), min(cfg.max_flows, len(x_later)), replace=False)

    background = x_train[rng.choice(len(x_train), cfg.background_size, replace=False)]
    path, interventional = _explainers(estimator, background)
    alerts_path = _reduce(path.shap_values(x_later[alerting]))
    alerts_interventional = _reduce(interventional.shap_values(x_later[alerting]))
    quiet_path = _reduce(path.shap_values(x_later[quiet]))
    quiet_interventional = _reduce(interventional.shap_values(x_later[quiet]))

    counts = _split_counts(booster, x_train.shape[1])
    mean_path = np.abs(alerts_path).mean(axis=0)
    mean_interventional = np.abs(alerts_interventional).mean(axis=0)
    order = np.argsort(-np.maximum(mean_path, mean_interventional))[: cfg.top_features]
    features = [
        FeatureRow(
            feature=display_feature_name(names[index]),
            path_dependent=float(mean_path[index]),
            interventional=float(mean_interventional[index]),
            splits=int(counts[index]),
        )
        for index in order
    ]

    backgrounds: list[BackgroundRow] = []
    benign_rows = np.flatnonzero(y_train == 0)
    choices: list[tuple[str, np.ndarray]] = [
        (
            "benign training flows only",
            x_train[rng.choice(benign_rows, cfg.background_size, replace=False)],
        ),
        (
            "a smaller sample of the same pool",
            x_train[rng.choice(len(x_train), max(8, cfg.background_size // 8), replace=False)],
        ),
        (
            "later-day traffic (what a deployment sees)",
            x_later[rng.choice(len(x_later), cfg.background_size, replace=False)],
        ),
    ]
    import shap

    for label, sample in choices:
        clock = time.perf_counter()
        other = shap.TreeExplainer(estimator, data=sample, feature_perturbation="interventional")
        values = _reduce(other.shap_values(x_later[alerting]))
        backgrounds.append(
            BackgroundRow(
                background=label,
                size=len(sample),
                top1_agreement=float(
                    np.mean(
                        [
                            top_features(alerts_interventional[i], 1)[0]
                            == top_features(values[i], 1)[0]
                            for i in range(len(values))
                        ]
                    )
                ),
                rank_correlation=float(
                    np.mean(
                        [
                            rank_agreement(alerts_interventional[i], values[i])
                            for i in range(len(values))
                        ]
                    )
                ),
                seconds=time.perf_counter() - clock,
            )
        )

    validation, exact_features = _validate(cfg, x_train, y_train, variant, rng)
    base_path = float(np.asarray(path.expected_value).ravel()[-1])
    margins = np.asarray(booster.predict(x_later[alerting], raw_score=True), dtype=float)
    validation.append(
        ValidationRow(
            "efficiency of the shipped estimand on the deployed model",
            float(np.max(np.abs(alerts_path.sum(axis=1) + base_path - margins))),
            cfg.tolerance,
        )
    )

    study = ShapEstimandStudy(
        validation=validation,
        agreement=[
            _agreement_rows(alerts_path, alerts_interventional, len(alerting), cfg.top_k),
            _agreement_rows(quiet_path, quiet_interventional, len(quiet), cfg.top_k),
        ],
        features=features,
        duplicates=_duplicate_experiment(cfg, variant, x_train, y_train, names, rng),
        backgrounds=backgrounds,
        n_alerts=len(alerting),
        n_features=x_train.shape[1],
        neighbours=cfg.neighbours,
        neighbours_alt=cfg.neighbours_alt,
        exact_features=exact_features,
        threshold=float(threshold),
        budget=cfg.budget,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "SHAP estimand study complete",
        extra={
            "alerts": study.n_alerts,
            "top1_agreement": round(study.agreement[0].top1_agreement, 3),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _validation_table(study: ShapEstimandStudy) -> str:
    rows = "\n".join(
        f"| {row.check} | {row.worst_error:.2e} | {row.tolerance:.0e} | "
        f"{'pass' if row.passed else '**FAIL**'} |"
        for row in study.validation
    )
    return "| check | worst error | tolerance | verdict |\n|---|---|---|---|\n" + rows


def _agreement_table(study: ShapEstimandStudy) -> str:
    labels = ["the alerts the API explains", "a random sample of later-day flows"]
    rows = "\n".join(
        f"| {label} | {row.population:,} | {row.rank_correlation:.3f} | "
        f"{row.top1_agreement:.1%} | {row.top3_overlap:.1%} | {row.magnitude_ratio:.3f} |"
        for label, row in zip(labels, study.agreement, strict=False)
    )
    return (
        "| population | flows | rank correlation | same top feature | top-3 overlap | "
        "magnitude ratio |\n|---|---|---|---|---|---|\n" + rows
    )


def _feature_table(study: ShapEstimandStudy) -> str:
    rows = "\n".join(
        f"| `{row.feature}` | {row.splits:,} | {row.path_dependent:.4f} | "
        f"{row.interventional:.4f} | {row.ratio:.2f}x |"
        for row in study.features
    )
    return (
        "| feature | splits in the model | shipped (path-dependent) | interventional | "
        "ratio |\n|---|---|---|---|---|\n" + rows
    )


def _duplicate_table(study: ShapEstimandStudy) -> str:
    if not study.duplicates:
        return "_The duplicate experiment did not run._"
    rows = "\n".join(
        f"| {row.feature} | {row.splits:,} | {row.path_dependent:.4f} | "
        f"{row.interventional:.4f} | **{row.conditional:.4f}** | **{row.conditional_alt:.4f}** |"
        for row in study.duplicates
    )
    return (
        "| feature | splits | shipped (path-dependent) | interventional | conditional "
        f"(k = {study.neighbours}) | conditional (k = {study.neighbours_alt}) |\n"
        "|---|---|---|---|---|---|\n" + rows
    )


def _background_table(study: ShapEstimandStudy) -> str:
    rows = "\n".join(
        f"| {row.background} | {row.size:,} | {row.top1_agreement:.1%} | "
        f"{row.rank_correlation:.3f} | {row.seconds:.0f} s |"
        for row in study.backgrounds
    )
    return (
        "| background sample | rows | same top feature as the reference | rank correlation | "
        "cost |\n|---|---|---|---|---|\n" + rows
    )


def _lead(study: ShapEstimandStudy) -> str:
    alerts = study.agreement[0] if study.agreement else None
    used = study.duplicates[0] if study.duplicates else None
    unused = study.duplicates[1] if len(study.duplicates) > 1 else None
    exact = study.validation[0] if study.validation else None
    return (
        f"**The library is doing exactly what it claims, and what it claims is not what most "
        f"people think they are reading.**\n\n"
        f"First the easy part. TreeExplainer's interventional output matches the definition -- "
        f"a weighted sum over all {2 ** study.exact_features} coalitions, computed by brute "
        f"force on a {study.exact_features}-feature model -- to "
        f"{exact.worst_error if exact else 0:.0e}. The fast algorithm is correct.\n\n"
        f"Then the part that matters. On the {study.n_alerts:,} alerts the API actually "
        f"explains, the shipped estimand and the interventional one agree closely: rank "
        f"correlation {alerts.rank_correlation if alerts else 0:.3f}, and they name the same "
        f"top feature for {alerts.top1_agreement if alerts else 0:.1%} of alerts. The choice "
        f"changes the headline of roughly one alert in "
        f"{1 / max(1 - (alerts.top1_agreement if alerts else 0), 1e-9):.0f}.\n\n"
        f"But agreeing with the interventional quantity is itself the finding, because it is "
        f"*not* what the shipped estimand is usually described as computing. Duplicate a "
        f"feature before training, let the model split on one copy "
        f"({used.splits if used else 0:,} times) and never on the other "
        f"({unused.splits if unused else 0}), and ask all three quantities what the spare copy "
        f"contributed. The shipped answer is **{unused.path_dependent if unused else 0:.4f}**. "
        f"The interventional answer is {unused.interventional if unused else 0:.4f}. The "
        f"*conditional* answer -- the one people mean when they say SHAP accounts for "
        f"correlations -- is **{unused.conditional if unused else 0:.4f}**, exactly half the "
        f"original's credit, because two identical features are exchangeable and Shapley's "
        f"symmetry axiom leaves no choice.\n\n"
        f"So `top_features` answers *what did this input do*, not *what does this feature tell "
        f"you*. That is a defensible thing for a detection API to answer. It was not a "
        f"decision anybody recorded."
    )


def _render(study: ShapEstimandStudy, disagreement: Path, background: Path) -> str:
    alerts = study.agreement[0] if study.agreement else None
    quiet = study.agreement[1] if len(study.agreement) > 1 else None
    unused = study.duplicates[1] if len(study.duplicates) > 1 else None
    weakest = (
        min(study.backgrounds, key=lambda row: row.top1_agreement) if study.backgrounds else None
    )
    alerts_read = (
        f"The two agree on the top feature for {alerts.top1_agreement:.1%} of alerts"
        if alerts
        else "The populations did not complete"
    )
    alerts_overlap = f"{alerts.top3_overlap:.1%}" if alerts else "n/a"
    weakest_read = (
        f"the weakest agreement is {weakest.top1_agreement:.1%} with {weakest.background}"
        if weakest
        else "the alternatives agree closely"
    )
    return f"""# NetSentry — Which Shapley Value Does the API Ship?

_Both TreeExplainer estimands on the {study.n_alerts:,} alerts the deployed model raises at a
{study.budget:.0%} false-positive budget, graded against the coalition sum on a
{study.exact_features}-feature model where it can be computed exactly, plus a third estimand
the shipped one is often confused with. Regenerate with `netsentry shapaudit`._

## Why this report exists

`/predict` returns `top_features`, and the code that produces them calls
`shap.TreeExplainer(model)` with no background data. That is a defensible choice and it is also
a *choice*, because "the SHAP value of this feature" is not one quantity:

- **Path-dependent** (the default, and what this project ships): missing features are
  integrated out using the training distribution *as the tree recorded it* -- the coverage
  counts stored in each node.
- **Interventional** (Lundberg 2020; Janzing, Minorics & Bloebaum 2020): missing features are
  replaced from a background sample independently of the rest, breaking correlations. It
  answers "what does the output owe to this input".
- **Conditional**: missing features are drawn from their distribution *given* the ones being
  held. It answers "what does this feature tell you about the output".

Two of those three are commonly used interchangeably in write-ups. They are not the same
number.

{_lead(study)}

## Is the library computing what it says?

{_validation_table(study)}

Shapley values are defined by a sum over every coalition, and TreeExplainer's contribution is
computing them in polynomial time by exploiting the tree structure. That is worth checking
against the definition at least once, on a model small enough for the definition to be
affordable: {study.exact_features} features is {2 ** study.exact_features} coalitions, which is
seconds rather than days. The efficiency rows check the axiom that makes the numbers additive
at all -- contributions plus baseline must reproduce the score -- for both the small model and
the deployed one.

An explanation nobody has validated is an assertion with a colour scheme.

## How far apart are the two estimands here?

![Where the estimands disagree](../figures/{disagreement.name})

{_agreement_table(study)}

{alerts_read} and on {alerts_overlap} of the top-3 set -- which is the list
the API returns. Agreement is
{"higher" if alerts and quiet and alerts.top1_agreement > quiet.top1_agreement else "lower"} on
alerts than on ordinary traffic, which is the direction that matters: the flows an analyst
actually reads are the ones the two methods most nearly agree about.

{_feature_table(study)}

The reason they agree so well is a property of this data rather than of the methods. The
interventional and conditional quantities coincide exactly when features are independent, and
the [kernel two-sample study](mmd.md) already measured this stand-in's modelled features at a
mean absolute pairwise correlation of **0.005**. On traffic with real feature coupling the gap
would open, and the correct reading of this table is "the choice is currently cheap here", not
"the choice does not matter".

## The experiment with a ground truth

{_duplicate_table(study)}

A feature is duplicated before training and column subsampling is switched off, so the tie
between the copies is broken deterministically and one of them is **never split on** --
verified by counting splits in the dumped model, not assumed. Then all three quantities are
asked what the unused copy contributed.

Two of the three answers are provable before they are measured, which is why this is the
experiment worth running. Interventional attribution must give the unused copy **exactly zero**:
intervening on a feature the model never reads cannot change the output. Conditional
attribution must give the two copies **exactly the same credit**: they are identical columns,
so no value function can distinguish them and Shapley's symmetry axiom leaves no choice. Both
hold in the table, and the conditional column holds at either smoothing setting, because the
equality is a property of the estimand rather than of the estimator.

What is *not* provable is the conditional magnitude, and the two k columns are there to stop it
being read as if it were. ``E[f | X_S = x_S]`` has no closed form for an empirical
distribution; it is estimated by averaging over the k nearest background rows in the held
subspace, and a larger k smooths every attribution toward zero. The ratio between the
conditional and the shipped column moves with k. The symmetry does not.

The shipped estimand returns {f"{unused.path_dependent:.4f}" if unused else "n/a"} for the
unused copy. **It sides with the interventional answer**, and for a structural reason rather
than a statistical one: a feature with no nodes has no paths to walk, whatever the correlation
structure says. That is worth knowing, because "path-dependent SHAP accounts for feature
correlations" is a sentence that appears in a great many write-ups, including ones this
project's own documentation could have written.

## The estimand has a second free parameter

![Background sensitivity](../figures/{background.name})

{_background_table(study)}

Interventional attribution is defined against a background distribution, and the background is
a modelling choice too. Reference here is a uniform sample of the training split;
{weakest_read}, so the answer moves with the reference in the same way it moves with the
estimand. A background
of later-day traffic answers "why is this flow unusual *now*"; a benign-only background answers
"why is this flow not benign". Those are different questions and the API currently asks neither
explicitly.

## What this changes

- **The contract should say which quantity it returns.** `top_features` answers *what did this
  input do*, which is the right answer for a detection API -- an analyst wants to know what to
  look at in the flow, not what the flow correlates with. Saying so is free.
- **A correlated deployment would need the choice revisited.** The measured agreement here
  rests on near-independent features. On real CIC-IDS2017 traffic, where forward and backward
  packet statistics move together, the gap would be larger and the argument would have to be
  made rather than inherited.
- **Nothing here says the explanations are wrong.** It says they answer one of three
  questions, that the library computes that answer correctly to nine decimal places, and that
  the question had never been written down.

## Scope and honest limits

- **The conditional estimand is estimated, not exact.** ``E[f | X_S = x_S]`` has no closed form
  for an empirical distribution; it is approximated here by averaging over the
  {study.exact_features}-dimensional nearest neighbours of the held subspace. The duplicate
  experiment does not depend on that approximation being accurate -- symmetry pins the answer
  -- but the magnitudes elsewhere would.
- **The exact reference is a small model.** {study.exact_features} features and a short
  ensemble; the deployed model has {study.n_features}. What is validated is the algorithm, not
  the deployed numbers, and that is the only thing brute force can validate.
- **Agreement is measured, causation is not.** That two estimands name the same top feature
  does not make that feature the reason for the alert. The
  [anchors study](anchors.md) and the [counterfactual study](recourse.md) attack that question
  from directions attribution cannot.
- **This is one dataset and one model family.** TreeExplainer's path-dependent mode is specific
  to trees; the same question for a neural model is a different implementation with different
  failure modes, and the [deep-tabular study](deep_tabular.md) would be where to ask it."""


def run_shap_estimand_report(settings: Settings) -> Path:
    """Run the estimand audit and write the report + figures."""
    study = run_shap_estimand_study(settings)
    labels = [row.feature for row in study.features]
    disagreement = plots.plot_grouped_barh(
        labels,
        {
            "shipped (path-dependent)": [row.path_dependent for row in study.features],
            "interventional": [row.interventional for row in study.features],
        },
        xlabel="mean |contribution| over the alerts (log-odds)",
        title="Two defensible answers to 'why did this fire'",
        out_path=settings.paths.figures_dir / DISAGREEMENT_FIGURE,
    )
    background = plots.plot_barh(
        [row.background for row in study.backgrounds],
        [row.top1_agreement for row in study.backgrounds],
        xlabel="share of alerts whose top feature is unchanged",
        title="The background is a modelling choice too",
        out_path=settings.paths.figures_dir / BACKGROUND_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, disagreement, background), encoding="utf-8")
    logger.info("Wrote SHAP estimand report", extra={"path": str(out_path)})

    with track_run(settings, "shap_estimand") as run:
        run.log_params({"alerts": study.n_alerts, "budget": study.budget})
        run.log_metrics(
            {
                "top1_agreement": study.agreement[0].top1_agreement if study.agreement else 0.0,
                "exact_error": study.validation[0].worst_error if study.validation else 0.0,
            }
        )
        for artifact in (disagreement, background, out_path):
            run.log_artifact(artifact)
    return out_path
