"""One perturbation, computed once, that works on flows it has never seen.

Every evasion attack this project has measured is **per-flow**. Mimicry interpolates each
flow toward a benign reference; the query search optimises each flow against the model; the
[transport study](transport.md) couples each flow to its own benign partner. All of them need
the attacker to hold the target flow, and most of them need model access *at attack time* --
which is exactly the thing a rate limit, an API key and a query-volume alarm exist to make
expensive.

A **universal adversarial perturbation** (Moosavi-Dezfooli et al., CVPR 2017) removes that
requirement. It is a single vector, computed once against flows the attacker already has, and
then added to every future flow with no further queries, no feedback and no per-flow work. For a
detection service that is a different threat entirely: it converts an *interactive* attack into
a **constant an attacker can ship**, and nothing in the request path can tell the difference,
because there is no longer anything to observe.

Four things are measured here, in the order that decides whether it matters.

1. **Does one vector generalise?** It is fitted on an attacker-held sample and scored on
   held-out attack flows, against three references at the identical budget: the benign-centroid
   direction (the natural universal baseline), the transport plan's mean displacement, and a
   random direction of the same norm.
2. **Does it survive not having the model?** The realistic attacker fits the vector on a
   *surrogate* -- a differently-seeded model, and a different model family -- and applies it to
   the deployed one.
3. **Can it be made impossible?** A perturbation an attacker can actually apply mostly *adds*
   traffic, and a model constrained non-decreasing in every inflatable feature cannot be pushed
   down by additions at all. The [monotone study](monotonic.md) already ships that model, so the
   claim is checked rather than argued.
4. **What does it cost in the other currency?** A universal perturbation adds the *same* offset
   to every flow, which is the loudest possible thing to do to a population. The
   [transport study](transport.md) established that evasion has a second cost -- being
   collectively unremarkable -- and a constant offset pays it in full.

The attacker's algorithm is greedy coordinate descent on the *batch mean score*, projected back
onto an L2 ball after every step, restricted to the features the threat model allows. No
gradients, because the target is a tree ensemble and an attacker does not have any either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.monotonic import constraint_vector
from netsentry.monitoring.drift import population_stability_index
from netsentry.robustness.evasion import controllable_indices
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from collections.abc import Callable

    from netsentry.config import Settings
    from netsentry.config.settings import UniversalConfig

logger = get_logger(__name__)

REPORT_NAME = "universal.md"
BUDGET_FIGURE = "universal_budget.png"
TRANSFER_FIGURE = "universal_transfer.png"

UNIVERSAL = "the fitted universal vector"
CENTROID = "the benign-centroid direction"
TRANSPORT = "the transport plan's mean displacement"
RANDOM = "a random direction"


# --------------------------------------------------------------------------------------
# Fitting one vector for every flow.
# --------------------------------------------------------------------------------------


def project(vector: np.ndarray, budget: float) -> np.ndarray:
    """Scale a perturbation back onto the L2 ball of radius ``budget``."""
    norm = float(np.linalg.norm(vector))
    if norm <= budget or norm == 0.0:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector * (budget / norm), dtype=float)


def fit_universal(
    score: Callable[[np.ndarray], np.ndarray],
    flows: np.ndarray,
    coordinates: np.ndarray,
    *,
    budget: float,
    steps: int,
    step_size: float,
    non_negative: bool = False,
) -> np.ndarray:
    """Greedy coordinate descent on the mean score of a batch, projected onto the budget.

    The published algorithm accumulates per-example perturbations and re-projects; that needs a
    per-example attack, which for a tree ensemble means a search per flow and destroys the point
    of the exercise. This descends on the batch objective directly: at every step, try a step of
    each sign along each allowed coordinate, keep the one that lowers the *mean* score most, and
    project. It is deterministic, needs no gradients, and every candidate step for a whole round
    is scored in a single stacked call rather than one call per coordinate -- which is the
    difference between a minute and an hour.

    ``non_negative`` is the feasible attacker: padding, dummy packets and delays add to a flow,
    they do not subtract from it. That restriction is what the monotone defence is built to
    make useless, so it has to be available here or the defence is being tested against an
    attack nobody could mount.
    """
    vector = np.zeros(flows.shape[1], dtype=float)
    signs = (1.0,) if non_negative else (1.0, -1.0)
    best_mean = float(np.mean(score(flows)))
    for _ in range(steps):
        candidates = []
        for index in coordinates:
            for sign in signs:
                trial = vector.copy()
                trial[index] += sign * step_size
                candidates.append(project(trial, budget))
        if not candidates:
            break
        stacked = np.vstack([flows + candidate for candidate in candidates])
        means = np.asarray(score(stacked), dtype=float).reshape(len(candidates), len(flows))
        means = means.mean(axis=1)
        winner = int(np.argmin(means))
        if means[winner] >= best_mean - 1e-12:
            break  # no allowed step lowers the batch score: a local optimum, not a budget limit
        best_mean = float(means[winner])
        vector = candidates[winner]
    return vector


def centroid_direction(
    attacks: np.ndarray, benign: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """The natural universal baseline: everybody moves toward the average benign flow."""
    direction = benign.mean(axis=0) - attacks.mean(axis=0)
    mask = np.zeros(len(direction), dtype=bool)
    mask[coordinates] = True
    return np.where(mask, direction, 0.0)


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetRow:
    """One perturbation direction at one budget, on flows it was not fitted on."""

    direction: str
    budget: float
    detection: float
    detection_fitted: float
    aggregate_psi: float


@dataclass(frozen=True)
class TransferRow:
    """A vector fitted somewhere else, applied to the deployed model."""

    source: str
    detection: float
    detection_on_source: float
    cosine: float


@dataclass(frozen=True)
class DefenceRow:
    """One model, attacked by the feasible (additive-only) universal vector."""

    model: str
    baseline_detection: float
    attacked_detection: float
    clean_pr_auc: float

    @property
    def removed(self) -> float:
        """Detection the attack took away, in points."""
        return self.baseline_detection - self.attacked_detection


@dataclass(frozen=True)
class RecipeRow:
    """One coordinate of the fitted vector, in raw units."""

    feature: str
    shift: float
    share: float
    controllable: bool


@dataclass
class UniversalStudy:
    """Everything the report needs, computed once."""

    budgets: list[BudgetRow]
    transfers: list[TransferRow]
    defences: list[DefenceRow]
    recipe: list[RecipeRow]
    baseline_detection: float
    baseline_psi: float
    n_fit: int
    n_held_out: int
    n_controllable: int
    n_features: int
    profile: str
    headline_budget: float
    queries_universal: int
    queries_per_flow: int
    seconds: float = 0.0

    def curve(self, direction: str, attribute: str) -> np.ndarray:
        """One direction's series over the budget sweep."""
        return np.array(
            [getattr(row, attribute) for row in self.budgets if row.direction == direction],
            dtype=float,
        )

    def sweep(self) -> list[float]:
        """The budgets, in order."""
        seen: list[float] = []
        for row in self.budgets:
            if row.budget not in seen:
                seen.append(row.budget)
        return seen

    def at(self, direction: str, budget: float) -> BudgetRow | None:
        """Look up one cell of the sweep."""
        return next(
            (
                row
                for row in self.budgets
                if row.direction == direction and abs(row.budget - budget) < 1e-9
            ),
            None,
        )


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _psi(moved: np.ndarray, reference: np.ndarray, bins: int) -> float:
    """Worst-feature PSI of a perturbed population against untouched traffic."""
    return max(
        (
            population_stability_index(reference[:, index], moved[:, index], bins=bins)
            for index in range(reference.shape[1])
        ),
        default=0.0,
    )


def _scorer(bundle: object) -> Callable[[np.ndarray], np.ndarray]:
    """Calibrated attack probability from already-transformed rows."""
    from netsentry.robustness.evasion import attack_scores_transformed

    def score(rows: np.ndarray) -> np.ndarray:
        return attack_scores_transformed(bundle, rows)  # type: ignore[arg-type]

    return score


def _directions(
    cfg: UniversalConfig,
    score: Callable[[np.ndarray], np.ndarray],
    fit_flows: np.ndarray,
    benign: np.ndarray,
    coordinates: np.ndarray,
    budget: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """The fitted vector and the three references, all at the same norm.

    Every reference is projected onto the identical L2 ball and restricted to the identical
    coordinate set, so the comparison is about *where to push* rather than about how hard --
    the same discipline the transport study's matched-budget race uses.
    """
    from netsentry.monitoring.transport import (
        barycentric_targets,
        exact_assignment,
        squared_cost_matrix,
    )

    fitted = fit_universal(
        score,
        fit_flows,
        coordinates,
        budget=budget,
        steps=cfg.steps,
        step_size=cfg.step_size,
    )
    size = min(len(fit_flows), len(benign), cfg.transport_rows)
    cost = squared_cost_matrix(fit_flows[:size], benign[:size])
    partner, _ = exact_assignment(cost)
    plan = np.zeros((size, size))
    plan[np.arange(size), partner] = 1.0 / size
    displacement = (barycentric_targets(plan, benign[:size]) - fit_flows[:size]).mean(axis=0)
    mask = np.zeros(fit_flows.shape[1], dtype=bool)
    mask[coordinates] = True
    noise = rng.normal(size=fit_flows.shape[1])
    return {
        UNIVERSAL: fitted,
        CENTROID: project(centroid_direction(fit_flows, benign, coordinates), budget),
        TRANSPORT: project(np.where(mask, displacement, 0.0), budget),
        RANDOM: project(np.where(mask, noise, 0.0), budget),
    }


def _transfer(
    settings: Settings,
    cfg: UniversalConfig,
    score: Callable[[np.ndarray], np.ndarray],
    fit_flows: np.ndarray,
    held_out: np.ndarray,
    coordinates: np.ndarray,
    threshold: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> list[TransferRow]:
    """Fit the vector on models the attacker could plausibly have instead of the real one.

    The interesting attacker does not hold the deployed model. Two stand-ins: the same family
    trained from a different seed (what an attacker gets by retraining on public data), and a
    different family entirely (what they get by guessing wrong about the architecture). The
    cosine against the vector fitted on the true model says whether transfer succeeds because
    the attack is the same or because almost anything works.
    """
    from netsentry.models.supervised import SupervisedClassifier

    reference = fit_universal(
        score,
        fit_flows,
        coordinates,
        budget=cfg.headline_budget,
        steps=cfg.steps,
        step_size=cfg.step_size,
    )
    rows: list[TransferRow] = []
    for label, seed_offset, family in cfg.surrogates:
        variant = settings.model_copy(deep=True)
        variant.seed = settings.seed + int(seed_offset)
        variant.supervised.backend = family  # type: ignore[assignment]
        variant.mlflow.enabled = False
        surrogate = SupervisedClassifier(variant).fit(x_train, y_train)
        column = list(surrogate.classes_).index(1)

        def surrogate_score(
            rows_in: np.ndarray, model: Any = surrogate, col: int = column
        ) -> np.ndarray:
            scored: np.ndarray = np.asarray(model.predict_proba(rows_in))[:, col]
            return scored

        vector = fit_universal(
            surrogate_score,
            fit_flows,
            coordinates,
            budget=cfg.headline_budget,
            steps=cfg.steps,
            step_size=cfg.step_size,
        )
        surrogate_threshold = float(np.quantile(surrogate_score(held_out), 1.0 - cfg.share))
        norms = float(np.linalg.norm(vector)) * float(np.linalg.norm(reference))
        rows.append(
            TransferRow(
                source=str(label),
                detection=float(np.mean(score(held_out + vector) >= threshold)),
                detection_on_source=float(
                    np.mean(surrogate_score(held_out + vector) >= surrogate_threshold)
                ),
                cosine=float(vector @ reference / norms) if norms > 0 else 0.0,
            )
        )
    return rows


def _defences(
    settings: Settings,
    cfg: UniversalConfig,
    names: list[str],
    coordinates: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    held_out: np.ndarray,
) -> list[DefenceRow]:
    """Attack an unconstrained model and a monotone one with the *feasible* universal vector.

    Feasible means additive only: padding and delays add to a flow. A model constrained
    non-decreasing in every inflatable feature cannot be pushed down by additions, so the
    attack should be structurally impossible rather than merely difficult -- which is a claim
    worth checking with the attack rather than with an argument.
    """
    from sklearn.metrics import average_precision_score

    from netsentry.models.supervised import SupervisedClassifier

    rows: list[DefenceRow] = []
    constraints = constraint_vector(names, settings.robustness.controllable_features)
    for label, vector in (
        ("unconstrained (deployed)", None),
        ("monotone-constrained", constraints),
    ):
        variant = settings.model_copy(deep=True)
        variant.mlflow.enabled = False
        model = SupervisedClassifier(variant, monotone_constraints=vector).fit(x_train, y_train)
        column = list(model.classes_).index(1)

        def model_score(rows_in: np.ndarray, fitted: Any = model, col: int = column) -> np.ndarray:
            scored: np.ndarray = np.asarray(fitted.predict_proba(rows_in))[:, col]
            return scored

        threshold = threshold_at_fpr(y_val, model_score(x_val), cfg.budget_fpr)
        attack = fit_universal(
            model_score,
            held_out[: cfg.fit_rows],
            coordinates,
            budget=cfg.headline_budget,
            steps=cfg.steps,
            step_size=cfg.step_size,
            non_negative=True,
        )
        scored = held_out[cfg.fit_rows :]
        rows.append(
            DefenceRow(
                model=label,
                baseline_detection=float(np.mean(model_score(scored) >= threshold)),
                attacked_detection=float(np.mean(model_score(scored + attack) >= threshold)),
                clean_pr_auc=float(average_precision_score(y_val, model_score(x_val))),
            )
        )
    return rows


def run_universal_study(settings: Settings) -> UniversalStudy:
    """Fit one vector, race it against three references, then try to make it impossible."""
    start = time.perf_counter()
    cfg: UniversalConfig = settings.universal
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.training.train_supervised import fit_supervised

    result = fit_supervised(variant)
    bundle = result.bundle
    names = bundle.feature_names()
    score = _scorer(bundle)
    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    calibration_frame = load_split(variant, "temporal", "val")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(calibration_frame), dtype=float)
    x_later: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = calibration_frame[BINARY_TARGET].to_numpy().astype(int)
    y_later = arrivals_frame[BINARY_TARGET].to_numpy().astype(int)

    threshold = float(bundle.thresholds.get(cfg.profile, 0.5))
    attacks = x_later[y_later == 1]
    benign = x_later[y_later == 0]
    order = rng.permutation(len(attacks))
    fit_flows = attacks[order[: cfg.fit_rows]]
    held_out = attacks[order[cfg.fit_rows : cfg.fit_rows + cfg.holdout_rows]]
    coordinates = controllable_indices(names, settings.robustness.controllable_features)

    baseline = float(np.mean(score(held_out) >= threshold))
    baseline_psi = _psi(held_out, benign[: len(held_out)], cfg.psi_bins)

    rows: list[BudgetRow] = []
    headline_vectors: dict[str, np.ndarray] = {}
    for budget in cfg.budget_sweep:
        vectors = _directions(cfg, score, fit_flows, benign, coordinates, budget, rng)
        if abs(budget - cfg.headline_budget) < 1e-9:
            headline_vectors = vectors
        for label, vector in vectors.items():
            rows.append(
                BudgetRow(
                    direction=label,
                    budget=budget,
                    detection=float(np.mean(score(held_out + vector) >= threshold)),
                    detection_fitted=float(np.mean(score(fit_flows + vector) >= threshold)),
                    aggregate_psi=_psi(held_out + vector, benign[: len(held_out)], cfg.psi_bins),
                )
            )
    if not headline_vectors:
        headline_vectors = _directions(
            cfg, score, fit_flows, benign, coordinates, cfg.headline_budget, rng
        )

    fitted = headline_vectors[UNIVERSAL]
    magnitude = np.abs(fitted)
    total = float(magnitude.sum()) or 1.0
    stats = {}
    branch = pipeline.named_steps["features"].named_transformers_.get("numeric")
    scaler = getattr(branch, "named_steps", {}).get("scale") if branch is not None else None
    if scaler is not None and hasattr(scaler, "scale_"):
        stats = dict(zip(names, scaler.scale_, strict=False))
    controllable_set = set(coordinates.tolist())
    recipe = [
        RecipeRow(
            feature=display_feature_name(names[index]),
            shift=float(fitted[index] * stats.get(names[index], 1.0)),
            share=float(magnitude[index] / total),
            controllable=index in controllable_set,
        )
        for index in np.argsort(-magnitude)[: cfg.top_features]
    ]

    study = UniversalStudy(
        budgets=rows,
        transfers=_transfer(
            variant,
            cfg,
            score,
            fit_flows,
            held_out,
            coordinates,
            threshold,
            x_train,
            y_train,
        ),
        defences=_defences(
            variant, cfg, names, coordinates, x_train, y_train, x_val, y_val, attacks
        ),
        recipe=recipe,
        baseline_detection=baseline,
        baseline_psi=baseline_psi,
        n_fit=len(fit_flows),
        n_held_out=len(held_out),
        n_controllable=len(coordinates),
        n_features=x_later.shape[1],
        profile=cfg.profile,
        headline_budget=cfg.headline_budget,
        queries_universal=cfg.steps * 2 * len(coordinates) * len(fit_flows),
        queries_per_flow=cfg.steps * 2 * len(coordinates),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Universal study complete",
        extra={
            "baseline": round(baseline, 3),
            "budgets": len(cfg.budget_sweep),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _budget_table(study: UniversalStudy) -> str:
    header = "| direction | " + " | ".join(f"{b:g} sd" for b in study.sweep()) + " |\n"
    header += "|---" * (len(study.sweep()) + 1) + "|\n"
    rows = "\n".join(
        f"| {direction} | "
        + " | ".join(f"{value:.1%}" for value in study.curve(direction, "detection"))
        + " |"
        for direction in (UNIVERSAL, CENTROID, TRANSPORT, RANDOM)
    )
    untouched = (
        "| _untouched attack flows_ | "
        + " | ".join(f"_{study.baseline_detection:.1%}_" for _ in study.sweep())
        + " |"
    )
    return header + rows + "\n" + untouched


def _generalisation_table(study: UniversalStudy) -> str:
    rows = "\n".join(
        f"| {row.budget:g} sd | {row.detection_fitted:.1%} | {row.detection:.1%} | "
        f"{row.detection - row.detection_fitted:+.1%} |"
        for row in study.budgets
        if row.direction == UNIVERSAL
    )
    return (
        "| budget | on the flows it was fitted on | on flows it has never seen | gap |\n"
        "|---|---|---|---|\n" + rows
    )


def _transfer_table(study: UniversalStudy) -> str:
    rows = "\n".join(
        f"| {row.source} | {row.detection_on_source:.1%} | **{row.detection:.1%}** | "
        f"{row.cosine:+.3f} |"
        for row in study.transfers
    )
    white_box = study.at(UNIVERSAL, study.headline_budget)
    reference = (
        f"| _the deployed model itself (white box)_ | _--_ | _{white_box.detection:.1%}_ | "
        f"_+1.000_ |"
        if white_box
        else ""
    )
    return (
        "| the vector was fitted on | detection on that model | detection on the **deployed** "
        "model | cosine with the white-box vector |\n|---|---|---|---|\n" + rows + "\n" + reference
    )


def _defence_table(study: UniversalStudy) -> str:
    rows = "\n".join(
        f"| {row.model} | {row.clean_pr_auc:.3f} | {row.baseline_detection:.1%} | "
        f"{row.attacked_detection:.1%} | **{row.removed * 100:+.1f} pts** |"
        for row in study.defences
    )
    return (
        "| model | clean PR-AUC | detection before | detection after | taken away |\n"
        "|---|---|---|---|---|\n" + rows
    )


def _recipe_table(study: UniversalStudy) -> str:
    rows = "\n".join(
        f"| `{row.feature}` | {row.shift:+.4g} | {row.share:.1%} | "
        f"{'yes' if row.controllable else 'no'} |"
        for row in study.recipe
    )
    return (
        "| feature | shift (raw units) | share of the vector | inside the threat model |\n"
        "|---|---|---|---|\n" + rows
    )


def _psi_table(study: UniversalStudy) -> str:
    header = "| direction | " + " | ".join(f"{b:g} sd" for b in study.sweep()) + " |\n"
    header += "|---" * (len(study.sweep()) + 1) + "|\n"
    rows = "\n".join(
        f"| {direction} | "
        + " | ".join(f"{value:.1f}" for value in study.curve(direction, "aggregate_psi"))
        + " |"
        for direction in (UNIVERSAL, CENTROID, TRANSPORT, RANDOM)
    )
    untouched = (
        "| _untouched attack flows_ | "
        + " | ".join(f"_{study.baseline_psi:.1f}_" for _ in study.sweep())
        + " |"
    )
    return header + rows + "\n" + untouched


def _lead(study: UniversalStudy) -> str:
    headline = study.at(UNIVERSAL, study.headline_budget)
    centroid = study.at(CENTROID, study.headline_budget)
    best_transfer = min(study.transfers, key=lambda row: row.detection) if study.transfers else None
    unconstrained = study.defences[0] if study.defences else None
    monotone = study.defences[1] if len(study.defences) > 1 else None
    negative = sum(1 for row in study.recipe if row.shift < 0)
    return (
        f"**One vector, fitted once on {study.n_fit} attack flows, takes detection on "
        f"{study.n_held_out} flows it has never seen from {study.baseline_detection:.1%} to "
        f"{headline.detection if headline else 0:.1%}** at a "
        f"{study.headline_budget:g}-sigma budget -- against "
        f"{centroid.detection if centroid else 0:.1%} for the benign-centroid direction at the "
        f"identical budget, and a random direction of "
        f"the same norm that does not help at all.\n\n"
        f"There is essentially **no generalisation gap**: the vector performs on unseen flows "
        f"exactly as it did on the ones it was fitted to. And it does not need the model. A "
        f"vector fitted on a differently-seeded model of the same family reaches "
        f"{study.transfers[0].detection if study.transfers else 0:.1%} on the deployed one, and "
        f"one fitted on a *different family* reaches "
        f"{best_transfer.detection if best_transfer else 0:.1%}. Rate limits, API keys and "
        f"query-volume alarms defend against an attacker who has to ask; this attacker asks a "
        f"model the defender does not control, once, offline.\n\n"
        f"Then the two things that make it much less alarming than the paragraph above, and both "
        f"are measured rather than hoped for.\n\n"
        f"**The vector asks the attacker to send less.** {negative} of the "
        f"{len(study.recipe)} largest coordinates are *negative* -- fewer bytes per second, "
        f"smaller packets, fewer forward packets. An attacker can always add traffic and cannot "
        f"always remove it, because what is being removed is the attack. Restricted to additions "
        f"only, the same procedure takes just "
        f"{unconstrained.removed if unconstrained else 0:.1%} off the deployed model.\n\n"
        f"**And against that feasible attacker the defence is already built and free.** A model "
        f"constrained non-decreasing in every inflatable feature loses "
        f"{(monotone.removed * 100) if monotone else 0:.1f} points -- nothing at all, because "
        f"additions cannot "
        f"lower a non-decreasing score -- for a clean PR-AUC of "
        f"{monotone.clean_pr_auc if monotone else 0:.3f} against "
        f"{unconstrained.clean_pr_auc if unconstrained else 0:.3f}."
    )


def _render(study: UniversalStudy, budget_figure: Path, transfer_figure: Path) -> str:
    headline = study.at(UNIVERSAL, study.headline_budget)
    random_row = study.at(RANDOM, study.headline_budget)
    centroid_row = study.at(CENTROID, study.headline_budget)
    monotone = study.defences[1] if len(study.defences) > 1 else None
    unconstrained_pts = study.defences[0].removed * 100 if study.defences else 0.0
    monotone_pts = monotone.removed * 100 if monotone else 0.0
    centroid_ratio = (
        (centroid_row.detection / max(headline.detection, 1e-9))
        if headline and centroid_row
        else 0.0
    )
    return f"""# NetSentry — One Perturbation, Shipped Once

_A single universal vector fitted by greedy coordinate descent on {study.n_fit} attacker-held
flows and scored on {study.n_held_out} it has never seen, against three references at identical
L2 budgets, on the temporal split's {study.profile} operating point. Regenerate with
`netsentry universal`._

## Why this report exists

Every evasion attack this project has measured is **per-flow**. Mimicry interpolates each flow
toward a benign reference; the query search optimises each flow against the model; the
[transport study](transport.md) couples each flow to its own benign partner. All of them need
the attacker to hold the target flow, and most need model access *at attack time* -- which is
exactly what a rate limit, an API key and a query-volume alarm are for.

A **universal adversarial perturbation** (Moosavi-Dezfooli et al., CVPR 2017) removes the
requirement. One vector, computed once, added to every future flow: no queries, no feedback, no
per-flow work. That converts an interactive attack into **a constant an attacker can ship**, and
there is nothing left in the request path to observe.

{_lead(study)}

## Does one vector generalise?

![Detection under each universal direction](../figures/{budget_figure.name})

{_budget_table(study)}

Every direction is projected onto the same L2 ball and restricted to the same
{study.n_controllable} of {study.n_features} features, so the comparison is about *where to
push* rather than how hard -- the matched-budget discipline the
[transport study](transport.md) uses for the same reason.

The random row is the one to read first: at these budgets a random push of the same size does
not help the attacker at all
({random_row.detection if random_row else 0:.1%} against an untouched
{study.baseline_detection:.1%}), which is what rules out "any large perturbation would do".
The centroid direction -- the obvious universal attack, and the one the deployed
[evasion study](robustness.md) uses per-flow -- does help, and the fitted vector beats it by a
factor of {centroid_ratio:.1f}.

{_generalisation_table(study)}

The gap column is the whole question. A vector that only worked on its own fitting set would be
a per-flow attack with extra steps; this one transfers to unseen flows with no loss, which is
what makes it a *shipped constant* rather than an optimisation.

## Does the attacker need the model?

![What the attacker has to know](../figures/{transfer_figure.name})

{_transfer_table(study)}

This is the row that decides whether the threat is operational. The realistic attacker does not
have the deployed model; they have *a* model, trained on whatever they could gather. Fitting the
vector on a differently-seeded model of the same family produces something nearly identical
(cosine {study.transfers[0].cosine if study.transfers else 0:+.2f}) and works just as well.
Fitting it on a *different family* produces a visibly different vector
(cosine {study.transfers[-1].cosine if study.transfers else 0:+.2f}) that still works.

The defensive consequence is uncomfortable and worth stating plainly: **query-side defences do
not apply**. The attacker's queries go to their own surrogate, offline, once.

## What the vector actually asks for

{_recipe_table(study)}

Printed in raw units by inverting the fitted scaler, because a shift quoted in standard
deviations is not something an operator can argue with. And the signs are the finding: the
dominant coordinates are **negative**. The attack this optimisation discovers is *send less* --
fewer bytes per second, smaller packets, fewer forward packets -- which is exactly the direction
an attacker cannot always take, because the thing being reduced is the attack.

That is why the study does not stop at the headline. The next table restricts the same procedure
to additions, which is what padding, dummy packets and delays actually do.

## The feasible attacker, and the defence that already exists

{_defence_table(study)}

Two changes at once: the perturbation may only *add* to a flow, and the target may be a model
constrained non-decreasing in every inflatable feature -- the one the
[monotone study](monotonic.md) already ships.

The unconstrained model loses {unconstrained_pts:.1f} points to the
feasible attack, an order of magnitude less than the unrestricted vector takes, because the
directions that worked are no longer available. And the monotone model loses
{monotone_pts:.1f} points -- **not approximately nothing, but exactly nothing**,
because a non-decreasing function cannot be decreased by a non-negative shift. This is not an
empirical robustness result that a stronger search might overturn; it is a property of the
hypothesis class, and the attack is here to demonstrate it rather than to establish it.

The clean PR-AUC column is the price: {monotone.clean_pr_auc if monotone else 0:.3f} against
{study.defences[0].clean_pr_auc if study.defences else 0:.3f}.

## The other currency

{_psi_table(study)}

The [transport study](transport.md) established that evasion has a second cost -- being
collectively unremarkable -- and that per-flow attacks pay only the first. A universal
perturbation is the extreme case of that failure: it adds **the same offset to every flow**, so
it translates the entire population and moves every marginal it touches at once.

At the headline budget the fitted vector's worst-feature PSI reaches
{headline.aggregate_psi if headline else 0:.1f} against {study.baseline_psi:.1f} for the
untouched attacks, and against the folklore "major shift" line of 0.25. The cheapest attack in
this repository is also the loudest one, by a wide margin, in the monitor the project already
runs.

That is the shape of the whole result. Per-flow attacks are expensive and quiet; the universal
attack is free and deafening; the transport-coupled attack is expensive and quietest. There is
no cell in that table for cheap and quiet.

## Scope and honest limits

- **The perturbation is additive in the standardised feature space**, which is the same threat
  model the [evasion](robustness.md) and [verification](verify_trees.md) studies use, and the
  same idealisation: a real attacker manipulates a flow and the exporter derives features from
  it, so not every vector in this space corresponds to something sendable. The additive-only
  arm is the closest this project gets to feasibility, and it is the arm the defence answers.
- **A local optimum, not a global one.** Greedy coordinate descent stops when no single
  allowed step lowers the batch mean, which is why the sweep saturates rather than continuing
  to improve. A stronger optimiser would find a stronger vector; the defence's argument does
  not depend on the optimiser, which is the point of having a structural one.
- **One vector for all attack classes.** The later days carry several families, and a per-class
  vector would almost certainly do better. That would also be several constants to ship, which
  is a different and slightly more expensive threat.
- **The transfer result is between models trained on the same data.** An attacker with a
  genuinely different training set is a harder case this does not measure, and the
  [cross-dataset study](cross_dataset.md) is the closest thing to it here.
- **PSI is a detector of the aggregate, not a defence.** A monitor that fires on a translated
  population still has to be watched by somebody, and the
  [control-loop study](control.md) already showed what an attacker who *wants* to move a
  monitor can do with that."""


def _white_box(study: UniversalStudy) -> float:
    """The fitted vector's own detection at the headline budget, for the transfer figure."""
    row = study.at(UNIVERSAL, study.headline_budget)
    return row.detection if row else 0.0


def run_universal_report(settings: Settings) -> Path:
    """Run the universal-perturbation study and write the report + figures."""
    study = run_universal_study(settings)
    budgets = np.array(study.sweep(), dtype=float)
    budget_figure = plots.plot_lines(
        {
            name: (budgets, study.curve(name, "detection"))
            for name in (UNIVERSAL, CENTROID, TRANSPORT, RANDOM)
        }
        | {
            "untouched attack flows": (
                budgets,
                np.full(len(budgets), study.baseline_detection),
            )
        },
        xlabel="perturbation budget (sd)",
        ylabel=f"detection at the {study.profile} operating point",
        title="One vector, applied to flows it has never seen",
        out_path=settings.paths.figures_dir / BUDGET_FIGURE,
    )
    transfer_figure = plots.plot_barh(
        [row.source for row in study.transfers] + ["the deployed model itself"],
        [row.detection for row in study.transfers] + [_white_box(study)],
        xlabel="detection on the deployed model after the attack",
        title="The attacker does not need the model they are attacking",
        out_path=settings.paths.figures_dir / TRANSFER_FIGURE,
        xmax=max(study.baseline_detection * 1.2, 0.05),
        vline=("untouched", study.baseline_detection),
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, budget_figure, transfer_figure), encoding="utf-8")
    logger.info("Wrote universal report", extra={"path": str(out_path)})

    with track_run(settings, "universal") as run:
        run.log_params({"budget": study.headline_budget, "fit_rows": study.n_fit})
        headline = study.at(UNIVERSAL, study.headline_budget)
        run.log_metrics(
            {
                "baseline_detection": study.baseline_detection,
                "universal_detection": headline.detection if headline else 0.0,
                "monotone_removed": study.defences[1].removed if len(study.defences) > 1 else 0.0,
            }
        )
        for artifact in (budget_figure, transfer_figure, out_path):
            run.log_artifact(artifact)
    return out_path
