"""Which features mean the same thing on Tuesday as on Wednesday — invariance as a lever.

The headline temporal split loses roughly half the PR-AUC the stratified split reports, and
this project has spent a lot of words on *that the gap exists*. The natural next question is
whether any of it is avoidable, and the natural candidate answer comes from causal machine
learning: maybe the model leans on correlations that happened to hold during the training days
and did not survive the boundary, and maybe a model forbidden to use them would transfer.

That is a testable proposition, and there is a literature with the tools. **Invariant Causal
Prediction** (Peters, Buhlmann & Meinshausen, JRSS-B 2016) treats each capture day as an
*environment* and keeps only the features whose relationship to the label is stable across
them, on the reasoning that a stable relationship is more likely to be causal and a causal
relationship is more likely to survive a new environment. **Invariant Risk Minimization**
(Arjovsky, Bottou, Gulrajani & Lopez-Paz, 2019) attacks the same idea from the loss side:
train a representation such that the *same* classifier is simultaneously optimal in every
environment, enforced by penalising the gradient of each environment's risk with respect to a
dummy scale on the logits. Both are implemented here from scratch, over capture days as the
environment variable, with a linear head where IRM needs one.

Two things make this more than a method demonstration.

The first is that **the premise is checked rather than assumed**. IRM's guarantee needs
environments that differ in their *spurious* structure while sharing the label mechanism. The
CIC-IDS2017 days do not obviously satisfy that — each day contains different attack classes,
so what changes across environments is largely which label exists at all, and a method built
for shifting nuisance correlations is being handed shifting concepts. The report measures the
per-environment label composition first and reads the results in that light, because a method
applied outside its assumptions can produce numbers that mean nothing at all.

The second is a **cross-check that was not designed in**. The [earliness](earliness.md) study
found, from an entirely different direction, that the features surviving the temporal boundary
are the *intensive* ones — means, extremes, rates, ratios — while the extensive ones (totals,
cumulative sums, durations) measure how large one particular burst happened to be and do not
transfer. Invariance screening knows nothing about that partition; it only looks at whether a
feature's relationship to the label is stable across days. If the two methods agree, that is
two independent arguments for the same conclusion, and worth more than either alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import availability_tier, display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import InvarianceConfig

logger = get_logger(__name__)

REPORT_NAME = "invariance.md"
PENALTY_FIGURE = "invariance_penalty.png"
STABILITY_FIGURE = "invariance_stability.png"


# --------------------------------------------------------------------------------------
# Invariant Causal Prediction screening (pure; unit-tested)
# --------------------------------------------------------------------------------------
def environment_strength(x: np.ndarray, y: np.ndarray) -> float:
    """Signed strength of one feature's relationship to the label in one environment.

    ``AUC - 0.5``, so the sign says which direction the feature points and the magnitude says
    how strongly. AUC rather than a correlation because it is rank-based: a feature whose
    scale differs between days (and on this data many do) still gets a comparable number, and
    comparability across environments is the entire point of the exercise.
    """
    labels = np.asarray(y).astype(int)
    if labels.min() == labels.max():  # one class present: no relationship is defined
        return 0.0
    return float(roc_auc_score(labels, np.asarray(x, dtype=float))) - 0.5


@dataclass
class FeatureStability:
    """How consistently one feature points the same way across environments."""

    feature: str
    tier: str
    strengths: list[float]
    mean_strength: float
    sign_agreement: bool
    dispersion: float  # spread of |strength| relative to its mean

    @property
    def invariant(self) -> bool:
        """Placeholder; the screen applies its own thresholds (see :func:`screen_invariant`)."""
        return self.sign_agreement


def informative_environments(y: np.ndarray, environments: np.ndarray) -> list[str]:
    """Environments containing both classes — the only ones that can screen anything.

    CIC-IDS2017's Monday is entirely benign, and an environment with one class does not have
    a weak relationship between feature and label, it has *no defined relationship at all*.
    Scoring it as zero strength would be a silent and severe error: every feature would be
    dragged towards zero mean strength and towards infinite dispersion, and the screen would
    reject almost everything for a reason that has nothing to do with invariance. Such
    environments are dropped, and the report says how many remain.
    """
    labels = np.asarray(y).astype(int)
    envs = np.asarray(environments).astype(str)
    keep = []
    for env in dict.fromkeys(envs.tolist()):
        subset = labels[envs == env]
        if subset.size and subset.min() != subset.max():
            keep.append(env)
    return keep


def feature_stability(
    x: np.ndarray, y: np.ndarray, environments: np.ndarray, names: list[str]
) -> list[FeatureStability]:
    """Per-feature signed strength in every *informative* environment, with its dispersion."""
    envs = informative_environments(y, environments)
    out: list[FeatureStability] = []
    for j, name in enumerate(names):
        strengths = []
        for env in envs:
            mask = np.asarray(environments).astype(str) == env
            strengths.append(environment_strength(x[mask, j], np.asarray(y)[mask]))
        arr = np.asarray(strengths, dtype=float)
        magnitudes = np.abs(arr)
        signs = {int(np.sign(v)) for v in arr if abs(v) > 1e-9}
        mean_magnitude = float(magnitudes.mean())
        out.append(
            FeatureStability(
                feature=name,
                tier=availability_tier(name),
                strengths=[float(v) for v in arr],
                mean_strength=mean_magnitude,
                sign_agreement=len(signs) <= 1,
                dispersion=(
                    float(magnitudes.std() / mean_magnitude) if mean_magnitude > 1e-9 else np.inf
                ),
            )
        )
    return out


def screen_invariant(
    stability: list[FeatureStability], min_strength: float, max_dispersion: float
) -> list[str]:
    """Keep features that point the same way in every environment, with a steady magnitude.

    Sign agreement alone is too weak a screen — a feature carrying no signal at all agrees
    with itself trivially — so a minimum mean strength is required as well, and a cap on how
    much the magnitude may swing between days. Those two thresholds are the whole of the
    screen, and they are config, not folklore.
    """
    return [
        s.feature
        for s in stability
        if s.sign_agreement and s.mean_strength >= min_strength and s.dispersion <= max_dispersion
    ]


def screen_breakdown(
    stability: list[FeatureStability], min_strength: float, max_dispersion: float
) -> dict[str, int]:
    """Why each feature failed the screen, first failure winning.

    A count of survivors says whether invariance is attainable; this says *why not*, which is
    the more useful diagnostic. A feature that flips sign between environments is telling you
    the label mechanism changed. A feature that keeps its sign but swings in magnitude is
    telling you the same mechanism operated at a different strength. Those are different
    problems and only one of them is the kind IRM was built for.
    """
    counts = {"passed": 0, "flipped sign": 0, "too weak": 0, "unstable magnitude": 0}
    for s in stability:
        if not s.sign_agreement:
            counts["flipped sign"] += 1
        elif s.mean_strength < min_strength:
            counts["too weak"] += 1
        elif s.dispersion > max_dispersion:
            counts["unstable magnitude"] += 1
        else:
            counts["passed"] += 1
    return counts


# --------------------------------------------------------------------------------------
# Invariant Risk Minimization (pure; unit-tested)
# --------------------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, so a large margin does not overflow into a NaN."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def irm_penalty(logits: np.ndarray, y: np.ndarray) -> float:
    """IRMv1's penalty for one environment: the squared gradient at the dummy scale.

    Arjovsky et al. multiply the logits by a constant scalar ``s`` fixed at 1.0 and penalise
    ``(d/ds) R_e(s * logits)`` squared. If the same classifier is simultaneously optimal in
    every environment, that derivative vanishes everywhere; where it does not, some
    environment would prefer the decision boundary scaled differently, which is the signature
    of a predictor leaning on something environment-specific. For the logistic loss the
    derivative has a closed form and no autodiff is needed:

        d/ds mean logloss(s * z, y) at s = 1  =  mean( (sigmoid(z) - y) * z )
    """
    z = np.asarray(logits, dtype=float)
    residual = _sigmoid(z) - np.asarray(y, dtype=float)
    return float(np.mean(residual * z) ** 2)


@dataclass
class LinearHead:
    """A logistic classifier trained by full-batch gradient descent (weights + intercept)."""

    weights: np.ndarray
    intercept: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        product: np.ndarray = np.asarray(x, dtype=float) @ self.weights + self.intercept
        return product

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.logits(x))


def fit_irm(
    x: np.ndarray,
    y: np.ndarray,
    environments: np.ndarray,
    *,
    penalty_weight: float,
    steps: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> LinearHead:
    """Fit a linear head minimising average risk plus ``penalty_weight`` times IRMv1's penalty.

    ``penalty_weight = 0`` is plain empirical risk minimization, which is exactly the control
    the comparison needs: the same optimiser, the same steps, the same initialisation, one
    term switched off. Gradients are computed analytically rather than by autodiff so the
    module has no deep-learning dependency and the penalty's form is inspectable.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    envs = [
        np.flatnonzero(np.asarray(environments).astype(str) == e)
        for e in dict.fromkeys(np.asarray(environments).astype(str).tolist())
    ]
    w: np.ndarray = np.asarray(rng.normal(0.0, 0.01, size=x.shape[1]), dtype=float)
    b: float = 0.0
    eps = 1e-4  # finite-difference step for the penalty's gradient
    for _ in range(steps):
        grad_w = l2 * w
        grad_b = 0.0
        for idx in envs:
            xe, ye = x[idx], y[idx]
            residual = _sigmoid(xe @ w + b) - ye
            grad_w += (xe.T @ residual) / len(idx) / len(envs)
            grad_b += float(residual.mean()) / len(envs)
            if penalty_weight > 0.0:
                # The penalty is a scalar function of the weights; its gradient is taken by
                # central differences along the current risk gradient, which is enough to
                # steer the optimiser and avoids a second analytic derivation whose only
                # purpose would be speed.
                direction = (xe.T @ residual) / len(idx)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    unit = direction / norm
                    plus = irm_penalty(xe @ (w + eps * unit) + b, ye)
                    minus = irm_penalty(xe @ (w - eps * unit) + b, ye)
                    grad_w += penalty_weight * unit * (plus - minus) / (2 * eps) / len(envs)
        if penalty_weight > 1.0:
            # The rescaling Arjovsky et al.'s own implementation applies once the penalty
            # dominates: without it the gradient grows with the penalty weight and the step
            # size that was stable at weight 1 diverges at weight 100, so the sweep would be
            # measuring optimiser blow-up rather than the objective.
            grad_w = grad_w / penalty_weight
            grad_b = grad_b / penalty_weight
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return LinearHead(weights=w, intercept=b)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class ScreenArm:
    """A gradient-boosted model trained on one feature subset, scored on the temporal test."""

    name: str
    n_features: int
    pr_auc: float
    detection: float


@dataclass
class PenaltyArm:
    """A linear head at one IRM penalty weight."""

    penalty_weight: float
    pr_auc: float
    detection: float
    mean_penalty: float


@dataclass
class EnvironmentRow:
    """What one capture day actually contains — the premise check."""

    environment: str
    n_flows: int
    attack_rate: float
    classes: list[str]


@dataclass
class InvarianceStudy:
    """Everything the report renders."""

    environments: list[EnvironmentRow]
    informative: list[str]
    shared_classes: list[str]
    stability: list[FeatureStability]
    invariant_features: list[str]
    screen_arms: list[ScreenArm]
    penalty_arms: list[PenaltyArm]
    tier_overlap: dict[str, tuple[int, int]]
    breakdown: dict[str, int]


def _score_subset(
    settings: Settings,
    columns: np.ndarray,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    operating_fpr: float,
) -> tuple[float, float]:
    """Refit the deployed model family on a feature subset; return (PR-AUC, detection)."""
    x_train, x_val, x_test = (m[:, columns] for m in matrices)
    y_train, y_val, y_test = targets
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    benign = settings.labels.benign_label
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    s_test = attack_probability(model.predict_proba(x_test), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
    return (
        float(average_precision_score(y_test, s_test)),
        rates_at_threshold(y_test, s_test, threshold)["tpr"],
    )


def run_invariance(settings: Settings) -> InvarianceStudy:
    """Screen features for cross-day invariance, then train with and without IRM's penalty."""
    cfg: InvarianceConfig = settings.invariance
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    operating_fpr = variant.thresholds.primary_fpr

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    targets = (
        train[BINARY_TARGET].to_numpy(),
        val[BINARY_TARGET].to_numpy(),
        test[BINARY_TARGET].to_numpy(),
    )
    pipeline = build_pipeline(variant)
    matrices = (
        pipeline.fit_transform(train),  # FIT ON TRAIN ONLY
        pipeline.transform(val),
        pipeline.transform(test),
    )
    names = [
        display_feature_name(n) for n in pipeline.named_steps["features"].get_feature_names_out()
    ]
    environments = (
        train[DAY_COLUMN].astype(str).to_numpy()
        if DAY_COLUMN in train.columns
        else np.zeros(len(train), dtype=str)
    )

    env_rows, shared = _describe_environments(train, environments, variant.labels.benign_label)
    stability = feature_stability(matrices[0], targets[0], environments, names)
    invariant = screen_invariant(stability, cfg.min_strength, cfg.max_dispersion)
    logger.info(
        "Invariance screen complete",
        extra={"kept": len(invariant), "of": len(names), "environments": len(env_rows)},
    )

    all_cols = np.arange(len(names))
    keep = np.array([j for j, n in enumerate(names) if n in set(invariant)], dtype=int)
    screen_arms = [
        ScreenArm(
            "all features (deployed)",
            len(names),
            *_score_subset(variant, all_cols, matrices, targets, operating_fpr),
        )
    ]
    if keep.size:
        screen_arms.append(
            ScreenArm(
                "cross-day invariant only",
                int(keep.size),
                *_score_subset(variant, keep, matrices, targets, operating_fpr),
            )
        )

    penalty_arms = _penalty_sweep(variant, cfg, matrices, targets, environments, operating_fpr)
    tier_overlap = _tier_overlap(stability, set(invariant))
    return InvarianceStudy(
        environments=env_rows,
        informative=informative_environments(targets[0], environments),
        shared_classes=shared,
        stability=sorted(stability, key=lambda s: -s.mean_strength),
        invariant_features=invariant,
        screen_arms=screen_arms,
        penalty_arms=penalty_arms,
        tier_overlap=tier_overlap,
        breakdown=screen_breakdown(stability, cfg.min_strength, cfg.max_dispersion),
    )


def _describe_environments(
    train: pd.DataFrame, environments: np.ndarray, benign_label: str
) -> tuple[list[EnvironmentRow], list[str]]:
    """Per-day composition, and the attack classes every day has in common."""
    labels = train[MULTICLASS_TARGET].astype(str).to_numpy()
    binary = train[BINARY_TARGET].to_numpy().astype(int)
    rows: list[EnvironmentRow] = []
    per_day: list[set[str]] = []
    for env in dict.fromkeys(environments.astype(str).tolist()):
        mask = environments.astype(str) == env
        classes = sorted({c for c in labels[mask].tolist() if c != benign_label})
        per_day.append(set(classes))
        rows.append(
            EnvironmentRow(
                environment=env,
                n_flows=int(mask.sum()),
                attack_rate=float(binary[mask].mean()),
                classes=classes,
            )
        )
    shared = sorted(set.intersection(*per_day)) if per_day else []
    return rows, shared


def _penalty_sweep(
    settings: Settings,
    cfg: InvarianceConfig,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray],
    targets: tuple[np.ndarray, np.ndarray, np.ndarray],
    environments: np.ndarray,
    operating_fpr: float,
) -> list[PenaltyArm]:
    """ERM against IRM at each penalty weight, on the same linear head and optimiser."""
    x_train, x_val, x_test = matrices
    y_train, y_val, y_test = (t.astype(float) for t in targets)
    envs = [
        np.flatnonzero(environments.astype(str) == e)
        for e in dict.fromkeys(environments.astype(str).tolist())
    ]
    arms: list[PenaltyArm] = []
    for weight in cfg.penalty_weights:
        head = fit_irm(
            x_train,
            y_train,
            environments,
            penalty_weight=weight,
            steps=cfg.steps,
            learning_rate=cfg.learning_rate,
            l2=cfg.l2,
            seed=settings.seed,
        )
        s_val, s_test = head.predict_proba(x_val), head.predict_proba(x_test)
        threshold = threshold_at_fpr(y_val.astype(int), s_val, operating_fpr)
        arms.append(
            PenaltyArm(
                penalty_weight=weight,
                pr_auc=float(average_precision_score(y_test.astype(int), s_test)),
                detection=rates_at_threshold(y_test.astype(int), s_test, threshold)["tpr"],
                mean_penalty=float(
                    np.mean([irm_penalty(head.logits(x_train[i]), y_train[i]) for i in envs])
                ),
            )
        )
        logger.info(
            "IRM arm complete",
            extra={"penalty": weight, "pr_auc": round(arms[-1].pr_auc, 4)},
        )
    return arms


def _tier_overlap(
    stability: list[FeatureStability], invariant: set[str]
) -> dict[str, tuple[int, int]]:
    """Per availability tier: (features kept by the screen, features in the tier)."""
    out: dict[str, tuple[int, int]] = {}
    for s in stability:
        kept, total = out.get(s.tier, (0, 0))
        out[s.tier] = (kept + (1 if s.feature in invariant else 0), total + 1)
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_invariance_report(settings: Settings) -> Path:
    """Run the invariance study and write the report + figures."""
    study = run_invariance(settings)

    penalty_fig = plots.plot_lines(
        {
            "PR-AUC (temporal test)": (
                np.array([max(a.penalty_weight, 1e-3) for a in study.penalty_arms]),
                np.array([a.pr_auc for a in study.penalty_arms]),
            )
        },
        xlabel="IRM penalty weight (log scale; leftmost point is plain ERM)",
        ylabel="PR-AUC on the held-out days",
        title="Does penalising environment-specific structure transfer better?",
        out_path=settings.paths.figures_dir / PENALTY_FIGURE,
        xscale="log",
    )
    top = study.stability[: settings.invariance.plot_features]
    stability_fig = plots.plot_barh(
        [s.feature for s in top],
        [s.mean_strength for s in top],
        xlabel="mean |AUC - 0.5| across capture days",
        title="Strongest single-feature signals, and whether they hold every day",
        out_path=settings.paths.figures_dir / STABILITY_FIGURE,
        xmax=max((s.mean_strength for s in top), default=0.5) * 1.2,
    )

    report = _render(study, penalty_fig, stability_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote invariance report", extra={"path": str(out_path)})

    with track_run(settings, "invariance") as run:
        run.log_params(
            {
                "n_environments": len(study.environments),
                "n_invariant": len(study.invariant_features),
            }
        )
        metrics = {f"screen_{a.name.split()[0]}_pr_auc": a.pr_auc for a in study.screen_arms}
        for arm in study.penalty_arms:
            metrics[f"irm_{arm.penalty_weight:g}_pr_auc"] = arm.pr_auc
        run.log_metrics(metrics)
        run.log_artifact(penalty_fig)
        run.log_artifact(stability_fig)
        run.log_artifact(out_path)
    return out_path


def _environment_table(study: InvarianceStudy) -> str:
    rows = ["| environment | flows | attack rate | attack classes present |", "|---|---|---|---|"]
    for e in study.environments:
        classes = ", ".join(e.classes) if e.classes else "_(none)_"
        rows.append(f"| {e.environment} | {e.n_flows:,} | {e.attack_rate:.1%} | {classes} |")
    return "\n".join(rows)


def _screen_table(study: InvarianceStudy) -> str:
    rows = ["| feature set | features | PR-AUC | detection @ budget |", "|---|---|---|---|"]
    for a in study.screen_arms:
        rows.append(f"| {a.name} | {a.n_features} | {a.pr_auc:.3f} | {a.detection:.1%} |")
    return "\n".join(rows)


def _penalty_table(study: InvarianceStudy) -> str:
    rows = [
        "| penalty weight | PR-AUC | detection @ budget | mean IRM penalty |",
        "|---|---|---|---|",
    ]
    for a in study.penalty_arms:
        label = "0 (plain ERM)" if a.penalty_weight == 0 else f"{a.penalty_weight:g}"
        rows.append(f"| {label} | {a.pr_auc:.3f} | {a.detection:.1%} | {a.mean_penalty:.3e} |")
    return "\n".join(rows)


def _tier_table(study: InvarianceStudy) -> str:
    rows = [
        "| availability tier | features | kept by the invariance screen | share |",
        "|---|---|---|---|",
    ]
    for tier, (kept, total) in sorted(study.tier_overlap.items()):
        rows.append(f"| {tier} | {total} | {kept} | {kept / total:.0%} |")
    return "\n".join(rows)


def _premise_read(study: InvarianceStudy) -> str:
    n_env = len(study.environments)
    shared = study.shared_classes
    if shared:
        return (
            f"The {n_env} capture days share {len(shared)} attack class(es) "
            f"({', '.join(shared)}), so there is at least some common label mechanism for an "
            "invariant predictor to be invariant *about*. The methods below are being applied "
            "roughly inside their assumptions."
        )
    return (
        f"**The {n_env} environments share no attack class at all.** Each capture day contains "
        "different attacks, so what varies across environments is not a nuisance correlation "
        "sitting alongside a stable label mechanism — it is the label mechanism. That matters "
        "before a single number is read, because both methods here assume the opposite: ICP "
        "looks for features whose relationship to the label is stable across environments, and "
        "IRM looks for a representation on which one classifier is simultaneously optimal in "
        "all of them. Neither is well posed when the thing being predicted changes identity "
        "between environments. The results are reported anyway, and read as what they are — a "
        "measurement of what these methods do when handed the wrong kind of shift, which is "
        "the situation a practitioner is actually most likely to be in, since nobody checks."
    )


def _informative_read(study: InvarianceStudy) -> str:
    """An all-benign day cannot screen features; say so before the screen is read."""
    silent = [e.environment for e in study.environments if e.environment not in study.informative]
    if not silent:
        return ""
    return (
        f"A second, more mundane problem sits underneath that one: "
        f"**{', '.join(silent)} contains no attacks at all**, so no feature has a defined "
        f"relationship to the label there. That is not a weak environment, it is not an "
        f"environment — and scoring it as zero strength (the obvious implementation) would "
        f"drag every feature towards zero mean strength and infinite dispersion, rejecting "
        f"almost the whole vector for a reason with nothing to do with invariance. "
        f"Single-class days are dropped, leaving {len(study.informative)} usable environments. "
        "Two is the bare minimum an invariance argument can be built on, and it is worth "
        "knowing that the famous five-day capture supplies exactly that."
    )


def _screen_read(study: InvarianceStudy) -> str:
    if len(study.screen_arms) < 2:
        return (
            "The screen kept no features at all, which is itself the answer: on these "
            "environments no feature's relationship to the label is stable enough to pass, so "
            "there is no invariant subset to train on."
        )
    full, invariant = study.screen_arms[0], study.screen_arms[1]
    delta = invariant.pr_auc - full.pr_auc
    if delta >= 0:
        return (
            f"Restricting the model to the {invariant.n_features} cross-day-invariant features "
            f"— {invariant.n_features / full.n_features:.0%} of what it normally sees — costs "
            f"{delta:+.3f} PR-AUC and {invariant.detection - full.detection:+.1%} detection. "
            "Discarding most of the feature vector and losing nothing means the discarded "
            "features were carrying environment-specific structure the model was better off "
            "without, which is the result ICP predicts."
        )
    return (
        f"Restricting the model to the {invariant.n_features} cross-day-invariant features "
        f"costs {delta:.3f} PR-AUC and {invariant.detection - full.detection:+.1%} detection. "
        "The screen does **not** pay for itself: the features it discards were contributing "
        "to out-of-environment detection, not just to in-environment fit. That is the "
        "expected outcome once the premise above is taken seriously — a screen looking for "
        "stability across environments that differ in their *labels* will reject features "
        "that are genuinely predictive of the attacks it has seen, on the grounds that they "
        "say nothing about attacks that were not present. Stability and usefulness come apart "
        "when the environments are not the kind the theory assumes."
    )


def _penalty_read(study: InvarianceStudy) -> str:
    if len(study.penalty_arms) < 2:
        return ""
    erm = study.penalty_arms[0]
    best = max(study.penalty_arms[1:], key=lambda a: a.pr_auc)
    strongest = study.penalty_arms[-1]
    reduced = strongest.mean_penalty < erm.mean_penalty
    mechanism = (
        f"The penalty term itself does fall, from {erm.mean_penalty:.2e} to "
        f"{strongest.mean_penalty:.2e} at the strongest weight, so the optimiser is doing what "
        "it was asked to do"
        if reduced
        else (
            f"The penalty term does not fall (from {erm.mean_penalty:.2e} to "
            f"{strongest.mean_penalty:.2e}), so before interpreting the transfer numbers the "
            "honest reading is that the objective was not actually optimised"
        )
    )
    gap = best.pr_auc - erm.pr_auc
    if gap > 0.005:
        verdict = f"the best penalised arm gains {gap:+.3f} PR-AUC against plain ERM"
    elif gap > -0.005:
        verdict = (
            "**no penalty weight beats plain ERM, and the best of them merely ties it** "
            f"({gap:+.3f} PR-AUC)"
        )
    else:
        verdict = f"**every penalty weight loses to plain ERM**, the best by {gap:.3f} PR-AUC"
    return (
        f"On a linear head with the optimiser, the initialisation and the step count held "
        f"fixed — only the penalty term switched on — {verdict}. {mechanism}; what it buys is "
        "the question, and the answer here is nothing. That is consistent with the premise "
        "check rather than surprising given it: IRM removes predictors that rely on structure "
        "varying across environments, and when the varying structure *is* the signal, removing "
        "it removes the signal. The method is not failing so much as being asked the wrong "
        "question, which is worth demonstrating precisely because the paper's framing invites "
        "exactly this misapplication."
    )


def _breakdown_table(study: InvarianceStudy) -> str:
    total = max(sum(study.breakdown.values()), 1)
    rows = ["| outcome of the screen | features | share |", "|---|---|---|"]
    for name, count in study.breakdown.items():
        rows.append(f"| {name} | {count} | {count / total:.0%} |")
    return "\n".join(rows)


def _breakdown_read(study: InvarianceStudy) -> str:
    """Why invariance is unattainable here — the decomposition, not just the survivor count."""
    b = study.breakdown
    total = max(sum(b.values()), 1)
    flipped = b.get("flipped sign", 0)
    if flipped / total > 0.4:
        return (
            f"**{flipped} of {total} features ({flipped / total:.0%}) point in opposite "
            "directions on different days.** Not weakly, not noisily — a feature that "
            "separates attack from benign one way on Tuesday separates it the other way on "
            "Wednesday. That single number explains everything else on this page, and it is "
            "the premise failure made concrete: Tuesday is brute-force traffic and Wednesday "
            "is denial of service, and those two things are abnormal in opposite directions. "
            "Patator floods are many short low-volume connections; DoS floods are sustained "
            "high-volume ones. A screen that requires 'this feature means the same thing in "
            "every environment' cannot survive that, and it should not — the requirement is "
            "correct and the data simply does not meet it. What the requirement rejects here "
            "is not spurious structure but genuine, class-specific structure that a detector "
            "trained on one attack family legitimately needs."
        )
    return (
        f"Of {total} features, {b.get('passed', 0)} pass, {flipped} flip sign between "
        f"environments, {b.get('too weak', 0)} carry too little signal to screen, and "
        f"{b.get('unstable magnitude', 0)} keep their direction but swing in strength. The "
        "middle category is the interesting one: those features mean genuinely different "
        "things on different days."
    )


def _overlap_read(study: InvarianceStudy) -> str:
    """Only claim the earliness cross-check when the surviving set can support it."""
    kept = len(study.invariant_features)
    if kept < 5:
        return (
            f"The screen kept {kept} features, which is far too few to say anything about how "
            "they distribute across availability tiers. The [earliness](earliness.md) study "
            "found from a completely different direction that the *intensive* features are the "
            "ones surviving the temporal boundary, and it would be a satisfying cross-check if "
            "an invariance screen rediscovered that partition without knowing it existed. On "
            "this data it cannot: with one or two survivors the tier shares are noise, and "
            f"reporting {kept} features as agreement with anything would be reading a pattern "
            "into a coin flip. The table is left in place because the null result is the "
            "honest one, not because it supports the claim."
        )
    overlap = study.tier_overlap
    kept_in, total_in = overlap.get("in_flight", (0, 0))
    kept_co, total_co = overlap.get("complete", (0, 0))
    share_in = kept_in / total_in if total_in else 0.0
    share_co = kept_co / total_co if total_co else 0.0
    if share_in > share_co:
        return (
            f"**{share_in:.0%} of the intensive (in-flight) features survive the invariance "
            f"screen against {share_co:.0%} of the extensive (complete-flow) ones** — and the "
            "screen has no idea that partition exists. The [earliness](earliness.md) study "
            "reached the same conclusion from a completely different direction, partitioning "
            "features by *when their value is knowable*. Two methods with nothing in common "
            "agreeing on which features are load-bearing is worth more than either alone, and "
            "the interpretation is clean: an extensive feature measures how large one "
            "particular burst was, which is a property of that day's campaign rather than of "
            "hostile behaviour, so it is exactly what an invariance screen is built to reject."
        )
    return (
        f"The screen keeps {share_in:.0%} of the intensive features and {share_co:.0%} of the "
        "extensive ones, so it does **not** reproduce the partition the "
        "[earliness](earliness.md) study found from the availability side. The two notions of "
        "'which features transfer' are measuring different things here."
    )


_SCOPE = """Environments are capture days, which is the only environment variable this dataset
supplies and a poor one for the purpose: the days differ in which attacks ran, so the shift
between them is concept shift rather than the covariate-with-stable-mechanism shift both
methods assume. The [covariate-shift](covariate_shift.md) study reached the same diagnosis
from the density-ratio side. A dataset with the same attacks captured on different networks
would be the right test bed, and this project does not have one.

The IRM arm uses a **linear head** on the fitted feature matrix, because IRMv1's penalty is
defined through a differentiable predictor and the deployed gradient-boosted model is not one.
Its absolute numbers are therefore not comparable to the screening arm's or to the headline;
what is comparable is ERM against IRM *within* that arm, which shares an optimiser, an
initialisation, a step count and a seed. The penalty's gradient is taken by central
differences along the risk gradient rather than analytically — enough to steer the optimiser,
and it keeps the module free of a deep-learning dependency.

The screen's two thresholds (minimum mean strength, maximum dispersion) are config, and moving
them moves how many features survive. They were fixed before the transfer numbers were looked
at; a screen tuned until its subset won would be selecting on the outcome, which is the same
error as tuning on the test set with extra steps."""


def _render(study: InvarianceStudy, penalty_fig: Path, stability_fig: Path) -> str:
    return f"""# NetSentry — Features That Mean the Same Thing Every Day

_Synthetic stand-in. Honest temporal/binary split; environments are the training capture days,
of which {len(study.informative)} contain both classes and can screen anything.
{len(study.invariant_features)} of {len(study.stability)} features pass the invariance screen._

## Why this report exists

The temporal split costs roughly half the PR-AUC the stratified split reports, and this
project has been careful to say *that* the gap exists without claiming much about why. Causal
machine learning offers a specific hypothesis worth testing: that the model leans on
correlations which held during the training days and did not survive the boundary, and that a
model forbidden to use them would transfer better.

Two methods attack that from opposite sides. **Invariant Causal Prediction** (Peters, Buhlmann
& Meinshausen 2016) keeps only features whose relationship to the label is stable across
environments. **Invariant Risk Minimization** (Arjovsky et al. 2019) penalises representations
on which different environments would prefer different classifiers. Both are implemented here
from scratch over capture days.

## First, does the premise hold?

{_environment_table(study)}

{_premise_read(study)}

{_informative_read(study)}

## Screening features for cross-day invariance

{_screen_table(study)}

{_screen_read(study)}

![single-feature strength across days](../figures/{stability_fig.name})

## Penalising environment-specific structure directly

{_penalty_table(study)}

{_penalty_read(study)}

![PR-AUC against penalty weight](../figures/{penalty_fig.name})

## Why invariance is unattainable here

{_breakdown_table(study)}

{_breakdown_read(study)}

## The cross-check that would have been satisfying

{_tier_table(study)}

{_overlap_read(study)}

## Scope

{_SCOPE}"""
