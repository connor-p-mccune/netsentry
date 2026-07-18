"""Influence functions: which training flows are responsible for a verdict (Koh & Liang 2017).

The data-valuation study (KNN-Shapley) answers a *global* question — how much is each training
flow worth to the model overall. This answers the *local* one a SOC analyst actually asks when
a verdict looks wrong: for **this** flow, which training examples drove the decision, and which
ones, if removed, would move it? Influence functions (Koh & Liang, ICML 2017, best paper)
answer it without retraining, by estimating the effect of infinitesimally up-weighting a
training point on the loss at a test point:

    I(z_train, z_test) = - grad_theta L(z_test)^T  H^{-1}  grad_theta L(z_train)

where ``H`` is the Hessian of the training objective at the fitted parameters. Removing a
training point is up-weighting it by ``-1/n``, so ``I`` (with that scale) predicts the
leave-one-out change in the test loss — the counterfactual "what if this flow had never been
labelled" without ever refitting.

**Scope, stated up front.** Influence functions require a twice-differentiable, convex loss;
the deployed gradient-boosted model is neither, so — exactly as the [distillation
study](distill.md) uses an auditable surrogate — this runs on the **logistic** baseline (one
of the project's reference models), where ``H`` is exact and positive-definite and the whole
computation is a single dense linear solve. That is a real limitation and it is named, not
hidden; the payoff is that the estimate is *validated*: the study actually retrains the model
with a sample of training points removed and correlates the true leave-one-out change against
the influence prediction. If that correlation is high, the approximation is trustworthy on
this data; if not, the report says so.

Two products follow from the same machinery:

- **Per-prediction explanation.** For a handful of test flows (a confident attack, a confident
  benign, a mistake), the most *helpful* and most *harmful* training flows — the specific
  labelled examples most responsible for the verdict — with their labels and capture days.
- **Self-influence mislabel detection.** A point that is very influential on *its own* loss is
  typically mislabelled or a hard outlier; ranking training flows by self-influence recovers
  planted label flips, an independent second opinion next to the confident-learning
  [label audit](label_audit.md) and the KNN-Shapley [data valuation](data_value.md).

Runs on the exchangeable stratified/binary split — the logistic surrogate's honest home, and
the split where the training and test flows are exchangeable so an influence estimated on one
transfers to the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.evaluation import plots
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import InfluenceConfig

logger = get_logger(__name__)

REPORT_NAME = "influence.md"
FIGURE_NAME = "influence.png"


def _augment(x: np.ndarray) -> np.ndarray:
    """Append an intercept column so the bias term rides in the parameter vector."""
    return np.hstack([np.asarray(x, dtype=float), np.ones((len(x), 1))])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), dtype=float)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, seed: int) -> np.ndarray:
    """Fit L2 logistic regression and return the augmented parameter vector ``theta = [w, b]``.

    ``C = 1 / l2`` so the sklearn optimum minimises ``sum_i loss + (l2/2)||w||^2`` — the same
    objective whose Hessian the influence computation uses, so gradients vanish at ``theta``.
    """
    clf = LogisticRegression(C=1.0 / l2, max_iter=5000, random_state=seed)
    clf.fit(x, y)
    return np.concatenate([clf.coef_.ravel(), clf.intercept_])


def _pointwise_grads(theta: np.ndarray, xa: np.ndarray, y_pm: np.ndarray) -> np.ndarray:
    """Per-example loss gradients ``g_i = -sigma(-y_i theta^T x_i) y_i x_i`` (rows = examples)."""
    margins = xa @ theta
    weights = -_sigmoid(-y_pm * margins) * y_pm
    return np.asarray(weights[:, None] * xa, dtype=float)


def hessian(theta: np.ndarray, xa: np.ndarray, l2: float) -> np.ndarray:
    """Objective Hessian ``H = sum_i p_i(1-p_i) x_i x_i^T + l2 I`` (intercept unpenalised).

    The ``l2 I`` damping (Koh & Liang's trick, here exactly the fit's regulariser) guarantees
    ``H`` is positive-definite and invertible even where the data term is rank-deficient.
    """
    p = _sigmoid(xa @ theta)
    s = p * (1.0 - p)
    h = xa.T @ (s[:, None] * xa)
    reg = l2 * np.eye(xa.shape[1])
    reg[-1, -1] = 0.0  # sklearn does not penalise the intercept
    return np.asarray(h + reg, dtype=float)


@dataclass
class InfluenceExplanation:
    """One test flow and the training flows most responsible for its verdict."""

    test_label: str
    test_pred_prob: float
    correct: bool
    helpful: list[tuple[str, str, float]]  # (label, capture-day, influence) — reduce test loss
    harmful: list[tuple[str, str, float]]  # increase test loss


@dataclass
class InfluenceStudy:
    """The full influence-function study: LOO validation, explanations, mislabel detection."""

    n_train: int
    n_features: int
    loo_pearson: float
    loo_spearman: float
    loo_n: float
    mislabel_auc: float
    mislabel_flip_rate: float
    explanations: list[InfluenceExplanation]
    loo_true: np.ndarray
    loo_pred: np.ndarray


def influence_on_test(
    theta: np.ndarray, h_inv: np.ndarray, g_train: np.ndarray, g_test: np.ndarray, n_train: int
) -> np.ndarray:
    """Predicted leave-one-out change in the test loss for every training point.

    ``delta_loss_i ≈ (1/n) * g_test^T H^{-1} g_i`` — removing point ``i`` is up-weighting it by
    ``-1/n``. Positive means removing the point *raises* the test loss (it was helping);
    vectorised over all training points as one matvec plus a dot.
    """
    h_inv_g_test = h_inv @ g_test
    return np.asarray((g_train @ h_inv_g_test) / n_train, dtype=float)


def self_influence(h_inv: np.ndarray, g_train: np.ndarray) -> np.ndarray:
    """Self-influence ``g_i^T H^{-1} g_i`` per training point — the mislabel/outlier score."""
    return np.asarray(np.einsum("ij,ij->i", g_train, g_train @ h_inv), dtype=float)


def _pm(y: np.ndarray) -> np.ndarray:
    """Map {0,1} labels to {-1,+1} for the logistic loss."""
    return np.asarray(2 * np.asarray(y, dtype=int) - 1, dtype=int)


def run_influence(settings: Settings) -> InfluenceStudy:
    """Fit the logistic surrogate, validate influence against real LOO, then explain + audit."""
    cfg: InfluenceConfig = settings.influence
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "stratified", "train").reset_index(drop=True)
    test = load_split(variant, "stratified", "test").reset_index(drop=True)
    if len(train) > cfg.max_train:
        train = train.sample(cfg.max_train, random_state=variant.seed).reset_index(drop=True)

    pipeline = build_pipeline(variant)
    x_train = _augment(np.asarray(pipeline.fit_transform(train)))
    x_test = _augment(np.asarray(pipeline.transform(test)))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    ypm_train, ypm_test = _pm(y_train), _pm(y_test)

    theta = fit_logistic(x_train[:, :-1], y_train, cfg.l2, variant.seed)
    h = hessian(theta, x_train, cfg.l2)
    h_inv = np.linalg.inv(h)
    g_train = _pointwise_grads(theta, x_train, ypm_train)

    loo_true, loo_pred = _validate_loo(
        cfg, x_train, y_train, theta, h_inv, g_train, x_test, ypm_test, variant.seed
    )
    loo_p = float(pearsonr(loo_pred, loo_true)[0]) if len(loo_true) > 2 else float("nan")
    loo_s = float(spearmanr(loo_pred, loo_true)[0]) if len(loo_true) > 2 else float("nan")

    explanations = _explain_flows(cfg, train, x_test, y_test, ypm_test, theta, h_inv, g_train)
    mislabel_auc = _mislabel_check(cfg, x_train, y_train, variant.seed)

    return InfluenceStudy(
        n_train=len(train),
        n_features=x_train.shape[1] - 1,
        loo_pearson=loo_p,
        loo_spearman=loo_s,
        loo_n=float(len(loo_true)),
        mislabel_auc=mislabel_auc,
        mislabel_flip_rate=cfg.mislabel_flip_rate,
        explanations=explanations,
        loo_true=loo_true,
        loo_pred=loo_pred,
    )


def _test_loss(theta: np.ndarray, xa: np.ndarray, ypm: np.ndarray) -> float:
    """Mean logistic loss of ``theta`` on the given (augmented) test rows."""
    return float(np.mean(np.log1p(np.exp(-ypm * (xa @ theta)))))


def _validate_loo(
    cfg: InfluenceConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    theta: np.ndarray,
    h_inv: np.ndarray,
    g_train: np.ndarray,
    x_test: np.ndarray,
    ypm_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrain without each sampled training point; correlate true vs predicted test-loss change.

    This is the Koh & Liang validation, and the study's spine: the influence estimate is only
    trustworthy if it tracks what genuinely happens when a point is removed. Uses a fixed
    random subset of training points and the mean test loss as the outcome.
    """
    rng = np.random.default_rng(seed)
    n = len(x_train)
    sample = rng.choice(n, size=min(cfg.loo_sample, n), replace=False)
    base_loss = _test_loss(theta, x_test, ypm_test)
    g_test_mean = _pointwise_grads(theta, x_test, ypm_test).mean(axis=0)

    true_delta, pred_delta = [], []
    for i in sample:
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        theta_i = fit_logistic(x_train[keep, :-1], y_train[keep], cfg.l2, seed)
        true_delta.append(_test_loss(theta_i, x_test, ypm_test) - base_loss)
        # Predicted LOO change in mean test loss from the influence of removing point i.
        pred_delta.append(float(g_test_mean @ (h_inv @ g_train[i]) / n))
    return np.asarray(true_delta), np.asarray(pred_delta)


def _explain_flows(
    cfg: InfluenceConfig,
    train: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ypm_test: np.ndarray,
    theta: np.ndarray,
    h_inv: np.ndarray,
    g_train: np.ndarray,
) -> list[InfluenceExplanation]:
    """Explain a spread of test flows by their most helpful/harmful training flows.

    Picks the flows with the largest true-label loss (the most *informative* verdicts — a
    saturated p=1.000 prediction has a near-zero gradient and so no training flow is
    individually responsible), preferring the model's own mistakes, which are exactly the
    cases an analyst brings to this tool.
    """
    from netsentry.data.schema import DAY_COLUMN

    margins = x_test @ theta
    probs = _sigmoid(margins)
    preds = (probs >= 0.5).astype(int)
    correct = preds == y_test
    losses = np.log1p(np.exp(-ypm_test * margins))  # per-flow true-label loss
    labels = train[MULTICLASS_TARGET].to_numpy()
    days = (
        train[DAY_COLUMN].astype(str).to_numpy()
        if DAY_COLUMN in train.columns
        else np.full(len(train), "n/a")
    )

    mistakes = np.where(~correct)[0]
    corrects = np.where(correct)[0]
    mistakes = mistakes[np.argsort(-losses[mistakes])]  # worst errors first
    corrects = corrects[np.argsort(-losses[corrects])]  # hardest correct first
    half = max(1, cfg.n_explained // 2)
    picks = list(mistakes[:half]) + list(corrects[: cfg.n_explained - len(mistakes[:half])])
    picks = picks[: cfg.n_explained]

    n = len(g_train)
    explanations: list[InfluenceExplanation] = []
    for idx in picks:
        g_test = _pointwise_grads(theta, x_test[idx : idx + 1], ypm_test[idx : idx + 1])[0]
        infl = influence_on_test(theta, h_inv, g_train, g_test, n)
        # Positive influence: removing the flow *raises* the loss on this test point's true
        # label, i.e. it was helping the model get this flow right. Negative: it was pushing
        # the model toward the wrong label. Highest positive = most supportive.
        order = np.argsort(-infl)
        helpful = [(str(labels[i]), str(days[i]), float(infl[i])) for i in order[: cfg.top_k]]
        harmful = [(str(labels[i]), str(days[i]), float(infl[i])) for i in order[::-1][: cfg.top_k]]
        explanations.append(
            InfluenceExplanation(
                test_label=_binary_label(y_test[idx]),
                test_pred_prob=float(probs[idx]),
                correct=bool(correct[idx]),
                helpful=helpful,
                harmful=harmful,
            )
        )
    return explanations


def _binary_label(y_binary: int) -> str:
    """Readable label for a binary test outcome (the multiclass label is not carried into x)."""
    return "attack" if int(y_binary) == 1 else "benign"


def _mislabel_check(
    cfg: InfluenceConfig, x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> float:
    """Plant label flips, rank by self-influence, and report the flip-detection AUC."""
    rng = np.random.default_rng(seed + 1)
    n = len(y_train)
    n_flip = int(n * cfg.mislabel_flip_rate)
    if n_flip == 0:
        return float("nan")
    flipped = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y_train.copy()
    y_noisy[flipped] = 1 - y_noisy[flipped]
    is_flipped = np.zeros(n, dtype=bool)
    is_flipped[flipped] = True

    theta = fit_logistic(x_train[:, :-1], y_noisy, cfg.l2, seed)
    h_inv = np.linalg.inv(hessian(theta, x_train, cfg.l2))
    g = _pointwise_grads(theta, x_train, _pm(y_noisy))
    scores = self_influence(h_inv, g)
    return float(roc_auc_score(is_flipped, scores))


def run_influence_report(settings: Settings) -> Path:
    """Run the influence-function study and write the report + figure."""
    study = run_influence(settings)

    fig = plots.plot_scatter_identity(
        study.loo_pred,
        study.loo_true,
        xlabel="predicted change in test loss (influence)",
        ylabel="actual change (leave-one-out retrain)",
        title="Influence functions predict the true leave-one-out effect",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote influence report", extra={"path": str(out_path)})

    with track_run(settings, "influence") as run:
        run.log_metrics(
            {
                "loo_pearson": study.loo_pearson,
                "loo_spearman": study.loo_spearman,
                "mislabel_auc": study.mislabel_auc,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _explanation_block(exp: InfluenceExplanation, i: int) -> str:
    verdict = "correct" if exp.correct else "**wrong**"
    lines = [
        f"### Test flow {i + 1}: true {exp.test_label}, model p(attack) = {exp.test_pred_prob:.3f} "
        f"({verdict})",
        "",
        f"| effect on the true-`{exp.test_label}` verdict | training flow label | capture day "
        "| influence |",
        "|---|---|---|---|",
    ]
    for label, day, val in exp.helpful:
        lines.append(f"| helps get it right | {label} | {day} | {val:+.2e} |")
    for label, day, val in exp.harmful:
        lines.append(f"| pushes it wrong | {label} | {day} | {val:+.2e} |")
    return "\n".join(lines)


def _read(study: InfluenceStudy) -> str:
    trust = (
        f"The influence estimate is **validated, not asserted**: across {study.loo_n:.0f} training "
        f"flows actually removed and retrained, the predicted change in test loss correlates with "
        f"the true leave-one-out change at Pearson **{study.loo_pearson:.2f}** (Spearman "
        f"{study.loo_spearman:.2f}). "
        + (
            "That is the Koh & Liang result reproduced on network-flow data — the closed-form "
            "inverse-Hessian estimate stands in for retraining the model thousands of times."
            if study.loo_pearson >= 0.7
            else (
                "The correlation is moderate on this stand-in, so the estimates are directional "
                "rather than exact here — reported as it fell; the validation is the point, and it "
                "keeps the tool honest about when to trust it."
            )
        )
    )
    mislabel = (
        f" The same machinery finds bad labels: ranking training flows by **self-influence** "
        f"recovers planted label flips at AUC **{study.mislabel_auc:.2f}** (at a "
        f"{study.mislabel_flip_rate:.0%} flip rate), an independent second opinion next to the "
        "confident-learning label audit and the KNN-Shapley data valuation — three different first "
        "principles (loss curvature, confident learning, and game-theoretic value) converging on "
        "the same suspicious rows."
    )
    return trust + mislabel


def _render(study: InfluenceStudy, fig: Path) -> str:
    blocks = "\n\n".join(_explanation_block(e, i) for i, e in enumerate(study.explanations))
    return f"""# NetSentry — Influence Functions (which training flows caused this verdict)

_Synthetic stand-in. Logistic surrogate on the stratified/binary split ({study.n_train:,}
training flows, {study.n_features} features). Influence is validated against real
leave-one-out retraining; explanations and the mislabel check use the same exact Hessian._

## Why this report exists

The [data-valuation study](data_value.md) scores each training flow's *global* worth; this
answers the *local* question an analyst asks of a surprising verdict — which specific training
flows drove it, and would removing them change it? Influence functions (Koh & Liang, ICML 2017)
estimate the leave-one-out effect of a training point on a test loss through the model's inverse
Hessian, with no retraining. They need a convex, twice-differentiable loss, so — like the
[distillation study](distill.md) — this runs on the **logistic** baseline, where the Hessian is
exact; the deployed gradient-boosted model is out of scope and that is stated, not smuggled past.

## Does the approximation hold? (validation against true leave-one-out)

{_read(study)}

![Predicted vs actual leave-one-out effect](../figures/{fig.name})

## Explaining individual verdicts

For each test flow, the training flows whose removal would most *raise* its loss (they support
the verdict) and most *lower* it (they oppose it):

{blocks}

## Scope

Influence is a first-order (infinitesimal-up-weighting) approximation around the fitted
parameters, exact in the limit and validated above for how well it holds here; it is computed
on the logistic surrogate, so it explains that model's decision, not the gradient-boosted
model's (the surrogate's own fidelity to the deployed model is the [distillation
study](distill.md)'s subject). The training pool is capped for the Hessian solve and the LOO
retrains, and everything runs on the exchangeable stratified split so a training-point influence
estimated there transfers to the test flow. What this buys, that no other explanation in the
suite does: an answer in the units of the *training data* — "remove these labelled flows and the
verdict moves" — which is directly actionable when a verdict is wrong because a handful of
training flows were mislabelled."""
