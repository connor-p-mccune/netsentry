"""Label-shift estimation and correction — recover the deployment prior with zero labels.

The base-rate study shows that the attack prevalence at deployment governs everything an
analyst's queue contains, and the [PPI study](ppi.md) estimates that prevalence from a
handful of labels. This study asks the harder question and then does something with the
answer: can the shifted prior be recovered from **entirely unlabelled** deployment traffic,
and can the classifier be *corrected* for it?

The setting is **label shift** (a.k.a. prior-probability shift): the class-conditional law
``p(x | y)`` is fixed — an attack looks like an attack — but the class prior ``p(y)`` moves,
because the fraction of malicious traffic on the wire is not the fraction in the training
mix. This is precisely the shift that resampling a test set to a new prevalence produces,
so it can be simulated exactly and the estimators validated against a known truth. Two cited
methods apply, and the study runs both because their assumptions differ:

- **BBSE** — Black-Box Shift Estimation (Lipton, Wang & Smola, ICML 2018). Using only the
  black-box predictor's *hard* labels, it solves the linear system ``C w = mu``, where
  ``C[i, j] = P(pred = i, true = j)`` is the source (validation) confusion matrix and
  ``mu[i] = P(pred = i)`` is the target's predicted-label distribution. The solution
  ``w[j] = q(y = j) / p(y = j)`` is the per-class importance weight, and the target prior is
  ``q(j) = w[j] p(j)``. Because it consumes only hard predictions and the confusion matrix,
  BBSE is **robust to miscalibration** — a badly-calibrated model still shifts consistently.
- **MLLS / EM** — the maximum-likelihood label-shift estimator (Saerens, Latinne &
  Decaestecker, 2002). It maximises the target log-likelihood over the prior by EM on the
  model's *soft* posteriors, so it is more **statistically efficient when the posteriors are
  calibrated**, and degrades when they are not.

Correction follows directly: the target posterior is the source posterior reweighted by the
class importance ``w`` and renormalised, ``q(y | x) ∝ w[y] p(y | x)``. The honest thing to
say up front is what correction does *not* buy — reweighting the two classes by constants is
a monotone transform of the attack score, so **ranking metrics (PR-AUC, ROC-AUC) do not
move**. What moves, and what the study measures, is **calibration**: an uncorrected model
reports probabilities anchored to the training prevalence, so at a shifted prior its scores
are systematically wrong; the correction repairs the Brier score and ECE, which is what any
threshold, cost calculation, or base-rate estimate downstream actually depends on.

Runs on the exchangeable stratified/binary split, because label shift assumes the target is
the source with only its prior changed — the exchangeability the temporal split deliberately
breaks (there the feature law shifts too, which is covariate/concept drift, a different
animal handled by the drift suite).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.calibration import brier_score, expected_calibration_error
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import LabelShiftConfig

logger = get_logger(__name__)

REPORT_NAME = "label_shift.md"
FIGURE_NAME = "label_shift.png"


def joint_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Source joint confusion matrix ``C[i, j] = P(pred = i, true = j)`` (sums to 1).

    BBSE's ``C``: the *joint* (not conditional) distribution of the black-box predictor's
    hard labels against the truth, estimated on labelled source (validation) data.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    c = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            c[i, j] = np.mean((yp == i) & (yt == j))
    return c


def bbse_weights(confusion: np.ndarray, mu_target: np.ndarray) -> np.ndarray:
    """Solve ``C w = mu`` for the per-class importance weights ``w[j] = q(j)/p(j)``.

    A near-singular ``C`` (a predictor no better than random has no shift signal) is caught
    by a least-squares fallback, and weights are clipped non-negative — a negative importance
    weight is meaningless and only ever a finite-sample artefact.
    """
    c = np.asarray(confusion, dtype=float)
    mu = np.asarray(mu_target, dtype=float)
    try:
        w = np.linalg.solve(c, mu)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(c, mu, rcond=None)
    return np.clip(w, 0.0, None)


def prior_from_weights(weights: np.ndarray, source_prior: np.ndarray) -> np.ndarray:
    """Target prior ``q(j) = w[j] p(j)``, renormalised to a distribution."""
    q = np.asarray(weights, dtype=float) * np.asarray(source_prior, dtype=float)
    total = q.sum()
    return q / total if total > 0 else np.asarray(source_prior, dtype=float)


def mlls_em_prior(
    posteriors: np.ndarray, source_prior: np.ndarray, max_iter: int, tol: float
) -> tuple[np.ndarray, int]:
    """MLE of the target prior by EM on soft posteriors (Saerens et al. 2002).

    Iterates ``q <- mean_x [ (q/p) . eta(x) / sum_k (q_k/p_k) eta_k(x) ]`` from the source
    prior until the prior stops moving. Returns the estimated prior and the iteration count.
    """
    eta = np.asarray(posteriors, dtype=float)
    p = np.asarray(source_prior, dtype=float)
    q = p.copy()
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        w = np.divide(q, p, out=np.zeros_like(q), where=p > 0)
        weighted = eta * w  # (n, k)
        denom = weighted.sum(axis=1, keepdims=True)
        gamma = np.divide(weighted, denom, out=np.zeros_like(weighted), where=denom > 0)
        q_new = gamma.mean(axis=0)
        q_new = q_new / q_new.sum() if q_new.sum() > 0 else q
        converged = bool(np.abs(q_new - q).max() < tol)
        q = q_new
        if converged:
            break
    return np.asarray(q, dtype=float), n_iter


def correct_posteriors(posteriors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reweight source posteriors by class importance and renormalise (``q(y|x) ∝ w[y] eta_y``)."""
    eta = np.asarray(posteriors, dtype=float)
    w = np.asarray(weights, dtype=float)
    reweighted = eta * w
    denom = reweighted.sum(axis=1, keepdims=True)
    out = np.divide(reweighted, denom, out=np.full_like(reweighted, 0.5), where=denom > 0)
    return np.asarray(out, dtype=float)


def resample_to_prior(
    y: np.ndarray, target_prior: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Row indices drawn (with replacement) so the attack fraction equals ``target_prior``.

    Sampling within each class preserves ``p(x | y)`` exactly and changes only ``p(y)`` — a
    label shift by construction, which is what makes the estimators' error measurable against
    a known truth.
    """
    y = np.asarray(y, dtype=int)
    attack_idx = np.where(y == 1)[0]
    benign_idx = np.where(y == 0)[0]
    n_attack = round(size * target_prior)
    n_benign = size - n_attack
    picks = np.concatenate(
        [
            rng.choice(attack_idx, size=n_attack, replace=True),
            rng.choice(benign_idx, size=n_benign, replace=True),
        ]
    )
    rng.shuffle(picks)
    return picks


@dataclass
class ShiftRow:
    """Estimation error and calibration at one true target prior, averaged over trials."""

    true_prior: float
    mae_bbse: float
    mae_mlls: float
    mae_naive: float  # predicted-positive-rate (biased by model error)
    mae_none: float  # assume the source prior (no correction)
    brier_uncorrected: float
    brier_corrected: float  # BBSE-corrected posteriors
    ece_uncorrected: float
    ece_corrected: float
    pr_auc: float  # unchanged by correction, reported to make that explicit


@dataclass
class LabelShiftStudy:
    """The full label-shift study over the target-prior sweep."""

    source_prior: float
    n_val: int
    n_test: int
    em_iters: int
    rows: list[ShiftRow]


def run_label_shift(settings: Settings) -> LabelShiftStudy:
    """Fit the exchangeable model, then estimate + correct label shift over a prior sweep."""
    cfg: LabelShiftConfig = settings.label_shift
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test").reset_index(drop=True)
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = model.classes_

    # Source confusion matrix from the black-box predictor's hard labels on validation.
    val_pred = np.asarray(model.predict(x_val)).astype(int)
    confusion = joint_confusion(y_val, val_pred, n_classes=2)
    source_prior = confusion.sum(axis=0)  # P(true = j) on val
    source_prior = source_prior / source_prior.sum()

    # Full-test posteriors, sampled per trial to a target prevalence.
    test_proba = model.predict_proba(x_test)  # (n, 2) columns aligned to `classes`
    attack_col = int(np.where(classes == 1)[0][0])
    test_pos = positive_scores(test_proba, classes)

    rng = np.random.default_rng(variant.seed)
    rows: list[ShiftRow] = []
    em_iters = 0
    for q_true in cfg.target_priors:
        errs: dict[str, list[float]] = {k: [] for k in ("bbse", "mlls", "naive", "none")}
        brier_u: list[float] = []
        brier_c: list[float] = []
        ece_u: list[float] = []
        ece_c: list[float] = []
        prauc: list[float] = []
        for _ in range(cfg.n_trials):
            idx = resample_to_prior(y_test, q_true, cfg.target_size, rng)
            proba = test_proba[idx]
            pos = test_pos[idx]
            y_t = y_test[idx]
            pred = (np.asarray(proba)[:, attack_col] >= 0.5).astype(int)
            mu = np.array([np.mean(pred == 0), np.mean(pred == 1)])

            w = bbse_weights(confusion, mu)
            q_bbse = prior_from_weights(w, source_prior)
            q_mlls, em_iters = mlls_em_prior(
                np.asarray(proba), source_prior, cfg.em_max_iter, cfg.em_tol
            )
            errs["bbse"].append(abs(q_bbse[1] - q_true))
            errs["mlls"].append(abs(q_mlls[1] - q_true))
            errs["naive"].append(abs(mu[1] - q_true))
            errs["none"].append(abs(source_prior[1] - q_true))

            corrected = correct_posteriors(np.asarray(proba), w)
            corr_pos = corrected[:, attack_col]
            brier_u.append(brier_score(y_t, pos))
            brier_c.append(brier_score(y_t, corr_pos))
            ece_u.append(expected_calibration_error(y_t, pos))
            ece_c.append(expected_calibration_error(y_t, corr_pos))
            prauc.append(float(average_precision_score(y_t, pos)))
        rows.append(
            ShiftRow(
                true_prior=q_true,
                mae_bbse=float(np.mean(errs["bbse"])),
                mae_mlls=float(np.mean(errs["mlls"])),
                mae_naive=float(np.mean(errs["naive"])),
                mae_none=float(np.mean(errs["none"])),
                brier_uncorrected=float(np.mean(brier_u)),
                brier_corrected=float(np.mean(brier_c)),
                ece_uncorrected=float(np.mean(ece_u)),
                ece_corrected=float(np.mean(ece_c)),
                pr_auc=float(np.mean(prauc)),
            )
        )
        logger.info(
            "Label-shift point",
            extra={
                "q_true": q_true,
                "mae_bbse": round(rows[-1].mae_bbse, 4),
                "mae_mlls": round(rows[-1].mae_mlls, 4),
            },
        )
    return LabelShiftStudy(
        source_prior=float(source_prior[1]),
        n_val=len(val),
        n_test=len(test),
        em_iters=em_iters,
        rows=rows,
    )


def run_label_shift_report(settings: Settings) -> Path:
    """Run the label-shift study and write the report + figure."""
    study = run_label_shift(settings)

    priors = np.array([r.true_prior for r in study.rows])
    fig = plots.plot_lines(
        {
            "BBSE (hard labels + confusion matrix)": (
                priors,
                np.array([r.mae_bbse for r in study.rows]),
            ),
            "MLLS / EM (soft posteriors)": (priors, np.array([r.mae_mlls for r in study.rows])),
            "naive predicted-positive rate": (
                priors,
                np.array([r.mae_naive for r in study.rows]),
            ),
            "no correction (assume source prior)": (
                priors,
                np.array([r.mae_none for r in study.rows]),
            ),
        },
        xlabel="true deployment attack prevalence",
        ylabel="mean absolute error of the prior estimate",
        title="Label-shift: recovering the deployment prior with zero labels",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote label-shift report", extra={"path": str(out_path)})

    with track_run(settings, "label_shift") as run:
        run.log_metrics(
            {
                "source_prior": study.source_prior,
                "mean_mae_bbse": float(np.mean([r.mae_bbse for r in study.rows])),
                "mean_mae_mlls": float(np.mean([r.mae_mlls for r in study.rows])),
                "mean_mae_none": float(np.mean([r.mae_none for r in study.rows])),
                "mean_brier_gain": float(
                    np.mean([r.brier_uncorrected - r.brier_corrected for r in study.rows])
                ),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _estimation_table(study: LabelShiftStudy) -> str:
    rows = [
        "| true prevalence | BBSE | MLLS/EM | naive pred-rate | no correction |",
        "|---|---|---|---|---|",
    ]
    for r in study.rows:
        rows.append(
            f"| {r.true_prior:.2f} | {r.mae_bbse:.4f} | {r.mae_mlls:.4f} "
            f"| {r.mae_naive:.4f} | {r.mae_none:.4f} |"
        )
    return "\n".join(rows)


def _calibration_table(study: LabelShiftStudy) -> str:
    rows = [
        "| true prevalence | Brier (uncorrected) | Brier (corrected) | ECE (uncorrected) "
        "| ECE (corrected) | PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in study.rows:
        rows.append(
            f"| {r.true_prior:.2f} | {r.brier_uncorrected:.4f} | {r.brier_corrected:.4f} "
            f"| {r.ece_uncorrected:.4f} | {r.ece_corrected:.4f} | {r.pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _read(study: LabelShiftStudy) -> str:
    mean_bbse = float(np.mean([r.mae_bbse for r in study.rows]))
    mean_mlls = float(np.mean([r.mae_mlls for r in study.rows]))
    mean_none = float(np.mean([r.mae_none for r in study.rows]))
    mean_naive = float(np.mean([r.mae_naive for r in study.rows]))
    best_name, best = ("BBSE", mean_bbse) if mean_bbse <= mean_mlls else ("MLLS/EM", mean_mlls)
    lift = mean_none / best if best > 0 else float("inf")

    est = (
        f"With no deployment labels, {best_name} recovers the attack prevalence to a mean "
        f"absolute error of **{best:.4f}** across the sweep — about **{lift:.0f}x** tighter "
        f"than assuming the training prior ({mean_none:.4f}), whose error just grows with the "
        f"size of the shift. The naive predicted-positive rate ({mean_naive:.4f}) sits between "
        "them: it moves in the right direction but inherits the model's own error, exactly the "
        "bias BBSE removes by inverting the confusion matrix. "
        + (
            f"BBSE and MLLS/EM land close here ({mean_bbse:.4f} vs {mean_mlls:.4f}): the "
            "stand-in model is calibrated enough for the likelihood estimator to keep up."
            if abs(mean_bbse - mean_mlls) < 0.01
            else (
                f"The two estimators separate sharply — MLLS/EM trails at {mean_mlls:.4f} — "
                "which is the textbook contrast, not a bug: MLLS assumes calibrated posteriors "
                "and this model is not calibrated enough for it, whereas BBSE reads only hard "
                "labels through the confusion matrix and is immune to exactly that. The moment "
                "estimator is the safer default, and this stand-in is why."
            )
        )
    )

    brier_gain = float(np.mean([r.brier_uncorrected - r.brier_corrected for r in study.rows]))
    biggest = max(study.rows, key=lambda r: r.brier_uncorrected - r.brier_corrected)
    hurt = [r for r in study.rows if r.brier_corrected > r.brier_uncorrected + 1e-4]
    near_source = (
        " Correction is not free everywhere: at prevalences near the source prior "
        f"({study.source_prior:.2f}) there is little shift to undo and the estimated weights "
        "add noise, so the "
        + ", ".join(f"{r.true_prior:.2f}" for r in hurt)
        + " row(s) come out marginally worse — the honest signature of a correction that should "
        "be applied in proportion to the measured shift, not reflexively."
        if hurt
        else ""
    )
    corr = (
        f" Correction does not touch ranking — PR-AUC is identical corrected or not, because "
        f"reweighting two classes by constants is monotone in the score — and the report shows "
        f"that column unchanged so the point is explicit. What it repairs is **calibration**: "
        f"averaged over the sweep the Brier score improves by {brier_gain:.4f}, and where "
        f"correction helps most (prevalence {biggest.true_prior:.2f}, against the "
        f"{study.source_prior:.2f} the model was trained at) it falls "
        f"{biggest.brier_uncorrected:.4f} -> "
        f"{biggest.brier_corrected:.4f} and ECE {biggest.ece_uncorrected:.4f} -> "
        f"{biggest.ece_corrected:.4f}. That is the number every downstream decision depends on — "
        "a threshold, a cost calculation, the base-rate estimate itself — so a model whose "
        "probabilities are anchored to the wrong prior is quietly wrong everywhere it is trusted, "
        "and this is the zero-label repair."
    )
    return est + corr + near_source


def _render(study: LabelShiftStudy, fig: Path) -> str:
    return f"""# NetSentry — Label-Shift Estimation & Correction (zero deployment labels)

_Synthetic stand-in. Stratified/binary model; the source (training) attack prevalence is
**{study.source_prior:.3f}**. Each row resamples the {study.n_test:,}-flow test set to a
target prevalence (preserving p(x | y), so it is a pure label shift) and averages over the
configured trials. MLLS/EM converged in ~{study.em_iters} iterations._

## Why this report exists

The [base-rate study](base_rate.md) shows the deployment prevalence governs what an
analyst's queue contains, and [PPI](ppi.md) estimates it from a few labels. Label shift is
the harder, label-free version: under the assumption that only the class prior changes
between training and deployment (the feature law ``p(x | y)`` fixed), the shifted prior is
recoverable from unlabelled traffic alone, and the classifier can be corrected for it. Two
cited estimators are run: **BBSE** (Lipton, Wang & Smola, ICML 2018) inverts the source
confusion matrix against the target's predicted-label distribution — hard labels only, so
robust to miscalibration — and **MLLS/EM** (Saerens et al. 2002) maximises the target
likelihood over the prior on soft posteriors — efficient when calibrated.

## Estimating the deployment prior (mean absolute error, lower is better)

{_estimation_table(study)}

![Label-shift prior estimation error](../figures/{fig.name})

## Correcting the classifier (calibration, not ranking)

{_calibration_table(study)}

{_read(study)}

## Scope

Label shift is not covariate or concept drift: it assumes an attack still *looks* like an
attack and only their proportion changes, which is why this study lives on the exchangeable
stratified split and the [drift suite](drift.md) — where the feature law itself moves — is
the complement, not a duplicate. The correction is exact only to the extent the assumption
holds; a real deployment mixes label shift with covariate shift, so the honest use is BBSE
as a **prior monitor** (its estimate drifting from the training prior is itself an alert,
and unlike a raw predicted-positive rate it is not fooled by the model's own error) feeding
the base-rate and cost calculations, with the drift detectors watching the other axis. The
2x2 confusion matrix must be non-singular — a detector no better than random carries no
shift signal — which is the label-shift analogue of every other study's "the model has to be
worth correcting" precondition."""
