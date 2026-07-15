"""Certified robustness via randomized smoothing — a provable radius, not a measured one.

The evasion study *measures* how far an attacker can push detection down; the hardening
study *reduces* that empirically. Neither one can say a flow is **provably** safe: an
absent attack is only evidence no one has found the attack yet. Randomized smoothing
(Cohen, Rosenfeld & Kolter, 2019) closes that gap with a certificate. Wrap the detector
in Gaussian noise — classify by majority vote over ``x + N(0, sigma^2 I)`` — and the
smoothed classifier comes with a **guaranteed** L2 radius: no perturbation smaller than

    R = sigma * Phi^-1(p_A)

can change its decision, where ``p_A`` is a high-confidence lower bound (Clopper-Pearson
over the Monte-Carlo votes) on the probability the base detector returns the majority
class under noise. Inside that ball the verdict is certified; an attacker cannot evade
without a larger perturbation, whether or not anyone has found one.

This is the formal-guarantee counterpart to the empirical evasion study, exactly as
differential privacy is the formal counterpart to the empirical membership audit: the
project's recurring arc of measuring a risk and then buying a certificate against it. It
is reported honestly, with the two conservatisms named. The certificate is against **any**
L2 perturbation, while the evasion attacker only moves the *controllable* feature subset,
so the certified radius is a strictly harder guarantee than the attack it is compared to;
and ``sigma`` trades clean detection for a larger certifiable radius — the standard
accuracy/robustness frontier, measured here rather than assumed. Radii are in the same
standardised-feature units as the evasion search budgets, so the two reports read against
each other directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta, norm

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "certify.md"
FIGURE_NAME = "certify.png"
ABSTAIN = -1  # Cohen's abstention: the vote is too close to certify either class


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided lower confidence bound on a binomial proportion (exact, Clopper-Pearson).

    Returns the largest p such that ``P(Binomial(n, p) >= k) <= alpha`` — the standard
    conservative lower bound randomized smoothing relies on so the certificate holds with
    probability at least ``1 - alpha``.
    """
    if k <= 0:
        return 0.0
    if k >= n:
        return float(alpha ** (1.0 / n))
    return float(beta.ppf(alpha, k, n - k + 1))


def certified_radius(attack_votes: int, n: int, sigma: float, alpha: float) -> tuple[int, float]:
    """Cohen's CERTIFY for the binary detector: return (predicted class, certified radius).

    ``attack_votes`` of ``n`` noisy draws returned "attack". The majority class is the
    prediction; its Monte-Carlo probability gets a Clopper-Pearson lower bound ``p_A``,
    and if ``p_A > 1/2`` the radius is ``sigma * Phi^-1(p_A)``. Otherwise the smoothed
    classifier abstains (radius 0), because the vote does not clear the confidence bar.
    """
    benign_votes = n - attack_votes
    if attack_votes >= benign_votes:
        cls, k = 1, attack_votes
    else:
        cls, k = 0, benign_votes
    p_lower = clopper_pearson_lower(k, n, alpha)
    if p_lower <= 0.5:
        return ABSTAIN, 0.0
    return cls, float(sigma * norm.ppf(p_lower))


@dataclass
class CertRow:
    """One flow's certification outcome under a given noise level."""

    true_label: int
    predicted: int  # 0 benign, 1 attack, or ABSTAIN
    radius: float

    @property
    def correct(self) -> bool:
        return self.predicted == self.true_label

    def certified_at(self, r: float) -> bool:
        """Correctly classified and provably robust to every L2 perturbation up to ``r``."""
        return self.correct and self.radius >= r


@dataclass
class SigmaResult:
    """Certification summary for one noise level over the evaluated flows."""

    sigma: float
    rows: list[CertRow]
    radii_grid: list[float]

    @property
    def abstain_rate(self) -> float:
        return float(np.mean([r.predicted == ABSTAIN for r in self.rows])) if self.rows else 0.0

    @property
    def clean_accuracy(self) -> float:
        """Smoothed-classifier accuracy ignoring certification (abstain counts as wrong)."""
        return float(np.mean([r.correct for r in self.rows])) if self.rows else 0.0

    @property
    def median_radius(self) -> float:
        certified = [r.radius for r in self.rows if r.radius > 0]
        return float(np.median(certified)) if certified else 0.0

    def certified_accuracy(self, r: float) -> float:
        return float(np.mean([row.certified_at(r) for row in self.rows])) if self.rows else 0.0


def _attack_score(
    model: SupervisedClassifier, classes: np.ndarray, benign: str, x: np.ndarray
) -> np.ndarray:
    return attack_probability(model.predict_proba(x), classes, benign)


def _attack_votes(
    model: SupervisedClassifier,
    classes: np.ndarray,
    benign: str,
    x_row: np.ndarray,
    threshold: float,
    sigma: float,
    n: int,
    rng: np.random.Generator,
    batch: int = 2000,
) -> int:
    """Count how many of ``n`` Gaussian-noise draws of ``x_row`` the base calls 'attack'."""
    votes = 0
    remaining = n
    d = len(x_row)
    while remaining > 0:
        b = min(batch, remaining)
        perturbed = x_row[None, :] + rng.standard_normal((b, d)) * sigma
        scores = _attack_score(model, classes, benign, perturbed)
        votes += int(np.sum(scores >= threshold))
        remaining -= b
    return votes


def run_certify(settings: Settings) -> list[SigmaResult]:
    """Certify a class-mixed sample of flows across the configured noise levels."""
    cfg = settings.certify
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")
    y_train = train[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    classes = np.asarray(model.model.classes_)
    benign = settings.labels.benign_label

    s_val = _attack_score(model, classes, benign, x_val)
    threshold = threshold_at_fpr(y_val, s_val, cfg.target_fpr)

    # A balanced sample so certified accuracy is not dominated by the benign majority.
    rng = np.random.default_rng(variant.seed)
    test = test.reset_index(drop=True)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    per_class = cfg.max_flows // 2
    idx = np.concatenate(
        [
            _sample_class(np.where(y_test == 1)[0], per_class, rng),
            _sample_class(np.where(y_test == 0)[0], per_class, rng),
        ]
    )
    x_eval = np.asarray(pipeline.transform(test.iloc[idx]))
    y_eval = y_test[idx]

    results: list[SigmaResult] = []
    for sigma in cfg.sigmas:
        rows: list[CertRow] = []
        for i in range(len(x_eval)):
            row_rng = np.random.default_rng(variant.seed + i)
            votes = _attack_votes(
                model, classes, benign, x_eval[i], threshold, sigma, cfg.n_samples, row_rng
            )
            predicted, radius = certified_radius(votes, cfg.n_samples, sigma, cfg.alpha)
            rows.append(CertRow(true_label=int(y_eval[i]), predicted=predicted, radius=radius))
        result = SigmaResult(sigma=sigma, rows=rows, radii_grid=cfg.radii_grid)
        results.append(result)
        logger.info(
            "Certified at sigma",
            extra={
                "sigma": sigma,
                "clean_acc": round(result.clean_accuracy, 3),
                "median_radius": round(result.median_radius, 3),
                "abstain": round(result.abstain_rate, 3),
            },
        )
    return results


def _sample_class(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) <= n:
        return pool
    return rng.choice(pool, size=n, replace=False)


def run_certify_report(settings: Settings) -> Path:
    """Run the certified-robustness study and write the report + figure."""
    results = run_certify(settings)

    series = {
        f"sigma = {r.sigma:g}": (
            np.array(r.radii_grid, dtype=float),
            np.array([r.certified_accuracy(g) for g in r.radii_grid]),
        )
        for r in results
    }
    fig = plots.plot_lines(
        series,
        xlabel="Certified L2 radius (standardised feature units)",
        ylabel="Certified accuracy (correct AND provably robust)",
        title="Certified robustness via randomized smoothing (Cohen et al. 2019)",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(results, settings, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote certification report", extra={"path": str(out_path)})

    with track_run(settings, "certify") as run:
        for r in results:
            run.log_metrics(
                {
                    f"clean_acc_sigma{r.sigma:g}": r.clean_accuracy,
                    f"median_radius_sigma{r.sigma:g}": r.median_radius,
                    f"abstain_sigma{r.sigma:g}": r.abstain_rate,
                }
            )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _table(results: list[SigmaResult]) -> str:
    grid = results[0].radii_grid
    header_r = " | ".join(f"cert@{g:g}" for g in grid)
    rows = [
        f"| sigma | smoothed acc. | abstain | median radius | {header_r} |",
        "|---|---|---|---|" + "---|" * len(grid),
    ]
    for r in results:
        cells = " | ".join(f"{r.certified_accuracy(g):.0%}" for g in grid)
        rows.append(
            f"| {r.sigma:g} | {r.clean_accuracy:.0%} | {r.abstain_rate:.0%} "
            f"| {r.median_radius:.3f} | {cells} |"
        )
    return "\n".join(rows)


def _read(results: list[SigmaResult]) -> str:
    best = max(results, key=lambda r: r.median_radius)
    small = max(results, key=lambda r: r.clean_accuracy)
    trades = small.sigma != best.sigma and best.clean_accuracy < small.clean_accuracy - 0.015
    frontier = (
        f"The accuracy/robustness frontier is visible in the table: the smallest noise "
        f"(sigma = {small.sigma:g}) keeps the most clean detection ({small.clean_accuracy:.0%}) "
        f"but certifies the smallest radii, while sigma = {best.sigma:g} certifies the largest "
        f"median radius ({best.median_radius:.3f}) at a clean-accuracy cost "
        f"({best.clean_accuracy:.0%}). "
        if trades
        else (
            f"On this stand-in the median certified radius peaks at {best.median_radius:.3f} "
            f"(sigma = {best.sigma:g}). "
        )
    )
    return (
        frontier
        + "Two conservatisms are worth stating plainly. The certificate is against **any** "
        "L2 perturbation, whereas the evasion study's attacker only moves the controllable "
        "feature subset — so a certified radius is a strictly stronger promise than the "
        "budget the attack needs, and the two numbers are not directly comparable, only "
        "read side by side. And randomized smoothing on an undefended gradient-boosted model "
        "certifies conservatively: the base was never trained to be stable under noise, which "
        "is why abstention is non-trivial and radii are modest. The standard next step (Cohen "
        "et al.) is to train the base on noise-augmented data so the smoothed classifier both "
        "abstains less and certifies farther — the measure-then-fix arc this project applies "
        "to evasion (hardening) and privacy (differential privacy), here for certified radius."
    )


def _render(results: list[SigmaResult], settings: Settings, fig: Path) -> str:
    cfg = settings.certify
    n_flows = len(results[0].rows) if results else 0
    return f"""# NetSentry - Certified Robustness (Randomized Smoothing)

_Synthetic stand-in. Stratified/binary model; {n_flows} class-balanced flows certified
with {cfg.n_samples:,} Monte-Carlo noise draws each at confidence 1 - {cfg.alpha:g}
(Clopper-Pearson). Radii are in standardised-feature L2 units, the same scale as the
evasion study's search budgets._

## Why this report exists

The evasion study measures how far detection can be pushed down by an attacker; the
hardening study reduces that empirically. Neither can *prove* a flow is safe — an absent
attack is only an attack not yet found. Randomized smoothing (Cohen, Rosenfeld & Kolter,
2019) gives a **provable** guarantee: classify each flow by majority vote under Gaussian
noise, and no L2 perturbation smaller than ``R = sigma * Phi^-1(p_A)`` can change the
verdict, where ``p_A`` is a Clopper-Pearson lower bound on the majority-vote probability.
This is the formal-guarantee counterpart to the empirical evasion study — the same role
differential privacy plays for the empirical membership audit.

## Certified accuracy vs radius

A flow counts as certified at radius ``r`` only if the smoothed classifier gets it right
**and** proves robustness to every L2 perturbation up to ``r``.

{_table(results)}

![Certified accuracy vs radius](../figures/{fig.name})

{_read(results)}

## Scope

Certification is a property of the *smoothed* classifier (majority vote under noise), not
the raw model the API serves — deploying it means accepting the clean-accuracy cost and
the per-flow sampling cost in the table. The guarantee is probabilistic (holds with
confidence 1 - {cfg.alpha:g} over the Monte-Carlo draws) and against all-feature L2
perturbations. It complements, rather than replaces, the empirical evasion and hardening
studies: those bound what a *known* attacker achieves against the deployed model; this
bounds what *any* attacker could achieve against the smoothed one."""
