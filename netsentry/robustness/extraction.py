"""Model-extraction (model-stealing) attack — the fourth classic adversarial axis.

Evasion is the inference-time adversary, poisoning the training-time one, and
membership inference the privacy one. This is the fourth attack in the standard
taxonomy and the one about **confidentiality of the model itself**: given only
query access to the deployed classifier — the exact access the ``/predict`` API
grants — can an attacker train a *surrogate* that replicates it, without ever
seeing a single ground-truth label (Tramer et al., 2016; Papernot et al., 2017)?

Two things make a stolen model matter on a NIDS, and both are measured here:

1. **The model is a stealable asset.** A surrogate trained purely on the victim's
   returned scores recovers most of its detection (PR-AUC approaching the
   victim's) and most of its decision boundary (high *fidelity* — agreement with
   the victim, not with the truth). The detector — tuning, thresholds, the
   behaviour a competitor or an attacker would pay for — leaks through the query
   interface.
2. **Stealing turns a black box into a white box for evasion.** Querying the
   victim to craft an evasion attack is costly and monitorable (it is exactly the
   query-search the robustness study rate-limits against). A stolen surrogate is
   free and unlimited to attack *offline*; the perturbations found on it
   **transfer** to the victim. So the extraction attack is the enabler behind the
   transfer-evasion threat, and this report prices that transfer directly.

The honest arc — the project's measure -> re-measure signature — is the **defense
axis**: the classic Tramer mitigation is to return *less* information. The victim
is queried three ways — full probabilities, probabilities rounded to a few
decimals, and the top-1 label only — and the fidelity/detection the attacker
keeps is measured for each. The kept finding is the same one the literature
reports: rounding and label-only *reduce* extraction fidelity but do not stop it,
because the decision boundary is recoverable from hard labels alone.

Runs on the exchangeable **stratified**/binary split: the attacker collects its
own same-distribution traffic to query with, so there is no temporal shift to
confound the surrogate. The feature representation (CIC flow features + a standard
scaler) is treated as public — the secret being stolen is the *model*, not the
featurisation — so victim and surrogate share the fitted pipeline; the surrogate
is a different, generic model family the attacker does not need to guess right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.robustness.evasion import attack_scores_transformed, controllable_indices
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

REPORT_NAME = "extraction.md"

# The three ways the victim can answer a query, ordered most -> least informative.
# This is the extraction *defense* axis (Tramer et al.): return less, leak less.
PROBABILITIES = "probabilities"
ROUNDED = "rounded"
LABEL_ONLY = "label_only"


def victim_scores(bundle: ModelBundle, x_transformed: np.ndarray) -> np.ndarray:
    """Raw attack probability the victim returns for already-transformed rows.

    Raw (uncalibrated) scores match the evaluation report's ranking convention;
    the calibrator is monotone, so it would not change fidelity or PR-AUC, only
    the numeric scale the surrogate regresses onto.
    """
    proba = np.asarray(bundle.model.predict_proba(x_transformed))
    benign = str(bundle.metadata.get("benign_label", "BENIGN"))
    return attack_probability(proba, bundle.classes, benign)


def answered_query(scores: np.ndarray, mode: str, round_decimals: int) -> np.ndarray:
    """The victim's response under a defense mode — the training signal the attacker gets."""
    if mode == PROBABILITIES:
        return np.asarray(scores, dtype=float)
    if mode == ROUNDED:
        return np.round(np.asarray(scores, dtype=float), round_decimals)
    if mode == LABEL_ONLY:
        return (np.asarray(scores, dtype=float) >= 0.5).astype(float)
    raise ValueError(f"unknown query mode: {mode}")


def train_surrogate(
    x_query: np.ndarray, answers: np.ndarray, mode: str, seed: int
) -> SurrogateScorer:
    """Fit the attacker's surrogate on ``(query, victim answer)`` pairs — no ground truth.

    Soft answers (probabilities / rounded) are matched by a regressor
    (distillation); a hard label-only answer is matched by a classifier. Both are
    a generic gradient-boosting family the attacker picks without knowing the
    victim's architecture.
    """
    if mode == LABEL_ONLY:
        labels = (np.asarray(answers) >= 0.5).astype(int)
        if len(np.unique(labels)) < 2:  # degenerate query pool: nothing to separate
            return SurrogateScorer(None, float(labels.mean()) if len(labels) else 0.0)
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(x_query, labels)
        return SurrogateScorer(clf, None)
    reg = HistGradientBoostingRegressor(random_state=seed)
    reg.fit(x_query, np.asarray(answers, dtype=float))
    return SurrogateScorer(reg, None)


@dataclass
class SurrogateScorer:
    """A fitted surrogate, exposing a single attack-score callable over transformed rows."""

    estimator: Any  # a fitted sklearn regressor/classifier, or None for a degenerate pool
    constant: float | None

    def score(self, x_transformed: np.ndarray) -> np.ndarray:
        """Attack score in [0, 1] for transformed rows (constant if the pool was degenerate)."""
        x = np.asarray(x_transformed)
        if self.estimator is None:
            return np.full(len(x), float(self.constant or 0.0))
        if isinstance(self.estimator, HistGradientBoostingClassifier):
            proba: np.ndarray = np.asarray(self.estimator.predict_proba(x), dtype=float)
            return proba[:, 1]
        preds: np.ndarray = np.asarray(self.estimator.predict(x), dtype=float)
        clipped: np.ndarray = np.clip(preds, 0.0, 1.0)
        return clipped


def fidelity(victim: np.ndarray, surrogate: np.ndarray) -> float:
    """Fraction of rows where the surrogate's decision matches the victim's (not the truth).

    Fidelity is the extraction-success metric: how faithfully the stolen model
    reproduces the victim's boundary. Both use the natural 0.5 argmax cut so the
    number is threshold-free and comparable across query modes.
    """
    v = np.asarray(victim) >= 0.5
    s = np.asarray(surrogate) >= 0.5
    return float(np.mean(v == s)) if len(v) else 0.0


def search_delta(
    score_fn: object,
    x: np.ndarray,
    ctrl_idx: np.ndarray,
    eps: float,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random-restart query search for the L2-bounded perturbation that minimises the score.

    The perturbation is confined to the attacker-controllable feature indices and
    to an L2 ball of radius ``eps`` (standardised units). ``score_fn`` maps
    transformed rows to attack scores — the victim (white-box on the real model)
    or the surrogate (the offline attack a stolen model enables). Returns the best
    per-row delta found, so the same perturbation can be replayed on either model
    to measure transfer.
    """
    assert callable(score_fn)
    n, d = x.shape
    best_delta = np.zeros((n, d))
    if eps == 0.0 or len(ctrl_idx) == 0:
        return best_delta
    best_score = np.asarray(score_fn(x), dtype=float)
    for _ in range(iterations):
        delta = np.zeros((n, d))
        step = rng.standard_normal((n, len(ctrl_idx)))
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        radius = eps * rng.uniform(size=(n, 1))  # search the ball interior, not just its shell
        delta[:, ctrl_idx] = step / norms * radius
        trial = np.asarray(score_fn(x + delta), dtype=float)
        improved = trial < best_score
        best_delta[improved] = delta[improved]
        best_score = np.minimum(best_score, trial)
    return best_delta


def _detection(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores) >= threshold)) if len(scores) else 0.0


@dataclass
class BudgetPoint:
    """Extraction quality at one query budget (full-probability queries)."""

    n_queries: int
    fidelity: float
    surrogate_pr_auc: float


@dataclass
class DefensePoint:
    """Extraction quality under one query-response defense mode, at the max budget."""

    mode: str
    fidelity: float
    surrogate_pr_auc: float


@dataclass
class TransferResult:
    """Evasion detection under each attack source, at the victim's operating point."""

    profile_fpr: float
    threshold: float
    baseline: float  # unperturbed attack detection (no evasion)
    random_control: float  # random perturbations of the same budget (no model used)
    transfer: float  # perturbations searched on the *surrogate*, scored on the victim
    white_box: float  # perturbations searched on the *victim* (the upper bound)
    eps: float
    n_attacks: int


@dataclass
class ExtractionStudy:
    """The full model-stealing study: budget sweep, defense comparison, transfer attack."""

    victim_pr_auc: float
    budgets: list[BudgetPoint]
    defenses: list[DefensePoint]
    transfer: TransferResult
    round_decimals: int


def _binary(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=int)


def run_extraction(settings: Settings) -> ExtractionStudy:
    """Steal the deployed model by query, then measure fidelity, task theft, and transfer."""
    cfg = settings.extraction
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    result = fit_supervised(variant)
    bundle = result.bundle
    classes = result.classes
    pipeline = bundle.pipeline

    # Threshold for the transfer operating point, chosen on validation (raw scores).
    s_val = attack_probability(result.proba_val, classes, settings.labels.benign_label)
    y_val = _binary(result.y_val)
    threshold = threshold_at_fpr(y_val, s_val, cfg.transfer_fpr)

    # The attacker's own same-distribution traffic: the held-out test split, carved
    # into an unlabelled *query pool* and a disjoint *eval* set. Labels on the eval
    # set are used only by *our* measurement (fidelity/PR-AUC), never by the attacker.
    test = load_split(variant, "stratified", "test").reset_index(drop=True)
    rng = np.random.default_rng(variant.seed)
    order = rng.permutation(len(test))
    max_budget = min(max(cfg.query_budgets), len(test) - 1)
    pool_idx = order[:max_budget]
    eval_idx = order[max_budget : max_budget + cfg.max_eval_rows]

    pool_df = test.iloc[pool_idx]
    eval_df = test.iloc[eval_idx]
    x_pool = np.asarray(pipeline.transform(pool_df))
    x_eval = np.asarray(pipeline.transform(eval_df))
    y_eval = (eval_df[BINARY_TARGET].to_numpy() == 1).astype(int)

    s_pool = victim_scores(bundle, x_pool)
    s_eval_victim = victim_scores(bundle, x_eval)
    victim_pr_auc = float(average_precision_score(y_eval, s_eval_victim))

    # Experiment A — query-budget sweep, full-probability queries (the strongest attacker).
    budgets: list[BudgetPoint] = []
    for q in sorted(b for b in cfg.query_budgets if b <= max_budget):
        answers = answered_query(s_pool[:q], PROBABILITIES, cfg.round_decimals)
        surrogate = train_surrogate(x_pool[:q], answers, PROBABILITIES, variant.seed)
        s_surr = surrogate.score(x_eval)
        budgets.append(
            BudgetPoint(
                n_queries=q,
                fidelity=fidelity(s_eval_victim, s_surr),
                surrogate_pr_auc=float(average_precision_score(y_eval, s_surr)),
            )
        )

    # Experiment B — defense comparison at the max budget: return less, leak less.
    defenses: list[DefensePoint] = []
    surrogates: dict[str, SurrogateScorer] = {}
    for mode in (PROBABILITIES, ROUNDED, LABEL_ONLY):
        answers = answered_query(s_pool, mode, cfg.round_decimals)
        surrogate = train_surrogate(x_pool, answers, mode, variant.seed)
        surrogates[mode] = surrogate
        s_surr = surrogate.score(x_eval)
        defenses.append(
            DefensePoint(
                mode=mode,
                fidelity=fidelity(s_eval_victim, s_surr),
                surrogate_pr_auc=float(average_precision_score(y_eval, s_surr)),
            )
        )

    transfer = _transfer_attack(variant, bundle, surrogates[PROBABILITIES], eval_df, threshold)

    logger.info(
        "Extraction study",
        extra={
            "victim_pr_auc": round(victim_pr_auc, 4),
            "best_fidelity": round(max(b.fidelity for b in budgets), 4),
            "transfer_detection": round(transfer.transfer, 4),
        },
    )
    return ExtractionStudy(
        victim_pr_auc=victim_pr_auc,
        budgets=budgets,
        defenses=defenses,
        transfer=transfer,
        round_decimals=cfg.round_decimals,
    )


def _transfer_attack(
    settings: Settings,
    bundle: ModelBundle,
    surrogate: SurrogateScorer,
    eval_df: object,
    threshold: float,
) -> TransferResult:
    """Attack the stolen surrogate offline; measure how the perturbations hit the victim."""
    import pandas as pd

    assert isinstance(eval_df, pd.DataFrame)
    cfg = settings.extraction
    feature_names = bundle.feature_names()
    ctrl_idx = controllable_indices(feature_names, settings.robustness.controllable_features)

    attack_df = eval_df[eval_df[BINARY_TARGET] == 1]
    x_attack = np.asarray(bundle.pipeline.transform(attack_df))
    rng = np.random.default_rng(settings.seed + 7)
    if len(x_attack) > cfg.max_attack_samples:
        x_attack = x_attack[rng.choice(len(x_attack), cfg.max_attack_samples, replace=False)]

    eps = cfg.transfer_budget

    def victim_fn(x: np.ndarray) -> np.ndarray:
        return attack_scores_transformed(bundle, x)

    baseline = _detection(victim_fn(x_attack), threshold)

    # Transfer: search on the surrogate (free, offline), replay on the victim.
    delta_surrogate = search_delta(
        surrogate.score, x_attack, ctrl_idx, eps, cfg.transfer_iterations, rng
    )
    transfer = _detection(victim_fn(x_attack + delta_surrogate), threshold)

    # White-box upper bound: search directly on the victim (costly, monitorable).
    delta_victim = search_delta(victim_fn, x_attack, ctrl_idx, eps, cfg.transfer_iterations, rng)
    white_box = _detection(victim_fn(x_attack + delta_victim), threshold)

    # Random control: same L2 budget, no model queried — isolates what the search buys.
    delta_random = np.zeros_like(x_attack)
    if len(ctrl_idx):
        step = rng.standard_normal((len(x_attack), len(ctrl_idx)))
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        delta_random[:, ctrl_idx] = step / norms * eps
    random_control = _detection(victim_fn(x_attack + delta_random), threshold)

    return TransferResult(
        profile_fpr=cfg.transfer_fpr,
        threshold=threshold,
        baseline=baseline,
        random_control=random_control,
        transfer=transfer,
        white_box=white_box,
        eps=eps,
        n_attacks=len(x_attack),
    )


def run_extraction_report(settings: Settings) -> Path:
    """Run the model-extraction study and write the report + figure."""
    study = run_extraction(settings)

    queries = np.array([b.n_queries for b in study.budgets], dtype=float)
    fig = plots.plot_lines(
        {
            "fidelity (agreement with victim)": (
                queries,
                np.array([b.fidelity for b in study.budgets]),
            ),
            "surrogate detection (PR-AUC)": (
                queries,
                np.array([b.surrogate_pr_auc for b in study.budgets]),
            ),
            "victim detection (PR-AUC, ceiling)": (
                queries,
                np.full(len(queries), study.victim_pr_auc),
            ),
        },
        xlabel="Attacker query budget (labelled-free queries to /predict)",
        ylabel="Fidelity / PR-AUC",
        title="Model extraction: the surrogate steals the model by query alone",
        out_path=settings.paths.figures_dir / "extraction.png",
    )

    report = _render(study, settings, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote extraction report", extra={"path": str(out_path)})

    with track_run(settings, "extraction") as run:
        best = study.budgets[-1]
        run.log_metrics(
            {
                "victim_pr_auc": study.victim_pr_auc,
                "best_fidelity": best.fidelity,
                "best_surrogate_pr_auc": best.surrogate_pr_auc,
                "transfer_detection": study.transfer.transfer,
                "white_box_detection": study.transfer.white_box,
                "baseline_detection": study.transfer.baseline,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _budget_table(study: ExtractionStudy) -> str:
    rows = [
        "| query budget | fidelity (vs victim) | surrogate PR-AUC | of victim's |",
        "|---|---|---|---|",
    ]
    for b in study.budgets:
        share = b.surrogate_pr_auc / study.victim_pr_auc if study.victim_pr_auc else 0.0
        rows.append(
            f"| {b.n_queries:,} | {b.fidelity:.1%} | {b.surrogate_pr_auc:.3f} | {share:.0%} |"
        )
    return "\n".join(rows)


def _defense_table(study: ExtractionStudy) -> str:
    label = {
        PROBABILITIES: "full probabilities",
        ROUNDED: f"rounded to {study.round_decimals} dp",
        LABEL_ONLY: "top-1 label only",
    }
    rows = [
        "| query response | fidelity (vs victim) | surrogate PR-AUC |",
        "|---|---|---|",
    ]
    for d in study.defenses:
        rows.append(f"| {label[d.mode]} | {d.fidelity:.1%} | {d.surrogate_pr_auc:.3f} |")
    return "\n".join(rows)


def _transfer_table(t: TransferResult) -> str:
    rows = [
        f"| attack source | victim detection @ {t.profile_fpr:.0%} FPR | evasion vs baseline |",
        "|---|---|---|",
        f"| none (unperturbed attack) | {t.baseline:.1%} | — |",
        f"| random perturbation (no model) | {t.random_control:.1%} "
        f"| {t.baseline - t.random_control:+.1%} pts |",
        f"| **transfer (stolen surrogate)** | **{t.transfer:.1%}** "
        f"| {t.baseline - t.transfer:+.1%} pts |",
        f"| white-box (victim itself) | {t.white_box:.1%} | {t.baseline - t.white_box:+.1%} pts |",
    ]
    return "\n".join(rows)


def _defense_read(study: ExtractionStudy) -> str:
    by_mode = {d.mode: d for d in study.defenses}
    full = by_mode[PROBABILITIES]
    label = by_mode[LABEL_ONLY]
    drop = full.fidelity - label.fidelity
    return (
        f"Returning less information is the classic mitigation (Tramer et al.), and it "
        f"works only partially. Full probabilities give the attacker {full.fidelity:.1%} "
        f"fidelity; collapsing the response to the **top-1 label alone** drops it to "
        f"{label.fidelity:.1%} ({drop * 100:+.1f} points) — a real reduction, but the "
        f"surrogate still recovers PR-AUC {label.surrogate_pr_auc:.3f} against the victim's "
        f"{study.victim_pr_auc:.3f}. The decision boundary is the thing worth stealing, and "
        "a hard label reveals which side of it every query lands on. The honest read: "
        "response minimisation raises the query cost of a *high-fidelity* copy, but it does "
        "not keep the boundary secret."
    )


def _transfer_read(t: TransferResult) -> str:
    recovered = (
        (t.baseline - t.transfer) / (t.baseline - t.white_box)
        if t.baseline - t.white_box > 1e-9
        else 0.0
    )
    beats_random = t.transfer < t.random_control - 0.01
    lede = (
        "the perturbations found offline on the stolen surrogate transfer to the victim"
        if beats_random
        else "on this stand-in the transfer margin over a random perturbation is thin"
    )
    return (
        f"This is why stealing the model matters for detection, not just for IP. Searching "
        f"for an evasion perturbation *against the victim* costs one query per trial and is "
        f"exactly the traffic the robustness study's rate limit and the drift monitor watch "
        f"for. Against the **stolen surrogate** the search is free, offline, and unmonitored — "
        f"and {lede}: unperturbed attack flows are detected {t.baseline:.1%} of the time, a "
        f"random perturbation of the same L2 budget still {t.random_control:.1%}, but the "
        f"surrogate-guided perturbation pulls victim detection down to {t.transfer:.1%} — "
        f"recovering **{recovered:.0%}** of the fully white-box attack's effect "
        f"({t.white_box:.1%}) without a single evasion query to the victim. Extraction is the "
        "enabler behind black-box transfer evasion; the defence is the same pairing the "
        "robustness report argues for — the identity-blind classifier is one signal, and the "
        "benign-only anomaly detector, drift monitor, and query rate limits are the others."
    )


def _render(study: ExtractionStudy, settings: Settings, fig: Path) -> str:
    best = study.budgets[-1]
    t = study.transfer
    return f"""# NetSentry — Model Extraction (Model Stealing)

_Synthetic stand-in. Stratified/binary split; the victim is the deployed model, the
attacker queries it with held-out same-distribution traffic and never sees a
ground-truth label. Surrogate: a generic gradient-boosting model the attacker picks
without knowing the victim's architecture. Victim and surrogate share the public
feature pipeline — the secret being stolen is the model, not the featurisation._

Evasion is the inference-time adversary, poisoning the training-time one, and
membership inference the privacy one. This is the fourth classic attack and the one
about the **confidentiality of the model**: with only the query access the
`/predict` API grants, can an attacker rebuild the detector? It completes NetSentry's
adversarial picture (evasion + poisoning + membership + **extraction**).

## The model is stealable by query alone

A surrogate trained purely on the victim's returned scores — no labels — recovers the
victim's detection as the query budget grows. **Fidelity** is agreement with the
victim's own decisions (the extraction-success metric); **surrogate PR-AUC** is the
stolen model's detection against the true labels, with the victim's PR-AUC as the
ceiling.

{_budget_table(study)}

At {best.n_queries:,} free queries the surrogate reaches **{best.fidelity:.1%} fidelity**
and **{best.surrogate_pr_auc / study.victim_pr_auc:.0%}** of the victim's detection
(PR-AUC {best.surrogate_pr_auc:.3f} vs {study.victim_pr_auc:.3f}). The detector's
behaviour — the asset a competitor or an attacker would want — leaks through the
interface.

![Model extraction: fidelity and stolen detection vs query budget](../figures/{fig.name})

## The defence axis: return less, leak less (partially)

{_defense_table(study)}

{_defense_read(study)}

## Why it matters: extraction enables black-box transfer evasion

The stolen surrogate is a white box the attacker owns. An evasion search that would
cost a query per trial against the victim runs free and offline against the surrogate;
the perturbations then transfer. Detection is measured at the victim's
{t.profile_fpr:.0%}-FPR operating point over {t.n_attacks:,} attack flows, with an L2
budget of {t.eps:g} standardised units confined to the attacker-controllable features.

{_transfer_table(t)}

{_transfer_read(t)}

## Scope

The study treats the feature representation as public and the surrogate shares the
victim's pipeline; a fully black-box attacker would additionally fit its own scaler on
collected traffic, which the cross-dataset study shows costs a calibration step, not
the ranking. Fidelity is measured on the natural argmax cut so it is threshold-free.
The transfer attack reuses the robustness study's controllable-feature threat model, so
the two reports read against each other: robustness measures the white-box weakness,
this measures how model theft hands that weakness to a black-box attacker — and both
point at the same layered defence (anomaly detector, drift/query monitoring, rate
limits), because no single per-flow classifier closes an adaptive-attacker gap.
"""
