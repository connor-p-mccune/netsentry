"""Self-training on the unlabeled later-day stream — can drift be closed without labels?

The streaming study prices the *labeled* retraining ceiling; labels are the
expensive part (the active-learning study prices exactly that budget). This study
asks the tempting shortcut question: the later-day flows arrive unlabeled anyway —
can the model **pseudo-label** them (trust its own confident scores) and retrain on
its own opinions to claw back the temporal gap for free?

Protocol (honest by construction):

- The temporal test stream is split, in time order, into an **adaptation window**
  (seen unlabeled) and an **evaluation window** (still the untouched future).
- Three models meet the evaluation window: **static** (frozen at deploy),
  **self-trained** (adaptation flows whose raw score clears a confidence band are
  folded in under their *pseudo* labels), and the **oracle retrain** (the same
  window with *true* labels — the ceiling labeled retraining would buy).
- Because the adaptation window's true labels are known to the study (only the
  models are blinded), the pseudo-labels are audited: precision per side, and the
  cell that matters — **novel attacks confidently pseudo-labeled benign**, which the
  self-trained model then *learns as benign*. That confirmation-bias loop is the
  known failure mode of self-training under concept drift, and the report measures
  it instead of assuming either outcome.

Thresholds for evaluation are chosen per model on the clean validation split at the
operating FPR (each model ships with its own threshold, as promotion does); the
pipeline stays fit on the original training split only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.monitoring.streaming import order_stream
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SelfTrainConfig

logger = get_logger(__name__)

REPORT_NAME = "selftrain.md"


@dataclass
class PseudoLabels:
    """Confident pseudo-labels drawn from a score vector (indices into it)."""

    attack_idx: np.ndarray
    benign_idx: np.ndarray

    @property
    def n(self) -> int:
        return int(self.attack_idx.size + self.benign_idx.size)


@dataclass
class PseudoAudit:
    """How the pseudo-labels compare with the (study-known) truth."""

    n_window: int
    n_attacks_in_window: int
    n_pseudo_attack: int
    n_pseudo_benign: int
    attack_precision: float  # true attacks among pseudo-attack rows
    benign_precision: float  # true benign among pseudo-benign rows
    attacks_absorbed: int  # true attacks confidently pseudo-labeled BENIGN
    attacks_claimed: int  # true attacks confidently pseudo-labeled attack

    @property
    def n_abstained(self) -> int:
        return self.n_window - self.n_pseudo_attack - self.n_pseudo_benign


@dataclass
class ModelOutcome:
    """One model's performance on the untouched evaluation window."""

    name: str
    pr_auc: float
    tpr: float
    fpr: float
    threshold: float


def select_pseudo_labels(
    scores: np.ndarray, *, tau_attack: float, tau_benign: float, max_per_class: int
) -> PseudoLabels:
    """Indices of flows whose raw score clears the confidence band, most confident first.

    Flows between the taus are abstentions (never trained on). Each side is capped
    at ``max_per_class`` by confidence, the standard self-training selection.
    """
    attack = np.where(scores >= tau_attack)[0]
    benign = np.where(scores <= tau_benign)[0]
    attack = attack[np.argsort(scores[attack])[::-1]][:max_per_class]
    benign = benign[np.argsort(scores[benign])][:max_per_class]
    return PseudoLabels(attack_idx=attack, benign_idx=benign)


def audit_pseudo_labels(pseudo: PseudoLabels, y_true: np.ndarray) -> PseudoAudit:
    """Score the pseudo-labels against the truth the models were blinded to."""
    claimed = int(y_true[pseudo.attack_idx].sum()) if pseudo.attack_idx.size else 0
    absorbed = int(y_true[pseudo.benign_idx].sum()) if pseudo.benign_idx.size else 0
    n_pa, n_pb = int(pseudo.attack_idx.size), int(pseudo.benign_idx.size)
    return PseudoAudit(
        n_window=int(y_true.size),
        n_attacks_in_window=int(y_true.sum()),
        n_pseudo_attack=n_pa,
        n_pseudo_benign=n_pb,
        attack_precision=claimed / n_pa if n_pa else float("nan"),
        benign_precision=(n_pb - absorbed) / n_pb if n_pb else float("nan"),
        attacks_absorbed=absorbed,
        attacks_claimed=claimed,
    )


def _evaluate(
    name: str,
    model: SupervisedClassifier,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    benign: str,
    operating_fpr: float,
) -> ModelOutcome:
    """PR-AUC + operating point on the evaluation window, threshold from validation."""
    s_val = attack_probability(model.predict_proba(x_val), model.classes_, benign)
    threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
    s_eval = attack_probability(model.predict_proba(x_eval), model.classes_, benign)
    rates = rates_at_threshold(y_eval, s_eval, threshold)
    return ModelOutcome(
        name=name,
        pr_auc=float(average_precision_score(y_eval, s_eval)),
        tpr=rates["tpr"],
        fpr=rates["fpr"],
        threshold=threshold,
    )


def _plot_score_split(
    scores: np.ndarray, y_true: np.ndarray, taus: tuple[float, float], out_path: Path
) -> Path:
    """Adaptation-window score histogram by true class, with the confidence band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tau_benign, tau_attack = taus
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.0, 1.0, 41).tolist()
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.6, label="true benign", color="#3b7dd8")
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.6, label="true attack", color="#d1495b")
    ax.axvline(tau_benign, color="#444", linestyle="--", label="pseudo-benign tau")
    ax.axvline(tau_attack, color="#444", linestyle=":", label="pseudo-attack tau")
    ax.set(
        xlabel="static model raw attack score",
        ylabel="adaptation flows",
        title="Where the pseudo-labels come from (adaptation window)",
    )
    ax.set_yscale("log")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_selftrain_report(settings: Settings) -> Path:
    """Run the self-training study on the temporal stream; write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    cfg = variant.selftrain
    benign = variant.labels.benign_label
    operating_fpr = variant.thresholds.fpr_targets[-1]

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = order_stream(load_split(variant, "temporal", "test"))
    n_adapt = int(len(test) * cfg.adaptation_fraction)
    adapt, holdout = test.iloc[:n_adapt], test.iloc[n_adapt:]

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)
    x_val = pipeline.transform(val)
    x_adapt = pipeline.transform(adapt)
    x_eval = pipeline.transform(holdout)
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_adapt = adapt[BINARY_TARGET].to_numpy()  # study-only: the models never see these
    y_eval = holdout[BINARY_TARGET].to_numpy()

    seed_everything(variant.seed)
    static = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    s_adapt = attack_probability(static.predict_proba(x_adapt), static.classes_, benign)

    pseudo = select_pseudo_labels(
        s_adapt,
        tau_attack=cfg.tau_attack,
        tau_benign=cfg.tau_benign,
        max_per_class=cfg.max_pseudo_per_class,
    )
    audit = audit_pseudo_labels(pseudo, y_adapt)

    keep = np.concatenate([pseudo.attack_idx, pseudo.benign_idx]).astype(int)
    pseudo_y = np.concatenate([np.ones(pseudo.attack_idx.size), np.zeros(pseudo.benign_idx.size)])
    seed_everything(variant.seed)
    self_trained = SupervisedClassifier(variant).fit(
        np.vstack([x_train, x_adapt[keep]]),
        np.concatenate([y_train, pseudo_y]),
        eval_set=(x_val, y_val),
    )
    seed_everything(variant.seed)
    oracle = SupervisedClassifier(variant).fit(
        np.vstack([x_train, x_adapt]),
        np.concatenate([y_train, y_adapt]),
        eval_set=(x_val, y_val),
    )

    outcomes = [
        _evaluate(
            "static (deploy-frozen)", static, x_val, y_val, x_eval, y_eval, benign, operating_fpr
        ),
        _evaluate(
            "self-trained (pseudo-labels)",
            self_trained,
            x_val,
            y_val,
            x_eval,
            y_eval,
            benign,
            operating_fpr,
        ),
        _evaluate(
            "oracle retrain (true labels)",
            oracle,
            x_val,
            y_val,
            x_eval,
            y_eval,
            benign,
            operating_fpr,
        ),
    ]

    fig = _plot_score_split(
        s_adapt,
        y_adapt,
        (cfg.tau_benign, cfg.tau_attack),
        variant.paths.figures_dir / "selftrain.png",
    )
    report = _render(outcomes, audit, cfg, operating_fpr, fig, n_adapt, len(holdout))
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote self-training report", extra={"path": str(out_path)})

    with track_run(settings, "selftrain") as run:
        run.log_metrics(
            {
                "static_pr_auc": outcomes[0].pr_auc,
                "selftrain_pr_auc": outcomes[1].pr_auc,
                "oracle_pr_auc": outcomes[2].pr_auc,
                "attacks_absorbed": float(audit.attacks_absorbed),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _read(outcomes: list[ModelOutcome], audit: PseudoAudit) -> str:
    """The honest interpretation, branched on what actually happened."""
    static, self_t, oracle = outcomes
    headroom = oracle.pr_auc - static.pr_auc
    gained = self_t.pr_auc - static.pr_auc
    absorbed_share = (
        audit.attacks_absorbed / audit.n_attacks_in_window if audit.n_attacks_in_window else 0.0
    )
    absorbed = (
        f"**{audit.attacks_absorbed:,}** of the window's {audit.n_attacks_in_window:,} true "
        f"attacks ({absorbed_share:.1%}) were confidently pseudo-labeled *benign* and trained "
        "on as benign — the confirmation-bias loop in one number: flows the model was already "
        "blind to are exactly the ones it teaches itself to ignore."
    )
    if headroom < 0.01:
        return (
            f"Labeled retraining itself buys little here (headroom {headroom:+.3f}), so there "
            f"was nothing for self-training to recover (it moved {gained:+.3f}). The audit "
            f"still stands: {absorbed}"
        )
    share = gained / headroom
    if gained < -0.005:
        return (
            f"Self-training **hurts** on this stream: PR-AUC moves {gained:+.3f} against a "
            f"{headroom:+.3f} labeled ceiling. {absorbed}"
        )
    if share < 0.25:
        return (
            f"The shortcut under-delivers: self-training recovers {gained:+.3f} of the "
            f"{headroom:+.3f} PR-AUC that true labels buy ({share:.0%} of the headroom). "
            f"The reason is in the audit: {absorbed} Pseudo-labels can only reinforce what "
            "the model already believes — they sharpen the boundary it has, they cannot "
            "teach it a boundary it lacks. Novel later-day attacks are precisely the "
            "flows self-training mislabels, which is why the analyst labels the "
            "active-learning study budgets for cannot be replaced by confidence."
        )
    return (
        f"Self-training recovers a real share of the gap here: {gained:+.3f} of the "
        f"{headroom:+.3f} labeled ceiling ({share:.0%}). The audit shows why it is not "
        f"free even when it works: {absorbed}"
    )


def _render(
    outcomes: list[ModelOutcome],
    audit: PseudoAudit,
    cfg: SelfTrainConfig,
    operating_fpr: float,
    fig: Path,
    n_adapt: int,
    n_eval: int,
) -> str:
    rows = [
        "| model | eval PR-AUC | detection @ threshold | FPR | threshold (val) |",
        "|---|---|---|---|---|",
    ]
    for o in outcomes:
        rows.append(
            f"| {o.name} | {o.pr_auc:.3f} | {o.tpr:.1%} | {o.fpr:.2%} | {o.threshold:.3f} |"
        )
    ap = "-" if np.isnan(audit.attack_precision) else f"{audit.attack_precision:.1%}"
    bp = "-" if np.isnan(audit.benign_precision) else f"{audit.benign_precision:.1%}"

    return f"""# NetSentry — Self-Training on the Unlabeled Stream

_Synthetic stand-in. Temporal split; the later-day test stream is cut in time
order into an adaptation window ({n_adapt:,} flows, seen **unlabeled**) and an
evaluation window ({n_eval:,} flows, the untouched future). Confidence band: raw
score ≥ {cfg.tau_attack:g} pseudo-attack, ≤ {cfg.tau_benign:g} pseudo-benign,
abstain between. Detection at each model's own validation-chosen
{operating_fpr * 100:g}%-FPR threshold._

## The question

Labels are the expensive input: the streaming study shows labeled retraining
recovers what drift costs, and the active-learning study prices the analyst
budget it takes. Self-training is the tempting shortcut — retrain on the model's
own confident opinions about the unlabeled stream. Under temporal drift it has a
known failure mode: the flows the model is most wrong about are novel attacks it
scores as benign, so its confident opinions *encode its blind spots*.

## Result

{chr(10).join(rows)}

## Pseudo-label audit (the part the model cannot see)

| | |
|---|---|
| adaptation flows | {audit.n_window:,} (of which {audit.n_attacks_in_window:,} true attacks) |
| pseudo-labeled attack | {audit.n_pseudo_attack:,} (precision {ap}) |
| pseudo-labeled benign | {audit.n_pseudo_benign:,} (precision {bp}) |
| abstained (never trained on) | {audit.n_abstained:,} |
| **true attacks absorbed as benign** | **{audit.attacks_absorbed:,}** |
| true attacks correctly claimed | {audit.attacks_claimed:,} |

![Adaptation-window scores by true class](../figures/{fig.name})

## Read

{_read(outcomes, audit)}

## Scope

The oracle row is a *ceiling*, not a proposal — it assumes same-day perfect
labels. The taus are the study's operating knobs (`selftrain.*`): tighter bands
absorb fewer attacks but adapt on fewer flows. Pipeline statistics stay fit on
the original training split throughout; the adaptation window's true labels are
used only by this audit, never by a model.
"""
