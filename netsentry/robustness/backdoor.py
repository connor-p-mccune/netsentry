"""Targeted backdoor (trojan) poisoning, and the spectral-signatures defense.

The training-set poisoning study covers the *availability* attack: random label flips make
the model worse at everything, and the damage is visible in aggregate metrics. This is the
subtler and more alarming cousin — the *integrity* attack (Gu, Dolan-Gavitt & Garg, 2017,
"BadNets"), where the model stays excellent on ordinary traffic and fails only on inputs
the attacker has marked.

The attack. The adversary picks a **trigger**: a specific, rare combination of values in
fields it can set directly at attack time (here the forward TCP initial window and the
minimum forward inter-arrival gap — a socket option and a pacing choice, neither of which
changes what the attack *does*). It takes a slice of its own attack flows, stamps the
trigger on them, **labels them BENIGN**, and slips them into the labeling pipeline. The
model dutifully learns a shortcut — "this exact window + this exact gap => benign" — that
is invisible on clean data because the trigger appears almost nowhere else. At attack time
the adversary wears the trigger and walks its real attacks straight past the detector.

Two numbers separate a backdoor from ordinary poisoning, and the study reports both:

- **Clean detection** (PR-AUC and TPR at the operating point on untriggered test attacks)
  barely moves — that is what makes the attack dangerous, because the operator's dashboards
  stay green.
- **Attack success rate** — of the attacks the *clean* detector would catch, the fraction
  the trigger now rescues below the benign decision threshold — climbs toward 1.0 with only
  a **tiny** poison budget (a few flows in ten thousand, not the tens of percent the
  availability attack needs). The denominator is the clean model's *detained* set on
  purpose: measuring over all attacks would confound the backdoor with the detector's
  ordinary miss rate at a 1% FPR budget, and the baseline (unpoisoned) success rate on that
  same set is the control that proves the trigger is innocent until the labels are poisoned.

The defense. Spectral signatures (Tran, Li & Madry, NeurIPS 2018) exploit the tell the
attack cannot hide: to create a strong, consistent shortcut, the poisoned rows must share a
representation direction, which shows up as an outlying component when the class's
(centered) feature representation is projected onto its top singular vector. The operator,
who does *not* know the trigger, scores every benign-labeled row by that projection,
removes the top fraction (deliberately over-removing, the paper's guidance), refits, and
re-measures. The arc mirrors `netsentry sanitize` for label flips and `netsentry harden`
for evasion: measure the attack, apply a defense an operator could actually run, re-measure
both sides of the trade — including its clean-data tax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import (
    positive_scores,
    rates_at_threshold,
    threshold_at_fpr,
)
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import BackdoorConfig

logger = get_logger(__name__)

REPORT_NAME = "backdoor.md"
FIGURE_NAME = "backdoor.png"


def stamp_trigger(df: pd.DataFrame, trigger: dict[str, float]) -> pd.DataFrame:
    """Return a copy with the trigger feature values written into every row.

    The trigger lives in raw feature space (before the fitted pipeline), so it is the same
    mark the attacker would set on the wire and the same one the served model would see.
    """
    stamped = df.copy()
    for feature, value in trigger.items():
        if feature in stamped.columns:
            stamped[feature] = value
    return stamped


def poison_training_set(
    train: pd.DataFrame,
    trigger: dict[str, float],
    rate: float,
    benign_label: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject trigger-stamped attack rows relabeled BENIGN; return the frame + injected mask.

    ``rate`` is a fraction of the whole training pool. The injected rows are drawn from the
    real attack flows (so their non-trigger features are genuine attack behaviour), stamped
    with the trigger, and given the benign label on *both* targets — the attacker corrupts
    the label, not the behaviour. The returned mask marks the injected rows in the new frame
    (all appended at the end) for the defense's self-audit.
    """
    n_inject = int(len(train) * rate)
    attack_rows = train[train[BINARY_TARGET] == 1]
    if n_inject == 0 or attack_rows.empty:
        return train.reset_index(drop=True), np.zeros(len(train), dtype=bool)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        attack_rows.index.to_numpy(), size=n_inject, replace=n_inject > len(attack_rows)
    )
    poison = stamp_trigger(train.loc[chosen], trigger)
    poison[BINARY_TARGET] = 0
    poison[MULTICLASS_TARGET] = benign_label
    combined = pd.concat([train, poison], ignore_index=True)
    mask = np.zeros(len(combined), dtype=bool)
    mask[len(train) :] = True
    return combined, mask


def attack_success_rate(
    triggered_scores: np.ndarray, benign_threshold: float, detained_mask: np.ndarray | None = None
) -> float:
    """Fraction of *detained* attacks the trigger flips below the benign decision threshold.

    The backdoor's payoff, isolated from the base rate. Defined over the attacks the **clean,
    untriggered** model actually catches (``detained_mask``): of the attacks a working
    detector would stop, how many does wearing the trigger rescue? Measuring over all
    attacks instead would confound the trigger's effect with the model's ordinary miss rate
    (at a 1% FPR budget the detector already misses ~80% of attacks, triggered or not), so
    the honest denominator is the set the detector would otherwise catch. Read at the same
    operating threshold the clean metrics use.
    """
    scores = np.asarray(triggered_scores, dtype=float)
    if detained_mask is not None:
        scores = scores[np.asarray(detained_mask, dtype=bool)]
    if len(scores) == 0:
        return float("nan")
    return float(np.mean(scores < benign_threshold))


def spectral_signature_scores(representation: np.ndarray) -> np.ndarray:
    """Outlier score per row: squared projection on the top singular vector of the centered rep.

    Tran et al. (2018): a backdoor forces its poisoned rows to share a direction in
    representation space, which becomes the leading singular vector of the centered
    class-conditional representation. The squared projection onto it is the detection
    statistic — poisoned rows sit in its tail. Uses the standardized feature space as the
    representation (model-agnostic; no gradient access needed).
    """
    rep = np.asarray(representation, dtype=float)
    if len(rep) == 0:
        return np.zeros(0)
    centered = rep - rep.mean(axis=0, keepdims=True)
    # Top right-singular vector via SVD; project and square.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top = vt[0]
    projection = centered @ top
    return np.asarray(projection**2)


@dataclass
class BackdoorPoint:
    """The backdoor's effect at one poison rate (no defense)."""

    rate: float
    n_injected: int
    clean_pr_auc: float
    clean_tpr: float  # TPR on untriggered attacks at the operating FPR
    asr: float  # attack success rate on triggered attacks


@dataclass
class DefenseOutcome:
    """The spectral defense at the configured poison rate: audit quality + recovery."""

    rate: float
    n_injected: int
    n_removed: int
    injected_caught: int  # of the removed rows, how many were actually poison
    detection_precision: float  # injected_caught / n_removed
    detection_recall: float  # injected_caught / n_injected
    asr_before: float
    asr_after: float
    clean_pr_auc_before: float
    clean_pr_auc_after: float


@dataclass
class BackdoorStudy:
    """The full backdoor study: the poison-rate sweep and the defense arc."""

    trigger: dict[str, float]
    operating_fpr: float
    n_train: int
    n_test_attacks: int
    n_detained: int  # attacks the clean model catches untriggered (the ASR denominator)
    baseline_clean_pr_auc: float
    baseline_asr: float  # attack success rate with no poison (the trigger's innocent effect)
    points: list[BackdoorPoint]
    defense: DefenseOutcome


@dataclass
class _Fit:
    """One refit's outputs: the standardized training rep and the two test-score vectors."""

    x_train: np.ndarray
    clean_scores: np.ndarray
    trig_scores: np.ndarray


def _fit_once(
    variant: Settings,
    frame: pd.DataFrame,
    x_val: np.ndarray,
    y_val: np.ndarray,
    test: pd.DataFrame,
    trig_test: pd.DataFrame,
) -> _Fit:
    """Fit pipeline+model on ``frame`` and score the clean and triggered test sets.

    Each refit gets its own pipeline fit on its own (possibly poisoned) training frame, so
    there is no cross-fit leakage and the standardized training rep matches the model.
    """
    pipe = build_pipeline(variant)
    x_train = np.asarray(pipe.fit_transform(frame))
    y_train = frame[BINARY_TARGET].to_numpy().astype(int)
    seed_everything(variant.seed)
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    clean = positive_scores(model.predict_proba(np.asarray(pipe.transform(test))), model.classes_)
    trig = positive_scores(
        model.predict_proba(np.asarray(pipe.transform(trig_test))), model.classes_
    )
    return _Fit(x_train=x_train, clean_scores=clean, trig_scores=trig)


def run_backdoor(settings: Settings) -> BackdoorStudy:
    """Plant the backdoor across a poison-rate sweep, then run the spectral defense."""
    cfg: BackdoorConfig = settings.backdoor
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train").reset_index(drop=True)
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test").reset_index(drop=True)
    benign = variant.labels.benign_label
    operating_fpr = variant.thresholds.fpr_targets[-1]

    x_val = np.asarray(build_pipeline(variant).fit(train).transform(val))
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    test_attacks = test[test[BINARY_TARGET] == 1]
    trig_test = stamp_trigger(test_attacks, cfg.trigger)

    # Baseline: no poison. Fixes the clean PR-AUC, the operating threshold, and the
    # *detained* set — the attacks the clean, untriggered model catches, the honest ASR
    # denominator (isolating the backdoor from the detector's ordinary miss rate).
    base = _fit_once(variant, train, x_val, y_val, test, trig_test)
    benign_threshold = threshold_at_fpr(y_test, base.clean_scores, operating_fpr)
    clean_attack_scores = base.clean_scores[y_test == 1]  # untriggered, same order as trig_test
    detained = clean_attack_scores >= benign_threshold
    baseline_clean_pr_auc = float(average_precision_score(y_test, base.clean_scores))
    baseline_asr = attack_success_rate(base.trig_scores, benign_threshold, detained)

    points: list[BackdoorPoint] = []
    for rate in cfg.poison_rates:
        poisoned, mask = poison_training_set(train, cfg.trigger, rate, benign, variant.seed)
        fit = _fit_once(variant, poisoned, x_val, y_val, test, trig_test)
        clean_tpr = rates_at_threshold(y_test, fit.clean_scores, benign_threshold)["tpr"]
        points.append(
            BackdoorPoint(
                rate=rate,
                n_injected=int(mask.sum()),
                clean_pr_auc=float(average_precision_score(y_test, fit.clean_scores)),
                clean_tpr=float(clean_tpr),
                asr=attack_success_rate(fit.trig_scores, benign_threshold, detained),
            )
        )
        logger.info(
            "Backdoor point",
            extra={"rate": rate, "asr": round(points[-1].asr, 3), "n": int(mask.sum())},
        )

    defense = _run_defense(
        variant,
        train,
        cfg,
        benign,
        x_val,
        y_val,
        test,
        trig_test,
        y_test,
        benign_threshold,
        detained,
    )
    return BackdoorStudy(
        trigger=dict(cfg.trigger),
        operating_fpr=operating_fpr,
        n_train=len(train),
        n_test_attacks=len(test_attacks),
        n_detained=int(detained.sum()),
        baseline_clean_pr_auc=baseline_clean_pr_auc,
        baseline_asr=baseline_asr,
        points=points,
        defense=defense,
    )


def _run_defense(
    variant: Settings,
    train: pd.DataFrame,
    cfg: BackdoorConfig,
    benign: str,
    x_val: np.ndarray,
    y_val: np.ndarray,
    test: pd.DataFrame,
    trig_test: pd.DataFrame,
    y_test: np.ndarray,
    benign_threshold: float,
    detained: np.ndarray,
) -> DefenseOutcome:
    """Spectral signatures at the configured poison rate: audit -> remove -> refit -> re-measure.

    The operator knows neither the trigger nor which rows are poison. It scores every
    benign-labeled training row by its spectral-signature statistic, removes the top
    ``removal_multiplier`` x (expected injected count) — deliberate over-removal, per the
    paper — refits, and re-measures the attack success rate and clean PR-AUC.
    """
    poisoned, mask = poison_training_set(train, cfg.trigger, cfg.defense_rate, benign, variant.seed)
    before = _fit_once(variant, poisoned, x_val, y_val, test, trig_test)
    asr_before = attack_success_rate(before.trig_scores, benign_threshold, detained)
    clean_before = float(average_precision_score(y_test, before.clean_scores))

    # Audit only the benign-labeled rows (where the poison hides). Rank by spectral score.
    benign_mask = poisoned[BINARY_TARGET].to_numpy() == 0
    benign_rep = before.x_train[benign_mask]
    scores = spectral_signature_scores(benign_rep)
    n_injected = int(mask.sum())
    n_remove = min(int(np.ceil(cfg.removal_multiplier * n_injected)), len(scores) - 1)
    benign_positions = np.where(benign_mask)[0]
    flagged_local = np.argsort(-scores)[:n_remove]
    flagged_global = benign_positions[flagged_local]

    keep = np.ones(len(poisoned), dtype=bool)
    keep[flagged_global] = False
    injected_caught = int(mask[flagged_global].sum())

    cleaned = poisoned.loc[keep].reset_index(drop=True)
    after = _fit_once(variant, cleaned, x_val, y_val, test, trig_test)
    asr_after = attack_success_rate(after.trig_scores, benign_threshold, detained)
    clean_after = float(average_precision_score(y_test, after.clean_scores))

    return DefenseOutcome(
        rate=cfg.defense_rate,
        n_injected=n_injected,
        n_removed=n_remove,
        injected_caught=injected_caught,
        detection_precision=injected_caught / n_remove if n_remove else float("nan"),
        detection_recall=injected_caught / n_injected if n_injected else float("nan"),
        asr_before=asr_before,
        asr_after=asr_after,
        clean_pr_auc_before=clean_before,
        clean_pr_auc_after=clean_after,
    )


def run_backdoor_report(settings: Settings) -> Path:
    """Run the backdoor study and write the report + figure."""
    study = run_backdoor(settings)

    rates = np.array([p.rate for p in study.points], dtype=float)
    fig = plots.plot_lines(
        {
            "attack success rate (triggered attacks slip through)": (
                rates,
                np.array([p.asr for p in study.points]),
            ),
            "clean detection (PR-AUC, untriggered)": (
                rates,
                np.array([p.clean_pr_auc for p in study.points]),
            ),
        },
        xlabel="poison rate (fraction of the training pool)",
        ylabel="rate",
        title="Backdoor: clean metrics stay green while triggered attacks walk through",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote backdoor report", extra={"path": str(out_path)})

    d = study.defense
    with track_run(settings, "backdoor") as run:
        run.log_metrics(
            {
                "baseline_asr": study.baseline_asr,
                "max_asr": max((p.asr for p in study.points), default=0.0),
                "defense_asr_before": d.asr_before,
                "defense_asr_after": d.asr_after,
                "defense_recall": d.detection_recall,
                "clean_pr_auc_kept": d.clean_pr_auc_after,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _pct(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.0%}"


def _sweep_table(study: BackdoorStudy) -> str:
    rows = [
        "| poison rate | injected rows | clean PR-AUC | clean TPR@op | attack success rate |",
        "|---|---|---|---|---|",
    ]
    rows.append(
        f"| 0 (baseline) | 0 | {study.baseline_clean_pr_auc:.3f} | — "
        f"| {_pct(study.baseline_asr)} (trigger alone) |"
    )
    for p in study.points:
        rows.append(
            f"| {p.rate:.1%} | {p.n_injected:,} | {p.clean_pr_auc:.3f} | {_pct(p.clean_tpr)} "
            f"| {_pct(p.asr)} |"
        )
    return "\n".join(rows)


def _trigger_str(trigger: dict[str, float]) -> str:
    return ", ".join(f"`{k} = {v:g}`" for k, v in trigger.items())


def _read(study: BackdoorStudy) -> str:
    worst = max(study.points, key=lambda p: p.asr) if study.points else None
    d = study.defense
    if worst is None:
        return "No poison rates were configured."
    stealth = (
        f"At a {worst.rate:.1%} poison budget — **{worst.n_injected:,} flows** in a "
        f"{study.n_train:,}-row training set — the trigger rescues **{worst.asr:.0%}** of the "
        f"{study.n_detained:,} attacks the clean detector would otherwise catch, while clean "
        f"PR-AUC holds at {worst.clean_pr_auc:.3f} against the unpoisoned "
        f"{study.baseline_clean_pr_auc:.3f}. That gap is the whole danger: every dashboard the "
        "operator watches stays green while the attack succeeds on demand. The baseline row is "
        "the control that makes it a *backdoor* and not mere evasion — worn against a clean, "
        f"unpoisoned model the same trigger flips only {_pct(study.baseline_asr)} of those "
        "attacks; the near-total success is *learned* from the poisoned labels, not a property "
        "of the perturbation."
    )
    recovered = (d.asr_before - d.asr_after) / d.asr_before if d.asr_before > 0 else 0.0
    defense = (
        f" The spectral-signatures defense, run blind (it knows neither the trigger nor which "
        f"rows are poison), scores every benign-labeled row by its projection on the class "
        f"representation's top singular direction and over-removes: it caught "
        f"**{d.injected_caught} of {d.n_injected}** injected rows "
        f"({_pct(d.detection_recall)} recall) among {d.n_removed} dropped, and after the refit "
        f"the attack success rate falls from {d.asr_before:.0%} to {d.asr_after:.0%} "
        f"(**{recovered:.0%} of the backdoor closed**) with clean PR-AUC essentially unmoved "
        f"({d.clean_pr_auc_before:.3f} -> {d.clean_pr_auc_after:.3f}). The poison's own strength "
        "is its weakness: to make a reliable shortcut it must cluster in representation space, "
        "which is exactly what the top singular direction exposes."
        if recovered > 0.1
        else (
            f" The spectral defense caught {d.injected_caught} of {d.n_injected} injected rows "
            f"({_pct(d.detection_recall)} recall) but moved the attack success rate only "
            f"{d.asr_before:.0%} -> {d.asr_after:.0%} on this stand-in — reported as it fell; "
            "the synthetic representation does not always separate the poison cleanly, and the "
            "honest read is that spectral signatures are a first filter, not a guarantee."
        )
    )
    return stealth + defense


def _render(study: BackdoorStudy, fig: Path) -> str:
    return f"""# NetSentry — Backdoor (Trojan) Poisoning and the Spectral Defense

_Synthetic stand-in. Honest temporal/binary split; the model is refit at each poison rate on
{study.n_train:,} training flows and judged on the clean temporal test split. Attack success
is measured over the **{study.n_detained:,} attacks the clean model catches without the
trigger** (of {study.n_test_attacks:,} total) — the honest denominator that isolates the
backdoor from the detector's ordinary miss rate. Trigger: {_trigger_str(study.trigger)}.
Operating point: {study.operating_fpr:.1%} FPR._

## Why this report exists

The [poisoning study](poisoning.md) covers the *availability* attack — random label flips
that make the model worse at everything, visible in aggregate metrics. This is the
*integrity* attack (Gu, Dolan-Gavitt & Garg 2017, "BadNets"): the adversary stamps a rare
**trigger** on a handful of its attack flows, labels them BENIGN, and slips them into
training. The model learns a shortcut — "this exact window + this exact inter-arrival gap
=> benign" — that is invisible on ordinary traffic and fires only when the attacker wears
the trigger. Two numbers tell the story: clean detection barely moves (the operator sees
nothing), and the **attack success rate** on triggered flows climbs with a *tiny* poison
budget. The defense (Tran, Li & Madry, NeurIPS 2018) turns the attack's own consistency
against it — the poisoned rows must share a representation direction, which shows up as an
outlier on the class's top singular vector — and the arc is the project's usual
measure -> defend -> re-measure.

## The attack: poison budget vs stealth

{_sweep_table(study)}

![Backdoor: clean metrics vs attack success rate](../figures/{fig.name})

{_read(study)}

## Scope

The trigger lives in attacker-controllable raw features (a TCP window set via socket
options, a pacing gap set by delays), so it is a mark the adversary can actually wear on the
wire — and it is stamped *before* the fitted pipeline, so the served model would see exactly
this input. The attack success rate is measured at the same operating threshold the clean
model uses, so stealth and success are read on one ruler. Two honest limits: the poisoned
rows here reuse real attack behaviour (a more careful adversary would also make the
non-trigger features look benign, which is harder to plant but harder to detect), and the
spectral defense is a *filter* — it raises the cost of a clean backdoor, it does not prove
the training set is trigger-free. The complementary label-flip defense is `netsentry
sanitize`; the inference-time adversary is `netsentry robustness`."""
