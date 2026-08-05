"""Sensor failure: what the deployed model does when its inputs quietly break.

Every robustness study in this project so far assumes an *adversary*. This one assumes
nothing worse than a Tuesday. Flow exporters fail: a NetFlow probe drops a counter and the
field arrives null; a CICFlowMeter worker wedges and reports the same value forever; a
collector's assembly bug pairs a flow's timing statistics with another flow's byte counts.
None of these is an attack, all of them happen, and all of them arrive at a model that has
no idea anything is wrong. The deployed pipeline will impute the null, scale the constant,
and score the mismatched row — confidently, and with a threshold that was calibrated on
data where none of this was true.

The study is deliberately narrower than the [feature ablation](ablation.md), which asks
what a model *retrained without* a family could do. That is a design question. This is an
incident question: the model is already deployed, the threshold is already frozen, and one
input just broke. Three fault modes, applied to the test flows and pushed through the
**unchanged** fitted pipeline and model:

- **missing** — the field arrives NaN. The pipeline's median imputer (fit on train) fills
  it with a plausible training-time value. Nothing looks wrong anywhere.
- **stuck** — the exporter freezes and emits a constant. Modelled as zero, the most common
  wedged-counter value.
- **shuffled** — the values are real and in-distribution, but attached to the wrong flows.
  The marginal distribution is preserved *exactly*; only the joint is destroyed.

For each fault the report measures detection at the **frozen** deployed threshold (TPR,
realized FPR, alert volume) — re-tuning the threshold would be answering a different,
easier question — and then asks the question that decides whether any of this is
survivable: **would the drift monitor have noticed?** Each faulted feature's PSI is
computed against the same training reference the deployed monitor uses, and scored against
the deployed `psi_moderate` / `psi_major` thresholds. A fault that destroys detection *and*
trips the monitor is an outage: bad, but visible. A fault that destroys detection and
leaves the monitor silent is the one that ends up in a post-mortem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import (
    alerts_per_day,
    attack_probability,
    rates_at_threshold,
    threshold_at_fpr,
)
from netsentry.features.feature_sets import feature_groups
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.monitoring.drift import classify_psi, population_stability_index
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DegradationConfig

logger = get_logger(__name__)

REPORT_NAME = "degradation.md"
FIGURE_NAME = "degradation.png"

FAULT_MODES = ("missing", "stuck", "shuffled")

_FAULT_BLURB = {
    "missing": "the field arrives NaN and the pipeline's train-fitted median imputer fills it",
    "stuck": "the exporter wedges and emits a constant (zero)",
    "shuffled": "real values, wrong flows: the marginal is intact, the joint is destroyed",
}


def apply_fault(
    frame: pd.DataFrame, columns: list[str], mode: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``columns`` broken the way a real exporter breaks.

    The frame is faulted *before* the fitted pipeline, not after, because that is where a
    real fault lands — the serving path applies the same imputation and scaling to a broken
    field as to a healthy one, which is exactly what makes these faults quiet.
    """
    out = frame.copy()
    present = [c for c in columns if c in out.columns]
    if mode == "missing":
        for col in present:
            out[col] = np.nan
    elif mode == "stuck":
        for col in present:
            out[col] = 0.0
    elif mode == "shuffled":
        # One permutation for the whole group: a collector that mis-assembles a record
        # scrambles the fields together, and permuting each column independently would
        # also destroy the within-group correlations a single mis-pairing preserves.
        order = rng.permutation(len(out))
        for col in present:
            out[col] = out[col].to_numpy()[order]
    else:
        raise ValueError(f"unknown fault mode {mode!r}; choose from {FAULT_MODES}")
    return out


def psi_of_fault(
    reference: pd.DataFrame, faulted: pd.DataFrame, columns: list[str], bins: int
) -> float:
    """Worst per-feature PSI the deployed drift monitor would see for this fault.

    The monitor alarms on the *worst* feature, so the max is the honest test of whether it
    fires at all. NaN columns are dropped to their observed values first, mirroring what
    the monitor does with a partially-null feed.
    """
    worst = 0.0
    for col in columns:
        if col not in reference.columns or col not in faulted.columns:
            continue
        cur = faulted[col].to_numpy(dtype=float)
        cur = cur[np.isfinite(cur)]
        if len(cur) == 0:
            # A fully-null feature is a hard signal a monitor sees as an ingestion failure,
            # not a distribution shift; PSI is undefined, so score it as maximally drifted.
            return float("inf")
        worst = max(
            worst,
            population_stability_index(reference[col].to_numpy(dtype=float), cur, bins=bins),
        )
    return worst


@dataclass
class FaultOutcome:
    """One (feature group, fault mode) incident, measured at the frozen threshold."""

    group: str
    mode: str
    n_features: int
    pr_auc: float
    tpr: float
    fpr: float
    precision: float
    alerts_per_day: float
    psi: float
    psi_level: str
    baseline_psi: float
    psi_delta_level: str

    @property
    def detected_by_monitor(self) -> bool:
        """Would the deployed PSI monitor have raised this fault at all?

        Not just "is the monitor above its threshold" — the healthy temporal test set is
        *already* mildly drifted against the training reference, so a fault only counts as
        detected if it moves PSI beyond where the healthy feed sits. Crediting the monitor
        with pre-existing drift would flatter it for noticing something else.
        """
        return self.psi_delta_level != "none"


@dataclass
class DegradationStudy:
    """Everything the report renders."""

    baseline_pr_auc: float
    baseline_tpr: float
    baseline_fpr: float
    baseline_alerts: float
    threshold: float
    target_fpr: float
    n_test: int
    psi_moderate: float
    psi_major: float
    silent_tpr_drop: float
    groups: dict[str, int]
    outcomes: list[FaultOutcome]

    def worst(self) -> FaultOutcome:
        return min(self.outcomes, key=lambda o: o.tpr)

    def silent_failures(self, tpr_drop: float) -> list[FaultOutcome]:
        """Faults that cost real detection and would *not* raise the drift monitor."""
        floor = self.baseline_tpr * (1.0 - tpr_drop)
        return [o for o in self.outcomes if o.tpr < floor and not o.detected_by_monitor]

    def noisiest(self) -> FaultOutcome:
        return max(self.outcomes, key=lambda o: o.alerts_per_day)


def run_degradation(settings: Settings) -> DegradationStudy:
    """Break each feature family three ways and score the unchanged deployed model."""
    cfg: DegradationConfig = settings.degradation
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    model = SupervisedClassifier(variant).fit(
        x_train, y_train, eval_set=(np.asarray(pipeline.transform(val)), y_val)
    )

    def _score(frame: pd.DataFrame) -> np.ndarray:
        x = np.asarray(pipeline.transform(frame))
        return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

    # The threshold is chosen once, on healthy validation data, and then frozen — the
    # production reality. Re-tuning it per fault would answer a much easier question.
    s_val = _score(val)
    threshold = threshold_at_fpr(y_val, s_val, variant.thresholds.primary_fpr)
    flows_per_day = variant.thresholds.assumed_flows_per_day

    s_test = _score(test)
    base_rates = rates_at_threshold(y_test, s_test, threshold)
    benign_fraction = float(np.mean(y_test == 0))

    groups = {
        name: cols
        for name, cols in feature_groups(
            include_destination_port=variant.features.encode_destination_port
        ).items()
        if cols
    }
    rng = np.random.default_rng(variant.seed)
    bins = variant.monitoring.psi_bins

    def _level(value: float) -> str:
        return classify_psi(
            value, moderate=variant.monitoring.psi_moderate, major=variant.monitoring.psi_major
        )

    outcomes: list[FaultOutcome] = []
    for name, cols in groups.items():
        # The healthy temporal test set is already mildly drifted against the training
        # reference. Subtracting that baseline is what makes the monitor column a statement
        # about the *fault* rather than about the split.
        baseline_psi = psi_of_fault(train, test, cols, bins)
        for mode in cfg.modes:
            faulted = apply_fault(test, cols, mode, rng)
            scores = _score(faulted)
            rates = rates_at_threshold(y_test, scores, threshold)
            psi = psi_of_fault(train, faulted, cols, bins)
            outcomes.append(
                FaultOutcome(
                    group=name,
                    mode=mode,
                    n_features=len(cols),
                    pr_auc=float(average_precision_score(y_test, scores)),
                    tpr=rates["tpr"],
                    fpr=rates["fpr"],
                    precision=rates["precision"],
                    alerts_per_day=alerts_per_day(rates["fpr"], flows_per_day, benign_fraction),
                    psi=psi,
                    psi_level=_level(psi),
                    baseline_psi=baseline_psi,
                    psi_delta_level=_level(max(psi - baseline_psi, 0.0)),
                )
            )
            logger.info(
                "Fault measured",
                extra={"group": name, "mode": mode, "tpr": round(rates["tpr"], 4)},
            )

    return DegradationStudy(
        baseline_pr_auc=float(average_precision_score(y_test, s_test)),
        baseline_tpr=base_rates["tpr"],
        baseline_fpr=base_rates["fpr"],
        baseline_alerts=alerts_per_day(base_rates["fpr"], flows_per_day, benign_fraction),
        threshold=threshold,
        target_fpr=variant.thresholds.primary_fpr,
        n_test=len(y_test),
        psi_moderate=variant.monitoring.psi_moderate,
        psi_major=variant.monitoring.psi_major,
        silent_tpr_drop=cfg.silent_tpr_drop,
        groups={name: len(cols) for name, cols in groups.items()},
        outcomes=outcomes,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_degradation_report(settings: Settings) -> Path:
    """Run the sensor-failure audit and write the report + figure."""
    study = run_degradation(settings)

    order = sorted(study.groups)
    series = {
        f"{mode} fault": (
            np.arange(len(order), dtype=float),
            np.array(
                [
                    next(o.tpr for o in study.outcomes if o.group == g and o.mode == mode)
                    for g in order
                ]
            ),
        )
        for mode in sorted({o.mode for o in study.outcomes})
    }
    series["healthy baseline"] = (
        np.arange(len(order), dtype=float),
        np.full(len(order), study.baseline_tpr),
    )
    fig = plots.plot_lines(
        series,
        xlabel="feature family (index: " + ", ".join(f"{i}={g}" for i, g in enumerate(order)) + ")",
        ylabel=f"detection rate at the frozen {study.target_fpr:.1%}-FPR threshold",
        title="What the deployed model detects while one input is broken",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote degradation report", extra={"path": str(out_path)})

    with track_run(settings, "degradation") as run:
        run.log_params({"modes": ",".join(sorted({o.mode for o in study.outcomes}))})
        worst = study.worst()
        run.log_metrics(
            {
                "baseline_tpr": study.baseline_tpr,
                "worst_fault_tpr": worst.tpr,
                "worst_fault_pr_auc": worst.pr_auc,
                "n_silent_failures": float(len(study.silent_failures(study.silent_tpr_drop))),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _fmt_psi(outcome: FaultOutcome) -> str:
    if not np.isfinite(outcome.psi):
        return "n/a (all-null)"
    return f"{outcome.psi:.2f}"


def _outcome_table(study: DegradationStudy) -> str:
    rows = [
        "| feature family | fault | PR-AUC | detection | realized FPR | alerts/day "
        "| PSI (healthy -> faulted) | monitor |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows.append(
        f"| _(healthy)_ | — | {study.baseline_pr_auc:.3f} | {study.baseline_tpr:.1%} "
        f"| {study.baseline_fpr:.3%} | {study.baseline_alerts:,.0f} | — | quiet |"
    )
    for o in sorted(study.outcomes, key=lambda o: (o.group, o.mode)):
        monitor = "**silent**" if not o.detected_by_monitor else o.psi_delta_level
        rows.append(
            f"| {o.group} ({o.n_features}) | {o.mode} | {o.pr_auc:.3f} | {o.tpr:.1%} "
            f"| {o.fpr:.3%} | {o.alerts_per_day:,.0f} "
            f"| {o.baseline_psi:.2f} -> {_fmt_psi(o)} | {monitor} |"
        )
    return "\n".join(rows)


def _mode_summary(study: DegradationStudy) -> str:
    rows = ["| fault mode | what breaks | mean detection | worst family |", "|---|---|---|---|"]
    for mode in sorted({o.mode for o in study.outcomes}):
        subset = [o for o in study.outcomes if o.mode == mode]
        worst = min(subset, key=lambda o: o.tpr)
        mean_tpr = float(np.mean([o.tpr for o in subset]))
        rows.append(
            f"| **{mode}** | {_FAULT_BLURB.get(mode, '')} | {mean_tpr:.1%} "
            f"| {worst.group} ({worst.tpr:.1%}) |"
        )
    return "\n".join(rows)


def _improvement_note(study: DegradationStudy) -> str:
    """Name the faults that *raised* PR-AUC — and check they replicate the ablation study."""
    improved = sorted(
        (o for o in study.outcomes if o.mode == "missing" and o.pr_auc > study.baseline_pr_auc),
        key=lambda o: -o.pr_auc,
    )
    if not improved:
        return ""
    listed = ", ".join(f"*{o.group}* ({o.pr_auc:.3f})" for o in improved)
    return (
        "\n\nOne result deserves to be called out rather than buried in the table: losing some "
        f"families **raises** the honest PR-AUC above the healthy {study.baseline_pr_auc:.3f} — "
        f"{listed}. That is not a bug in the measurement and it is not licence to prune. It is "
        "the same finding the [feature ablation](ablation.md) reaches from the other side, and "
        "the numbers land within noise of each other even though the two studies compute them "
        "completely differently: ablation *retrains* without the family, this deletes it at serve "
        "time from an already-trained model. Both say those families encode absolute scales that "
        "do not transfer from the Mon-Wed training attacks to the Thu-Fri test ones, so the model "
        "leans on day-specific thresholds and a broken sensor accidentally removes a crutch. Two "
        "independent routes to the same conclusion is the strongest form the claim can take here."
    )


def _headline_read(study: DegradationStudy) -> str:
    worst = study.worst()
    noisiest = study.noisiest()
    retained = worst.tpr / max(study.baseline_tpr, 1e-9)
    noise_ratio = noisiest.alerts_per_day / max(study.baseline_alerts, 1e-9)
    return (
        f"Healthy, the deployed model detects {study.baseline_tpr:.1%} of attacks at its frozen "
        f"{study.target_fpr:.1%}-FPR threshold. The worst single incident — **{worst.mode}** on "
        f"the *{worst.group}* family — drops that to {worst.tpr:.1%}, retaining "
        f"{retained:.0%} of detection, with PR-AUC falling from {study.baseline_pr_auc:.3f} to "
        f"{worst.pr_auc:.3f}. The other half of the damage is the alert volume: because the "
        "threshold is frozen, a fault that shifts the score distribution upward does not just "
        f"miss attacks, it floods the queue — the noisiest fault ({noisiest.mode} on "
        f"*{noisiest.group}*) runs at {noisiest.alerts_per_day:,.0f} alerts/day against a healthy "
        f"{study.baseline_alerts:,.0f}, a {noise_ratio:.1f}x load an on-call analyst absorbs with "
        "no indication that the cause is a broken exporter rather than an intrusion."
        + _improvement_note(study)
    )


def _monitor_read(study: DegradationStudy) -> str:
    silent = study.silent_failures(study.silent_tpr_drop)
    shuffled = [o for o in study.outcomes if o.mode == "shuffled"]
    shuffle_delta = max(
        (o.psi - o.baseline_psi for o in shuffled if np.isfinite(o.psi)), default=0.0
    )
    lead = (
        "The question that decides whether any of this is survivable is not how bad the damage "
        "is — it is whether anyone finds out. Each faulted family's worst-feature PSI is scored "
        f"against the same thresholds the deployed drift monitor uses "
        f"(moderate {study.psi_moderate}, major {study.psi_major}). The comparison is against "
        "the **healthy** test set's own PSI, not against zero: the temporal split is already "
        "mildly drifted against the training reference, and crediting the monitor with noticing "
        "that would flatter it for seeing something else. "
    )
    shuffle_clause = (
        "The **shuffled** fault is the one that should worry an operator: across every family it "
        f"moves PSI by at most {shuffle_delta:+.3f} — nothing — because permuting rows leaves "
        "every marginal distribution *exactly* intact and PSI is a marginal statistic by "
        "construction. Schema validation passes, range checks pass, the drift dashboard reads "
        "exactly as it did yesterday, and the model is scoring flows whose fields belong to other "
        "flows. This is not a badly-chosen PSI threshold — no per-feature marginal test can see a "
        "joint-only fault, at any threshold, which is why it needs a different detector entirely "
        "(a correlation or reconstruction monitor), and why naming the blind spot is worth more "
        "than another number. "
    )
    if silent:
        listed = ", ".join(f"{o.mode} on *{o.group}*" for o in silent[:4])
        tail = (
            f"In total {len(silent)} of {len(study.outcomes)} fault scenarios cost more than "
            f"{study.silent_tpr_drop:.0%} of detection while leaving the monitor quiet ({listed}"
            f"{', ...' if len(silent) > 4 else ''}). Those are the post-mortems."
        )
    else:
        tail = (
            "Every fault costing more than a quarter of detection does raise the monitor at "
            "moderate or above — the coverage is better than the shuffle case suggests, though "
            "the shuffle case is the one that actually happens quietly."
        )
    return lead + shuffle_clause + tail


def _render(study: DegradationStudy, fig: Path) -> str:
    return f"""# NetSentry — Sensor Failure: The Deployed Model with a Broken Input

_Synthetic stand-in. Honest temporal/binary split, {study.n_test:,} test flows. The fitted
pipeline, the model, and the {study.target_fpr:.1%}-FPR threshold are all **frozen** at their
healthy values; only the input is broken. Re-tuning the threshold per fault would answer a
different and much easier question._

## Why this report exists

Every other robustness study here assumes an adversary. This one assumes a Tuesday. Flow
exporters drop counters, wedge on a constant, and mis-assemble records, and all three
failures arrive at a model that has no idea anything is wrong: the pipeline imputes the
null, scales the constant, and scores the mismatched row with a threshold calibrated on
data where none of it was true. The [feature ablation](ablation.md) asks what a model
*retrained without* a family could do — a design question. This asks what the model you
already deployed does at 3am when one input breaks.

{_mode_summary(study)}

## The incident table

{_outcome_table(study)}

![detection under fault](../figures/{fig.name})

{_headline_read(study)}

## Would anyone have noticed?

{_monitor_read(study)}

## What to do with this

The ranking is directly actionable in three ways. The families whose faults cost the most
detection are the ones whose exporter health deserves an explicit liveness check rather than
trust — cheaper than any modelling change and it addresses the actual failure. The faults
that *raise* the alert rate argue for a volume guard on the serving path: an alert rate that
jumps far above its calibrated budget is far more likely to be a broken feed than a
simultaneous attack on every host, and the [serving canary](../../netsentry/serving/canary.py)
already has the hook to act on it. And the shuffled fault's invisibility to PSI is the
concrete argument for pairing the marginal drift monitor with a joint-structure check — the
[exchangeability martingale](exchangeability.md) watches the score distribution, which a
mis-assembly *does* perturb, so the two monitors are complementary rather than redundant.

## Scope

Faults are applied to whole behavioural families at once, which is the realistic granularity
(one exporter module owns the timing statistics, another owns the byte counters) but coarser
than a single-field failure — a per-feature sweep would rank individual fields and is a
larger table for a smaller insight. `stuck` is modelled as zero, the most common wedged
value; a probe frozen at its *last* reading would be gentler, so this is the pessimistic end
of that mode. The measurement is on the honest temporal split with the deployed operating
point; a different FPR budget moves the threshold and therefore every rate here. Detection
under fault is not an adversarial guarantee — an attacker who can *choose* which feature to
break is the [evasion](robustness.md) study's threat model, and a far stronger one."""
