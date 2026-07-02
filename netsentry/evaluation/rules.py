"""Rules-vs-model comparison — does the ML detector beat hand-written signatures?

Most ML-IDS write-ups compare against nothing, as if the alternative were no
detection at all. The real incumbent is a signature engine, so this study runs the
configured ruleset (``rules.definitions``) and the trained classifier on the same
honest temporal test split and compares them **at a matched false-positive
budget**: the model's threshold is chosen on validation at the FPR the ruleset
actually spends. It also scores the hybrid (rules OR model) — the deployment a SOC
would actually run — and breaks detection down per attack class, where the
structural weakness of signatures (zero recall on anything nobody wrote a rule
for) becomes visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_curve

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import (
    alerts_per_day,
    positive_scores,
    rates_at_threshold,
    threshold_at_fpr,
)
from netsentry.evaluation.slices import ClassSlice, per_class_detection
from netsentry.log import get_logger
from netsentry.models.rules import RuleEngine
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "rules.md"


@dataclass
class RuleStats:
    """One rule's outcome on the test split."""

    name: str
    description: str
    fired: int
    precision: float
    recall_all_attacks: float
    dominant_hit: str


@dataclass
class SystemPoint:
    """One detection system's operating point on the test split."""

    name: str
    tpr: float
    fpr: float
    precision: float
    alerts_day: float


def rule_statistics(
    matches: pd.DataFrame, y_bin: np.ndarray, labels: np.ndarray, engine: RuleEngine
) -> list[RuleStats]:
    """Per-rule fired count, precision, attack recall, and dominant true-positive class."""
    y_bin = np.asarray(y_bin).astype(int)
    labels = np.asarray(labels).astype(str)
    n_attacks = max(int(y_bin.sum()), 1)
    stats: list[RuleStats] = []
    for rule in engine.definitions:
        mask = matches[rule.name].to_numpy()
        fired = int(mask.sum())
        hits = mask & (y_bin == 1)
        precision = float(hits.sum() / fired) if fired else 0.0
        recall = float(hits.sum() / n_attacks)
        if hits.any():
            values, counts = np.unique(labels[hits], return_counts=True)
            dominant = str(values[np.argmax(counts)])
        else:
            dominant = "—"
        stats.append(RuleStats(rule.name, rule.description, fired, precision, recall, dominant))
    return stats


def _system_point(
    name: str, y_bin: np.ndarray, decisions: np.ndarray, flows_per_day: int
) -> SystemPoint:
    """Operating point of a binary decision vector (reusing the metric core)."""
    rates = rates_at_threshold(y_bin, decisions.astype(float), 0.5)
    benign_fraction = float(np.mean(np.asarray(y_bin) == 0))
    return SystemPoint(
        name=name,
        tpr=rates["tpr"],
        fpr=rates["fpr"],
        precision=rates["precision"],
        alerts_day=alerts_per_day(rates["fpr"], flows_per_day, benign_fraction),
    )


def run_rules_report(settings: Settings) -> Path:
    """Run the ruleset and the temporal model on the same split; write the comparison."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    engine = RuleEngine(settings.rules.definitions)
    val = load_split(settings, "temporal", "val")
    test = load_split(settings, "temporal", "test")
    y_val = result.y_val.astype(int)
    y_test = result.y_test.astype(int)
    labels = test[MULTICLASS_TARGET].to_numpy()
    flows_per_day = settings.thresholds.assumed_flows_per_day

    # Rules are threshold-free, so their FPR is whatever the ruleset spends. Measure
    # it on validation and give the model the *same* budget (threshold chosen on val,
    # never on test) — the only fair single-point comparison between the two systems.
    rules_val = engine.decisions(val)
    rules_budget = float(np.mean(rules_val[y_val == 0])) if (y_val == 0).any() else 0.0
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    thr_matched = threshold_at_fpr(y_val, s_val, rules_budget)
    thr_primary = threshold_at_fpr(y_val, s_val, settings.thresholds.primary_fpr)

    matches_test = engine.matches(test)
    rules_dec = matches_test.to_numpy().any(axis=1)
    model_matched_dec = s_test >= thr_matched
    hybrid_dec = (s_test >= thr_primary) | rules_dec

    points = [
        _system_point("rules (union)", y_test, rules_dec, flows_per_day),
        _system_point("model @ matched FPR budget", y_test, model_matched_dec, flows_per_day),
        _system_point(
            f"hybrid (rules OR model @ {settings.thresholds.primary_fpr * 100:g}% FPR)",
            y_test,
            hybrid_dec,
            flows_per_day,
        ),
    ]
    per_rule = rule_statistics(matches_test, y_test, labels, engine)

    benign = settings.labels.benign_label
    class_rules = per_class_detection(labels, rules_dec.astype(float), 0.5, benign)
    class_model = per_class_detection(labels, s_test, thr_matched, benign)

    fig = _plot(y_test, s_test, points, settings.paths.figures_dir / "rules.png")
    report = _render(per_rule, points, class_rules, class_model, rules_budget, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote rules report",
        extra={"path": str(out_path), "rules_tpr": round(points[0].tpr, 4)},
    )

    with track_run(settings, "rules_baseline") as run:
        run.log_metrics(
            {
                "rules_tpr": points[0].tpr,
                "rules_fpr": points[0].fpr,
                "model_tpr_at_matched_fpr": points[1].tpr,
                "hybrid_tpr": points[2].tpr,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(
    y_test: np.ndarray, s_test: np.ndarray, points: list[SystemPoint], out_path: Path
) -> Path:
    """Model detection-vs-FPR curve (zoomed to the operational region) + system points."""
    fpr, tpr, _ = roc_curve(y_test, s_test)
    xmax = min(1.0, max(0.02, 2.5 * max(p.fpr for p in points)))
    keep = fpr <= xmax
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "model (score sweep)": (fpr[keep], tpr[keep])
    }
    for p in points:
        series[p.name] = (np.array([p.fpr]), np.array([p.tpr]))
    return plots.plot_lines(
        series,
        xlabel="False positive rate",
        ylabel="Detection rate (TPR)",
        title="Rules vs model (temporal test, operational region)",
        out_path=out_path,
    )


def _read(points: list[SystemPoint], zero_classes: list[str]) -> str:
    """Sign-aware prose so the report can never contradict its own numbers."""
    rules, model, hybrid = points
    if model.tpr >= rules.tpr:
        head = (
            f"At the ruleset's own false-positive budget ({rules.fpr * 100:.2f}%), the model "
            f"detects **{model.tpr * 100:.1f}%** of attacks vs the rules' "
            f"{rules.tpr * 100:.1f}% — the learned detector wins at the matched operating point."
        )
    else:
        head = (
            f"At the matched budget the tuned signatures edge out the model "
            f"({rules.tpr * 100:.1f}% vs {model.tpr * 100:.1f}%) on this attack mix — "
            "unsurprising where the mix is dominated by exactly the patterns the rules "
            "encode. Coverage is where signatures lose, not the single operating point."
        )
    coverage = (
        "Every attack class without a signature is invisible to the rules: "
        + ", ".join(f"**{c}**" for c in zero_classes)
        + " have ~0% rule detection."
        if zero_classes
        else "On this run every class was touched by at least one rule — real traffic "
        "(and real novel attacks) will not be so obliging."
    )
    return f"""{head}

{coverage} The deeper differences are structural: a ruleset has **no dial** (its
FPR is fixed by whoever wrote the thresholds, while the model trades precision for
recall along a curve), no probability (so no cost-optimal or conformal layer can
sit on top), and a maintenance loop measured in analyst time per rule. The hybrid
row shows the two are complements, not rivals: rules OR model detects
{hybrid.tpr * 100:.1f}% at {hybrid.fpr * 100:.2f}% FPR — signatures give cheap
precision on known tools, the model covers the space between signatures, and the
anomaly detector covers what neither has seen."""


def _render(
    per_rule: list[RuleStats],
    points: list[SystemPoint],
    class_rules: list[ClassSlice],
    class_model: list[ClassSlice],
    rules_budget: float,
    fig: Path,
) -> str:
    rule_rows = [
        "| rule | encodes | fired | precision | recall (all attacks) | dominant hit |",
        "|---|---|---|---|---|---|",
    ]
    for r in per_rule:
        rule_rows.append(
            f"| `{r.name}` | {r.description} | {r.fired:,} | {r.precision * 100:.1f}% "
            f"| {r.recall_all_attacks * 100:.1f}% | {r.dominant_hit} |"
        )

    point_rows = [
        "| system | detection (TPR) | FPR | precision | alerts/day @ 1M flows |",
        "|---|---|---|---|---|",
    ]
    for p in points:
        point_rows.append(
            f"| {p.name} | **{p.tpr * 100:.1f}%** | {p.fpr * 100:.2f}% "
            f"| {p.precision * 100:.1f}% | {p.alerts_day:,.0f} |"
        )

    model_by_class = {s.label: s.detection for s in class_model}
    class_rows = [
        "| attack class | test support | rules | model (matched budget) |",
        "|---|---|---|---|",
    ]
    zero_classes: list[str] = []
    for s in sorted(class_rules, key=lambda s: s.detection, reverse=True):
        model_det = model_by_class.get(s.label, 0.0)
        class_rows.append(
            f"| {s.label} | {s.support:,} | {s.detection * 100:.1f}% | {model_det * 100:.1f}% |"
        )
        if s.detection < 0.005:
            zero_classes.append(s.label)

    return f"""# NetSentry — Rules-vs-Model Baseline

_Synthetic stand-in. The configured signature ruleset (`rules.definitions`) and the
temporal-split binary classifier, evaluated on the same honest temporal test split.
The model's comparison threshold is chosen on **validation** at the ruleset's own
false-positive budget ({rules_budget * 100:.2f}% on validation), so neither system
touches test before the comparison._

## Why compare against rules

A signature engine is the incumbent, and it is genuinely hard to beat on the
patterns it encodes: rules are auditable, port-scoped (they may use `Destination
Port` — the very context the ML model deliberately drops to avoid memorising it),
and free of training data, so there is nothing to leak. An ML detector that cannot
beat six hand-written thresholds at the same false-positive cost has no business in
the pipeline.

## Per-rule performance (temporal test)

{chr(10).join(rule_rows)}

## Systems at a matched false-positive budget

{chr(10).join(point_rows)}

![Rules vs model](../figures/{fig.name})

## Per-class detection

{chr(10).join(class_rows)}

## Read

{_read(points, zero_classes)}
"""
