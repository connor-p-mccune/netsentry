"""Run the full analysis suite and write an index — one-command reproducibility.

Regenerates every model-analysis report (operational evaluation + calibration,
cost-sensitive thresholds, conformal prediction, adversarial robustness, drift) and
writes an ``INDEX.md`` linking them with one-line summaries and a pass/fail status.
Each report is run defensively, so one failure does not abort the rest — the index
records which succeeded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.evaluation.ablation import run_ablation_report
from netsentry.evaluation.active_learning import run_active_learning_report
from netsentry.evaluation.alert_queue import run_alert_queue_report
from netsentry.evaluation.conformal import run_conformal_report
from netsentry.evaluation.cost import run_cost_report
from netsentry.evaluation.label_audit import run_label_audit_report
from netsentry.evaluation.lodo import run_lodo_report
from netsentry.evaluation.novelty import run_novelty_report
from netsentry.evaluation.report import run_evaluation
from netsentry.evaluation.rules import run_rules_report
from netsentry.evaluation.slices import run_slices_report
from netsentry.evaluation.subgroups import run_subgroups_report
from netsentry.explain.counterfactual import run_recourse_report
from netsentry.governance.provenance import run_provenance_report
from netsentry.intel.navigator import run_navigator_export
from netsentry.intel.report import run_mitre_report
from netsentry.log import get_logger
from netsentry.monitoring.report import run_drift_report, run_drift_tests_report
from netsentry.monitoring.streaming import run_streaming_report
from netsentry.robustness.hardening import run_hardening_report
from netsentry.robustness.poisoning import run_poisoning_report
from netsentry.robustness.report import run_robustness_report

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

INDEX_NAME = "INDEX.md"

# (title, description, output filename, runner). Runners take only Settings.
_ANALYSES: list[tuple[str, str, str, Callable[[Settings], Path]]] = [
    (
        "Operational evaluation",
        "PR-AUC, TPR@FPR, per-class, calibration",
        "evaluation.md",
        run_evaluation,
    ),
    ("Cost-sensitive thresholds", "decision-theoretic operating point", "cost.md", run_cost_report),
    (
        "Alert-queue capacity",
        "detection vs analyst budget; lift over random triage",
        "alert_queue.md",
        run_alert_queue_report,
    ),
    (
        "Conformal prediction",
        "coverage guarantee + selective alerting",
        "conformal.md",
        run_conformal_report,
    ),
    (
        "Adversarial robustness",
        "evasion (mimicry + query search)",
        "robustness.md",
        run_robustness_report,
    ),
    (
        "Training-set poisoning",
        "label flips + benign-pool contamination",
        "poisoning.md",
        run_poisoning_report,
    ),
    (
        "Adversarial hardening",
        "adversarial training vs mimicry, re-measured",
        "hardening.md",
        run_hardening_report,
    ),
    (
        "Label-noise audit",
        "confident-learning flags + planted-flip self-validation",
        "label_audit.md",
        run_label_audit_report,
    ),
    ("Drift monitoring", "feature/score PSI, train vs test", "drift.md", run_drift_report),
    (
        "Statistical drift",
        "per-feature KS+FDR, online Page-Hinkley/DDM",
        "drift_tests.md",
        run_drift_tests_report,
    ),
    (
        "Prequential streaming",
        "static vs retrained model on the later-day stream",
        "streaming.md",
        run_streaming_report,
    ),
    (
        "Per-class detection",
        "which temporal-split attacks are caught",
        "slices.md",
        run_slices_report,
    ),
    (
        "Per-service parity",
        "detection/false-alarm equity across services",
        "subgroups.md",
        run_subgroups_report,
    ),
    (
        "Novelty distance",
        "detection vs distance-to-training; the split gap decomposed",
        "novelty.md",
        run_novelty_report,
    ),
    (
        "Leave-one-day-out",
        "temporal sensitivity: every day takes a turn as the future",
        "lodo.md",
        run_lodo_report,
    ),
    (
        "Rules-vs-model baseline",
        "hand-written signatures at a matched FPR budget",
        "rules.md",
        run_rules_report,
    ),
    (
        "Feature-group ablation",
        "which behavioural families carry detection",
        "ablation.md",
        run_ablation_report,
    ),
    (
        "Counterfactual recourse",
        "minimal change that clears a hit",
        "recourse.md",
        run_recourse_report,
    ),
    (
        "Active learning",
        "uncertainty vs random labeling efficiency",
        "active_learning.md",
        run_active_learning_report,
    ),
    ("MITRE ATT&CK coverage", "attack class -> tactic/technique", "mitre.md", run_mitre_report),
    (
        "ATT&CK Navigator layer",
        "detection coverage as a loadable Navigator layer",
        "attack_navigator_layer.json",
        run_navigator_export,
    ),
    (
        "Provenance & supply chain",
        "CycloneDX SBOM + model-integrity manifest",
        "provenance.md",
        run_provenance_report,
    ),
]


@dataclass
class AnalysisEntry:
    """The outcome of running one analysis in the suite."""

    title: str
    description: str
    filename: str
    ok: bool
    error: str | None = None


def write_index(reports_dir: Path, entries: list[AnalysisEntry]) -> Path:
    """Write the analysis index linking each report with its status."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NetSentry — Analysis Index",
        "",
        f"_Regenerated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} via "
        "`netsentry analyze`. Synthetic stand-in unless run on the real dataset._",
        "",
        "| report | what it covers | status |",
        "|---|---|---|",
    ]
    for e in entries:
        status = f"[open]({e.filename})" if e.ok else f"failed — {e.error}"
        lines.append(f"| {e.title} | {e.description} | {status} |")
    lines.append("")
    out = reports_dir / INDEX_NAME
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_full_analysis(settings: Settings) -> Path:
    """Run every analysis report and write the index; return the index path."""
    entries: list[AnalysisEntry] = []
    for title, description, filename, runner in _ANALYSES:
        try:
            runner(settings)
            entries.append(AnalysisEntry(title, description, filename, ok=True))
            logger.info("Analysis done", extra={"report": title})
        except Exception as exc:  # one report failing must not abort the suite
            logger.warning("Analysis failed (%s): %s", title, exc)
            entries.append(AnalysisEntry(title, description, filename, ok=False, error=str(exc)))
    index = write_index(settings.paths.reports_dir, entries)
    n_ok = sum(e.ok for e in entries)
    logger.info(
        "Wrote analysis index", extra={"path": str(index), "ok": n_ok, "total": len(entries)}
    )
    return index
