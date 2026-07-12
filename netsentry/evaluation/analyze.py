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
from netsentry.evaluation.adaptive_conformal import run_adaptive_conformal_report
from netsentry.evaluation.alert_queue import run_alert_queue_report
from netsentry.evaluation.baserate import run_base_rate_report
from netsentry.evaluation.campaigns import run_campaigns_report
from netsentry.evaluation.conformal import run_conformal_report
from netsentry.evaluation.cost import run_cost_report
from netsentry.evaluation.gate import run_gate
from netsentry.evaluation.label_audit import run_label_audit_report
from netsentry.evaluation.leaderboard import run_leaderboard_report
from netsentry.evaluation.lodo import run_lodo_report
from netsentry.evaluation.novelty import run_novelty_report
from netsentry.evaluation.report import run_evaluation
from netsentry.evaluation.rules import run_rules_report
from netsentry.evaluation.seed_variance import run_seed_variance_report
from netsentry.evaluation.slices import run_slices_report
from netsentry.evaluation.socsim import run_socsim_report
from netsentry.evaluation.subgroups import run_subgroups_report
from netsentry.explain.counterfactual import run_recourse_report
from netsentry.explain.distill import run_distill_report
from netsentry.explain.exemplars import run_exemplars_report
from netsentry.explain.importance_stability import run_importance_stability_report
from netsentry.governance.provenance import run_provenance_report
from netsentry.intel.navigator import run_navigator_export
from netsentry.intel.report import run_mitre_report
from netsentry.intel.sigma import run_sigma_export
from netsentry.log import get_logger
from netsentry.monitoring.refresh import run_refresh_report
from netsentry.monitoring.report import run_drift_report, run_drift_tests_report
from netsentry.monitoring.retrain_policy import run_retrain_policy_report
from netsentry.monitoring.streaming import run_streaming_report
from netsentry.robustness.hardening import run_hardening_report
from netsentry.robustness.poisoning import run_poisoning_report
from netsentry.robustness.report import run_robustness_report
from netsentry.robustness.sanitize import run_sanitize_report
from netsentry.training.selftrain import run_selftrain_report

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

INDEX_NAME = "INDEX.md"


def _run_gate_report(settings: Settings) -> Path:
    """Adapter: the gate writes its report either way; enforcement is the CLI's job."""
    out, _ = run_gate(settings)
    return out


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
        "SOC queue simulation",
        "FIFO vs score-priority attack-SLA under queueing load",
        "socsim.md",
        run_socsim_report,
    ),
    (
        "Base-rate stress test",
        "alert precision vs production prevalence (Axelsson 1999)",
        "base_rate.md",
        run_base_rate_report,
    ),
    (
        "Conformal prediction",
        "coverage guarantee + selective alerting",
        "conformal.md",
        run_conformal_report,
    ),
    (
        "Adaptive conformal",
        "coverage restored online under drift (ACI)",
        "adaptive_conformal.md",
        run_adaptive_conformal_report,
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
        "Poisoning defense",
        "audit-and-drop sanitization vs label flips, re-measured",
        "poisoning_defense.md",
        run_sanitize_report,
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
        "Retrain-trigger policy",
        "when to retrain: never / periodic / drift-triggered / every batch",
        "retrain_policy.md",
        run_retrain_policy_report,
    ),
    (
        "Threshold refresh",
        "the label-cheap lever vs retraining; budget compliance under drift",
        "refresh.md",
        run_refresh_report,
    ),
    (
        "Self-training",
        "pseudo-labels on the unlabeled stream vs the labeled ceiling",
        "selftrain.md",
        run_selftrain_report,
    ),
    (
        "Model-family leaderboard",
        "every family through one honest protocol; the gap replicates",
        "leaderboard.md",
        run_leaderboard_report,
    ),
    (
        "Per-class detection",
        "which temporal-split attacks are caught",
        "slices.md",
        run_slices_report,
    ),
    (
        "Campaign detection",
        "the (day, class) operation as the unit: first alerts and silent campaigns",
        "campaigns.md",
        run_campaigns_report,
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
        "Importance stability",
        "are the shipped explanations stable across refits",
        "importance_stability.md",
        run_importance_stability_report,
    ),
    (
        "Exemplar explanations",
        "do the nearest known training flows vouch for the alerts",
        "exemplars.md",
        run_exemplars_report,
    ),
    (
        "Surrogate distillation",
        "the model's closest auditable imitation, with fidelity priced",
        "distill.md",
        run_distill_report,
    ),
    (
        "Active learning",
        "uncertainty vs random labeling efficiency",
        "active_learning.md",
        run_active_learning_report,
    ),
    (
        "Seed sensitivity",
        "the training-noise floor under every reported metric",
        "seed_variance.md",
        run_seed_variance_report,
    ),
    (
        "Release gate",
        "honesty invariants + metric floors the candidate must clear",
        "gate.md",
        _run_gate_report,
    ),
    ("MITRE ATT&CK coverage", "attack class -> tactic/technique", "mitre.md", run_mitre_report),
    (
        "ATT&CK Navigator layer",
        "detection coverage as a loadable Navigator layer",
        "attack_navigator_layer.json",
        run_navigator_export,
    ),
    (
        "Sigma detection rules",
        "the signature baseline exported as portable Sigma rules",
        "sigma/README.md",
        run_sigma_export,
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
