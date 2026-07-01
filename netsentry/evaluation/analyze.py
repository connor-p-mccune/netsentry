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

from netsentry.evaluation.conformal import run_conformal_report
from netsentry.evaluation.cost import run_cost_report
from netsentry.evaluation.report import run_evaluation
from netsentry.explain.counterfactual import run_recourse_report
from netsentry.intel.report import run_mitre_report
from netsentry.log import get_logger
from netsentry.monitoring.report import run_drift_report
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
    ("Drift monitoring", "feature/score PSI, train vs test", "drift.md", run_drift_report),
    (
        "Counterfactual recourse",
        "minimal change that clears a hit",
        "recourse.md",
        run_recourse_report,
    ),
    ("MITRE ATT&CK coverage", "attack class -> tactic/technique", "mitre.md", run_mitre_report),
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
