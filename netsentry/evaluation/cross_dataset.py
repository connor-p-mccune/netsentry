"""Cross-dataset generalization study.

Score a foreign-schema dataset with the CIC-trained model (no retraining, no
peeking — the exact bundle you would serve) and contrast it honestly with
in-domain performance. The expected, healthy outcome is a **drop**: a model that
generalised perfectly across schemas would be suspicious, not impressive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.cross_dataset import adapt_foreign_to_cic, generate_foreign
from netsentry.data.split import load_split
from netsentry.evaluation import metrics as M
from netsentry.log import get_logger
from netsentry.models.registry import ModelBundle, latest_bundle, load_bundle
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "cross_dataset.md"


@dataclass
class _Row:
    name: str
    pr_auc: float
    roc_auc: float
    tpr: float
    prevalence: float


def _scores(bundle: ModelBundle, frame: pd.DataFrame, benign: str) -> np.ndarray:
    return M.attack_probability(bundle.predict_proba(frame), bundle.classes, benign)


def _row(name: str, y: np.ndarray, scores: np.ndarray, fpr: float) -> _Row:
    summary = M.binary_summary(y, scores)
    _, tpr = M.tpr_at_fpr(y, scores, fpr)
    return _Row(
        name, summary["pr_auc"], summary.get("roc_auc", float("nan")), tpr, float(np.mean(y))
    )


def run_cross_dataset_eval(settings: Settings, *, bundle: ModelBundle | None = None) -> Path:
    """Score the foreign dataset, write the contrast report, log to MLflow."""
    if bundle is None:
        path = settings.serving.artifact_path or latest_bundle(settings)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                "No model bundle found. Train one with `netsentry train supervised` first."
            )
        bundle = load_bundle(Path(path))
    benign = settings.labels.benign_label

    in_domain = load_split(settings, "temporal", "test")
    y_in = in_domain[BINARY_TARGET].to_numpy().astype(int)
    s_in = _scores(bundle, in_domain, benign)

    foreign = generate_foreign(settings)
    adapted = adapt_foreign_to_cic(foreign, settings)
    y_x = adapted[BINARY_TARGET].to_numpy().astype(int)
    s_x = _scores(bundle, adapted, benign)

    fpr = settings.thresholds.primary_fpr
    in_row = _row("in-domain (CIC temporal test)", y_in, s_in, fpr)
    cross_row = _row(f"cross ({settings.crossdata.name})", y_x, s_x, fpr)
    gap = in_row.pr_auc - cross_row.pr_auc

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(in_row, cross_row, gap, fpr, settings), encoding="utf-8")
    logger.info(
        "Wrote cross-dataset report", extra={"path": str(out_path), "pr_auc_gap": round(gap, 4)}
    )

    with track_run(settings, "cross_dataset") as run:
        run.log_metrics(
            {
                "in_domain_pr_auc": in_row.pr_auc,
                "cross_pr_auc": cross_row.pr_auc,
                "cross_pr_auc_gap": gap,
                "cross_tpr_at_primary_fpr": cross_row.tpr,
            }
        )
        run.log_artifact(out_path)
    return out_path


def _narrative(gap: float) -> str:
    """Sign-aware prose so the report can never contradict its own numbers."""
    if gap >= 0.05:
        return (
            "The cross-dataset number is lower — the honest, expected result. A NetFlow "
            "record exposes only a handful of counters, so most CIC features have no "
            "equivalent and are imputed; detection transfers solely through shared "
            "behaviour (packet/byte volumes and rates). A model that scored identically "
            "across schemas would be suspicious, not impressive — reporting the drop is "
            "what separates learned attack behaviour from a memorised capture."
        )
    if gap <= -0.05:
        return (
            "The cross score is *higher* than in-domain — which here is a property of the "
            "**synthetic** stand-in, not a real generalisation win, and exactly the kind of "
            "too-good result this project treats as a flag to investigate. The foreign "
            "attacks are high-volume flows the CIC model detects easily, whereas the "
            "in-domain temporal test holds genuinely novel later-day attacks. On real "
            "UNSW-NB15 / NF-*-v2 data expect the gap to invert into the usual drop; that is "
            "the number worth trusting."
        )
    return (
        "In-domain and cross scores are close. Detection transfers through the few shared "
        "behavioural features (volumes and rates); the rest of the CIC feature space is "
        "absent from a NetFlow schema and imputed. Treat the synthetic magnitude as "
        "illustrative — real UNSW-NB15 / NF-*-v2 numbers are the ones to trust."
    )


def _render(in_row: _Row, cross_row: _Row, gap: float, fpr: float, settings: Settings) -> str:
    header = f"| dataset | PR-AUC | ROC-AUC | TPR @ {fpr * 100:.1f}% FPR | attack prevalence |"
    rows = [header, "|---|---|---|---|---|"]
    for r in (in_row, cross_row):
        rows.append(
            f"| {r.name} | {r.pr_auc:.3f} | {r.roc_auc:.3f} "
            f"| {r.tpr * 100:.1f}% | {r.prevalence:.2f} |"
        )
    table = "\n".join(rows)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# NetSentry — Cross-Dataset Generalization

_Generated {generated}. The CIC-trained model is scored, unchanged, on a foreign
**NetFlow-schema** dataset adapted into CIC features. Both datasets here are
synthetic stand-ins; the methodology — not the absolute number — is the point._

## Result

{table}

- **PR-AUC: in-domain {in_row.pr_auc:.3f} → cross {cross_row.pr_auc:.3f} (gap {gap:+.3f}).**

{_narrative(gap)}

## Method

- Foreign data: `{settings.crossdata.name}` — a NetFlow-style schema (in/out
  packets & bytes, duration, port) whose attacks are DoS/DDoS-like high-volume
  flows.
- Adapter: rename/unit-convert the shared quantities, derive a few CIC rates, and
  leave every unmatched CIC feature NaN for the fitted pipeline to impute. No
  retraining and no peeking — the production bundle is scored exactly as served.
- For real numbers, point the adapter at UNSW-NB15 or the NetFlow `NF-*-v2`
  releases; the commands and framing are identical.
"""
