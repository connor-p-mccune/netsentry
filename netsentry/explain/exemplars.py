"""Exemplar (case-based) explanations: "this flow looks like these known flows".

SHAP answers *which features* drove a score; the analyst's next question is
usually *have we seen this before?* Exemplar retrieval answers it with cases:
the k nearest **training** flows in the model's own standardized feature space,
with their labels, capture days, and distances. A flagged flow whose neighbours
are all DoS Hulk from Wednesday is a different triage item from one whose
neighbours are benign web sessions — same score, different story.

The audit here asks whether that story can be *trusted* before shipping it:

- **Agreement.** Among alerted test flows, is the alert precision higher when
  the nearest training exemplars agree (majority attack) than when they don't?
  If yes, neighbour agreement is a free re-ranking signal on top of the score.
- **Distance as novelty.** Missed attacks should sit farther from the training
  set than caught ones (the novelty study's geometry, replayed per flow) — the
  distance column doubles as a "this is unlike anything we trained on" flag.

The index is a class-balanced, seeded subsample of the training split, so rare
attack classes are represented rather than drowned by benign volume, and the
whole structure stays small enough to embed in a serving bundle (float32,
hundreds of rows per class). Retrieval is exact brute-force Euclidean — at this
index size an ANN structure would be complexity without payoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data import schema
from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "exemplars.md"


@dataclass
class ExemplarIndex:
    """A compact, class-balanced case base in the pipeline's standardized space."""

    matrix: np.ndarray  # (n, d) float32, standardized features
    labels: np.ndarray  # (n,) consolidated class labels (display)
    days: np.ndarray  # (n,) capture day per exemplar

    def query(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact k-NN: (distances, indices), each (n_queries, k), nearest first."""
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        # (a - b)^2 expansion keeps memory at n_q x n_index floats.
        sq = (q**2).sum(axis=1)[:, None] + (self.matrix**2).sum(axis=1)[None, :]
        sq -= 2.0 * (q @ self.matrix.T)
        distances = np.sqrt(np.maximum(sq, 0.0))
        k = min(k, self.matrix.shape[0])
        idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        row = np.arange(len(q))[:, None]
        order = np.argsort(distances[row, idx], axis=1)
        idx = idx[row, order]
        return distances[row, idx], idx

    def to_payload(self) -> dict[str, Any]:
        """A plain-container form safe to stash in bundle metadata."""
        return {
            "matrix": np.asarray(self.matrix, dtype=np.float32),
            "labels": [str(v) for v in self.labels],
            "days": [str(v) for v in self.days],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExemplarIndex:
        return cls(
            matrix=np.asarray(payload["matrix"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=str),
            days=np.asarray(payload["days"], dtype=str),
        )


def build_exemplar_index(
    matrix: np.ndarray,
    labels: np.ndarray,
    days: np.ndarray,
    per_class: int,
    seed: int,
) -> ExemplarIndex:
    """Class-balanced, seeded subsample: every label keeps up to ``per_class`` rows.

    Balance is the point — a proportional sample would be ~80% benign and the
    rare attack classes (the cases an analyst most needs to recognise) would
    have no representatives to match against.
    """
    labels = np.asarray(labels).astype(str)
    days = np.asarray(days).astype(str)
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for label in sorted(set(labels.tolist())):
        rows = np.where(labels == label)[0]
        if len(rows) > per_class:
            rows = rng.choice(rows, size=per_class, replace=False)
        keep.append(np.sort(rows))
    order = np.concatenate(keep) if keep else np.array([], dtype=int)
    return ExemplarIndex(
        matrix=np.asarray(matrix, dtype=np.float32)[order],
        labels=labels[order],
        days=days[order],
    )


def exemplar_support(neighbor_is_attack: np.ndarray) -> np.ndarray:
    """Majority vote per query row; a tie is *not* support (conservative)."""
    votes = np.asarray(neighbor_is_attack, dtype=float)
    return np.asarray(votes.mean(axis=1) > 0.5)


def run_exemplars_report(settings: Settings) -> Path:
    """Audit exemplar agreement and distance on the temporal split; write the report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)
    cfg = settings.exemplars

    train = load_split(variant, "temporal", "train")
    test = load_split(variant, "temporal", "test")
    x_train = result.bundle.pipeline.transform(train)
    x_test = result.bundle.pipeline.transform(test)
    benign = variant.labels.benign_label

    index = build_exemplar_index(
        x_train,
        train[MULTICLASS_TARGET].to_numpy(),
        (
            train[schema.DAY_COLUMN].to_numpy()
            if schema.DAY_COLUMN in train.columns
            else np.full(len(train), "?")
        ),
        cfg.per_class,
        variant.seed,
    )

    # Decisions at the loose operating point (raw scores, the eval convention).
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)
    y_test = result.y_test.astype(int)
    budget = sorted(variant.thresholds.fpr_targets)[-1]
    threshold = threshold_at_fpr(result.y_val.astype(int), s_val, budget)
    alerts = s_test >= threshold

    distances, idx = index.query(np.asarray(x_test), cfg.k)
    neighbor_attack = index.labels[idx] != benign
    supported = exemplar_support(neighbor_attack)
    nn_distance = distances[:, 0]

    sup_alerts = alerts & supported
    unsup_alerts = alerts & ~supported
    prec_all = float(y_test[alerts].mean()) if alerts.any() else float("nan")
    prec_sup = float(y_test[sup_alerts].mean()) if sup_alerts.any() else float("nan")
    prec_unsup = float(y_test[unsup_alerts].mean()) if unsup_alerts.any() else float("nan")

    caught = (y_test == 1) & alerts
    missed = (y_test == 1) & ~alerts
    dist_caught = float(nn_distance[caught].mean()) if caught.any() else float("nan")
    dist_missed = float(nn_distance[missed].mean()) if missed.any() else float("nan")

    fig = plots.plot_barh(
        ["all alerts", "exemplar-supported", "exemplar-unsupported"],
        [prec_all, prec_sup, prec_unsup],
        xlabel=f"alert precision @ {budget * 100:g}% FPR budget",
        title="Do the nearest known cases vouch for the alert?",
        out_path=settings.paths.figures_dir / "exemplars.png",
    )

    examples = _example_rows(test, s_test, alerts, distances, idx, index, cfg.examples, benign)
    report = _render(
        settings,
        budget,
        (prec_all, prec_sup, prec_unsup),
        (float(sup_alerts.sum()), float(alerts.sum())),
        (dist_caught, dist_missed),
        examples,
        len(index.labels),
        fig,
    )
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote exemplars report", extra={"path": str(out_path)})

    with track_run(settings, "exemplars") as run:
        run.log_params({"per_class": cfg.per_class, "k": cfg.k})
        run.log_metrics(
            {
                "alert_precision": prec_all,
                "alert_precision_supported": prec_sup,
                "alert_precision_unsupported": prec_unsup,
                "nn_distance_caught": dist_caught,
                "nn_distance_missed": dist_missed,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _example_rows(
    test: Any,
    s_test: np.ndarray,
    alerts: np.ndarray,
    distances: np.ndarray,
    idx: np.ndarray,
    index: ExemplarIndex,
    n_examples: int,
    benign: str,
) -> list[str]:
    """The report's 'what an analyst would see' table: top alerts + their cases."""
    rows = [
        "| alert (raw score) | true label | nearest training cases (label @ distance) |",
        "|---|---|---|",
    ]
    alert_positions = np.where(alerts)[0]
    top = alert_positions[np.argsort(s_test[alert_positions])[::-1][:n_examples]]
    truth = test[MULTICLASS_TARGET].to_numpy()
    for t in top:
        cases = ", ".join(
            f"{index.labels[j]} @ {d:.1f}"
            for d, j in zip(distances[t, :3], idx[t, :3], strict=True)
        )
        label = str(truth[t])
        label = label if label != benign else f"{benign} (false alert)"
        rows.append(f"| {s_test[t]:.3f} | {label} | {cases} |")
    return rows


def _render(
    settings: Settings,
    budget: float,
    precisions: tuple[float, float, float],
    support_counts: tuple[float, float],
    nn_dists: tuple[float, float],
    examples: list[str],
    index_size: int,
    fig: Path,
) -> str:
    cfg = settings.exemplars
    prec_all, prec_sup, prec_unsup = precisions
    n_sup, n_alerts = support_counts
    dist_caught, dist_missed = nn_dists
    sup_share = n_sup / n_alerts if n_alerts else float("nan")

    n_unsup = int(n_alerts - n_sup)
    if np.isfinite(prec_sup) and np.isfinite(prec_unsup) and prec_sup > prec_unsup + 0.05:
        agreement_read = (
            f"Neighbour agreement points the right way: alerts whose {cfg.k} nearest training "
            f"cases vote attack are {prec_sup:.0%} precise ({int(n_sup)} alerts), against "
            f"{prec_unsup:.0%} when the neighbourhood disagrees ({n_unsup} alerts; all alerts "
            f"{prec_all:.0%}). Read the bucket sizes before the percentages — the disagreeing "
            "bucket is small here, so the gap is directional evidence for triage ordering "
            "(corroborated alerts first), not a calibrated re-ranker."
        )
    elif np.isfinite(prec_sup) and np.isfinite(prec_unsup):
        agreement_read = (
            f"On this stand-in, neighbour agreement adds little over the score: supported "
            f"alerts are {prec_sup:.0%} precise vs {prec_unsup:.0%} unsupported (all: "
            f"{prec_all:.0%}). The retrieval is still worth shipping for *explanation* — "
            "'similar to these known cases' is checkable in a way a bare probability is not — "
            "but it should not be sold as a triage re-ranker here."
        )
    else:
        agreement_read = (
            "One of the alert buckets is empty at this operating point, so the agreement "
            "comparison is not defined on this run; the examples table below still shows "
            "the retrieval contract."
        )

    if np.isfinite(dist_caught) and np.isfinite(dist_missed) and dist_missed > dist_caught * 1.1:
        distance_read = (
            f"Missed attacks sit farther from the training set (mean nearest-neighbour "
            f"distance {dist_missed:.1f}) than caught ones ({dist_caught:.1f}) — the novelty "
            "study's geometry, replayed per flow. A large distance on a *cleared* flow is "
            "therefore a cheap unfamiliarity flag, complementing the anomaly detector."
        )
    elif np.isfinite(dist_caught) and np.isfinite(dist_missed):
        distance_read = (
            f"Distance does **not** separate caught from missed attacks here (caught "
            f"{dist_caught:.1f} vs missed {dist_missed:.1f} mean NN distance) — consistent "
            "with the novelty study's stand-in finding that the hard attacks hug the benign "
            "manifold rather than sitting far away. On real burst-structured data the "
            "expectation flips; the report states what this data shows."
        )
    else:
        distance_read = "No attacks on one side of the threshold; distance comparison skipped."

    return f"""# NetSentry — Exemplar Explanations (the case-based *have we seen this?*)

_Synthetic stand-in. Temporal split; binary model, raw scores, threshold chosen
on validation at the {budget * 100:g}% FPR budget. The case base is a
class-balanced sample of the training split ({index_size} exemplars,
{cfg.per_class}/class cap) in the fitted pipeline's standardized feature space;
retrieval is exact k-NN (k = {cfg.k}), distances in standardized units._

## Why cases, when SHAP already explains?

SHAP attributes the score to features; it cannot tell an analyst whether the
flow resembles anything the model has actually seen. Exemplars answer with
precedent — the nearest labeled training flows — which is checkable evidence: an
analyst can pull those cases and compare. This report audits whether the
retrieval earns trust before the API ships it.

## Does the neighbourhood vouch for the alerts?

{sup_share:.0%} of alerts are exemplar-supported (majority of {cfg.k} neighbours
vote attack; ties count against, conservatively).

![Alert precision by exemplar support](../figures/{fig.name})

{agreement_read}

## Distance as an unfamiliarity flag

{distance_read}

## What an analyst would see

{chr(10).join(examples)}

## Scope

Exemplars explain by *similarity in the model's feature space*, not mechanism —
two flows can be near neighbours for reasons an analyst would not consider
related, and the space itself is the standardized CIC features, so the
explanation inherits every representational limit the model has. The case base
is a subsample: absence of a near neighbour means "nothing similar in the
sample", not "nothing similar in training". Distances are in standardized units
and comparable only within one pipeline fit.
"""
