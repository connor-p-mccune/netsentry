"""Evaluation figures, saved under ``docs/figures``.

Matplotlib is imported lazily (with the headless Agg backend) so importing this
module stays cheap and it works in CI/containers without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from netsentry.evaluation.metrics import rates_at_threshold
from netsentry.log import get_logger

logger = get_logger(__name__)

ScoreCurves = dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (y_true, scores)


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig: Any, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
    logger.info("Wrote figure", extra={"path": str(out_path)})
    return out_path


def plot_pr_curves(curves: ScoreCurves, out_path: Path) -> Path:
    """Precision-recall curves (one line per split) — the headline comparison."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_roc_curves(curves: ScoreCurves, out_path: Path) -> Path:
    """ROC curves (optimistic under imbalance; reported for completeness)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, scores) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_threshold_curve(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> Path:
    """Precision / recall(TPR) / FPR versus decision threshold."""
    plt = _plt()
    thresholds = np.quantile(np.unique(scores), np.linspace(0, 1, 100))
    precision, recall, fpr = [], [], []
    for thr in thresholds:
        rates = rates_at_threshold(y_true, scores, float(thr))
        precision.append(rates["precision"])
        recall.append(rates["tpr"])
        fpr.append(rates["fpr"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall (TPR)")
    ax.plot(thresholds, fpr, label="FPR")
    ax.set(xlabel="Decision threshold", ylabel="Rate", title="Threshold trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> Path:
    """Row-normalised confusion matrix heatmap."""
    plt = _plt()
    with np.errstate(invalid="ignore"):
        normed = cm / cm.sum(axis=1, keepdims=True)
    normed = np.nan_to_num(normed)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)
