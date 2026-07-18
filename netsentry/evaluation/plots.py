"""Evaluation figures, saved under ``docs/figures``.

Matplotlib is imported lazily (with the headless Agg backend) so importing this
module stays cheap and it works in CI/containers without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def plot_lines(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    vlines: dict[str, float] | None = None,
    xscale: str | None = None,
    yscale: str | None = None,
) -> Path:
    """Generic ``name -> (x, y)`` line chart (robustness/cost/conformal curves).

    ``xscale``/``yscale`` pass through to matplotlib (e.g. ``"log"`` for a
    prevalence sweep or a realized-FPR axis spanning orders of magnitude, where a
    linear axis would crush one end).
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (x, y) in series.items():
        ax.plot(np.asarray(x), np.asarray(y), marker="o", label=name)
    for label, xpos in (vlines or {}).items():
        ax.axvline(xpos, color="gray", linestyle=":", alpha=0.8, label=label)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_barh(
    labels: list[str],
    values: list[float],
    *,
    xlabel: str,
    title: str,
    out_path: Path,
    xmax: float | None = None,
    vline: tuple[str, float] | None = None,
) -> Path:
    """Horizontal bar chart (e.g. per-class detection). Highest bar on top.

    ``xmax`` overrides the axis maximum (default ``max(1.0, max(values))``, suited to
    rates in [0, 1]); pass it for small-magnitude series like per-service FPR so the
    bars are visible. ``vline`` draws a labelled vertical reference line — e.g. the
    global FPR budget the per-service bars are being compared against.
    """
    plt = _plt()
    order = np.argsort(values)
    names = [labels[i] for i in order]
    vals = [values[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
    ax.barh(names, vals, color="#3b7dd8")
    ax.set(xlabel=xlabel, title=title)
    ax.set_xlim(0, xmax if xmax is not None else max(1.0, max(vals) if vals else 1.0))
    if vline is not None:
        line_label, xpos = vline
        ax.axvline(xpos, color="#d1495b", linestyle="--", alpha=0.9, label=line_label)
        ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, out_path)


def plot_hist_overlay(
    series: dict[str, np.ndarray],
    *,
    xlabel: str,
    title: str,
    out_path: Path,
    bins: int = 40,
    vline: float | None = None,
) -> Path:
    """Overlaid, density-normalised histograms — e.g. clean-vs-mislabelled value spreads.

    Each series is drawn semi-transparent over a shared, data-spanning bin edge set so
    two populations (with very different counts) are visually comparable. ``vline``
    draws a labelled reference (e.g. the zero line separating helpful from harmful
    training points).
    """
    plt = _plt()
    finite = [np.asarray(v)[np.isfinite(v)] for v in series.values()]
    pooled = np.concatenate([v for v in finite if len(v)]) if finite else np.array([0.0, 1.0])
    edges = np.histogram_bin_edges(pooled, bins=bins)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in series.items():
        v = np.asarray(values)
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=edges, alpha=0.55, density=True, label=name)
    if vline is not None:
        ax.axvline(
            vline, color="#d1495b", linestyle="--", alpha=0.9, label="zero (harmful | helpful)"
        )
    ax.set(xlabel=xlabel, ylabel="density", title=title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_scatter_identity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> Path:
    """Scatter with a y = x reference line — for predicted-vs-actual validation plots.

    Points on the diagonal mean the prediction matched the ground truth exactly; the
    reference line spans the combined data range so systematic bias reads as a tilt off it.
    """
    plt = _plt()
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xa, ya, s=22, alpha=0.7, color="#3b7dd8")
    finite = np.concatenate([xa[np.isfinite(xa)], ya[np.isfinite(ya)]])
    if len(finite):
        lo, hi = float(finite.min()), float(finite.max())
        ax.plot([lo, hi], [lo, hi], color="#d1495b", linestyle="--", alpha=0.9, label="y = x")
        ax.legend(loc="upper left")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    *,
    title: str,
    out_path: Path,
    cbar_label: str = "value",
) -> Path:
    """Annotated square heatmap (e.g. a feature-interaction matrix). Values in each cell.

    Off-diagonal cells carry the pairwise value; the diagonal is left blank (a feature
    does not interact with itself). Text colour flips on a mid-scale threshold so the
    numbers stay legible on both dark and light cells.
    """
    plt = _plt()
    n = len(labels)
    vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(figsize=(max(5.0, n * 0.95), max(4.0, n * 0.85)))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.55 * vmax else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    return _save(fig, out_path)


def plot_pdp_grid(
    panels: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]],
    *,
    out_path: Path,
    ylabel: str = "attack probability",
    ncols: int = 2,
) -> Path:
    """Small-multiples partial-dependence panels: PDP (bold) over faint ICE curves.

    Each panel is ``(feature, grid_x, pdp_y, ice_matrix|None)`` where ``ice_matrix``
    is ``(n_ice, len(grid))``. The y-axis (attack probability) is shared across panels
    so effect sizes are comparable at a glance.
    """
    plt = _plt()
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False)

    lo, hi = 1.0, 0.0
    for _, _, pdp, ice in panels:
        vals = [np.asarray(pdp, dtype=float)]
        if ice is not None and len(ice):
            vals.append(np.asarray(ice, dtype=float).ravel())
        stacked = np.concatenate(vals)
        lo, hi = min(lo, float(stacked.min())), max(hi, float(stacked.max()))
    pad = max(0.02, (hi - lo) * 0.1)
    ylim = (max(0.0, lo - pad), min(1.0, hi + pad))

    for i, (feature, grid, pdp, ice) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        grid = np.asarray(grid, dtype=float)
        if ice is not None and len(ice):
            for row in np.asarray(ice, dtype=float):
                ax.plot(grid, row, color="#9bb8d3", alpha=0.25, linewidth=0.7)
        ax.plot(grid, np.asarray(pdp, dtype=float), color="#d1495b", linewidth=2.2, label="PDP")
        ax.set(xlabel=feature, ylabel=ylabel, title=feature)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    for j in range(n, nrows * ncols):  # hide unused panels
        axes[j // ncols][j % ncols].axis("off")
    return _save(fig, out_path)


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


def plot_reliability_curve(curves: ScoreCurves, out_path: Path, n_bins: int = 10) -> Path:
    """Reliability diagram: mean predicted probability vs observed frequency.

    A perfectly calibrated score lies on the diagonal; bowing below it means the
    score is over-confident. One line per series (e.g. raw vs calibrated).
    """
    from netsentry.evaluation.calibration import reliability_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    for name, (y_true, scores) in curves.items():
        mean_pred, observed, _ = reliability_curve(y_true, scores, n_bins)
        ax.plot(mean_pred, observed, marker="o", label=name)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed attack frequency",
        title="Reliability diagram",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
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
