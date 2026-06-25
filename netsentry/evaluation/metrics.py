"""Operational metrics: PR-AUC, ROC-AUC, per-class P/R/F1, TPR@fixed-FPR, alerts/day.

Accuracy is deliberately never a headline metric here (an all-benign predictor
scores ~80% and catches nothing). The operating point is the important one: pick
a decision threshold on the **validation** set at a target false-positive budget,
then report the true-positive (detection) rate at that threshold on **test** —
framed the way a SOC would ("at a 0.1% FP budget, we detect N% of attacks").
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def positive_scores(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Extract the attack-class probability column from a proba matrix."""
    classes = np.asarray(classes)
    matches = np.where(classes == 1)[0]
    pos = int(matches[0]) if len(matches) else proba.shape[1] - 1
    return np.asarray(proba[:, pos])


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Highest score threshold whose FPR does not exceed ``target_fpr``.

    Choosing the highest such threshold yields the best detection rate that still
    respects the false-positive budget. Computed on the set passed in (use the
    validation set to pick an operating point, then apply it to test).
    """
    fpr, _tpr, thresholds = roc_curve(y_true, scores)
    within_budget = np.where(fpr <= target_fpr)[0]
    if len(within_budget) == 0:
        return float(thresholds[0])
    return float(thresholds[within_budget[-1]])


def rates_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Confusion-derived rates when predicting positive iff score >= threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"tpr": tpr, "fpr": fpr, "precision": precision, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Return (threshold, TPR) at the largest FPR not exceeding ``target_fpr``."""
    threshold = threshold_at_fpr(y_true, scores, target_fpr)
    tpr = rates_at_threshold(y_true, scores, threshold)["tpr"]
    return threshold, tpr


def alerts_per_day(fpr: float, flows_per_day: int, benign_fraction: float = 1.0) -> float:
    """Rough false-alert volume per day at a given FPR — the alert-fatigue lens."""
    return float(fpr * flows_per_day * benign_fraction)


def operating_point(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    target_fpr: float,
    flows_per_day: int,
) -> dict[str, float]:
    """Pick a threshold on validation at ``target_fpr``; evaluate it on test."""
    threshold = threshold_at_fpr(y_val, scores_val, target_fpr)
    rates = rates_at_threshold(y_test, scores_test, threshold)
    benign_fraction = float(np.mean(np.asarray(y_test) == 0))
    return {
        "target_fpr": target_fpr,
        "threshold": threshold,
        "tpr": rates["tpr"],
        "fpr": rates["fpr"],
        "precision": rates["precision"],
        "alerts_per_day": alerts_per_day(rates["fpr"], flows_per_day, benign_fraction),
    }


def binary_summary(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """PR-AUC (primary) and ROC-AUC for a binary attack/benign score."""
    y_true = np.asarray(y_true)
    summary = {"pr_auc": float(average_precision_score(y_true, scores))}
    # ROC-AUC is undefined if only one class is present in y_true.
    if len(np.unique(y_true)) > 1:
        summary["roc_auc"] = float(roc_auc_score(y_true, scores))
    return summary


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-class precision/recall/F1/support, plus macro and weighted F1."""
    if labels is None:
        labels = sorted({str(v) for v in np.unique(np.concatenate([y_true, y_pred]))})
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        report[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report["__macro__"] = {
        "precision": float(macro[0]),
        "recall": float(macro[1]),
        "f1": float(macro[2]),
    }
    report["__weighted__"] = {
        "precision": float(weighted[0]),
        "recall": float(weighted[1]),
        "f1": float(weighted[2]),
    }
    return report


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> np.ndarray:
    """Confusion matrix with rows=true, cols=pred, ordered by ``labels``."""
    return np.asarray(confusion_matrix(y_true, y_pred, labels=labels))
