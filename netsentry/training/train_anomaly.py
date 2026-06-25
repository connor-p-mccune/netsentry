"""Train benign-only anomaly detectors and evaluate leave-one-attack-out.

The detector trains on benign traffic only and is scored on an attack class it
never saw — the "novel/zero-day" story. We report, per held-out attack, the
detection rate at a benign-calibrated FPR budget and the PR-AUC of the anomaly
score, then a short ensemble comparison (supervised + anomaly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data import schema
from netsentry.data.clean import CLEAN_FILENAME, MULTICLASS_TARGET
from netsentry.data.split import leave_one_attack_out
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.anomaly import build_anomaly_detector
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

ANOMALY_REPORT = "anomaly.md"


def _available_detectors(settings: Settings) -> list[str]:
    kinds = []
    for kind in settings.anomaly.detectors:
        if kind == "autoencoder" and not is_available("torch"):
            logger.warning("Torch not installed; skipping autoencoder detector")
            continue
        kinds.append(kind)
    return kinds


def evaluate_loao(settings: Settings, df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """Leave-one-attack-out detection per detector: {kind: {attack: {metrics}}}."""
    counts = df[MULTICLASS_TARGET].value_counts()
    attacks = [
        a for a in schema.attack_labels() if counts.get(a, 0) >= settings.anomaly.loao_min_samples
    ]
    kinds = _available_detectors(settings)
    results: dict[str, dict[str, dict[str, float]]] = {kind: {} for kind in kinds}

    for attack in attacks:
        split = leave_one_attack_out(df, attack, settings)
        pipeline = build_pipeline(settings)
        x_train = pipeline.fit_transform(split.train)  # benign only
        x_val = pipeline.transform(split.val)  # benign only
        x_test = pipeline.transform(split.test)  # benign + held-out attack
        y_test = (split.test[MULTICLASS_TARGET].to_numpy() == attack).astype(int)

        for kind in kinds:
            detector = build_anomaly_detector(settings, kind).fit(x_train)
            detector.calibrate_threshold(x_val, settings.anomaly.target_fpr)
            scores = detector.score(x_test)
            detection = rates_at_threshold(y_test, scores, detector.threshold)["tpr"]
            results[kind][attack] = {
                "detection_at_fpr": float(detection),
                "pr_auc": float(average_precision_score(y_test, scores)),
            }
        logger.info("LOAO evaluated", extra={"attack": attack, "n_attack": int(counts[attack])})
    return results


def ensemble_comparison(settings: Settings, df: pd.DataFrame) -> dict[str, float]:
    """On the temporal test (novel attacks), compare supervised vs anomaly vs both."""
    result = fit_supervised(settings.model_copy(deep=True))
    sup_scores = positive_scores(result.proba_test, result.classes)
    y_test = result.y_test.astype(int)

    # Anomaly score on the same temporal test, fit on temporal benign train.
    from netsentry.data.split import load_split

    train = load_split(settings, "temporal", "train")
    test = load_split(settings, "temporal", "test")
    benign_train = train[train[MULTICLASS_TARGET] == settings.labels.benign_label]
    pipeline = build_pipeline(settings)
    pipeline.fit(benign_train)
    detector = build_anomaly_detector(settings, "iforest").fit(pipeline.transform(benign_train))
    anomaly_scores = detector.score(pipeline.transform(test))

    def _norm(values: np.ndarray) -> np.ndarray:
        ranks = pd.Series(values).rank().to_numpy()
        return np.asarray(ranks / len(ranks))

    combined = 0.5 * _norm(sup_scores) + 0.5 * _norm(anomaly_scores)
    return {
        "supervised_only": float(average_precision_score(y_test, sup_scores)),
        "anomaly_only": float(average_precision_score(y_test, anomaly_scores)),
        "ensemble": float(average_precision_score(y_test, combined)),
    }


def train_anomaly(settings: Settings) -> dict[str, Any]:
    """Train benign-only detectors, evaluate leave-one-attack-out, write the report."""
    seed_everything(settings.seed)
    clean_path = settings.paths.data_processed / CLEAN_FILENAME
    if not clean_path.exists():
        raise FileNotFoundError(f"{clean_path} not found. Run `netsentry prep` first.")
    df = pd.read_parquet(clean_path)

    loao = evaluate_loao(settings, df)
    try:
        ensemble = ensemble_comparison(settings, df)
    except FileNotFoundError:
        ensemble = {}  # temporal splits not persisted; ensemble section omitted

    report_path = settings.paths.reports_dir / ANOMALY_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render(settings, loao, ensemble), encoding="utf-8")
    logger.info("Wrote anomaly report", extra={"path": str(report_path)})

    with track_run(settings, "anomaly_loao") as run:
        for kind, per_attack in loao.items():
            if per_attack:
                avg = float(np.mean([m["detection_at_fpr"] for m in per_attack.values()]))
                run.log_metrics({f"{kind}_avg_detection": avg})
        if ensemble:
            run.log_metrics({f"ensemble_{k}_pr_auc": v for k, v in ensemble.items()})
        run.log_artifact(report_path)

    return {"loao": loao, "ensemble": ensemble, "report": str(report_path)}


def _render(
    settings: Settings,
    loao: dict[str, dict[str, dict[str, float]]],
    ensemble: dict[str, float],
) -> str:
    fpr_pct = settings.anomaly.target_fpr * 100
    lines = [
        "# NetSentry — Anomaly Detection (novel attacks)",
        "",
        "_Benign-only detectors evaluated **leave-one-attack-out**: trained on benign "
        "traffic, scored on an attack class held out entirely. Synthetic stand-in data "
        "unless run on the real dataset._",
        "",
        f"Detection rate is measured at a **{fpr_pct:.1f}% benign false-positive budget** "
        "(threshold calibrated on a benign validation set).",
        "",
    ]
    for kind, per_attack in loao.items():
        if not per_attack:
            continue
        lines += [
            f"## {kind}",
            "",
            "| held-out attack | detection @ FPR | anomaly PR-AUC |",
            "|---|---|---|",
        ]
        for attack, m in per_attack.items():
            lines.append(f"| {attack} | {m['detection_at_fpr'] * 100:.1f}% | {m['pr_auc']:.3f} |")
        avg_det = np.mean([m["detection_at_fpr"] for m in per_attack.values()])
        avg_ap = np.mean([m["pr_auc"] for m in per_attack.values()])
        lines.append(f"| **average** | **{avg_det * 100:.1f}%** | **{avg_ap:.3f}** |")
        lines.append("")

    if ensemble:
        lines += [
            "## Ensemble — supervised + anomaly on the temporal test",
            "",
            "PR-AUC on later-day (partly novel) attacks:",
            "",
            "| scorer | PR-AUC |",
            "|---|---|",
            f"| supervised only | {ensemble['supervised_only']:.3f} |",
            f"| anomaly only | {ensemble['anomaly_only']:.3f} |",
            f"| **ensemble (rank-mean)** | **{ensemble['ensemble']:.3f}** |",
            "",
            "Combining a supervised classifier (known attacks) with a benign-only "
            "anomaly detector (novel attacks) is the production pattern: neither alone "
            "covers both regimes.",
        ]
    return "\n".join(lines) + "\n"
