"""Batch file scoring: score a CSV/Parquet of flows to a predictions file.

The same `InferenceEngine` the API uses, driven over a file so the model is usable
without standing up the service — the common "score yesterday's flows" workflow.
Non-feature columns in the input (labels, identifiers) are ignored; missing feature
columns are imputed by the fitted pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.log import get_logger
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.serving.schemas import PredictionResponse

logger = get_logger(__name__)

OUTPUT_COLUMNS = [
    "predicted_class",
    "is_attack",
    "attack_probability",
    "anomaly_score",
    "is_anomaly",
    "recommended_action",
    "mitre_technique",
    "top_feature",
]


def _to_row(r: PredictionResponse) -> dict[str, object]:
    return {
        "predicted_class": r.predicted_class,
        "is_attack": r.is_attack,
        "attack_probability": round(r.attack_probability, 6),
        "anomaly_score": None if r.anomaly_score is None else round(r.anomaly_score, 6),
        "is_anomaly": r.is_anomaly,
        "recommended_action": r.recommended_action,
        "mitre_technique": r.mitre["technique_id"] if r.mitre else None,
        "top_feature": r.top_features[0].feature if r.top_features else None,
    }


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def score_dataframe(
    engine: InferenceEngine, df: pd.DataFrame, *, profile: str | None, batch_size: int
) -> pd.DataFrame:
    """Score every row and return a predictions frame (one row per input flow)."""
    flows = df.to_dict("records")
    rows: list[dict[str, object]] = []
    for start in range(0, len(flows), batch_size):
        chunk = flows[start : start + batch_size]
        rows.extend(_to_row(r) for r in engine.predict(chunk, profile=profile))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def score_file(
    settings: Settings,
    input_path: Path,
    output_path: Path,
    *,
    profile: str | None = None,
    batch_size: int | None = None,
) -> dict[str, int]:
    """Score a flow file end-to-end; write predictions and return summary counts."""
    engine = InferenceEngine(settings)
    df = _read(input_path)
    predictions = score_dataframe(
        engine, df, profile=profile, batch_size=batch_size or settings.serving.max_batch_size
    )
    _write(predictions, output_path)
    stats = {"scored": len(predictions), "flagged": int(predictions["is_attack"].sum())}
    logger.info("Scored flow file", extra={**stats, "output": str(output_path)})
    return stats
