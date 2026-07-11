"""Zeek conn.log ingestion: score the logs a network team already has.

Zeek (formerly Bro) is the de-facto open-source network security monitor, and
`conn.log` — one record per connection — is its most widely deployed output.
This adapter maps conn.log records (classic tab-separated logs with `#fields`
headers, or JSON-lines from the json-streaming writers; sniffed automatically)
into the CIC feature schema and scores them through the same
:class:`~netsentry.serving.inference.InferenceEngine` the API uses, so "point
NetSentry at your Zeek logs" is one command rather than a feature-engineering
project.

The honesty that comes with it, stated up front: conn.log carries connection
totals (duration, bytes, packets per direction) but none of the intra-flow
detail many CIC features encode (inter-arrival timing, per-packet sizes, TCP
window/header fields), so only the volume/rate/shape subset is mapped — the
rest are left missing and **imputed with training medians by the fitted
pipeline**, exactly the regime the cross-dataset study measures. Expect its
finding to apply here too: the score *ranking* transfers through the shared
behavioural features, but a fixed-FPR operating point calibrated on full CIC
features degrades — re-choose thresholds on labeled local traffic (the
threshold-refresh machinery) before trusting a budget. Flag counts derived from
Zeek's `history` string are lower bounds (history records event letters, not
reliable counts) and are documented as such.

Flow identity (endpoints, ports, protocol, the Zeek UID for pivoting back into
other Zeek logs) rides along as output metadata and never enters the model —
the same contract as the pcap path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.log import get_logger
from netsentry.serving.batch import OUTPUT_COLUMNS, score_dataframe
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

# Identity/pivot metadata carried beside the verdicts (never model input).
ZEEK_META_COLUMNS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Zeek UID"]

_UNSET = "-"


class ZeekReadError(ValueError):
    """The file is not a readable Zeek conn.log."""


def read_conn_log(path: Path) -> list[dict[str, str]]:
    """Parse a Zeek conn.log (TSV with ``#fields`` header, or JSON lines).

    Returns one string-valued mapping per record; unset TSV fields (``-`` by
    default) are dropped so downstream numeric parsing treats them as missing.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return _read_json_lines(text)
    if stripped.startswith("#"):
        return _read_tsv(text)
    raise ZeekReadError(
        f"{path.name} does not look like a Zeek log (expected '#'-headed TSV or JSON lines)"
    )


def _read_json_lines(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        records.append({str(k): str(v) for k, v in obj.items() if v is not None})
    return records


def _read_tsv(text: str) -> list[dict[str, str]]:
    separator = "\t"
    unset = _UNSET
    fields: list[str] | None = None
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("#separator"):
            # e.g. "#separator \x09" — the value is space-separated and escaped.
            value = line.split(" ", 1)[1].strip()
            separator = value.encode().decode("unicode_escape")
        elif line.startswith("#unset_field"):
            unset = line.split(separator, 1)[1].strip()
        elif line.startswith("#fields"):
            fields = line.split(separator)[1:]
        elif line.startswith("#"):
            continue  # path/open/close/types/empty_field metadata
        elif line.strip():
            if fields is None:
                raise ZeekReadError("data row before any #fields header")
            values = line.split(separator)
            records.append(
                {name: value for name, value in zip(fields, values, strict=False) if value != unset}
            )
    if fields is None:
        raise ZeekReadError("no #fields header found")
    return records


def _num(record: dict[str, str], key: str) -> float:
    value = record.get(key)
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _rate(numerator: float, seconds: float) -> float:
    """Per-second rate with the cleaning module's division policy: 0-duration -> NaN."""
    if not np.isfinite(numerator) or not np.isfinite(seconds) or seconds <= 0:
        return float("nan")
    return numerator / seconds


def _mean(total: float, count: float) -> float:
    """CICFlowMeter's convention: a mean over zero packets is 0, not undefined."""
    if not np.isfinite(total) or not np.isfinite(count):
        return float("nan")
    return total / count if count > 0 else 0.0


def zeek_record_to_cic(record: dict[str, str]) -> dict[str, float]:
    """Map one conn.log record onto the CIC columns it can honestly speak for.

    fwd = originator, bwd = responder. Only volume/rate/shape features are
    computable from connection totals; everything else stays missing for the
    fitted pipeline to impute. ``history``-derived flag counts are lower bounds.
    """
    duration_s = _num(record, "duration")
    fwd_pkts = _num(record, "orig_pkts")
    bwd_pkts = _num(record, "resp_pkts")
    fwd_bytes = _num(record, "orig_bytes")
    bwd_bytes = _num(record, "resp_bytes")
    total_pkts = fwd_pkts + bwd_pkts
    total_bytes = fwd_bytes + bwd_bytes
    history = record.get("history", "")

    fwd_mean = _mean(fwd_bytes, fwd_pkts)
    bwd_mean = _mean(bwd_bytes, bwd_pkts)
    row: dict[str, float] = {
        "Destination Port": _num(record, "id.resp_p"),
        "Flow Duration": duration_s * 1_000_000 if np.isfinite(duration_s) else float("nan"),
        "Total Fwd Packets": fwd_pkts,
        "Total Backward Packets": bwd_pkts,
        "Total Length of Fwd Packets": fwd_bytes,
        "Total Length of Bwd Packets": bwd_bytes,
        "Fwd Packet Length Mean": fwd_mean,
        "Bwd Packet Length Mean": bwd_mean,
        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,
        "Flow Bytes/s": _rate(total_bytes, duration_s),
        "Flow Packets/s": _rate(total_pkts, duration_s),
        "Fwd Packets/s": _rate(fwd_pkts, duration_s),
        "Bwd Packets/s": _rate(bwd_pkts, duration_s),
        "Packet Length Mean": _mean(total_bytes, total_pkts),
        "Average Packet Size": _mean(total_bytes, total_pkts),
        "Down/Up Ratio": (bwd_pkts / fwd_pkts if np.isfinite(fwd_pkts) and fwd_pkts > 0 else 0.0),
        "Subflow Fwd Packets": fwd_pkts,
        "Subflow Fwd Bytes": fwd_bytes,
        "Subflow Bwd Packets": bwd_pkts,
        "Subflow Bwd Bytes": bwd_bytes,
        # history letters record connection events; counts are lower bounds.
        "SYN Flag Count": float(history.lower().count("s")),
        "FIN Flag Count": float(history.lower().count("f")),
        "RST Flag Count": float(history.lower().count("r")),
    }
    return {k: v for k, v in row.items() if np.isfinite(v)}


def zeek_to_cic(records: list[dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, metadata) frames for a batch of conn.log records.

    The features frame carries every canonical CIC column (unmapped -> NaN, so
    the fitted pipeline's train-median imputation applies); metadata carries the
    flow identity for the output file, never for the model.
    """
    feature_rows = [
        {col: row.get(col, float("nan")) for col in schema.FEATURE_COLUMNS}
        for row in (zeek_record_to_cic(r) for r in records)
    ]
    meta_rows = [
        {
            "Src IP": r.get("id.orig_h", ""),
            "Src Port": r.get("id.orig_p", ""),
            "Dst IP": r.get("id.resp_h", ""),
            "Dst Port": r.get("id.resp_p", ""),
            "Protocol": r.get("proto", ""),
            "Zeek UID": r.get("uid", ""),
        }
        for r in records
    ]
    features = pd.DataFrame(feature_rows, columns=list(schema.FEATURE_COLUMNS))
    meta = pd.DataFrame(meta_rows, columns=ZEEK_META_COLUMNS)
    return features, meta


def mapped_feature_count(features: pd.DataFrame) -> int:
    """How many CIC columns the log could actually speak for (any non-NaN value)."""
    return int(features.notna().any(axis=0).sum())


def score_zeek_log(
    settings: Settings,
    input_path: Path,
    output_path: Path,
    *,
    flows_out: Path | None = None,
    profile: str | None = None,
) -> dict[str, int]:
    """conn.log in, verdicts out — through the identical engine the API serves."""
    records = read_conn_log(input_path)
    features, meta = zeek_to_cic(records)
    if flows_out is not None:
        _write(pd.concat([meta, features], axis=1), flows_out)

    if features.empty:
        result = pd.DataFrame(columns=ZEEK_META_COLUMNS + OUTPUT_COLUMNS)
        flagged = 0
    else:
        engine = InferenceEngine(settings)
        predictions = score_dataframe(
            engine, features, profile=profile, batch_size=settings.serving.max_batch_size
        )
        result = pd.concat([meta.reset_index(drop=True), predictions], axis=1)
        flagged = int(predictions["is_attack"].sum())
    _write(result, output_path)

    mapped = mapped_feature_count(features)
    stats = {"connections": len(records), "flagged": flagged, "mapped_features": mapped}
    logger.info(
        "Scored Zeek conn.log",
        extra={
            **stats,
            "total_features": len(schema.FEATURE_COLUMNS),
            "note": "unmapped features imputed from training medians; "
            "re-choose thresholds on labeled local traffic before trusting a budget",
        },
    )
    return stats


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
