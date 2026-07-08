"""Packet-to-verdict scoring: a PCAP in, a predictions file out.

Chains the capture stack (parse -> assemble -> CIC features) into the same
:class:`~netsentry.serving.inference.InferenceEngine` the API uses, so a raw
capture gets the identical calibrated, explained, threshold-profiled verdicts a
posted flow record would — the flow identity (IPs, ports, protocol) rides along
as output metadata but never enters the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.capture.flows import FLOW_META_COLUMNS, extract_flows
from netsentry.capture.pcap import read_pcap
from netsentry.log import get_logger
from netsentry.serving.batch import OUTPUT_COLUMNS, score_dataframe
from netsentry.serving.inference import InferenceEngine

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def score_capture(
    settings: Settings,
    pcap_path: Path,
    output_path: Path,
    *,
    flows_out: Path | None = None,
    profile: str | None = None,
) -> dict[str, int]:
    """Score a packet capture end-to-end; return summary counts.

    Writes one row per assembled flow: the flow's identity metadata followed by
    the full prediction contract. ``flows_out`` optionally persists the raw
    extracted feature rows (metadata + CIC columns) for inspection or replay
    through ``netsentry score``.
    """
    packets, stats = read_pcap(pcap_path)
    features, meta = extract_flows(packets, settings)
    if flows_out is not None:
        _write(pd.concat([meta, features], axis=1), flows_out)

    if features.empty:
        result = pd.DataFrame(columns=FLOW_META_COLUMNS + OUTPUT_COLUMNS)
    else:
        engine = InferenceEngine(settings)
        predictions = score_dataframe(
            engine, features, profile=profile, batch_size=settings.serving.max_batch_size
        )
        result = pd.concat([meta.reset_index(drop=True), predictions], axis=1)
    _write(result, output_path)

    summary = {
        "packets": stats.packets_total,
        "parsed": stats.packets_parsed,
        "flows": len(result),
        "flagged": int(result["is_attack"].sum()) if len(result) else 0,
    }
    logger.info("Scored capture", extra={**summary, "output": str(output_path)})
    return summary
