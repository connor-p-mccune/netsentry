"""Beaconing / C2 periodicity detection over connection timelines.

The supervised classifier scores each flow **in isolation**, and by design drops
every identifier (IPs, ports, timestamp) so it cannot memorise them. That makes it
blind to a whole class of malicious behaviour that is invisible in one flow and
obvious across many: **beaconing** — a compromised host calling home to a
command-and-control server on a regular cadence (ATT&CK Command and Control,
T1071). No single callback looks anomalous; the *regularity of the schedule* is the
tell.

``netsentry beacon`` fills that gap with an unsupervised, identity-aware analytic —
the cross-flow complement to the per-flow model, the way the signature ruleset is
its interpretable complement. It groups connections by talker pair (``Src IP`` ->
``Dst IP``, optionally per destination port), and for each pair with enough events
scores the **regularity** of the inter-arrival times: a robust dispersion
(median-absolute-deviation over the median interval) turned into a 0..1 score where
1.0 is a perfectly periodic beacon and 0.0 is bursty, human-looking traffic.

The honest scoping, stated in the report: this is a **hunt lead generator, not a
verdict**. A legitimate periodic service — NTP, a monitoring poll, a cron job, a
software update check — is also regular and will score high; the analytic surfaces
*candidates* ranked by regularity for a human to triage, and adds no detection to
the model's per-flow verdicts. It consumes the timestamp/identity columns as
metadata only — exactly the fields the model is forbidden to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "beacon_report.md"
DEMO_REPORT_NAME = "beacon_demo.md"

_SRC_IP, _DST_IP, _DST_PORT = "Src IP", "Dst IP", "Dst Port"


def inter_arrival_times(timestamps: np.ndarray) -> np.ndarray:
    """Sorted, non-negative gaps (seconds) between consecutive connection times."""
    ordered = np.sort(np.asarray(timestamps, dtype=float))
    return np.diff(ordered)


def regularity_score(iats: np.ndarray) -> float:
    """Beacon regularity in ``[0, 1]`` from inter-arrival-time dispersion.

    Uses a robust dispersion — median absolute deviation over the median interval —
    so a handful of jittered or missed callbacks does not tank an otherwise-periodic
    schedule the way the standard-deviation-based coefficient of variation would. A
    perfectly periodic series has MAD 0 and scores 1.0; bursty human traffic has
    dispersion >= 1 and scores 0.0.
    """
    iats = np.asarray(iats, dtype=float)
    if iats.size == 0:
        return 0.0
    median = float(np.median(iats))
    if median <= 0:
        return 0.0
    mad = float(np.median(np.abs(iats - median)))
    return float(np.clip(1.0 - mad / median, 0.0, 1.0))


def coefficient_of_variation(iats: np.ndarray) -> float:
    """Std/mean of the inter-arrival times (reported alongside the robust score)."""
    iats = np.asarray(iats, dtype=float)
    if iats.size == 0:
        return 0.0
    mean = float(np.mean(iats))
    return float(np.std(iats) / mean) if mean > 0 else 0.0


@dataclass
class BeaconCandidate:
    """One talker pair scored for beacon-like periodicity."""

    source: str
    destination: str
    port: int | None
    n_connections: int
    median_interval_s: float
    cv: float
    score: float

    @property
    def pair(self) -> str:
        dst = f"{self.destination}:{self.port}" if self.port is not None else self.destination
        return f"{self.source} -> {dst}"


def _to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Coerce a timestamp column to epoch seconds (numeric passthrough or parsed)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:  # already epoch-like
        epoch: np.ndarray = numeric.to_numpy(dtype=float)
        return epoch
    parsed = pd.to_datetime(series, errors="coerce")
    result: np.ndarray = parsed.astype("int64").to_numpy() / 1e9
    return result


def detect_beacons(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    min_events: int,
    by_port: bool,
) -> list[BeaconCandidate]:
    """Rank talker pairs by beacon-like regularity of their connection times.

    Pairs with fewer than ``min_events`` connections are skipped — periodicity is
    not judgeable from a handful of points. Returns candidates sorted by score
    (descending), then connection count.
    """
    for column in (_SRC_IP, _DST_IP, timestamp_column):
        if column not in df.columns:
            raise ValueError(f"beacon detection needs a {column!r} column")

    frame = df[[_SRC_IP, _DST_IP, timestamp_column]].copy()
    if by_port and _DST_PORT in df.columns:
        frame[_DST_PORT] = df[_DST_PORT]
    frame["_epoch"] = _to_epoch_seconds(df[timestamp_column])
    frame = frame[np.isfinite(frame["_epoch"])]

    group_keys = [_SRC_IP, _DST_IP] + (
        [_DST_PORT] if by_port and _DST_PORT in frame.columns else []
    )
    candidates: list[BeaconCandidate] = []
    for key, block in frame.groupby(group_keys):
        times = block["_epoch"].to_numpy(dtype=float)
        if times.size < min_events:
            continue
        iats = inter_arrival_times(times)
        keys = key if isinstance(key, tuple) else (key,)
        port = int(keys[2]) if len(keys) > 2 else None
        candidates.append(
            BeaconCandidate(
                source=str(keys[0]),
                destination=str(keys[1]),
                port=port,
                n_connections=int(times.size),
                median_interval_s=float(np.median(iats)),
                cv=coefficient_of_variation(iats),
                score=regularity_score(iats),
            )
        )
    candidates.sort(key=lambda c: (c.score, c.n_connections), reverse=True)
    return candidates


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _format_interval(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def render_report(
    candidates: list[BeaconCandidate], *, threshold: float, top_n: int, demo: bool
) -> str:
    """Render the ranked beacon candidates and the honest scoping note."""
    flagged = [c for c in candidates if c.score >= threshold]
    rows = ["| talker pair | connections | period | regularity | CV |", "|---|---|---|---|---|"]
    for c in candidates[:top_n]:
        mark = " **[flag]**" if c.score >= threshold else ""
        rows.append(
            f"| `{c.pair}`{mark} | {c.n_connections} | {_format_interval(c.median_interval_s)} "
            f"| {c.score:.3f} | {c.cv:.2f} |"
        )

    demo_note = ""
    if demo and flagged:
        top = flagged[0]
        demo_note = (
            f"\nThe synthetic capture plants a single periodic beacon; the detector "
            f"ranks it first (`{top.pair}`, regularity {top.score:.3f}, period "
            f"{_format_interval(top.median_interval_s)}) above the jittery benign "
            f"talkers — the mechanic, on data with a known answer.\n"
        )

    return f"""# NetSentry — Beaconing / C2 Periodicity{" (synthetic demo)" if demo else ""}

The classifier scores each flow in isolation and drops every identifier, so it is
blind to **beaconing** — a host calling a command-and-control server on a fixed
cadence (MITRE ATT&CK **Command and Control**, e.g. T1071 Application Layer
Protocol). No single callback looks anomalous; the regularity of the schedule is
the tell. This analytic scores each talker pair's connection-time regularity, the
cross-flow complement to the per-flow model.

**Candidates:** {len(candidates)} talker pair(s) with enough events; **{len(flagged)}**
scored at or above the {threshold:.2f} regularity flag line.
{demo_note}
{chr(10).join(rows)}

## How to read this

Regularity is a robust dispersion of the inter-arrival times (median absolute
deviation over the median interval), in `[0, 1]`: **1.0 is a perfectly periodic
beacon, 0.0 is bursty human traffic.** The coefficient of variation (std/mean of
the intervals) is shown alongside — a beacon has CV near zero.

This is a **hunt lead generator, not a verdict.** A legitimate periodic service —
NTP, a monitoring poll, a cron job, a software-update check — is also regular and
will score high; the analytic surfaces *candidates* ranked by regularity for a
human to triage, and adds no detection to the model's per-flow verdicts. It reads
the timestamp and identity columns as metadata only — exactly the fields the model
is forbidden to see — which is why it can catch what the model, by construction,
cannot.
"""


def synthesize_beacon_flows(seed: int, *, horizon_s: float = 6 * 3600.0) -> pd.DataFrame:
    """Deterministic capture: jittery benign talkers plus one planted periodic beacon.

    Used by ``--demo`` and the tests — a ground-truth harness with a known answer,
    no model or dataset required (the analytic is pure timing).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    # Benign talkers: bursty, human-like arrivals (exponential gaps, high dispersion).
    for host in range(6):
        src = f"10.0.0.{10 + host}"
        dst = f"93.184.216.{34 + host}"
        port = int(rng.choice([80, 443]))
        t = float(rng.uniform(0, 600))
        while t < horizon_s:
            rows.append({"Src IP": src, "Dst IP": dst, "Dst Port": port, "_t": t})
            t += float(rng.exponential(400.0)) + 5.0

    # Planted beacon: a fixed 60 s cadence with small jitter to a non-standard port.
    beacon_src, beacon_dst, beacon_port, period = "10.0.0.7", "45.77.12.9", 8443, 60.0
    t = 30.0
    while t < horizon_s:
        jitter = float(rng.normal(0.0, 1.5))
        rows.append(
            {"Src IP": beacon_src, "Dst IP": beacon_dst, "Dst Port": beacon_port, "_t": t + jitter}
        )
        t += period

    frame = pd.DataFrame(rows)
    # Turn the relative offsets into an absolute epoch timeline, then shuffle rows so
    # the detector must recover order from the timestamp, not the row layout.
    frame["Timestamp"] = 1_500_000_000.0 + frame["_t"]
    frame = frame.drop(columns="_t")
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_beacon_report(
    settings: Settings,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    demo: bool = False,
) -> Path:
    """Detect beacons in a flow file (or the synthetic demo) and write the report."""
    cfg = settings.beacon
    if demo:
        df = synthesize_beacon_flows(settings.seed)
        default_out = settings.paths.reports_dir / DEMO_REPORT_NAME
    elif input_path is not None:
        df = _read(input_path)
        default_out = settings.paths.reports_dir / REPORT_NAME
    else:
        raise ValueError("provide --input a flow file or use --demo")

    candidates = detect_beacons(
        df,
        timestamp_column=cfg.timestamp_column,
        min_events=cfg.min_events,
        by_port=cfg.by_port,
    )
    report = render_report(candidates, threshold=cfg.score_threshold, top_n=cfg.top_n, demo=demo)
    out_path = output_path or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info(
        "Wrote beacon report",
        extra={
            "path": str(out_path),
            "candidates": len(candidates),
            "flagged": sum(c.score >= cfg.score_threshold for c in candidates),
        },
    )
    return out_path
