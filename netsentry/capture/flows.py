"""Bidirectional flow assembly: decoded packets -> CIC-schema feature rows.

Reimplements the CICFlowMeter aggregation the training data was built with, so a
raw capture can be scored by the same pipeline+model bundle with **zero serving
skew**: the output columns are exactly :data:`netsentry.data.schema.FEATURE_COLUMNS`
(the schema module is the single source of truth), and known departures from
CICFlowMeter are deliberate and documented:

- **Bulk features are emitted as 0.** CICFlowMeter's bulk heuristic produces 0 for
  the overwhelming majority of flows; the model learns nothing from them here.
- **Rates on zero-duration flows are NaN**, mirroring cleaning's Inf->NaN policy
  (the fitted pipeline imputes them with train medians), rather than the raw
  dataset's ``Infinity`` strings.
- **Flow termination** is idle-timeout plus TCP close (RST, or FIN seen in both
  directions); CICFlowMeter additionally splits on an *active* timeout.

Direction is defined by the flow's first packet (its sender is "forward"), which
matches CICFlowMeter. ``Destination Port`` is included in the row: the feature
pipeline drops it from the model per the leakage policy, and the serving layer
uses it only as routing metadata for the ``per_service`` threshold profile.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from netsentry.capture.pcap import TCP_FIN, TCP_RST, PacketRecord
from netsentry.data.schema import FEATURE_COLUMNS
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

# Metadata columns describing each assembled flow (never model features; the
# engine selects only its trained input columns).
FLOW_META_COLUMNS = [
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Start Time (us)",
    "Packets",
]

_FLAG_COLUMNS: dict[str, int] = {
    "FIN Flag Count": 0x01,
    "SYN Flag Count": 0x02,
    "RST Flag Count": 0x04,
    "PSH Flag Count": 0x08,
    "ACK Flag Count": 0x10,
    "URG Flag Count": 0x20,
    "ECE Flag Count": 0x40,
    "CWE Flag Count": 0x80,  # the CIC column name for the CWR bit
}


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    """(max, min, mean, population std) with CIC's 0.0 for empty inputs."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return max(values), min(values), mean, math.sqrt(var)


def _iat(timestamps: list[int]) -> list[float]:
    return [float(b - a) for a, b in itertools.pairwise(timestamps)]


@dataclass
class _Flow:
    """Accumulator for one bidirectional flow in progress."""

    first: PacketRecord
    fwd: list[PacketRecord] = field(default_factory=list)
    bwd: list[PacketRecord] = field(default_factory=list)
    fin_fwd: bool = False
    fin_bwd: bool = False
    closed: bool = False  # RST seen, or FIN in both directions

    @property
    def last_timestamp(self) -> int:
        latest = self.first.timestamp_us
        if self.fwd:
            latest = max(latest, self.fwd[-1].timestamp_us)
        if self.bwd:
            latest = max(latest, self.bwd[-1].timestamp_us)
        return latest

    def add(self, packet: PacketRecord) -> None:
        forward = (packet.src_ip, packet.src_port) == (self.first.src_ip, self.first.src_port)
        (self.fwd if forward else self.bwd).append(packet)
        if packet.tcp_flags & TCP_RST:
            self.closed = True
        if packet.tcp_flags & TCP_FIN:
            if forward:
                self.fin_fwd = True
            else:
                self.fin_bwd = True
            if self.fin_fwd and self.fin_bwd:
                self.closed = True

    def _active_idle(
        self, timestamps: list[int], activity_timeout_us: int
    ) -> tuple[list[float], list[float]]:
        """Split the packet timeline into active periods separated by idle gaps."""
        active: list[float] = []
        idle: list[float] = []
        start = prev = timestamps[0]
        for ts in timestamps[1:]:
            gap = ts - prev
            if gap > activity_timeout_us:
                if prev > start:
                    active.append(float(prev - start))
                idle.append(float(gap))
                start = ts
            prev = ts
        if prev > start:
            active.append(float(prev - start))
        return active, idle

    def to_features(self, activity_timeout_us: int) -> dict[str, float]:
        """Compute the full CIC feature row for this flow."""
        all_packets = sorted(self.fwd + self.bwd, key=lambda p: p.timestamp_us)
        timestamps = [p.timestamp_us for p in all_packets]
        duration = float(timestamps[-1] - timestamps[0])
        duration_s = duration / 1_000_000.0

        fwd_sizes = [float(p.payload_bytes) for p in self.fwd]
        bwd_sizes = [float(p.payload_bytes) for p in self.bwd]
        all_sizes = [float(p.payload_bytes) for p in all_packets]
        fwd_bytes, bwd_bytes = sum(fwd_sizes), sum(bwd_sizes)

        f_max, f_min, f_mean, f_std = _stats(fwd_sizes)
        b_max, b_min, b_mean, b_std = _stats(bwd_sizes)
        a_max, a_min, a_mean, a_std = _stats(all_sizes)

        flow_iat = _iat(timestamps)
        fwd_iat = _iat([p.timestamp_us for p in self.fwd])
        bwd_iat = _iat([p.timestamp_us for p in self.bwd])
        fi_max, fi_min, fi_mean, fi_std = _stats(flow_iat)
        ff_max, ff_min, ff_mean, ff_std = _stats(fwd_iat)
        bb_max, bb_min, bb_mean, bb_std = _stats(bwd_iat)

        # Zero-duration (single-instant) flows have undefined rates; emit NaN so
        # the fitted pipeline imputes them exactly as cleaning treats the raw
        # dataset's Infinity values.
        def rate(total: float) -> float:
            return total / duration_s if duration_s > 0 else float("nan")

        active, idle = self._active_idle(timestamps, activity_timeout_us)
        act_max, act_min, act_mean, act_std = _stats(active)
        idl_max, idl_min, idl_mean, idl_std = _stats(idle)

        tcp = self.first.protocol == "TCP"
        init_win_fwd = float(self.fwd[0].window) if tcp and self.fwd else -1.0
        init_win_bwd = float(self.bwd[0].window) if tcp and self.bwd else -1.0

        row: dict[str, float] = {
            "Destination Port": float(self.first.dst_port),
            "Flow Duration": duration,
            "Total Fwd Packets": float(len(self.fwd)),
            "Total Backward Packets": float(len(self.bwd)),
            "Total Length of Fwd Packets": fwd_bytes,
            "Total Length of Bwd Packets": bwd_bytes,
            "Fwd Packet Length Max": f_max,
            "Fwd Packet Length Min": f_min,
            "Fwd Packet Length Mean": f_mean,
            "Fwd Packet Length Std": f_std,
            "Bwd Packet Length Max": b_max,
            "Bwd Packet Length Min": b_min,
            "Bwd Packet Length Mean": b_mean,
            "Bwd Packet Length Std": b_std,
            "Flow Bytes/s": rate(fwd_bytes + bwd_bytes),
            "Flow Packets/s": rate(float(len(all_packets))),
            "Flow IAT Mean": fi_mean,
            "Flow IAT Std": fi_std,
            "Flow IAT Max": fi_max,
            "Flow IAT Min": fi_min,
            "Fwd IAT Total": float(sum(fwd_iat)),
            "Fwd IAT Mean": ff_mean,
            "Fwd IAT Std": ff_std,
            "Fwd IAT Max": ff_max,
            "Fwd IAT Min": ff_min,
            "Bwd IAT Total": float(sum(bwd_iat)),
            "Bwd IAT Mean": bb_mean,
            "Bwd IAT Std": bb_std,
            "Bwd IAT Max": bb_max,
            "Bwd IAT Min": bb_min,
            "Fwd PSH Flags": float(sum(1 for p in self.fwd if p.tcp_flags & 0x08)),
            "Bwd PSH Flags": float(sum(1 for p in self.bwd if p.tcp_flags & 0x08)),
            "Fwd URG Flags": float(sum(1 for p in self.fwd if p.tcp_flags & 0x20)),
            "Bwd URG Flags": float(sum(1 for p in self.bwd if p.tcp_flags & 0x20)),
            "Fwd Header Length": float(sum(p.header_bytes for p in self.fwd)),
            "Bwd Header Length": float(sum(p.header_bytes for p in self.bwd)),
            "Fwd Packets/s": rate(float(len(self.fwd))),
            "Bwd Packets/s": rate(float(len(self.bwd))),
            "Min Packet Length": a_min,
            "Max Packet Length": a_max,
            "Packet Length Mean": a_mean,
            "Packet Length Std": a_std,
            "Packet Length Variance": a_std**2,
            "Down/Up Ratio": float(len(self.bwd)) / len(self.fwd) if self.fwd else 0.0,
            "Average Packet Size": a_mean,
            "Avg Fwd Segment Size": f_mean,
            "Avg Bwd Segment Size": b_mean,
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": float(len(self.fwd)),
            "Subflow Fwd Bytes": fwd_bytes,
            "Subflow Bwd Packets": float(len(self.bwd)),
            "Subflow Bwd Bytes": bwd_bytes,
            "Init_Win_bytes_forward": init_win_fwd,
            "Init_Win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": float(sum(1 for p in self.fwd if p.payload_bytes > 0)),
            "min_seg_size_forward": float(min((p.header_bytes for p in self.fwd), default=0)),
            "Active Mean": act_mean,
            "Active Std": act_std,
            "Active Max": act_max,
            "Active Min": act_min,
            "Idle Mean": idl_mean,
            "Idle Std": idl_std,
            "Idle Max": idl_max,
            "Idle Min": idl_min,
        }
        for column, bit in _FLAG_COLUMNS.items():
            row[column] = float(sum(1 for p in all_packets if p.tcp_flags & bit))
        missing = set(FEATURE_COLUMNS) - set(row)
        assert not missing, f"flow features missing schema columns: {sorted(missing)}"
        return row

    def metadata(self) -> dict[str, object]:
        """Human-facing flow identity for the scored output (never model input)."""
        return {
            "Src IP": self.first.src_ip,
            "Src Port": self.first.src_port,
            "Dst IP": self.first.dst_ip,
            "Dst Port": self.first.dst_port,
            "Protocol": self.first.protocol,
            "Start Time (us)": self.first.timestamp_us,
            "Packets": len(self.fwd) + len(self.bwd),
        }


def _flow_key(p: PacketRecord) -> tuple[str, tuple[tuple[str, int], tuple[str, int]]]:
    """Direction-agnostic 5-tuple key: both directions land on one flow."""
    endpoints = tuple(sorted([(p.src_ip, p.src_port), (p.dst_ip, p.dst_port)]))
    return p.protocol, endpoints  # type: ignore[return-value]


class FlowAssembler:
    """Groups decoded packets into bidirectional flows with timeout/close splitting."""

    def __init__(self, *, flow_timeout_us: int, activity_timeout_us: int) -> None:
        self.flow_timeout_us = flow_timeout_us
        self.activity_timeout_us = activity_timeout_us
        self._active: dict[object, _Flow] = {}
        self._done: list[_Flow] = []

    def add(self, packet: PacketRecord) -> None:
        key = _flow_key(packet)
        flow = self._active.get(key)
        if flow is not None:
            expired = packet.timestamp_us - flow.last_timestamp > self.flow_timeout_us
            if expired or flow.closed:
                self._done.append(flow)
                flow = None
        if flow is None:
            flow = _Flow(first=packet)
            self._active[key] = flow
        flow.add(packet)

    def flows(self) -> list[_Flow]:
        """All completed and still-open flows, in first-packet order."""
        everything = self._done + list(self._active.values())
        return sorted(everything, key=lambda f: f.first.timestamp_us)


def extract_flows(
    packets: list[PacketRecord], settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble packets into flows; return (feature frame, metadata frame).

    The feature frame carries exactly the canonical CIC columns (in schema
    order); the metadata frame identifies each flow for the human reading the
    scored output. Row *i* of each describes the same flow.
    """
    assembler = FlowAssembler(
        flow_timeout_us=settings.capture.flow_timeout_us,
        activity_timeout_us=settings.capture.activity_timeout_us,
    )
    for packet in sorted(packets, key=lambda p: p.timestamp_us):
        assembler.add(packet)
    flows = assembler.flows()
    features = pd.DataFrame(
        [f.to_features(settings.capture.activity_timeout_us) for f in flows],
        columns=list(FEATURE_COLUMNS),
    )
    meta = pd.DataFrame([f.metadata() for f in flows], columns=FLOW_META_COLUMNS)
    logger.info("Assembled flows", extra={"packets": len(packets), "flows": len(flows)})
    return features, meta
