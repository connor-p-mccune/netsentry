"""Capture stack: pcap parsing, flow assembly, and CIC feature fidelity.

The demo builders in :mod:`netsentry.capture.demo` double as the ground truth
here: frames are constructed field-by-field, so every assertion below checks the
parser and assembler against known on-wire values, not against themselves.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.capture.demo import ethernet_ipv4_packet, pcap_bytes, write_demo_pcap
from netsentry.capture.flows import FLOW_META_COLUMNS, FlowAssembler, extract_flows
from netsentry.capture.pcap import (
    TCP_ACK,
    TCP_FIN,
    TCP_RST,
    TCP_SYN,
    PacketRecord,
    PcapReadError,
    read_pcap,
)
from netsentry.config import Settings
from netsentry.data.schema import FEATURE_COLUMNS


def _write(tmp_path: Path, records: list[tuple[int, bytes]], name: str = "t.pcap") -> Path:
    path = tmp_path / name
    path.write_bytes(pcap_bytes(records))
    return path


def _tcp(ts: int = 0, **kw: object) -> tuple[int, bytes]:
    base: dict[str, object] = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80)
    base.update(kw)
    return ts, ethernet_ipv4_packet(**base)  # type: ignore[arg-type]


# --- pcap reader --------------------------------------------------------------


def test_reads_tcp_packet_fields(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [_tcp(ts=1_500_000, payload_len=100, tcp_flags=TCP_SYN | TCP_ACK, window=4321)],
    )
    packets, stats = read_pcap(path)
    assert stats.packets_total == stats.packets_parsed == 1
    (p,) = packets
    assert (p.src_ip, p.dst_ip, p.src_port, p.dst_port) == ("10.0.0.1", "10.0.0.2", 1234, 80)
    assert p.protocol == "TCP"
    assert p.timestamp_us == 1_500_000
    assert p.payload_bytes == 100
    assert p.header_bytes == 20
    assert p.tcp_flags == TCP_SYN | TCP_ACK
    assert p.window == 4321


def test_reads_udp_packet(tmp_path: Path) -> None:
    path = _write(tmp_path, [_tcp(protocol="UDP", dst_port=53, payload_len=48)])
    (p,), _ = read_pcap(path)
    assert p.protocol == "UDP"
    assert p.payload_bytes == 48
    assert p.header_bytes == 8
    assert p.window == -1  # the CIC 'not set' sentinel for non-TCP


def test_big_endian_and_nanosecond_variants(tmp_path: Path) -> None:
    _, frame = _tcp(payload_len=10)
    # Big-endian microsecond header; one record with ts = 2s + 7us.
    header = struct.pack(">IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    record = struct.pack(">IIII", 2, 7, len(frame), len(frame)) + frame
    big = tmp_path / "big.pcap"
    big.write_bytes(header + record)
    (p,), _ = read_pcap(big)
    assert p.timestamp_us == 2_000_007

    # Little-endian nanosecond magic: fractional part is divided down to us.
    header = struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, 1)
    record = struct.pack("<IIII", 2, 7_000, len(frame), len(frame)) + frame
    nsec = tmp_path / "nsec.pcap"
    nsec.write_bytes(header + record)
    (p,), _ = read_pcap(nsec)
    assert p.timestamp_us == 2_000_007


def test_non_ip_frames_are_counted_and_skipped(tmp_path: Path) -> None:
    arp = b"\x02" * 6 + b"\x04" * 6 + struct.pack("!H", 0x0806) + bytes(28)
    _, good = _tcp()
    path = _write(tmp_path, [(0, arp), (10, good)])
    packets, stats = read_pcap(path)
    assert len(packets) == 1
    assert stats.skipped_non_ip == 1


def test_truncated_final_record_is_tolerated(tmp_path: Path) -> None:
    path = _write(tmp_path, [_tcp(ts=0), _tcp(ts=10)])
    data = path.read_bytes()
    path.write_bytes(data[:-8])  # kill the capture mid-record
    packets, stats = read_pcap(path)
    assert len(packets) == 1
    assert stats.truncated


def test_rejects_non_pcap_and_pcapng(tmp_path: Path) -> None:
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00\x01\x02\x03" + bytes(40))
    with pytest.raises(PcapReadError, match="not a pcap"):
        read_pcap(junk)
    ng = tmp_path / "ng.pcapng"
    ng.write_bytes(b"\x0a\x0d\x0d\x0a" + bytes(40))
    with pytest.raises(PcapReadError, match="pcapng"):
        read_pcap(ng)


# --- flow assembly ------------------------------------------------------------


def _record(
    ts: int,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    sport: int = 1000,
    dport: int = 80,
    payload: int = 0,
    flags: int = TCP_ACK,
    window: int = 8192,
) -> PacketRecord:
    return PacketRecord(
        timestamp_us=ts,
        src_ip=src,
        dst_ip=dst,
        src_port=sport,
        dst_port=dport,
        protocol="TCP",
        payload_bytes=payload,
        header_bytes=20,
        tcp_flags=flags,
        window=window,
    )


def _session_features(settings: Settings) -> pd.Series:
    packets = [
        _record(0, flags=TCP_SYN, window=1000, payload=0),
        _record(100, src="10.0.0.2", dst="10.0.0.1", sport=80, dport=1000, window=2000),
        _record(200, payload=300),
        _record(400, src="10.0.0.2", dst="10.0.0.1", sport=80, dport=1000, payload=900),
    ]
    features, meta = extract_flows(packets, settings)
    assert len(features) == 1
    assert meta.loc[0, "Src IP"] == "10.0.0.1"  # direction = first packet's sender
    return features.iloc[0]


def test_bidirectional_flow_features(settings: Settings) -> None:
    row = _session_features(settings)
    assert row["Destination Port"] == 80
    assert row["Flow Duration"] == 400
    assert row["Total Fwd Packets"] == 2
    assert row["Total Backward Packets"] == 2
    assert row["Total Length of Fwd Packets"] == 300
    assert row["Total Length of Bwd Packets"] == 900
    assert row["SYN Flag Count"] == 1
    assert row["Init_Win_bytes_forward"] == 1000
    assert row["Init_Win_bytes_backward"] == 2000
    assert row["act_data_pkt_fwd"] == 1  # only one forward packet carried payload
    assert row["Flow IAT Mean"] == pytest.approx((100 + 100 + 200) / 3)
    assert row["Down/Up Ratio"] == 1.0
    # 1200 payload bytes over 400us
    assert row["Flow Bytes/s"] == pytest.approx(1200 / 400e-6)


def test_emits_every_schema_column_with_no_extras(settings: Settings) -> None:
    row = _session_features(settings)
    assert list(row.index) == list(FEATURE_COLUMNS)


def test_zero_duration_flow_rates_are_nan_not_inf(settings: Settings) -> None:
    features, _ = extract_flows([_record(0, flags=TCP_SYN)], settings)
    row = features.iloc[0]
    assert math.isnan(row["Flow Bytes/s"])
    assert math.isnan(row["Flow Packets/s"])
    assert not np.isinf(features.to_numpy(dtype=float)).any()


def test_idle_gap_splits_active_periods(settings: Settings) -> None:
    settings.capture.activity_timeout_us = 1_000
    packets = [_record(0), _record(500), _record(10_000), _record(10_400)]
    features, _ = extract_flows(packets, settings)
    row = features.iloc[0]
    assert row["Idle Mean"] == 9_500
    assert row["Active Mean"] == pytest.approx((500 + 400) / 2)


def test_flow_timeout_starts_a_new_flow(settings: Settings) -> None:
    settings.capture.flow_timeout_us = 1_000_000
    assembler = FlowAssembler(flow_timeout_us=1_000_000, activity_timeout_us=5_000_000)
    assembler.add(_record(0))
    assembler.add(_record(2_000_001))  # > timeout after the first
    assert len(assembler.flows()) == 2


def test_tcp_close_then_reuse_starts_a_new_flow(settings: Settings) -> None:
    assembler = FlowAssembler(flow_timeout_us=120_000_000, activity_timeout_us=5_000_000)
    assembler.add(_record(0, flags=TCP_RST))
    assembler.add(_record(100))  # same 5-tuple, after RST
    assert len(assembler.flows()) == 2


def test_fin_both_directions_closes_flow() -> None:
    assembler = FlowAssembler(flow_timeout_us=120_000_000, activity_timeout_us=5_000_000)
    assembler.add(_record(0, flags=TCP_FIN | TCP_ACK))
    assembler.add(
        _record(100, src="10.0.0.2", dst="10.0.0.1", sport=80, dport=1000, flags=TCP_FIN | TCP_ACK)
    )
    assembler.add(_record(200))
    assert len(assembler.flows()) == 2


# --- demo capture end-to-end (extraction only; scoring is an integration test) -


def test_demo_capture_extracts_recognisable_shapes(tmp_path: Path, settings: Settings) -> None:
    path = write_demo_pcap(tmp_path / "demo.pcap", seed=7)
    packets, stats = read_pcap(path)
    assert stats.packets_parsed == stats.packets_total > 400
    features, meta = extract_flows(packets, settings)
    assert list(meta.columns) == FLOW_META_COLUMNS
    assert len(features) > 40  # 20 sessions + 40 sweep probes + flood + DNS

    # The SYN sweep appears as many single-SYN flows from the scanning source.
    sweep = meta["Src IP"] == "203.0.113.66"
    assert sweep.sum() == 40
    assert (features.loc[sweep, "SYN Flag Count"] == 1).all()

    # The flood is one flow with a high packet rate.
    flood = features[meta["Src IP"] == "198.51.100.7"]
    assert len(flood) == 1
    assert flood.iloc[0]["Flow Packets/s"] > 100
