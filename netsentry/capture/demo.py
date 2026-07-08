"""Synthetic capture writer: a small, deterministic pcap for demos and tests.

Builds a classic-pcap byte stream from scratch (Ethernet/IPv4/TCP+UDP via
``struct``) containing recognisable traffic shapes — ordinary HTTP/DNS sessions,
a SYN port sweep, and a high-rate flood — so the packet-to-verdict path can be
exercised end-to-end with no capture hardware, no real traffic, and no binary
fixture checked into the repo. The same builders serve as the test harness for
the reader (they are the ground truth the parser is asserted against).
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

from netsentry.capture.pcap import (
    LINKTYPE_ETHERNET,
    TCP_ACK,
    TCP_FIN,
    TCP_PSH,
    TCP_SYN,
)
from netsentry.log import get_logger

logger = get_logger(__name__)

_GLOBAL_HEADER = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET)


def _ip_bytes(dotted: str) -> bytes:
    return bytes(int(part) for part in dotted.split("."))


def ethernet_ipv4_packet(
    *,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    protocol: str = "TCP",
    payload_len: int = 0,
    tcp_flags: int = TCP_ACK,
    window: int = 8192,
) -> bytes:
    """One Ethernet/IPv4/TCP-or-UDP frame with a zero-filled payload."""
    if protocol == "TCP":
        transport = struct.pack(
            "!HHIIBBHHH", src_port, dst_port, 0, 0, (5 << 4), tcp_flags, window, 0, 0
        )
    elif protocol == "UDP":
        transport = struct.pack("!HHHH", src_port, dst_port, 8 + payload_len, 0)
    else:
        raise ValueError(f"unsupported protocol {protocol!r}")
    total_length = 20 + len(transport) + payload_len
    proto_num = 6 if protocol == "TCP" else 17
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        (4 << 4) | 5,
        0,
        total_length,
        0,
        0,
        64,
        proto_num,
        0,
        _ip_bytes(src_ip),
        _ip_bytes(dst_ip),
    )
    ethernet = b"\x02" * 6 + b"\x04" * 6 + struct.pack("!H", 0x0800)
    return ethernet + ip_header + transport + bytes(payload_len)


def pcap_record(timestamp_us: int, frame: bytes) -> bytes:
    """A classic-pcap record header + frame (little-endian, microsecond format)."""
    return (
        struct.pack(
            "<IIII", timestamp_us // 1_000_000, timestamp_us % 1_000_000, len(frame), len(frame)
        )
        + frame
    )


def pcap_bytes(records: list[tuple[int, bytes]]) -> bytes:
    """A complete classic-pcap file from (timestamp_us, frame) pairs."""
    return _GLOBAL_HEADER + b"".join(pcap_record(ts, frame) for ts, frame in records)


def _tcp_session(
    rng: random.Random,
    start_us: int,
    client: str,
    server: str,
    client_port: int,
    server_port: int,
    exchanges: int,
) -> list[tuple[int, bytes]]:
    """A plausible request/response TCP session: handshake, data both ways, close."""

    def pkt(src: str, dst: str, sport: int, dport: int, **kw: int) -> bytes:
        return ethernet_ipv4_packet(
            src_ip=src, dst_ip=dst, src_port=sport, dst_port=dport, **kw  # type: ignore[arg-type]
        )

    ts = start_us
    frames = [
        (ts, pkt(client, server, client_port, server_port, tcp_flags=TCP_SYN)),
        (ts + 200, pkt(server, client, server_port, client_port, tcp_flags=TCP_SYN | TCP_ACK)),
        (ts + 400, pkt(client, server, client_port, server_port, tcp_flags=TCP_ACK)),
    ]
    ts += 400
    for _ in range(exchanges):
        ts += rng.randint(2_000, 60_000)
        request = rng.randint(80, 600)
        response = rng.randint(400, 6000)
        frames.append(
            (
                ts,
                pkt(
                    client,
                    server,
                    client_port,
                    server_port,
                    payload_len=request,
                    tcp_flags=TCP_PSH | TCP_ACK,
                ),
            )
        )
        frames.append(
            (
                ts + rng.randint(500, 4_000),
                pkt(
                    server,
                    client,
                    server_port,
                    client_port,
                    payload_len=response,
                    tcp_flags=TCP_PSH | TCP_ACK,
                ),
            )
        )
    ts += rng.randint(2_000, 30_000)
    frames.append((ts, pkt(client, server, client_port, server_port, tcp_flags=TCP_FIN | TCP_ACK)))
    frames.append(
        (ts + 300, pkt(server, client, server_port, client_port, tcp_flags=TCP_FIN | TCP_ACK))
    )
    return frames


def write_demo_pcap(path: Path, *, seed: int = 42) -> Path:
    """Write the deterministic demo capture; return its path.

    Contents: ~20 benign web/DNS sessions, a 40-port SYN sweep from one source
    (PortScan-shaped), and a one-second high-rate flood at a web server
    (DoS-shaped) — enough texture for the extractor, the model, and the demo to
    have something worth disagreeing about.
    """
    rng = random.Random(seed)
    records: list[tuple[int, bytes]] = []
    ts = 1_000_000

    # Benign browsing sessions and a few DNS lookups.
    for i in range(20):
        client = f"10.0.0.{10 + i}"
        records.extend(
            _tcp_session(
                rng,
                ts + rng.randint(0, 3_000_000),
                client,
                "93.184.216.34",
                49_000 + i,
                443 if i % 3 else 80,
                exchanges=rng.randint(2, 6),
            )
        )
        if i % 4 == 0:
            query_ts = ts + rng.randint(0, 3_000_000)
            records.append(
                (
                    query_ts,
                    ethernet_ipv4_packet(
                        src_ip=client,
                        dst_ip="10.0.0.1",
                        src_port=50_000 + i,
                        dst_port=53,
                        protocol="UDP",
                        payload_len=rng.randint(30, 60),
                    ),
                )
            )
            records.append(
                (
                    query_ts + rng.randint(300, 2_000),
                    ethernet_ipv4_packet(
                        src_ip="10.0.0.1",
                        dst_ip=client,
                        src_port=53,
                        dst_port=50_000 + i,
                        protocol="UDP",
                        payload_len=rng.randint(60, 200),
                    ),
                )
            )

    # A SYN sweep: one source probing 40 ports, one tiny packet each.
    sweep_start = ts + 4_000_000
    for port in range(1, 41):
        records.append(
            (
                sweep_start + port * 1_500,
                ethernet_ipv4_packet(
                    src_ip="203.0.113.66",
                    dst_ip="10.0.0.5",
                    src_port=61_000,
                    dst_port=port,
                    tcp_flags=TCP_SYN,
                    window=1024,
                ),
            )
        )

    # A flood: 400 packets in ~1s at one web server from one source.
    flood_start = ts + 6_000_000
    for i in range(400):
        records.append(
            (
                flood_start + i * 2_500,
                ethernet_ipv4_packet(
                    src_ip="198.51.100.7",
                    dst_ip="10.0.0.8",
                    src_port=40_000,
                    dst_port=80,
                    payload_len=1200,
                    tcp_flags=TCP_PSH | TCP_ACK,
                ),
            )
        )

    records.sort(key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcap_bytes(records))
    logger.info("Wrote demo capture", extra={"path": str(path), "packets": len(records)})
    return path
