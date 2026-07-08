"""Pure-stdlib reader for classic libpcap capture files.

Parses the classic pcap container (both byte orders, microsecond and nanosecond
timestamp variants) and decodes Ethernet / raw-IP link layers down to the TCP/UDP
transport header — everything the CIC flow features need, with no capture-library
dependency. Frames that are not IPv4 TCP/UDP (ARP, IPv6, ICMP, ...) are counted
and skipped, never fatal: a real capture is full of traffic a flow NIDS does not
model, and dropping the reader into a pipeline must not require a clean capture.

The pcapng container is out of scope (``tshark -F pcap`` converts); that
limitation is raised as a clear error rather than a parse failure.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from netsentry.log import get_logger

logger = get_logger(__name__)

# Classic pcap magic numbers. The byte order of the *file* is discovered from
# which variant matches; the "swapped" forms mean the writer's endianness
# differs from the canonical representation.
_MAGIC_USEC = 0xA1B2C3D4
_MAGIC_NSEC = 0xA1B23C4D
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

_GLOBAL_HEADER_LEN = 24
_RECORD_HEADER_LEN = 16

# Link types this reader decodes. Ethernet is the overwhelmingly common case;
# LINKTYPE_RAW captures (no link header, straight into the IP packet) come from
# tunnels and some capture tools.
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101

_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_VLAN = 0x8100

_IPPROTO_TCP = 6
_IPPROTO_UDP = 17

# TCP flag bits (byte 13 of the TCP header).
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20
TCP_ECE = 0x40
TCP_CWR = 0x80


class PcapReadError(ValueError):
    """The file is not a readable classic-pcap capture."""


@dataclass(frozen=True)
class PacketRecord:
    """One decoded IPv4 TCP/UDP packet — the unit the flow assembler consumes.

    ``header_bytes`` is the transport-header length (the CIC ``Fwd/Bwd Header
    Length`` semantics) and ``payload_bytes`` the transport payload, computed
    from the IP total length so a snaplen-truncated capture still yields the
    on-wire sizes.
    """

    timestamp_us: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # "TCP" | "UDP"
    payload_bytes: int
    header_bytes: int
    tcp_flags: int = 0  # raw flag bits; 0 for UDP
    window: int = -1  # TCP receive window; -1 (the CIC 'not set' sentinel) for UDP


@dataclass
class PcapStats:
    """Bookkeeping for one read: how much of the capture the reader understood."""

    packets_total: int = 0
    packets_parsed: int = 0
    skipped_non_ip: int = 0
    skipped_non_tcp_udp: int = 0
    skipped_malformed: int = 0
    truncated: bool = False  # capture ended mid-record (a killed capture session)
    linktype: int = LINKTYPE_ETHERNET
    notes: list[str] = field(default_factory=list)


def _ipv4(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _detect_format(magic_bytes: bytes) -> tuple[str, int]:
    """Return (struct byte-order prefix, timestamp divisor to microseconds)."""
    if magic_bytes == _PCAPNG_MAGIC:
        raise PcapReadError(
            "pcapng container detected; convert to classic pcap first "
            "(e.g. `tshark -F pcap -w out.pcap -r in.pcapng`)."
        )
    for order in ("<", ">"):
        (magic,) = struct.unpack(f"{order}I", magic_bytes)
        if magic == _MAGIC_USEC:
            return order, 1
        if magic == _MAGIC_NSEC:
            return order, 1000
    raise PcapReadError(f"not a pcap file (magic {magic_bytes.hex()})")


def _decode_ethernet(frame: bytes) -> bytes | None:
    """Strip the Ethernet (and one optional 802.1Q VLAN) header; None if not IPv4."""
    if len(frame) < 14:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    offset = 14
    if ethertype == _ETHERTYPE_VLAN:
        if len(frame) < 18:
            return None
        ethertype = struct.unpack("!H", frame[16:18])[0]
        offset = 18
    if ethertype != _ETHERTYPE_IPV4:
        return None
    return frame[offset:]


def _decode_packet(packet: bytes, timestamp_us: int, stats: PcapStats) -> PacketRecord | None:
    """Decode an IPv4 packet into a PacketRecord; count and skip what we can't."""
    if len(packet) < 20:
        stats.skipped_malformed += 1
        return None
    version_ihl = packet[0]
    if version_ihl >> 4 != 4:
        stats.skipped_non_ip += 1
        return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(packet) < ihl:
        stats.skipped_malformed += 1
        return None
    total_length = struct.unpack("!H", packet[2:4])[0]
    protocol = packet[9]
    src_ip = _ipv4(packet[12:16])
    dst_ip = _ipv4(packet[16:20])
    transport = packet[ihl:]

    if protocol == _IPPROTO_TCP:
        if len(transport) < 20:
            stats.skipped_malformed += 1
            return None
        src_port, dst_port = struct.unpack("!HH", transport[:4])
        data_offset = (transport[12] >> 4) * 4
        flags = transport[13]
        window = struct.unpack("!H", transport[14:16])[0]
        payload = max(total_length - ihl - data_offset, 0)
        return PacketRecord(
            timestamp_us=timestamp_us,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol="TCP",
            payload_bytes=payload,
            header_bytes=data_offset,
            tcp_flags=flags,
            window=window,
        )
    if protocol == _IPPROTO_UDP:
        if len(transport) < 8:
            stats.skipped_malformed += 1
            return None
        src_port, dst_port = struct.unpack("!HH", transport[:4])
        payload = max(total_length - ihl - 8, 0)
        return PacketRecord(
            timestamp_us=timestamp_us,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol="UDP",
            payload_bytes=payload,
            header_bytes=8,
        )
    stats.skipped_non_tcp_udp += 1
    return None


def _read_records(
    handle: BinaryIO, order: str, ts_divisor: int, linktype: int, stats: PcapStats
) -> list[PacketRecord]:
    packets: list[PacketRecord] = []
    record_header = struct.Struct(f"{order}IIII")
    while True:
        header = handle.read(_RECORD_HEADER_LEN)
        if not header:
            break
        if len(header) < _RECORD_HEADER_LEN:
            stats.truncated = True
            break
        ts_sec, ts_frac, incl_len, _orig_len = record_header.unpack(header)
        frame = handle.read(incl_len)
        if len(frame) < incl_len:
            stats.truncated = True
            break
        stats.packets_total += 1
        timestamp_us = ts_sec * 1_000_000 + ts_frac // ts_divisor
        if linktype == LINKTYPE_ETHERNET:
            ip_packet = _decode_ethernet(frame)
            if ip_packet is None:
                stats.skipped_non_ip += 1
                continue
        else:  # LINKTYPE_RAW: the frame *is* the IP packet
            ip_packet = frame
        record = _decode_packet(ip_packet, timestamp_us, stats)
        if record is not None:
            stats.packets_parsed += 1
            packets.append(record)
    return packets


def read_pcap(path: Path) -> tuple[list[PacketRecord], PcapStats]:
    """Read a classic pcap file into decoded TCP/UDP packet records.

    Returns the packets in capture order plus read statistics. Raises
    :class:`PcapReadError` only for a file that is not a pcap at all (or is
    pcapng); per-packet problems are counted in the stats and skipped.
    """
    with path.open("rb") as handle:
        magic_bytes = handle.read(4)
        if len(magic_bytes) < 4:
            raise PcapReadError("file too short to be a pcap capture")
        order, ts_divisor = _detect_format(magic_bytes)
        rest = handle.read(_GLOBAL_HEADER_LEN - 4)
        if len(rest) < _GLOBAL_HEADER_LEN - 4:
            raise PcapReadError("truncated pcap global header")
        # version(2+2), thiszone(4), sigfigs(4), snaplen(4), network(4)
        linktype = struct.unpack(f"{order}I", rest[16:20])[0]
        if linktype not in (LINKTYPE_ETHERNET, LINKTYPE_RAW):
            raise PcapReadError(
                f"unsupported link type {linktype}; supported: Ethernet (1), raw IP (101)"
            )
        stats = PcapStats(linktype=linktype)
        packets = _read_records(handle, order, ts_divisor, linktype, stats)
    if stats.truncated:
        stats.notes.append("capture ended mid-record; trailing packet discarded")
    logger.info(
        "Read pcap",
        extra={
            "path": str(path),
            "packets": stats.packets_total,
            "parsed": stats.packets_parsed,
            "skipped_non_ip": stats.skipped_non_ip,
            "skipped_non_tcp_udp": stats.skipped_non_tcp_udp,
        },
    )
    return packets, stats
