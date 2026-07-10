"""Pure-stdlib readers for packet captures: classic libpcap and pcapng.

Parses both capture containers — classic pcap (both byte orders, microsecond and
nanosecond timestamp variants) and pcapng (section/interface/packet blocks, both
byte orders, per-interface ``if_tsresol`` timestamp resolution, multi-section
files) — and decodes Ethernet / raw-IP link layers down to the TCP/UDP transport
header: everything the CIC flow features need, with no capture-library
dependency. ``read_capture`` sniffs the container by magic and dispatches.

Frames that are not IPv4 TCP/UDP (ARP, IPv6, ICMP, ...) are counted and skipped,
never fatal: a real capture is full of traffic a flow NIDS does not model, and
dropping the reader into a pipeline must not require a clean capture. The same
posture holds at the pcapng block level — unknown block types (name resolution,
statistics, custom) are skipped by their declared length, and packets on an
interface with an unsupported link type are counted rather than aborting the
read.
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
# The pcapng Section Header Block type is a byte-order palindrome on purpose, so
# it reads the same before the byte-order magic inside the block is known.
_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"
_PCAPNG_BOM = 0x1A2B3C4D

_GLOBAL_HEADER_LEN = 24
_RECORD_HEADER_LEN = 16

# pcapng block types (the ones with packet-relevant content; others are skipped).
_BLOCK_IDB = 0x00000001  # Interface Description
_BLOCK_OBSOLETE_PACKET = 0x00000002  # pre-standard Packet Block (skipped, noted)
_BLOCK_SPB = 0x00000003  # Simple Packet (no timestamp, no interface options)
_BLOCK_EPB = 0x00000006  # Enhanced Packet
_OPT_ENDOFOPT = 0
_OPT_IF_TSRESOL = 9
_DEFAULT_TICKS_PER_SECOND = 1_000_000  # pcapng default resolution: microseconds

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
    skipped_unsupported_link: int = 0  # pcapng: packets on a non-Ethernet/raw interface
    truncated: bool = False  # capture ended mid-record (a killed capture session)
    linktype: int = LINKTYPE_ETHERNET
    notes: list[str] = field(default_factory=list)


def _ipv4(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _detect_format(magic_bytes: bytes) -> tuple[str, int]:
    """Return (struct byte-order prefix, timestamp divisor to microseconds)."""
    if magic_bytes == _PCAPNG_MAGIC:
        raise PcapReadError(
            "pcapng container: this is the classic-pcap reader — "
            "use read_pcapng() or read_capture() (the CLI dispatches automatically)."
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


def _decode_frame(
    frame: bytes, linktype: int, timestamp_us: int, stats: PcapStats
) -> PacketRecord | None:
    """Link-layer dispatch shared by both container readers (counts in ``stats``)."""
    stats.packets_total += 1
    if linktype == LINKTYPE_ETHERNET:
        ip_packet = _decode_ethernet(frame)
        if ip_packet is None:
            stats.skipped_non_ip += 1
            return None
    else:  # LINKTYPE_RAW: the frame *is* the IP packet
        ip_packet = frame
    record = _decode_packet(ip_packet, timestamp_us, stats)
    if record is not None:
        stats.packets_parsed += 1
    return record


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
        timestamp_us = ts_sec * 1_000_000 + ts_frac // ts_divisor
        record = _decode_frame(frame, linktype, timestamp_us, stats)
        if record is not None:
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


def _if_tsresol_ticks(options: bytes, order: str) -> int:
    """Ticks-per-second from an IDB options blob (pcapng default: microseconds).

    ``if_tsresol`` is one byte: MSB clear means 10^-v seconds per tick, MSB set
    means 2^-v — both converted here to a ticks-per-second frequency so the
    packet loop can do one integer division to microseconds.
    """
    offset = 0
    while offset + 4 <= len(options):
        code, olen = struct.unpack(f"{order}HH", options[offset : offset + 4])
        offset += 4
        if code == _OPT_ENDOFOPT:
            break
        value = options[offset : offset + olen]
        offset += (olen + 3) & ~3  # option values pad to 32-bit boundaries
        if code == _OPT_IF_TSRESOL and len(value) >= 1:
            v = value[0]
            return int(2 ** (v & 0x7F)) if v & 0x80 else int(10 ** (v & 0x7F))
    return _DEFAULT_TICKS_PER_SECOND


def _pcapng_section_order(bom_bytes: bytes) -> str:
    """Byte order of one pcapng section, from the SHB's byte-order magic."""
    if struct.unpack("<I", bom_bytes)[0] == _PCAPNG_BOM:
        return "<"
    if struct.unpack(">I", bom_bytes)[0] == _PCAPNG_BOM:
        return ">"
    raise PcapReadError(f"pcapng section header with unreadable byte-order magic {bom_bytes.hex()}")


def read_pcapng(path: Path) -> tuple[list[PacketRecord], PcapStats]:
    """Read a pcapng capture into decoded TCP/UDP packet records.

    Handles Section Header / Interface Description / Enhanced Packet / Simple
    Packet blocks in either byte order, per-interface ``if_tsresol`` timestamp
    resolution, and concatenated sections (interface numbering resets per
    section, as the spec requires). Unknown block types are skipped by their
    declared length; packets on interfaces with an unsupported link type are
    counted, not fatal.
    """
    packets: list[PacketRecord] = []
    stats = PcapStats(linktype=LINKTYPE_ETHERNET)
    order = "<"
    interfaces: list[tuple[int, int]] = []  # per current section: (linktype, ticks/s)
    noted: set[str] = set()

    def note(message: str) -> None:
        if message not in noted:
            noted.add(message)
            stats.notes.append(message)

    with path.open("rb") as handle:
        if handle.read(4) != _PCAPNG_MAGIC:
            raise PcapReadError("not a pcapng capture")
        handle.seek(0)
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) < 8:
                stats.truncated = True
                break
            if header[:4] == _PCAPNG_MAGIC:  # a (new) section begins
                bom_bytes = handle.read(4)
                if len(bom_bytes) < 4:
                    stats.truncated = True
                    break
                order = _pcapng_section_order(bom_bytes)
                total_len = struct.unpack(f"{order}I", header[4:8])[0]
                # SHB minimum: type+len+BOM+version+section-length+trailing len = 28.
                if total_len < 28 or total_len % 4:
                    raise PcapReadError(f"corrupt pcapng section header (len {total_len})")
                rest = handle.read(total_len - 12)
                if len(rest) < total_len - 12:
                    stats.truncated = True
                    break
                interfaces = []  # interface IDs are section-scoped
                continue
            block_type, total_len = struct.unpack(f"{order}II", header)
            if total_len < 12 or total_len % 4:
                raise PcapReadError(f"corrupt pcapng block (type {block_type:#x}, len {total_len})")
            body = handle.read(total_len - 8)
            if len(body) < total_len - 8:
                stats.truncated = True
                break
            payload = body[:-4]  # strip the trailing block-length copy

            if block_type == _BLOCK_IDB and len(payload) >= 8:
                linktype = struct.unpack(f"{order}H", payload[0:2])[0]
                interfaces.append((linktype, _if_tsresol_ticks(payload[8:], order)))
                if linktype not in (LINKTYPE_ETHERNET, LINKTYPE_RAW):
                    note(f"interface {len(interfaces) - 1}: unsupported link type {linktype}")
            elif block_type == _BLOCK_EPB and len(payload) >= 20:
                iface, ts_high, ts_low, cap_len, _orig = struct.unpack(
                    f"{order}IIIII", payload[:20]
                )
                if iface >= len(interfaces):
                    stats.skipped_malformed += 1
                    note("packet block references an undeclared interface")
                    continue
                linktype, ticks = interfaces[iface]
                if linktype not in (LINKTYPE_ETHERNET, LINKTYPE_RAW):
                    stats.skipped_unsupported_link += 1
                    continue
                frame = payload[20 : 20 + cap_len]
                timestamp_us = ((ts_high << 32) | ts_low) * 1_000_000 // ticks
                record = _decode_frame(frame, linktype, timestamp_us, stats)
                if record is not None:
                    packets.append(record)
            elif block_type == _BLOCK_SPB and len(payload) >= 4:
                if not interfaces:
                    stats.skipped_malformed += 1
                    note("simple packet block before any interface description")
                    continue
                linktype, _ticks = interfaces[0]  # SPB is defined against interface 0
                if linktype not in (LINKTYPE_ETHERNET, LINKTYPE_RAW):
                    stats.skipped_unsupported_link += 1
                    continue
                (orig_len,) = struct.unpack(f"{order}I", payload[:4])
                frame = payload[4 : 4 + min(orig_len, len(payload) - 4)]
                note("simple packet blocks carry no timestamps; time features degrade")
                record = _decode_frame(frame, linktype, 0, stats)
                if record is not None:
                    packets.append(record)
            elif block_type == _BLOCK_OBSOLETE_PACKET:
                stats.skipped_malformed += 1
                note("obsolete (pre-standard) packet block skipped")
            # All other block types (name resolution, statistics, custom) are
            # metadata: skipped by their declared length, already consumed above.

    if stats.truncated:
        stats.notes.append("capture ended mid-block; trailing data discarded")
    if interfaces:
        stats.linktype = interfaces[0][0]
    logger.info(
        "Read pcapng",
        extra={
            "path": str(path),
            "packets": stats.packets_total,
            "parsed": stats.packets_parsed,
            "skipped_non_ip": stats.skipped_non_ip,
            "skipped_unsupported_link": stats.skipped_unsupported_link,
        },
    )
    return packets, stats


def read_capture(path: Path) -> tuple[list[PacketRecord], PcapStats]:
    """Read a capture in either container, sniffed by magic (pcap or pcapng)."""
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == _PCAPNG_MAGIC:
        return read_pcapng(path)
    return read_pcap(path)
