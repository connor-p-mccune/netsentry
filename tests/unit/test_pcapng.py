"""pcapng reader: blocks built field-by-field so the parser is checked against
known on-wire values, not against itself (the same discipline as the classic
pcap tests)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from netsentry.capture.demo import ethernet_ipv4_packet, pcap_bytes
from netsentry.capture.pcap import (
    TCP_SYN,
    PcapReadError,
    read_capture,
    read_pcap,
    read_pcapng,
)

_BOM = 0x1A2B3C4D


def _block(block_type: int, body: bytes, order: str = "<") -> bytes:
    total = 12 + len(body)
    return struct.pack(f"{order}II", block_type, total) + body + struct.pack(f"{order}I", total)


def _shb(order: str = "<") -> bytes:
    body = struct.pack(f"{order}IHHq", _BOM, 1, 0, -1)  # BOM, v1.0, unknown section length
    return _block(0x0A0D0D0A, body, order)


def _idb(order: str = "<", linktype: int = 1, tsresol: int | None = None) -> bytes:
    body = struct.pack(f"{order}HHI", linktype, 0, 65535)  # linktype, reserved, snaplen
    if tsresol is not None:
        body += struct.pack(f"{order}HH", 9, 1) + bytes([tsresol]) + b"\x00\x00\x00"
        body += struct.pack(f"{order}HH", 0, 0)  # opt_endofopt
    return _block(0x00000001, body, order)


def _epb(frame: bytes, ticks: int, order: str = "<", iface: int = 0) -> bytes:
    body = struct.pack(
        f"{order}IIIII", iface, ticks >> 32, ticks & 0xFFFFFFFF, len(frame), len(frame)
    )
    body += frame + b"\x00" * ((-len(frame)) % 4)
    return _block(0x00000006, body, order)


def _spb(frame: bytes, order: str = "<") -> bytes:
    body = struct.pack(f"{order}I", len(frame)) + frame + b"\x00" * ((-len(frame)) % 4)
    return _block(0x00000003, body, order)


def _frame(**kw: object) -> bytes:
    base: dict[str, object] = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80)
    base.update(kw)
    return ethernet_ipv4_packet(**base)  # type: ignore[arg-type]


def _write(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "t.pcapng"
    path.write_bytes(payload)
    return path


def test_reads_enhanced_packet_block_fields(tmp_path: Path) -> None:
    frame = _frame(payload_len=100, tcp_flags=TCP_SYN, window=4321)
    path = _write(tmp_path, _shb() + _idb() + _epb(frame, ticks=1_500_000))
    packets, stats = read_pcapng(path)
    assert stats.packets_total == stats.packets_parsed == 1
    (p,) = packets
    assert (p.src_ip, p.dst_ip, p.src_port, p.dst_port) == ("10.0.0.1", "10.0.0.2", 1234, 80)
    assert p.timestamp_us == 1_500_000  # default if_tsresol is microseconds
    assert p.payload_bytes == 100
    assert p.tcp_flags == TCP_SYN
    assert p.window == 4321


def test_big_endian_section_parses(tmp_path: Path) -> None:
    frame = _frame(payload_len=7)
    path = _write(tmp_path, _shb(">") + _idb(">") + _epb(frame, ticks=42, order=">"))
    (p,), stats = read_pcapng(path)
    assert p.timestamp_us == 42
    assert p.payload_bytes == 7
    assert not stats.truncated


def test_if_tsresol_converts_to_microseconds(tmp_path: Path) -> None:
    frame = _frame(payload_len=1)
    # 10^-9 (nanoseconds): 1_500_000_000 ticks -> 1_500_000 us.
    nano = _write(tmp_path, _shb() + _idb(tsresol=9) + _epb(frame, ticks=1_500_000_000))
    (p,), _ = read_pcapng(nano)
    assert p.timestamp_us == 1_500_000
    # 2^-10 (MSB set, v=10): 2048 ticks = 2 seconds.
    two = tmp_path / "two.pcapng"
    two.write_bytes(_shb() + _idb(tsresol=0x80 | 10) + _epb(frame, ticks=2048))
    (p2,), _ = read_pcapng(two)
    assert p2.timestamp_us == 2_000_000


def test_unknown_blocks_are_skipped_by_length(tmp_path: Path) -> None:
    frame = _frame(payload_len=3)
    name_resolution = _block(0x00000004, b"\x00" * 8)  # NRB: metadata, not packets
    custom = _block(0x0BAD0BAD, b"\x00" * 16)
    path = _write(tmp_path, _shb() + name_resolution + _idb() + custom + _epb(frame, ticks=5))
    packets, stats = read_pcapng(path)
    assert len(packets) == 1
    assert stats.packets_parsed == 1


def test_unsupported_interface_skips_its_packets_only(tmp_path: Path) -> None:
    frame = _frame(payload_len=2)
    wifi_idb = _idb(linktype=105)  # IEEE 802.11: not decoded
    payload = _shb() + _idb() + wifi_idb + _epb(frame, ticks=1, iface=1) + _epb(frame, ticks=2)
    packets, stats = read_pcapng(_write(tmp_path, payload))
    assert len(packets) == 1  # only the Ethernet interface's packet
    assert stats.skipped_unsupported_link == 1
    assert any("unsupported link type 105" in n for n in stats.notes)


def test_simple_packet_block_parses_without_timestamp(tmp_path: Path) -> None:
    frame = _frame(payload_len=9)
    packets, stats = read_pcapng(_write(tmp_path, _shb() + _idb() + _spb(frame)))
    (p,) = packets
    assert p.timestamp_us == 0
    assert p.payload_bytes == 9
    assert any("no timestamps" in n for n in stats.notes)


def test_new_section_resets_interface_numbering(tmp_path: Path) -> None:
    frame = _frame(payload_len=4)
    # Two concatenated sections, each with its own IDB 0; both packets must parse.
    payload = _shb() + _idb() + _epb(frame, ticks=1) + _shb() + _idb() + _epb(frame, ticks=2)
    packets, _ = read_pcapng(_write(tmp_path, payload))
    assert len(packets) == 2


def test_truncated_capture_sets_flag_not_crash(tmp_path: Path) -> None:
    frame = _frame(payload_len=1)
    whole = _shb() + _idb() + _epb(frame, ticks=1)
    packets, stats = read_pcapng(_write(tmp_path, whole[:-6]))  # cut mid-block
    assert stats.truncated
    assert packets == []  # the cut block is discarded, not misparsed


def test_read_capture_dispatches_on_magic(tmp_path: Path) -> None:
    frame = _frame(payload_len=5)
    ng = _write(tmp_path, _shb() + _idb() + _epb(frame, ticks=7))
    classic = tmp_path / "c.pcap"
    classic.write_bytes(pcap_bytes([(7, frame)]))
    for path in (ng, classic):
        (p,), _ = read_capture(path)
        assert p.timestamp_us == 7
    garbage = tmp_path / "g.bin"
    garbage.write_bytes(b"\x00" * 64)
    with pytest.raises(PcapReadError):
        read_capture(garbage)


def test_classic_reader_still_names_pcapng_clearly(tmp_path: Path) -> None:
    path = _write(tmp_path, _shb() + _idb())
    with pytest.raises(PcapReadError, match="pcapng"):
        read_pcap(path)
