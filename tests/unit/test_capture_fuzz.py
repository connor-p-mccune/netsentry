"""Fuzzing the untrusted-input capture parser: the never-crash / count-and-skip contract.

A parser of attacker-supplied binary — a ``.pcap`` dropped on a sensor, or a capture
uploaded to be scored — is a classic memory-safety and denial-of-service surface: a
crafted length field or a truncated record is exactly how such a parser is made to
crash, hang, or over-allocate. NetSentry's reader is pure-stdlib and documents a
"skip-don't-die" posture; these tests *assert* it with Hypothesis.

The contract: on **arbitrary bytes** the parser must either return a
``(records, stats)`` pair or raise the single typed, controlled error
(:class:`PcapReadError`) — never an uncaught ``struct.error`` / ``IndexError``, never
an unbounded allocation, never a hang. The fuzzer drives three regimes: free-form
bytes, a valid container magic followed by garbage (so the deep header/record/block
decoders take the malformed input), and structured mutations of a *valid* capture
(truncations and byte flips, reaching the packet-decode paths a bare-random fuzzer
almost never hits). Inputs are size-bounded, so an O(n) parse over a file handle is
guaranteed to terminate.
"""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from netsentry.capture.demo import write_demo_pcap
from netsentry.capture.pcap import PcapReadError, PcapStats, read_capture

# Fuzzing does file I/O + binary parsing per example, so it lives in the slow tier
# (run by `make test` / CI), keeping the fast dev loop snappy — it is a thoroughness
# guard, not a per-edit check.
pytestmark = pytest.mark.slow

# Classic pcap (both byte orders) + pcapng container magics — a correct magic gets past
# the format sniff, so the bytes after it reach the deep decoders.
_MAGICS = [b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a"]

_FUZZ = settings(
    max_examples=200,
    deadline=None,  # per-example wall-clock varies with disk I/O; the size bound is the guard
    suppress_health_check=[HealthCheck.too_slow],
)

# One reused scratch file for every fuzzed example (creating/deleting a temp file per
# example dominates the runtime on Windows; the parser reads from a path).
_fd, _scratch_name = tempfile.mkstemp(suffix=".pcap")
os.close(_fd)
_SCRATCH = Path(_scratch_name)
atexit.register(lambda: _SCRATCH.unlink(missing_ok=True))


def _assert_contract(data: bytes) -> None:
    """The parser returns a well-typed result or raises only PcapReadError."""
    _SCRATCH.write_bytes(data)
    try:
        records, stats = read_capture(_SCRATCH)
    except PcapReadError:
        return  # the single permitted failure mode
    assert isinstance(records, list)
    assert isinstance(stats, PcapStats)


def _demo_capture_bytes() -> bytes:
    """Bytes of a valid Ethernet demo capture, to mutate toward the decode paths."""
    fd, name = tempfile.mkstemp(suffix=".pcap")
    os.close(fd)
    path = Path(name)
    try:
        write_demo_pcap(path, seed=1)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


_VALID = _demo_capture_bytes()


@_FUZZ
@given(data=st.binary(max_size=1024))
def test_arbitrary_bytes_never_crash(data: bytes) -> None:
    _assert_contract(data)


@_FUZZ
@given(prefix=st.sampled_from(_MAGICS), body=st.binary(max_size=512))
def test_valid_magic_plus_garbage_never_crash(prefix: bytes, body: bytes) -> None:
    _assert_contract(prefix + body)


@_FUZZ
@given(payload=st.data())
def test_mutated_valid_capture_never_crash(payload: st.DataObject) -> None:
    buf = bytearray(_VALID)
    # Truncate at an arbitrary offset (short reads mid-header/record).
    buf = buf[: payload.draw(st.integers(min_value=0, max_value=len(buf)))]
    # Flip a handful of bytes (corrupt lengths, link types, offsets).
    if buf:
        for _ in range(payload.draw(st.integers(min_value=0, max_value=16))):
            i = payload.draw(st.integers(min_value=0, max_value=len(buf) - 1))
            buf[i] = payload.draw(st.integers(min_value=0, max_value=255))
    _assert_contract(bytes(buf))


def test_valid_capture_still_parses() -> None:
    """Sanity: the un-mutated demo capture reads back cleanly (the fuzzer's baseline)."""
    _SCRATCH.write_bytes(_VALID)
    records, stats = read_capture(_SCRATCH)
    assert isinstance(records, list) and len(records) > 0
    assert isinstance(stats, PcapStats)
