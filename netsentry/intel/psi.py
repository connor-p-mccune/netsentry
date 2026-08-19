"""Ask another organisation whether they have seen this indicator, without telling them.

This project can already export what it detects: [Sigma rules](sigma/README.md) for a SIEM,
[STIX 2.1 bundles](../reports/mitre.md) for an intel platform. Both assume the sharing decision
has already been made. The step before it is the one nobody instruments — **to ask a peer "have
you also seen 203.0.113.7?", you have to tell them you are interested in 203.0.113.7**, which
is a statement about your incident. Sharing communities solve this socially (trust groups,
Traffic Light Protocol markings) and technically they usually do not solve it at all.

The technical answer is **private set intersection**: two parties learn which indicators they
have in common and nothing else about the rest. The construction here is the classical
Diffie-Hellman one (Meadows 1986; Huberman, Franklin & Hogg 1999), built on the same RFC 3526
group the [secure-aggregation study](secagg.md) already verifies by Miller-Rabin:

1. Each party maps every indicator into the group's prime-order subgroup, `M(x) = H(x)^2 mod p`,
   and raises it to its own secret exponent.
2. The blinded sets are exchanged and blinded *again* with the other party's exponent.
3. `M(x)^(ab)` equals `M(y)^(ba)` exactly when `x = y`, so equality of the double-blinded values
   reveals the intersection — and a value blinded by an exponent you do not hold is a uniform
   group element, which is what makes everything else invisible.

Three things get measured rather than asserted:

- **What the usual practice actually leaks.** Sharing SHA-256 hashes of indicators is common,
  intuitive, and not private: the space of IPv4 addresses is `2^32`, which is small. The hash
  rate is measured on this machine, the *complete* attack is executed against a smaller
  indicator space (ports) to show the machinery end to end, and the IPv4 figure follows as
  arithmetic rather than as a claim. Salting does not help for the reason people forget — both
  parties must use the same salt, so every participant already has it.
- **The attack on PSI itself.** A dishonest participant can inflate its input set: submit ten
  thousand candidate indicators instead of its hundred real ones and the protocol faithfully
  reports which of the ten thousand the peer holds. This is executed, its yield measured, and
  the mitigations named — set-size limits, cardinality-only variants, or an authorised
  indicator source.
- **What it costs.** Modular exponentiations, wire bytes and wall-clock as the lists grow,
  against the hash-exchange it replaces, with the honest note that a deployment would use an
  elliptic-curve group and pay roughly an order of magnitude less.
"""

from __future__ import annotations

import hashlib
import ipaddress
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.secagg import DH_PRIME
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import PSIConfig

logger = get_logger(__name__)

REPORT_NAME = "psi.md"
FIGURE_NAME = "psi_cost.png"

#: Group elements are 2048 bits, so one blinded indicator costs this many bytes on the wire.
GROUP_BYTES = 256

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# The protocol.
# --------------------------------------------------------------------------------------


def hash_to_group(item: str) -> int:
    """Map an indicator into the prime-order subgroup of the RFC 3526 group.

    Squaring is what does the work: the group has order `2q`, so squaring any element lands in
    the subgroup of order `q` and removes the one bit of structure (the Legendre symbol) that
    would otherwise leak. Hashing first is what makes the map one-way — without it, an
    indicator's blinded value would be algebraically related to its neighbours'.
    """
    digest = hashlib.sha256(item.encode("utf-8")).digest()
    return pow(int.from_bytes(digest, "big") % DH_PRIME, 2, DH_PRIME)


def blind(elements: list[int], exponent: int) -> list[int]:
    """Raise every element to a secret exponent. This is the only operation either party runs."""
    return [pow(element, exponent, DH_PRIME) for element in elements]


def secret_exponent(rng: random.Random) -> int:
    """A 256-bit exponent, the same width the secure-aggregation study uses."""
    return rng.randrange(2, 1 << 256)


@dataclass
class PSIResult:
    """What one run of the protocol produced, and what it cost."""

    intersection: list[str]
    truth: list[str]
    a_size: int
    b_size: int
    exponentiations: int
    wire_bytes: int
    seconds: float

    @property
    def exact(self) -> bool:
        """Did the protocol recover the intersection exactly — no misses, no false matches?"""
        return sorted(self.intersection) == sorted(self.truth)


def private_set_intersection(
    a_items: list[str], b_items: list[str], rng: random.Random
) -> PSIResult:
    """Run the two-party protocol and return the intersection the *initiator* learns.

    The output is one-sided on purpose: A learns which of its own indicators B also holds, and
    B learns nothing at all. A shuffles its set before sending — the position of an element in
    a list is metadata, and a party that orders its indicators by first-seen time is leaking a
    timeline — and keeps the permutation so it can map the answer back.

    B returns A's double-blinded values **in the order it received them**, and that ordering is
    the whole difference between two protocols. Preserved, A can tell *which* of its own
    indicators matched (this function). Shuffled, A can only count them, which is
    cardinality-only PSI — a strictly weaker disclosure, and the mitigation the inflation
    attack below is asking for.
    """
    start = time.perf_counter()
    alpha = secret_exponent(rng)
    beta = secret_exponent(rng)

    # 1. A blinds its own set and sends it, shuffled, keeping the permutation.
    a_blinded = blind([hash_to_group(item) for item in a_items], alpha)
    order = list(range(len(a_blinded)))
    rng.shuffle(order)
    a_on_wire = [a_blinded[i] for i in order]

    # 2. B blinds what it received again (order preserved) and sends its own set blinded once
    #    and shuffled. B never sees a plaintext indicator and never learns which value matched.
    a_double = blind(a_on_wire, beta)
    b_blinded = blind([hash_to_group(item) for item in b_items], beta)
    rng.shuffle(b_blinded)

    # 3. A blinds B's set with its own exponent. M(x)^(ab) == M(y)^(ba) exactly when x == y, so
    #    a positional hit in `a_double` names one of A's own indicators through the permutation.
    b_double = set(blind(b_blinded, alpha))
    intersection = [
        a_items[original]
        for position, original in enumerate(order)
        if a_double[position] in b_double
    ]

    truth = sorted(set(a_items) & set(b_items))
    exponentiations = 2 * len(a_items) + 2 * len(b_items)
    wire_bytes = GROUP_BYTES * (len(a_items) * 2 + len(b_items))
    return PSIResult(
        intersection=sorted(intersection),
        truth=truth,
        a_size=len(a_items),
        b_size=len(b_items),
        exponentiations=exponentiations,
        wire_bytes=wire_bytes,
        seconds=time.perf_counter() - start,
    )


# --------------------------------------------------------------------------------------
# What the usual practice leaks.
# --------------------------------------------------------------------------------------


def measure_hash_rate(samples: int) -> float:
    """SHA-256 hashes per second on this machine — the attacker's only cost."""
    start = time.perf_counter()
    for value in range(samples):
        hashlib.sha256(str(value).encode("utf-8")).digest()
    elapsed = max(time.perf_counter() - start, _EPS)
    return samples / elapsed


@dataclass
class DictionaryAttack:
    """One executed preimage recovery against a hashed indicator list."""

    space: str
    universe: int
    recovered: int
    total: int
    seconds: float
    salted: bool

    @property
    def share(self) -> float:
        return self.recovered / max(self.total, 1)


def recover_hashed_ports(hashed: set[str], salt: str = "") -> DictionaryAttack:
    """Enumerate the whole 16-bit port space against a hashed list — the complete attack.

    Ports are used for the fully-executed demonstration because their universe is small enough
    to exhaust in milliseconds. Nothing about the attack changes for addresses except how long
    it runs, which is the point the IPv4 arithmetic then makes.
    """
    start = time.perf_counter()
    recovered = 0
    for port in range(65536):
        digest = hashlib.sha256(f"{salt}{port}".encode()).hexdigest()
        if digest in hashed:
            recovered += 1
    return DictionaryAttack(
        space="TCP/UDP port (2^16)",
        universe=65536,
        recovered=recovered,
        total=len(hashed),
        seconds=time.perf_counter() - start,
        salted=bool(salt),
    )


def enumerate_addresses(hashed: set[str], count: int, start_address: int) -> DictionaryAttack:
    """Enumerate ``count`` consecutive IPv4 addresses — a real, timed slice of the full attack.

    Timed with the address formatting included, because that is what the attacker actually
    pays: a bare `sha256(int)` benchmark overstates the rate several-fold and would make the
    extrapolation flattering rather than honest. The dotted quad is built arithmetically for
    the same reason `ipaddress.IPv4Address` is not used — the standard library's parser costs
    more than the hash it feeds.
    """
    started = time.perf_counter()
    recovered = 0
    digest = hashlib.sha256
    for value in range(start_address, start_address + count):
        address = f"{value >> 24 & 255}.{value >> 16 & 255}.{value >> 8 & 255}.{value & 255}"
        if digest(address.encode()).hexdigest() in hashed:
            recovered += 1
    return DictionaryAttack(
        space=f"IPv4, {count:,} consecutive addresses",
        universe=count,
        recovered=recovered,
        total=len(hashed),
        seconds=time.perf_counter() - started,
        salted=False,
    )


def full_space_seconds(rate: float, bits: int = 32) -> float:
    """Wall-clock to exhaust a space of ``2^bits`` at the measured hash rate."""
    return float(2**bits) / max(rate, _EPS)


# --------------------------------------------------------------------------------------
# The attack on the protocol itself.
# --------------------------------------------------------------------------------------


@dataclass
class InflationAttack:
    """A dishonest participant submitting a padded input set."""

    submitted: int
    honest: int
    learned: int
    peer_set: int
    universe_hits: int  # how many of the peer's indicators the submitted universe contained

    @property
    def peer_share(self) -> float:
        return self.learned / max(self.peer_set, 1)

    @property
    def yield_rate(self) -> float:
        """Share of the reachable indicators the attack actually extracted."""
        return self.learned / max(self.universe_hits, 1)


def inflation_attack(
    honest_items: list[str], candidates: list[str], b_items: list[str], rng: random.Random
) -> InflationAttack:
    """Run the protocol with A's set padded by a candidate universe it does not hold.

    The protocol is secure against an honest-but-*curious* participant and says nothing about a
    dishonest one: it faithfully reveals the intersection of the sets it was given. A party
    that submits a large candidate universe therefore learns the peer's membership across that
    whole universe, and the protocol emits no signal that the input was inflated.

    The measured quantity is deliberately *not* "how good is the attacker's guess" — that
    depends on how enumerable the indicator space is, and on this stand-in's independently
    drawn addresses it would be zero. It is the **yield**: of the peer indicators the submitted
    universe happens to contain, how many does the attack extract? The answer is all of them,
    at any size, which is what makes a size cap the only defence inside the protocol.
    """
    padded = list(dict.fromkeys([*honest_items, *candidates]))
    result = private_set_intersection(padded, b_items, rng)
    return InflationAttack(
        submitted=len(padded),
        honest=len(honest_items),
        learned=len(result.intersection),
        peer_set=len(b_items),
        universe_hits=len(set(padded) & set(b_items)),
    )


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class CostRow:
    """Protocol cost at one list size, against the hash exchange it replaces."""

    size: int
    exponentiations: int
    psi_bytes: int
    hash_bytes: int
    psi_seconds: float


@dataclass
class PSIStudy:
    """Everything the report renders."""

    result: PSIResult
    overlap: int
    hash_rate: float
    port_attack: DictionaryAttack
    salted_port_attack: DictionaryAttack
    address_attack: DictionaryAttack
    ipv4_seconds: float
    inflation: list[InflationAttack]
    cost: list[CostRow]
    a_day: str
    b_day: str


def _indicator_lists(settings: Settings, cfg: PSIConfig) -> tuple[list[str], list[str], int]:
    """Build two organisations' indicator lists from the capture's attack destinations.

    The addresses are metadata the model never sees — the same posture the
    [beaconing](beacon_demo.md) and [host-graph](graph_demo.md) analytics take. The *overlap* is
    constructed rather than found: this stand-in draws each row's addresses independently, so
    two organisations share nothing by construction and a measured intersection would be a
    measurement of the generator. The realistic scenario it stands in for is two victims of one
    actor, who share exactly the infrastructure that actor reused.
    """
    import pandas as pd

    raw = settings.paths.data_raw
    frames = {}
    for day in (cfg.org_a_day, cfg.org_b_day):
        matches = sorted(raw.glob(f"{day}*.csv"))
        if not matches:
            raise FileNotFoundError(f"no raw capture for {day} under {raw}")
        frame = pd.read_csv(matches[0])
        frame.columns = [column.strip() for column in frame.columns]
        frames[day] = frame

    def _attack_destinations(frame: pd.DataFrame) -> list[str]:
        attacks = frame[frame["Label"].astype(str).str.upper() != "BENIGN"]
        return list(dict.fromkeys(attacks["Destination IP"].astype(str).tolist()))

    a_all = _attack_destinations(frames[cfg.org_a_day])[: cfg.list_size]
    b_own = _attack_destinations(frames[cfg.org_b_day])
    overlap = min(cfg.overlap, len(a_all), cfg.list_size)
    b_items = list(dict.fromkeys([*a_all[:overlap], *b_own]))[: cfg.list_size]
    return a_all, b_items, overlap


def run_psi_study(settings: Settings) -> PSIStudy:
    """Run the protocol, the attack on the practice it replaces, and the attack on itself."""
    cfg: PSIConfig = settings.psi
    seed_everything(settings.seed)
    rng = random.Random(settings.seed)

    a_items, b_items, overlap = _indicator_lists(settings, cfg)
    result = private_set_intersection(a_items, b_items, rng)
    logger.info(
        "PSI complete",
        extra={"a": len(a_items), "b": len(b_items), "intersection": len(result.intersection)},
    )

    # What the usual practice leaks, executed.
    rate = measure_hash_rate(cfg.hash_samples)
    ports = rng.sample(range(1024, 65536), cfg.port_indicators)
    hashed_ports = {hashlib.sha256(str(port).encode()).hexdigest() for port in ports}
    port_attack = recover_hashed_ports(hashed_ports)
    salt = "netsentry-sharing-group"
    salted_ports = {hashlib.sha256(f"{salt}{port}".encode()).hexdigest() for port in ports}
    salted_attack = recover_hashed_ports(salted_ports, salt=salt)

    hashed_addresses = {hashlib.sha256(item.encode()).hexdigest() for item in a_items}
    start_address = int(ipaddress.IPv4Address(a_items[0])) if a_items else 0
    address_attack = enumerate_addresses(hashed_addresses, cfg.address_sample, start_address)

    # The attack on the protocol itself.
    # The candidate universe is *constructed* to contain a known fraction of the peer's
    # indicators, and the report says so: how enumerable a real indicator space is depends on
    # the indicator type, and on this stand-in's independently drawn addresses a genuine guess
    # would hit nothing. What is being measured is the protocol's response to inflation, not
    # the attacker's imagination.
    inflation = []
    for size in cfg.inflation_sizes:
        reachable = b_items[: max(int(size * cfg.universe_hit_rate), 1)]
        decoys = [str(ipaddress.IPv4Address(rng.randrange(1 << 32))) for _ in range(size)]
        candidates = list(dict.fromkeys([*reachable, *decoys]))[:size]
        inflation.append(inflation_attack(a_items[: cfg.honest_size], candidates, b_items, rng))

    # Cost.
    cost = []
    for size in cfg.cost_sizes:
        sample_a = a_items[:size] if len(a_items) >= size else [f"10.0.0.{i}" for i in range(size)]
        sample_b = b_items[:size] if len(b_items) >= size else [f"10.1.0.{i}" for i in range(size)]
        run = private_set_intersection(sample_a, sample_b, rng)
        cost.append(
            CostRow(
                size=size,
                exponentiations=run.exponentiations,
                psi_bytes=run.wire_bytes,
                hash_bytes=32 * (len(sample_a) + len(sample_b)),
                psi_seconds=run.seconds,
            )
        )
        logger.info("PSI cost measured", extra={"size": size})

    return PSIStudy(
        result=result,
        overlap=overlap,
        hash_rate=rate,
        port_attack=port_attack,
        salted_port_attack=salted_attack,
        address_attack=address_attack,
        ipv4_seconds=full_space_seconds(
            address_attack.universe / max(address_attack.seconds, _EPS), 32
        ),
        inflation=inflation,
        cost=cost,
        a_day=cfg.org_a_day,
        b_day=cfg.org_b_day,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_psi_report(settings: Settings) -> Path:
    """Run the indicator-sharing study and write the report + figure."""
    study = run_psi_study(settings)
    sizes = np.array([row.size for row in study.cost], dtype=float)
    figure = plots.plot_lines(
        {
            "private set intersection": (
                sizes,
                np.array([row.psi_seconds for row in study.cost], dtype=float),
            ),
        },
        xlabel="indicators per organisation",
        ylabel="seconds for one exchange",
        title="What privacy costs, as the indicator lists grow",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote PSI report", extra={"path": str(out_path)})

    with track_run(settings, "psi") as run:
        run.log_params({"a_items": study.result.a_size, "b_items": study.result.b_size})
        run.log_metrics(
            {
                "intersection": float(len(study.result.intersection)),
                "exact": float(study.result.exact),
                "hash_rate": study.hash_rate,
                "ipv4_full_space_seconds": study.ipv4_seconds,
            }
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _attack_table(study: PSIStudy) -> str:
    rows = [
        "| hashed indicator space | universe | preimages recovered | time | verdict |",
        "|---|---|---|---|---|",
    ]
    for attack in (study.port_attack, study.salted_port_attack, study.address_attack):
        label = attack.space + (" — **salted**" if attack.salted else "")
        if attack.share >= 0.999:
            verdict = "**fully recovered** (the space was exhausted)"
        else:
            coverage = attack.universe / float(2**32)
            verdict = (
                f"a {coverage:.4%} slice of the space, so the rest is arithmetic rather than "
                "an obstacle"
            )
        rows.append(
            f"| {label} | {attack.universe:,} | {attack.recovered:,} / {attack.total:,} | "
            f"{attack.seconds:.2f} s | {verdict} |"
        )
    return "\n".join(rows)


def _inflation_table(study: PSIStudy) -> str:
    rows = [
        "| indicators submitted | genuinely held | reachable in the universe | learned | "
        "yield | share of the peer's list |",
        "|---|---|---|---|---|---|",
    ]
    for attack in study.inflation:
        rows.append(
            f"| {attack.submitted:,} | {attack.honest:,} | {attack.universe_hits:,} | "
            f"{attack.learned:,} | **{attack.yield_rate:.0%}** | {attack.peer_share:.0%} |"
        )
    return "\n".join(rows)


def _cost_table(study: PSIStudy) -> str:
    rows = [
        "| indicators each | modular exponentiations | PSI wire bytes | hash-exchange bytes | "
        "overhead | seconds |",
        "|---|---|---|---|---|---|",
    ]
    for row in study.cost:
        rows.append(
            f"| {row.size:,} | {row.exponentiations:,} | {row.psi_bytes:,} | "
            f"{row.hash_bytes:,} | {row.psi_bytes / max(row.hash_bytes, 1):.0f}x | "
            f"{row.psi_seconds:.2f} |"
        )
    return "\n".join(rows)


def _correctness_read(study: PSIStudy) -> str:
    result = study.result
    verdict = (
        "recovers the intersection **exactly** — every shared indicator found, nothing else "
        "reported"
        if result.exact
        else "did **not** recover the intersection exactly, which is a bug in the "
        "implementation rather than a property of the protocol"
    )
    return (
        f"Organisation A holds {result.a_size:,} indicators derived from its own detections on "
        f"{study.a_day}; organisation B holds {result.b_size:,} from {study.b_day}, of which "
        f"{study.overlap} are infrastructure the same actor reused against both. The protocol "
        f"{verdict}: {len(result.intersection)} of {len(result.truth)} shared indicators, in "
        f"{result.seconds:.2f} seconds.\n\n"
        "What B learns is nothing. What A learns is which of *its own* indicators B also holds "
        "— not how many others B has beyond the list size, not what they are. Every value B "
        "sends is `M(y)^beta` for a secret `beta` A does not have, which is a uniform element "
        "of a 2048-bit prime-order subgroup: there is no dictionary attack against it, because "
        "the attacker cannot compute the blinded form of a guess."
    )


def _hashing_read(study: PSIStudy) -> str:
    hours = study.ipv4_seconds / 3600.0
    return (
        "The practice this replaces is exchanging **hashes** of indicators, which feels private "
        "and is not. A hash is only one-way when the input is unguessable, and an IPv4 address "
        f"is a 32-bit number. Enumerating addresses against a hashed list runs at "
        f"{study.address_attack.universe / max(study.address_attack.seconds, _EPS):,.0f} "
        "candidates a second here — measured with the address formatting included, because a "
        f"bare `sha256` benchmark ({study.hash_rate:,.0f}/s) overstates what the attacker gets "
        f"— so the **entire IPv4 space costs {hours:.1f} hours on one laptop core**, after "
        "which every address in the list is recovered with certainty. Not probably: certainly, "
        "because the space is finite and the map is deterministic. A compiled implementation or "
        "a GPU turns hours into minutes, and neither is exotic.\n\n"
        "The table runs the *complete* attack against a smaller space so the machinery is "
        "visible rather than argued, plus a real timed slice of the address space:\n\n"
        f"{_attack_table(study)}\n\n"
        "The salted row is the one worth dwelling on, because 'just salt it' is the standard "
        "response and it does not work here. A salt defeats precomputation by an *outsider*; in "
        "an indicator-sharing group every participant must use the **same** salt or no two "
        "hashes would ever match, so every participant — and anyone who joins, or who obtains "
        "the group's documentation — can run exactly the attack above. The salted list falls in "
        f"{study.salted_port_attack.seconds:.2f} seconds, the same as the unsalted one."
    )


def _inflation_read(study: PSIStudy) -> str:
    if not study.inflation:
        return ""
    largest = max(study.inflation, key=lambda a: a.submitted)
    return (
        "The protocol is secure against an honest-but-*curious* participant and says nothing "
        "about a dishonest one, so the obvious attack is to lie about the input. A party that "
        f"submits {largest.submitted:,} candidate indicators instead of the "
        f"{largest.honest:,} it actually holds gets back every one of them that the peer also "
        f"has — {largest.learned:,} hits, a **{largest.yield_rate:.0%} yield** on the "
        "reachable indicators and no signal to the peer that anything unusual happened. The "
        "cryptography performs perfectly throughout. Nothing was broken; the assumption that "
        "inputs are truthful was never in force.\n\n"
        "One thing this table is *not* measuring, and the distinction matters: how good an "
        "attacker's guesses are. That depends entirely on how enumerable the indicator type is "
        "— trivial for addresses, hopeless for long random tokens — and on this stand-in, whose "
        "addresses are drawn independently per flow, a genuine guess would hit nothing at all. "
        "The candidate universes below are constructed to contain a known share of the peer's "
        "list precisely so the measured quantity is the *protocol's* response to inflation: it "
        "has none. The yield is total at every size.\n\n"
        f"{_inflation_table(study)}\n\n"
        "The mitigations are all outside the protocol, which is the honest way to state them: "
        "cap the input size and make both parties commit to it before the exchange; use a "
        "cardinality-only variant (PSI-CA) so the initiator learns *how many* indicators are "
        "shared and not which; or require indicators to be signed by a source that will not "
        "sign a guess. A sharing agreement that specifies none of these has adopted a protocol "
        "and not a policy."
    )


def _cost_read(study: PSIStudy) -> str:
    if not study.cost:
        return ""
    small, large = study.cost[0], study.cost[-1]
    return (
        f"{_cost_table(study)}\n\n"
        f"Privacy costs {large.psi_bytes / max(large.hash_bytes, 1):.0f}x the bandwidth of a "
        f"hash exchange (a 2048-bit group element against a 256-bit digest) and "
        f"{large.psi_seconds:.1f} seconds of CPU at {large.size:,} indicators a side, against "
        f"{small.psi_seconds:.2f} seconds at {small.size:,}. The scaling is linear in the list "
        "size and entirely dominated by modular exponentiation — every other step is a hash or "
        "a set lookup.\n\n"
        "The honest engineering note is that this group is the slow choice. A 2048-bit "
        "finite-field exponentiation is roughly an order of magnitude more work than the "
        "equivalent operation on a 256-bit elliptic curve, and the same protocol runs unchanged "
        "on one; it is used here because the group is already verified in this repository's "
        "test suite and because a from-scratch curve implementation would be a much larger "
        "surface to get subtly wrong. Read the seconds as an upper bound on a laptop, not as "
        "the cost of the idea."
    )


def _render(study: PSIStudy, figure: Path) -> str:
    return f"""# NetSentry — Asking Without Telling: Private Indicator Sharing

_Diffie-Hellman private set intersection over RFC 3526 group 14, run between two organisations'
indicator lists ({study.result.a_size:,} and {study.result.b_size:,} addresses), with the
dictionary attack on hash-based sharing executed and the inflation attack on PSI executed too._

## Why this report exists

This project already exports what it detects — [Sigma rules](sigma/README.md) for a SIEM, STIX
2.1 bundles for an intel platform — and both assume the decision to share has been made. The
step before it is the one nobody instruments: **asking a peer whether they have seen an
indicator tells them you are interested in it**, which is a statement about your incident.

## Does it work?

{_correctness_read(study)}

## What the usual practice leaks

{_hashing_read(study)}

## The attack on the protocol itself

{_inflation_read(study)}

## What it costs

![Cost by list size](../figures/{figure.name})

{_cost_read(study)}

## Scope and honest limits

- **The overlap is constructed, and it has to be.** This stand-in draws each flow's addresses
  independently, so two organisations' indicator lists intersect in exactly zero elements — a
  measured intersection here would be a measurement of the generator. The lists are built from
  real attack destinations in the capture and then given a documented overlap, standing in for
  the scenario that makes sharing worth doing: two victims of one actor, sharing the
  infrastructure that actor reused.
- **One-sided output, honest-but-curious model.** A learns the intersection and B learns
  nothing; making it two-sided is one extra message. Neither party is protected against the
  other lying about its input, which is the inflation attack above, and neither is protected
  against a party that simply publishes the result afterwards.
- **Set sizes leak.** Both parties learn the other's list size, which is not nothing: a peer
  whose indicator list triples overnight is a peer having an incident. Padding to a fixed size
  is the standard fix and costs exactly what the padding costs.
- **This is set intersection, not intelligence.** Two organisations sharing an IP address are
  sharing the weakest possible indicator; the [MITRE mapping](mitre.md) and
  [incident reports](incident_demo.md) are where behaviour rather than infrastructure gets
  described, and behaviour does not fit in a set-intersection protocol.
- **The group is verified, the implementation is not certified.** The modulus and generator are
  checked by Miller-Rabin in the test suite, the exponents come from the seeded project RNG so
  the report is reproducible, and a deployment would need an OS CSPRNG, an authenticated
  channel, and constant-time arithmetic — none of which is what this report measures."""
