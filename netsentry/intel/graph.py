"""Host-communication-graph analytics — the structural attacks a per-flow model can't see.

The supervised classifier scores each flow **in isolation** and drops every
identifier, so — exactly like beaconing — a whole class of malicious behaviour is
invisible to it because the signal lives *between* flows, in the topology of who
talks to whom. Two patterns matter most and neither exists inside a single flow:

- **Scan fan-out** (ATT&CK Discovery, T1046 Network Service Discovery / T1595
  Active Scanning). One source touching many distinct destinations is a horizontal
  sweep; one source touching many distinct ports is a vertical sweep. A single scan
  probe looks like an unremarkable short flow — which is why the temporal model
  misses PortScan, a later-day class it never trained on (see the per-class slices)
  — but the *fan-out* across the graph is unambiguous.
- **Lateral-movement chains** (ATT&CK Lateral Movement, T1021 Remote Services). A
  host that is first *reached* and then *initiates* connections deeper into the
  network is pivoting; a directed path ``a -> b -> c -> ...`` of such internal hops
  is the movement chain a responder wants to see whole.

``netsentry graph`` is the cross-flow, identity-aware complement to the per-flow
model — the topology mirror of how ``netsentry beacon`` is its timing mirror. Like
beacon, it reads the identity columns (``Src IP`` / ``Dst IP`` / ``Dst Port``) as
**metadata only** — the fields the model is forbidden to see — and, like beacon, the
honest scoping is written into the report: this is a **hunt-lead generator, not a
verdict**. A vulnerability scanner, a monitoring host, a backup server, or a jump box
all fan out or pivot legitimately; the analytic surfaces ranked candidates for a
human, and adds no detection to the model's per-flow verdicts.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "graph_report.md"
DEMO_REPORT_NAME = "graph_demo.md"

_SRC_IP, _DST_IP, _DST_PORT = "Src IP", "Dst IP", "Dst Port"


# RFC1918 private ranges — the "internal network". Deliberately *not*
# ``ip_address.is_private``, which also matches documentation/reserved ranges
# (e.g. TEST-NET 203.0.113.0/24) an operator would treat as external.
_RFC1918: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def is_internal(host: str) -> bool:
    """Whether a host is an RFC1918 private address (used only for annotation/scoring).

    Lateral movement is internal->internal; the entry hop is external->internal. The
    core detection works on any host labels — internal/external only sharpens the
    chain search toward movement *into* the network and away from ordinary egress.
    """
    try:
        ip = ipaddress.ip_address(str(host))
    except ValueError:
        return False
    return any(ip in net for net in _RFC1918)


@dataclass
class ScanSource:
    """One source ranked by how far it fans out across the graph."""

    source: str
    distinct_destinations: int
    distinct_ports: int
    total_flows: int

    @property
    def fan_out(self) -> int:
        """The larger of the horizontal (hosts) and vertical (ports) spread."""
        return max(self.distinct_destinations, self.distinct_ports)

    @property
    def kind(self) -> str:
        """Whether the spread reads as a horizontal (host) or vertical (port) sweep."""
        if self.distinct_destinations >= self.distinct_ports:
            return "horizontal"
        return "vertical"


@dataclass
class LateralChain:
    """A directed path of internal hops — a candidate movement chain."""

    path: list[str] = field(default_factory=list)

    @property
    def hosts(self) -> int:
        return len(self.path)

    @property
    def internal_hops(self) -> int:
        """Transitions whose destination is an internal host (movement inward)."""
        return sum(1 for i in range(1, len(self.path)) if is_internal(self.path[i]))

    @property
    def score(self) -> tuple[int, int]:
        """Rank key: more hosts first, then more internal hops."""
        return (self.hosts, self.internal_hops)

    def render(self) -> str:
        return " -> ".join(f"`{h}`" for h in self.path)


def detect_scans(df: pd.DataFrame, *, min_fanout: int, by_port: bool) -> list[ScanSource]:
    """Rank sources by fan-out (distinct destinations and/or distinct ports).

    A source is a scan candidate when either its distinct-destination count
    (horizontal sweep) or its distinct-port count (vertical sweep) reaches
    ``min_fanout``. Returns candidates sorted by fan-out (descending).
    """
    for column in (_SRC_IP, _DST_IP):
        if column not in df.columns:
            raise ValueError(f"graph analytics needs a {column!r} column")
    has_ports = by_port and _DST_PORT in df.columns

    scans: list[ScanSource] = []
    for source, block in df.groupby(_SRC_IP):
        distinct_dsts = int(block[_DST_IP].nunique())
        distinct_ports = int(block[_DST_PORT].nunique()) if has_ports else 0
        source_scan = ScanSource(str(source), distinct_dsts, distinct_ports, len(block))
        if source_scan.fan_out >= min_fanout:
            scans.append(source_scan)
    scans.sort(key=lambda s: (s.fan_out, s.total_flows), reverse=True)
    return scans


def build_adjacency(df: pd.DataFrame) -> dict[str, set[str]]:
    """Directed out-neighbour sets ``src -> {dst, ...}`` over the flow frame."""
    out_edges: dict[str, set[str]] = {}
    for src, dst in zip(df[_SRC_IP].astype(str), df[_DST_IP].astype(str), strict=True):
        if src == dst:
            continue
        out_edges.setdefault(src, set()).add(dst)
    return out_edges


def _longest_internal_path(start: str, out_edges: dict[str, set[str]], max_depth: int) -> list[str]:
    """Deepest simple path from ``start`` that moves along internal->internal hops.

    The head may be external (the entry into the network); every subsequent hop must
    land on an internal host, so ordinary egress to the internet cannot form a chain.
    Depth-first with a visited set (simple paths only) and a depth cap, so the search
    is bounded on large graphs.
    """
    best: list[str] = [start]

    def walk(node: str, path: list[str], seen: set[str]) -> None:
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        if len(path) >= max_depth:
            return
        for nxt in sorted(out_edges.get(node, ())):
            if nxt in seen or not is_internal(nxt):
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(nxt, path, seen)
            path.pop()
            seen.remove(nxt)

    walk(start, [start], {start})
    return best


def detect_lateral_chains(
    df: pd.DataFrame, *, min_chain_hosts: int, max_depth: int, max_starts: int
) -> list[LateralChain]:
    """Find directed internal-movement chains and return the longest, deduplicated.

    Start nodes are the graph's *entries* (hosts nobody connects to — an external
    attacker or a first-compromised host), capped at ``max_starts`` for runtime.
    Chains shorter than ``min_chain_hosts`` hosts are dropped, and a chain wholly
    contained in a longer one is suppressed so the report shows movement paths, not
    every prefix of them.
    """
    out_edges = build_adjacency(df)
    reached: set[str] = set()
    for dsts in out_edges.values():
        reached |= dsts
    entries = sorted(node for node in out_edges if node not in reached) or sorted(out_edges)

    chains: list[LateralChain] = []
    for start in entries[:max_starts]:
        path = _longest_internal_path(start, out_edges, max_depth)
        if len(path) >= min_chain_hosts:
            chains.append(LateralChain(path))

    chains.sort(key=lambda c: c.score, reverse=True)
    kept: list[LateralChain] = []
    seen_paths: list[list[str]] = []
    for chain in chains:
        if any(_is_subpath(chain.path, longer) for longer in seen_paths):
            continue
        kept.append(chain)
        seen_paths.append(chain.path)
    return kept


def _is_subpath(short: list[str], long: list[str]) -> bool:
    """Whether ``short`` appears as a contiguous run inside ``long``."""
    window = len(short)
    return any(long[i : i + window] == short for i in range(len(long) - window + 1))


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def render_report(
    scans: list[ScanSource],
    chains: list[LateralChain],
    *,
    min_fanout: int,
    top_n: int,
    demo: bool,
) -> str:
    """Render the ranked scan and lateral-movement candidates plus the scoping note."""
    scan_rows = [
        "| source | kind | distinct hosts | distinct ports | flows |",
        "|---|---|---|---|---|",
    ]
    for s in scans[:top_n]:
        scan_rows.append(
            f"| `{s.source}` | {s.kind} | {s.distinct_destinations} | {s.distinct_ports} "
            f"| {s.total_flows} |"
        )
    if len(scan_rows) == 2:
        scan_rows.append("| _(none above the fan-out line)_ | | | | |")

    chain_rows = ["| movement chain | hosts | internal hops |", "|---|---|---|"]
    for c in chains[:top_n]:
        chain_rows.append(f"| {c.render()} | {c.hosts} | {c.internal_hops} |")
    if len(chain_rows) == 2:
        chain_rows.append("| _(no internal movement chain found)_ | | |")

    demo_note = ""
    if demo:
        top_scan = scans[0] if scans else None
        top_chain = chains[0] if chains else None
        bits = []
        if top_scan is not None:
            bits.append(
                f"the planted sweep (`{top_scan.source}`, {top_scan.fan_out} distinct "
                f"{'hosts' if top_scan.kind == 'horizontal' else 'ports'}) tops the scan table"
            )
        if top_chain is not None:
            bits.append(
                f"the planted pivot ({top_chain.render()}) is recovered whole "
                f"({top_chain.hosts} hosts, {top_chain.internal_hops} internal hops)"
            )
        if bits:
            demo_note = (
                "\nThe synthetic capture plants a horizontal sweep, a vertical sweep, and one "
                "multi-hop pivot among benign talkers that only reach the internet; "
                + " and ".join(bits)
                + " — the mechanic, on data with a known answer.\n"
            )

    return f"""# NetSentry — Host-Graph Analytics{" (synthetic demo)" if demo else ""}

The classifier scores each flow in isolation and drops every identifier, so it is
blind to the two attacks whose signal lives in the **graph** of who talks to whom:
**scanning** (one host fanning out across many destinations or ports — ATT&CK
Discovery, T1046) and **lateral movement** (a reached host pivoting deeper — ATT&CK
Lateral Movement, T1021). This analytic reconstructs the host communication graph
and surfaces both — the cross-flow, topology-aware complement to the per-flow model,
the way beaconing is its timing-aware complement.
{demo_note}
## Scan fan-out (Discovery)

Sources whose distinct-destination (horizontal) or distinct-port (vertical) spread
reaches the {min_fanout}-target fan-out line. A single scan probe is an unremarkable
short flow — which is exactly why the temporal model misses PortScan, a later-day
class it never trained on — but the spread across the graph is not subtle.

{chr(10).join(scan_rows)}

## Lateral-movement chains (Lateral Movement)

Directed paths that enter the network and then hop internal-to-internal. Ordinary
egress to the internet cannot form a chain (the destination is external), so a long
internal path is the pivot pattern a responder wants to see whole rather than as
disconnected alerts.

{chr(10).join(chain_rows)}

## How to read this

This is a **hunt-lead generator, not a verdict.** A vulnerability scanner, a
monitoring poller, a backup server, or an administrator's jump box all fan out or
pivot as a matter of course and will surface here; the analytic ranks *candidates*
by structural suspicion for a human to triage, and adds no detection to the model's
per-flow verdicts. It reads the ``Src IP`` / ``Dst IP`` / ``Dst Port`` columns as
metadata only — exactly the identifiers the model is forbidden to see — which is why
it can catch what the model, by construction, cannot. Internal/external is an RFC1918
heuristic; on a real deployment substitute the site's own address plan.
"""


def synthesize_graph_flows(seed: int) -> pd.DataFrame:
    """Deterministic capture: benign egress talkers plus a planted scan and pivot chain.

    Used by ``--demo`` and the tests — a ground-truth harness with a known answer, no
    model or dataset required (the analytic is pure topology). Rows are shuffled so the
    detector must reconstruct the graph from the identity columns, not the row order.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    # Benign internal hosts: each talks to one or two external services only (egress).
    for host in range(12):
        src = f"10.0.0.{20 + host}"
        for _ in range(int(rng.integers(3, 9))):
            dst = f"93.184.216.{int(rng.integers(1, 60))}"
            port = int(rng.choice([80, 443, 53]))
            rows.append({_SRC_IP: src, _DST_IP: dst, _DST_PORT: port})

    # Planted horizontal sweep: one external host touches 40 distinct internal hosts.
    scanner = "203.0.113.9"
    for target in range(40):
        rows.append({_SRC_IP: scanner, _DST_IP: f"10.0.0.{100 + target}", _DST_PORT: 445})

    # Planted vertical sweep: one external host hits 30 distinct ports on a single host.
    vscanner = "203.0.113.7"
    for port in range(30):
        rows.append({_SRC_IP: vscanner, _DST_IP: "10.0.0.10", _DST_PORT: 1024 + port})

    # Planted lateral-movement chain: external entry -> pivot -> pivot -> two targets.
    chain = [
        ("203.0.113.5", "10.0.0.5", 22),
        ("10.0.0.5", "10.0.0.20", 3389),
        ("10.0.0.20", "10.0.0.50", 445),
        ("10.0.0.50", "10.0.0.51", 445),
    ]
    for src, dst, port in chain:
        rows.append({_SRC_IP: src, _DST_IP: dst, _DST_PORT: port})

    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_graph_report(
    settings: Settings,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    demo: bool = False,
) -> Path:
    """Build the host graph from a flow file (or the synthetic demo) and write the report."""
    cfg = settings.graph
    if demo:
        df = synthesize_graph_flows(settings.seed)
        default_out = settings.paths.reports_dir / DEMO_REPORT_NAME
    elif input_path is not None:
        df = _read(input_path)
        default_out = settings.paths.reports_dir / REPORT_NAME
    else:
        raise ValueError("provide --input a flow file or use --demo")

    scans = detect_scans(df, min_fanout=cfg.min_fanout, by_port=cfg.by_port)
    chains = detect_lateral_chains(
        df,
        min_chain_hosts=cfg.min_chain_hosts,
        max_depth=cfg.max_depth,
        max_starts=cfg.max_starts,
    )
    report = render_report(scans, chains, min_fanout=cfg.min_fanout, top_n=cfg.top_n, demo=demo)
    out_path = output_path or default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    n_scans, n_chains = len(scans), len(chains)
    logger.info(
        "Wrote host-graph report",
        extra={"path": str(out_path), "scans": n_scans, "chains": n_chains},
    )
    return out_path
