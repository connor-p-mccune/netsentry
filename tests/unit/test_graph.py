"""Host-graph analytics: fan-out ranking, internal-chain recovery, and the demo."""

from __future__ import annotations

import pandas as pd

from netsentry.config import Settings
from netsentry.intel.graph import (
    LateralChain,
    build_adjacency,
    detect_lateral_chains,
    detect_scans,
    is_internal,
    render_report,
    run_graph_report,
    synthesize_graph_flows,
)


def test_is_internal_classifies_rfc1918() -> None:
    assert is_internal("10.0.0.5")
    assert is_internal("192.168.1.1")
    assert is_internal("172.16.4.9")
    assert not is_internal("203.0.113.5")
    assert not is_internal("not-an-ip")  # hostnames are treated as external, never crash


def test_horizontal_sweep_is_flagged_and_classified() -> None:
    df = pd.DataFrame(
        {
            "Src IP": ["203.0.113.9"] * 25,
            "Dst IP": [f"10.0.0.{i}" for i in range(25)],
            "Dst Port": [445] * 25,
        }
    )
    scans = detect_scans(df, min_fanout=20, by_port=True)
    assert scans and scans[0].source == "203.0.113.9"
    assert scans[0].distinct_destinations == 25
    assert scans[0].kind == "horizontal"


def test_vertical_sweep_is_detected_by_port_spread() -> None:
    df = pd.DataFrame(
        {
            "Src IP": ["203.0.113.7"] * 30,
            "Dst IP": ["10.0.0.10"] * 30,
            "Dst Port": list(range(1024, 1054)),
        }
    )
    scans = detect_scans(df, min_fanout=20, by_port=True)
    assert scans and scans[0].kind == "vertical"
    assert scans[0].distinct_ports == 30
    assert scans[0].distinct_destinations == 1


def test_quiet_egress_talker_is_not_a_scan() -> None:
    df = pd.DataFrame(
        {
            "Src IP": ["10.0.0.20"] * 6,
            "Dst IP": ["93.184.216.34", "93.184.216.35"] * 3,
            "Dst Port": [443] * 6,
        }
    )
    assert detect_scans(df, min_fanout=20, by_port=True) == []


def test_adjacency_ignores_self_loops() -> None:
    df = pd.DataFrame({"Src IP": ["a", "a", "b"], "Dst IP": ["a", "b", "c"]})
    out_edges = build_adjacency(df)
    assert out_edges["a"] == {"b"}  # the a->a self-loop is dropped
    assert out_edges["b"] == {"c"}


def test_internal_chain_is_recovered_whole() -> None:
    df = pd.DataFrame(
        {
            "Src IP": ["203.0.113.5", "10.0.0.5", "10.0.0.20", "10.0.0.50"],
            "Dst IP": ["10.0.0.5", "10.0.0.20", "10.0.0.50", "10.0.0.51"],
            "Dst Port": [22, 3389, 445, 445],
        }
    )
    chains = detect_lateral_chains(df, min_chain_hosts=3, max_depth=8, max_starts=500)
    assert chains
    top = chains[0]
    assert top.path == ["203.0.113.5", "10.0.0.5", "10.0.0.20", "10.0.0.50", "10.0.0.51"]
    assert top.hosts == 5
    assert top.internal_hops == 4  # every hop after the external entry lands internal


def test_egress_to_internet_forms_no_chain() -> None:
    # A hub talking only to external hosts must not read as lateral movement.
    df = pd.DataFrame(
        {
            "Src IP": ["10.0.0.20", "10.0.0.21", "10.0.0.22"],
            "Dst IP": ["93.184.216.1", "93.184.216.2", "93.184.216.3"],
            "Dst Port": [443, 443, 443],
        }
    )
    assert detect_lateral_chains(df, min_chain_hosts=3, max_depth=8, max_starts=500) == []


def test_chain_score_prefers_length_then_internal_hops() -> None:
    short = LateralChain(["a", "10.0.0.1", "10.0.0.2"])
    longer = LateralChain(["a", "10.0.0.1", "10.0.0.2", "10.0.0.3"])
    assert longer.score > short.score


def test_demo_frame_surfaces_both_planted_patterns() -> None:
    df = synthesize_graph_flows(seed=7)
    scans = detect_scans(df, min_fanout=20, by_port=True)
    chains = detect_lateral_chains(df, min_chain_hosts=3, max_depth=8, max_starts=500)
    sources = {s.source for s in scans}
    assert "203.0.113.9" in sources  # horizontal sweep
    assert "203.0.113.7" in sources  # vertical sweep
    assert chains and chains[0].path[0] == "203.0.113.5"  # the planted pivot entry
    assert chains[0].hosts == 5


def test_detection_is_order_invariant() -> None:
    df = synthesize_graph_flows(seed=3)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = detect_lateral_chains(df, min_chain_hosts=3, max_depth=8, max_starts=500)
    b = detect_lateral_chains(shuffled, min_chain_hosts=3, max_depth=8, max_starts=500)
    assert [c.path for c in a] == [c.path for c in b]


def test_report_renders_scoping_note_and_planted_hits() -> None:
    df = synthesize_graph_flows(seed=7)
    scans = detect_scans(df, min_fanout=20, by_port=True)
    chains = detect_lateral_chains(df, min_chain_hosts=3, max_depth=8, max_starts=500)
    report = render_report(scans, chains, min_fanout=20, top_n=20, demo=True)
    assert "# NetSentry — Host-Graph Analytics" in report
    assert "hunt-lead generator, not a verdict" in report
    assert "203.0.113.9" in report and "203.0.113.5" in report


def test_run_graph_demo_writes_report(tmp_path) -> None:
    settings = Settings()
    settings.paths.reports_dir = tmp_path / "reports"
    out = run_graph_report(settings, demo=True)
    assert out.exists() and out.name == "graph_demo.md"
