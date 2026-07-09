"""Campaign grouping and detection accounting (pure logic)."""

from __future__ import annotations

import numpy as np

from netsentry.evaluation.campaigns import CampaignOutcome, campaign_outcomes


def _stream() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiny stream: a Thursday web campaign, a Friday scan, benign in between."""
    labels = np.array(
        ["BENIGN", "Web Attack", "Web Attack", "BENIGN", "Web Attack", "PortScan", "PortScan"]
    )
    days = np.array(
        ["Thursday", "Thursday", "Thursday", "Thursday", "Thursday", "Friday", "Friday"]
    )
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.85, 0.3, 0.2])
    return labels, days, scores


def test_groups_by_day_and_class_in_day_order() -> None:
    labels, days, scores = _stream()
    outcomes = campaign_outcomes(labels, days, scores, threshold=0.8, benign_label="BENIGN")
    assert [(o.label, o.day) for o in outcomes] == [
        ("Web Attack", "Thursday"),
        ("PortScan", "Friday"),
    ]


def test_first_alert_counts_only_the_campaigns_own_flows() -> None:
    labels, days, scores = _stream()
    web, scan = campaign_outcomes(labels, days, scores, threshold=0.8, benign_label="BENIGN")
    # Web Attack flows in stream order score 0.2, 0.9, 0.85: the first alert is
    # its 2nd flow — the benign 0.95 in between must not advance the counter.
    assert web.flows == 3
    assert web.alerts == 2
    assert web.first_alert_flow == 2
    assert web.flow_detection == 2 / 3
    # The scan never crosses the threshold: silent campaign.
    assert scan.alerts == 0
    assert scan.first_alert_flow is None
    assert not scan.detected(1)


def test_detected_at_k_is_the_conservative_reading() -> None:
    outcome = CampaignOutcome(label="DDoS", day="Friday", flows=100, alerts=3, first_alert_flow=7)
    assert outcome.detected(1)
    assert outcome.detected(3)
    assert not outcome.detected(5)


def test_same_class_on_two_days_is_two_campaigns() -> None:
    labels = np.array(["Bot", "Bot"])
    days = np.array(["Thursday", "Friday"])
    outcomes = campaign_outcomes(labels, days, np.array([0.9, 0.1]), 0.5, "BENIGN")
    assert len(outcomes) == 2
    assert outcomes[0].detected(1) and not outcomes[1].detected(1)
