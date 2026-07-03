"""Per-service parity tests: the port->service map, the per-service TPR/FPR slicing
at one global threshold (including support floors and NaN handling for services with
no attacks/benign flows), and the equalized-odds-style parity gap."""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.evaluation.subgroups import (
    OTHER_SERVICE,
    parity_gap,
    service_of,
    service_slices,
    wilson_interval,
)


def test_service_of_maps_well_known_ports() -> None:
    assert service_of(22) == "SSH"
    assert service_of(80) == "HTTP"
    assert service_of(8080) == "HTTP"  # alt-HTTP folds into the same service
    assert service_of(443) == "HTTPS"
    assert service_of(21) == "FTP"


def test_service_of_buckets_ephemeral_and_garbage() -> None:
    assert service_of(51423) == OTHER_SERVICE  # ephemeral / sprayed scan port
    assert service_of(float("nan")) == OTHER_SERVICE
    assert service_of(float("inf")) == OTHER_SERVICE


def test_service_slices_computes_tpr_fpr_per_service() -> None:
    # SSH: 2 benign (one flagged) + 2 attacks (both flagged); HTTP: 2 benign (clean)
    # + 2 attacks (one flagged). Threshold 0.5, no support floor.
    ports = np.array([22, 22, 22, 22, 80, 80, 80, 80])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([0.6, 0.1, 0.9, 0.8, 0.2, 0.3, 0.7, 0.1])
    slices = {s.service: s for s in service_slices(y, scores, ports, 0.5, min_support=1)}

    ssh, http = slices["SSH"], slices["HTTP"]
    assert (ssh.n_benign, ssh.n_attack) == (2, 2)
    assert ssh.detection == 1.0 and ssh.fpr == 0.5
    assert http.detection == 0.5 and http.fpr == 0.0
    assert (ssh.true_positives, http.true_positives) == (2, 1)
    # The single false positive in the whole set belongs to SSH.
    assert ssh.false_positives == 1 and ssh.alert_share == 1.0
    assert http.false_positives == 0 and http.alert_share == 0.0


def test_service_slices_respects_min_support() -> None:
    ports = np.array([22, 22, 22, 443])  # HTTPS has only one flow
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.9, 0.9])
    services = {s.service for s in service_slices(y, scores, ports, 0.5, min_support=2)}
    assert services == {"SSH"}


def test_service_with_no_attacks_reports_nan_detection() -> None:
    ports = np.array([53, 53, 53])
    y = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.9])
    (dns,) = service_slices(y, scores, ports, 0.5, min_support=1)
    assert dns.service == "DNS"
    assert np.isnan(dns.detection) and dns.fpr == 1 / 3


def test_alert_shares_sum_to_one_when_any_fp_exists() -> None:
    ports = np.array([22, 80, 443, 53])
    y = np.zeros(4, dtype=int)
    scores = np.array([0.9, 0.9, 0.1, 0.9])  # 3 false positives across 3 services
    slices = service_slices(y, scores, ports, 0.5, min_support=1)
    assert np.isclose(sum(s.alert_share for s in slices), 1.0)


def test_parity_gap_is_max_minus_min_over_finite_values() -> None:
    assert parity_gap([0.1, 0.4, float("nan")]) == pytest.approx(0.3)
    assert parity_gap([0.2]) == 0.0  # fewer than two finite values -> no gap
    assert parity_gap([]) == 0.0


def test_wilson_interval_brackets_the_rate_and_stays_in_unit_range() -> None:
    low, high = wilson_interval(10, 100)
    assert 0.0 <= low < 0.1 < high <= 1.0
    # Zero successes must not collapse to a degenerate [0, 0] interval.
    z_low, z_high = wilson_interval(0, 100)
    assert z_low == 0.0 and z_high > 0.0


def test_wilson_interval_narrows_with_support() -> None:
    small = wilson_interval(5, 50)
    large = wilson_interval(500, 5000)  # same 10% rate, 100x the support
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_is_nan_for_empty_group() -> None:
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)
