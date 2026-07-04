"""Per-service threshold routing: the flow's Destination Port (request metadata,
never a model feature) selects its service's validation-calibrated threshold, with
a global fallback for absent ports, unmapped services, and malformed bundles."""

from __future__ import annotations

from netsentry.data.schema import DESTINATION_PORT
from netsentry.data.services import PER_SERVICE_PROFILE, service_of
from netsentry.serving.inference import resolve_service_thresholds

CONFIG: dict[str, object] = {
    "target_fpr": 0.001,
    "global": 0.9,
    "thresholds": {"SSH": 0.7, "HTTP": 0.95},
}


def test_port_routes_to_its_service_threshold() -> None:
    flows: list[dict[str, float]] = [
        {DESTINATION_PORT: 22, "Flow Duration": 1.0},
        {DESTINATION_PORT: 8080, "Flow Duration": 1.0},  # alt-HTTP folds into HTTP
    ]
    assert resolve_service_thresholds(flows, CONFIG, fallback=0.5) == [0.7, 0.95]


def test_unmapped_service_falls_back_to_global() -> None:
    flows: list[dict[str, float]] = [{DESTINATION_PORT: 443, "Flow Duration": 1.0}]
    # HTTPS has no calibrated entry in this bundle -> the profile's global cut.
    assert resolve_service_thresholds(flows, CONFIG, fallback=0.5) == [0.9]


def test_missing_port_falls_back_to_global() -> None:
    flows: list[dict[str, float]] = [{"Flow Duration": 1.0}]
    assert resolve_service_thresholds(flows, CONFIG, fallback=0.5) == [0.9]


def test_malformed_config_degrades_to_fallback() -> None:
    flows: list[dict[str, float]] = [{DESTINATION_PORT: 22, "Flow Duration": 1.0}]
    assert resolve_service_thresholds(flows, {}, fallback=0.5) == [0.5]
    assert resolve_service_thresholds(flows, {"thresholds": "bogus"}, fallback=0.5) == [0.5]


def test_profile_name_and_port_map_are_shared() -> None:
    # The audit and the serving layer must agree on both the profile name and the
    # port->service map; a drifted copy would silently misroute thresholds.
    assert PER_SERVICE_PROFILE == "per_service"
    assert service_of(22) == "SSH" and service_of(8080) == "HTTP"
