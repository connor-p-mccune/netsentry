"""Well-known port -> coarse service mapping — shared reference data.

Used by the per-service parity audit (``evaluation/subgroups``) to slice the test
set, and by the serving bundle/inference layer to route the ``per_service``
threshold profile. IANA-registered assignments: reference data (like the ATT&CK
mapping), not a per-deployment knob. Ports outside the map fall into
``other/ephemeral`` — which is exactly where a port scan's sprayed destinations and
odd C2 ports land, itself a meaningful bucket.

The port is deliberately **not** a model feature (the memorisation leak documented
in ``.claude/rules/ml.md``); here it only names a slice or selects an operating
threshold, and never enters a prediction.
"""

from __future__ import annotations

PORT_SERVICE: dict[int, str] = {
    20: "FTP",
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    465: "SMTP",
    587: "SMTP",
    53: "DNS",
    80: "HTTP",
    8000: "HTTP",
    8080: "HTTP",
    110: "POP3",
    995: "POP3",
    143: "IMAP",
    993: "IMAP",
    443: "HTTPS",
    8443: "HTTPS",
    139: "SMB",
    445: "SMB",
    3389: "RDP",
}
OTHER_SERVICE = "other/ephemeral"

# Threshold-profile name under which the serving bundle stores per-service
# operating thresholds (the productised fix for the parity audit's finding).
PER_SERVICE_PROFILE = "per_service"


def service_of(port: float) -> str:
    """Map a destination port to a coarse service name (well-known ports; else 'other').

    Non-finite or non-integer ports (missing/garbled rows) fall into the 'other'
    bucket rather than raising — callers use this to slice or route, not to gate.
    """
    try:
        number = int(port)
    except (ValueError, TypeError, OverflowError):
        return OTHER_SERVICE
    return PORT_SERVICE.get(number, OTHER_SERVICE)
