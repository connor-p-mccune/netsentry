"""Single source of truth for the CIC-IDS2017 schema.

Defines the canonical CICFlowMeter feature columns, the identifier/leaky columns
that MUST be dropped before modelling, the label vocabulary, and the per-day
attack layout used by the temporal split. Cleaning, the feature pipeline, the
synthetic generator, and the tests all import from here so the leakage contract
and column names live in exactly one place.
"""

from __future__ import annotations

# --- Identifiers / leakage --------------------------------------------------
# These identify the flow or capture session, not its behaviour. They trivially
# encode the label on CIC-IDS2017 (a given attack reuses a handful of IPs/ports
# within one capture) and MUST never reach a model. Not all variants of the CSVs
# contain every one of these; cleaning drops whatever is present.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "External IP",
    "Fwd Header Length.1",  # known duplicate column; redundant, dropped
)

# `Destination Port` is borderline: predictive, but it lets the model memorise
# "attack X always hit port Y" instead of behaviour. Handled deliberately — kept
# out of the headline model and optionally encoded as a categorical.
DESTINATION_PORT = "Destination Port"

LABEL_COLUMN = "Label"
DAY_COLUMN = "Day"

# --- Canonical feature columns (the 78 CICFlowMeter statistics) --------------
FEATURE_COLUMNS: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)

# Columns containing Inf (division by zero on instantaneous flows).
RATE_COLUMNS: tuple[str, ...] = ("Flow Bytes/s", "Flow Packets/s")

# --- Labels -----------------------------------------------------------------
BENIGN_LABEL = "BENIGN"

# Raw labels as they appear in CIC-IDS2017 (after whitespace/dash normalisation).
RAW_LABELS: tuple[str, ...] = (
    "BENIGN",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Bot",
    "PortScan",
    "DDoS",
)

# Which CIC-IDS2017 day each attack was captured on (drives the temporal split
# and the synthetic generator).
DAY_ORDER: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
LABEL_DAYS: dict[str, str] = {
    "FTP-Patator": "Tuesday",
    "SSH-Patator": "Tuesday",
    "DoS slowloris": "Wednesday",
    "DoS Slowhttptest": "Wednesday",
    "DoS Hulk": "Wednesday",
    "DoS GoldenEye": "Wednesday",
    "Heartbleed": "Wednesday",
    "Web Attack - Brute Force": "Thursday",
    "Web Attack - XSS": "Thursday",
    "Web Attack - Sql Injection": "Thursday",
    "Infiltration": "Thursday",
    "Bot": "Friday",
    "PortScan": "Friday",
    "DDoS": "Friday",
}


def identifier_columns() -> list[str]:
    """Columns that identify the flow/session, not its behaviour (must be dropped)."""
    return list(IDENTIFIER_COLUMNS)


def feature_columns(*, include_destination_port: bool = False) -> list[str]:
    """Canonical feature columns, optionally excluding the borderline port column."""
    if include_destination_port:
        return list(FEATURE_COLUMNS)
    return [c for c in FEATURE_COLUMNS if c != DESTINATION_PORT]


def label_values() -> list[str]:
    """The raw label vocabulary (BENIGN plus the attack classes)."""
    return list(RAW_LABELS)


def attack_labels() -> list[str]:
    """The attack classes (everything except BENIGN)."""
    return [label for label in RAW_LABELS if label != BENIGN_LABEL]


def is_attack(label: str) -> bool:
    """Whether a (normalised) label denotes an attack."""
    return label != BENIGN_LABEL


def day_from_filename(filename: str) -> str | None:
    """Infer the capture day from a CIC-IDS2017 CSV filename, if recognisable."""
    lowered = filename.lower()
    for day in DAY_ORDER:
        if day.lower() in lowered:
            return day
    return None
