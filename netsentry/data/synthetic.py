"""Synthetic CIC-IDS2017-shaped data.

When the real dataset is unavailable (it requires registration with the CIC and
is large), this generates a schema-faithful stand-in so the full pipeline, the
tests, and the CI smoke train can run. It deliberately reproduces the dataset's
character:

- the exact column schema, including identifier columns to be dropped;
- class imbalance with rare classes (Heartbleed, Infiltration);
- class-conditional feature signal that is learnable but **overlapping**, so the
  honest metrics look like a real problem, not a leaked 99.99%;
- the well-known data defects (Inf in rate columns, -1 sentinels, duplicates);
- a per-day attack layout so the temporal split is meaningful;
- ``Destination Port`` correlated with attack class, so the port-leakage concern
  can actually be demonstrated.

Synthetic data is always logged and labelled as such; it is never presented as a
real-world result.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

# Heavy-tailed scale per feature (defaults to 1.0). Only a handful need tuning to
# land in a realistic order of magnitude.
_BASE_SCALE: dict[str, float] = {
    "Flow Duration": 1.0e5,
    "Flow Bytes/s": 1.0e3,
    "Flow Packets/s": 1.0e2,
    "Total Fwd Packets": 1.0e1,
    "Total Backward Packets": 1.0e1,
    "Total Length of Fwd Packets": 5.0e2,
    "Total Length of Bwd Packets": 5.0e2,
    "Flow IAT Mean": 1.0e4,
    "Flow IAT Max": 1.0e4,
    "Fwd IAT Total": 1.0e4,
    "Init_Win_bytes_forward": 8.192e3,
    "Init_Win_bytes_backward": 8.192e3,
    "Fwd Packet Length Mean": 5.0e1,
    "Bwd Packet Length Mean": 5.0e1,
    "Bwd Packet Length Max": 5.0e1,
    "Average Packet Size": 5.0e1,
    "Min Packet Length": 4.0e1,
    "Max Packet Length": 6.0e1,
}

# Multiplicative shifts applied to signal features per attack class. Factors are
# modest and overlap across classes on purpose.
_PROFILES: dict[str, dict[str, float]] = {
    "DoS Hulk": {"Flow Packets/s": 8, "Total Fwd Packets": 6, "Flow Bytes/s": 7},
    "DoS GoldenEye": {"Flow Duration": 4, "Flow IAT Max": 4, "Total Fwd Packets": 3},
    "DoS slowloris": {"Flow Duration": 6, "Flow IAT Mean": 5, "Total Fwd Packets": 0.5},
    "DoS Slowhttptest": {"Flow Duration": 5, "Flow IAT Mean": 4},
    "DDoS": {"Flow Packets/s": 10, "Total Backward Packets": 7, "Flow Bytes/s": 8},
    "PortScan": {"Flow Duration": 0.2, "Total Fwd Packets": 0.3, "SYN Flag Count": 5},
    "Bot": {"Flow Bytes/s": 2, "Bwd Packet Length Mean": 2, "Down/Up Ratio": 3},
    "FTP-Patator": {"Total Fwd Packets": 2, "SYN Flag Count": 3},
    "SSH-Patator": {"Total Fwd Packets": 2.5, "SYN Flag Count": 3},
    "Web Attack - Brute Force": {"Total Fwd Packets": 2, "Fwd Packet Length Mean": 1.5},
    "Web Attack - XSS": {"Fwd Packet Length Mean": 1.8},
    "Web Attack - Sql Injection": {"Fwd Packet Length Mean": 2.0},
    "Heartbleed": {"Bwd Packet Length Max": 6, "Total Length of Bwd Packets": 8},
    "Infiltration": {"Flow Duration": 2, "Bwd Packet Length Mean": 1.5},
}

# Relative frequency of each attack class (rare classes are genuinely rare).
_ATTACK_WEIGHTS: dict[str, float] = {
    "DoS Hulk": 1.0,
    "PortScan": 0.9,
    "DDoS": 0.7,
    "DoS GoldenEye": 0.3,
    "FTP-Patator": 0.2,
    "SSH-Patator": 0.18,
    "DoS slowloris": 0.15,
    "DoS Slowhttptest": 0.14,
    "Bot": 0.1,
    "Web Attack - Brute Force": 0.05,
    "Web Attack - XSS": 0.025,
    "Infiltration": 0.01,
    "Web Attack - Sql Injection": 0.008,
    "Heartbleed": 0.004,
}

# Characteristic destination port per attack class (PortScan sprays many).
_ATTACK_PORTS: dict[str, int] = {
    "FTP-Patator": 21,
    "SSH-Patator": 22,
    "Web Attack - Brute Force": 80,
    "Web Attack - XSS": 80,
    "Web Attack - Sql Injection": 80,
    "DoS Hulk": 80,
    "DoS GoldenEye": 80,
    "DoS slowloris": 80,
    "DoS Slowhttptest": 80,
    "DDoS": 80,
    "Bot": 8080,
    "Heartbleed": 443,
    "Infiltration": 444,
}
_BENIGN_PORTS = np.array([80, 443, 53, 22, 25, 110, 143, 993, 8080])


def _random_ips(rng: np.random.Generator, n: int) -> np.ndarray:
    octets = rng.integers(1, 255, size=(n, 4))
    return np.array([".".join(map(str, row)) for row in octets])


def generate_synthetic(
    settings: Settings, *, rows: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Generate a labelled, schema-faithful synthetic flow DataFrame.

    Returns a DataFrame with the identifier columns, the 78 feature columns, a
    ``Label`` column, and a ``Day`` column (used by the temporal split).
    """
    n = rows if rows is not None else settings.data.synthetic_rows
    rng = np.random.default_rng(seed if seed is not None else settings.seed)
    attack_fraction = settings.data.synthetic_attack_fraction

    labels = _draw_labels(rng, n, attack_fraction)
    frame = _draw_features(rng, labels)
    _apply_class_signal(rng, frame, labels)
    _add_identifiers_and_ports(rng, frame, labels)
    _inject_defects(rng, frame)

    frame[schema.LABEL_COLUMN] = labels
    frame[schema.DAY_COLUMN] = _assign_days(rng, labels)

    n_attacks = int((labels != schema.BENIGN_LABEL).sum())
    logger.info(
        "Generated SYNTHETIC dataset (not real CIC-IDS2017)",
        extra={"rows": len(frame), "attacks": n_attacks, "classes": int(pd.unique(labels).size)},
    )
    return frame


def _draw_labels(rng: np.random.Generator, n: int, attack_fraction: float) -> np.ndarray:
    attacks = list(_ATTACK_WEIGHTS)
    weights = np.array([_ATTACK_WEIGHTS[a] for a in attacks], dtype=float)
    weights /= weights.sum()
    all_labels = [schema.BENIGN_LABEL, *attacks]
    probs = np.concatenate([[1.0 - attack_fraction], weights * attack_fraction])
    return rng.choice(all_labels, size=n, p=probs)


def _draw_features(rng: np.random.Generator, labels: np.ndarray) -> pd.DataFrame:
    n = len(labels)
    data: dict[str, np.ndarray] = {}
    for feature in schema.FEATURE_COLUMNS:
        if feature == schema.DESTINATION_PORT:
            continue  # filled in by _add_identifiers_and_ports
        scale = _BASE_SCALE.get(feature, 1.0)
        data[feature] = rng.lognormal(mean=0.0, sigma=1.0, size=n) * scale
    return pd.DataFrame(data)


def _apply_class_signal(rng: np.random.Generator, frame: pd.DataFrame, labels: np.ndarray) -> None:
    for label, profile in _PROFILES.items():
        mask = labels == label
        if not mask.any():
            continue
        count = int(mask.sum())
        for feature, factor in profile.items():
            jitter = rng.normal(1.0, 0.25, size=count).clip(0.1, None)
            frame.loc[mask, feature] = frame.loc[mask, feature].to_numpy() * factor * jitter


def _add_identifiers_and_ports(
    rng: np.random.Generator, frame: pd.DataFrame, labels: np.ndarray
) -> None:
    n = len(labels)
    dst_ports = np.empty(n, dtype=np.int64)
    for i, label in enumerate(labels):
        if label == schema.BENIGN_LABEL:
            dst_ports[i] = int(rng.choice(_BENIGN_PORTS))
        elif label == "PortScan":
            dst_ports[i] = int(rng.integers(1, 65535))
        else:
            dst_ports[i] = _ATTACK_PORTS.get(label, 80)
    frame[schema.DESTINATION_PORT] = dst_ports

    src_ip = _random_ips(rng, n)
    dst_ip = _random_ips(rng, n)
    src_port = rng.integers(1024, 65535, size=n)
    frame["Flow ID"] = [
        f"{s}-{d}-{sp}-{dp}-6"
        for s, d, sp, dp in zip(src_ip, dst_ip, src_port, dst_ports, strict=True)
    ]
    frame["Source IP"] = src_ip
    frame["Source Port"] = src_port
    frame["Destination IP"] = dst_ip
    base = np.datetime64("2017-07-03T09:00:00")
    offsets = rng.integers(0, 8 * 3600, size=n).astype("timedelta64[s]")
    frame["Timestamp"] = (base + offsets).astype(str)


def _inject_defects(rng: np.random.Generator, frame: pd.DataFrame) -> None:
    n = len(frame)
    # Instantaneous flows: zero duration -> Inf in the per-second rate columns.
    inf_mask = rng.random(n) < 0.02
    frame.loc[inf_mask, "Flow Duration"] = 0.0
    for col in schema.RATE_COLUMNS:
        frame.loc[inf_mask, col] = np.inf
    # "Not set" window size -> -1 sentinel (not a real byte count).
    sentinel_mask = rng.random(n) < 0.05
    frame.loc[sentinel_mask, "Init_Win_bytes_forward"] = -1
    # A few genuine NaNs.
    nan_mask = rng.random(n) < 0.005
    frame.loc[nan_mask, "Flow IAT Std"] = np.nan


def _assign_days(rng: np.random.Generator, labels: np.ndarray) -> np.ndarray:
    days = np.empty(len(labels), dtype=object)
    benign = labels == schema.BENIGN_LABEL
    days[benign] = rng.choice(schema.DAY_ORDER, size=int(benign.sum()))
    for i, label in enumerate(labels):
        if not benign[i]:
            days[i] = schema.LABEL_DAYS.get(label, "Wednesday")
    return days


def write_synthetic_raw(settings: Settings, *, rows: int | None = None) -> list[Path]:
    """Generate synthetic data and write it as per-day CSVs into ``data/raw``.

    The ``Day`` column is dropped on write (mirroring the real per-day CSVs, whose
    day is encoded in the filename) and a couple of headers are given the
    dataset's characteristic leading whitespace so cleaning is exercised.
    """
    raw_dir = settings.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic(settings, rows=rows)

    written: list[Path] = []
    for day, group in frame.groupby(schema.DAY_COLUMN, sort=False):
        out = group.drop(columns=[schema.DAY_COLUMN]).copy()
        # Mimic the real CSVs' leading-whitespace header defect on a few columns.
        out = out.rename(columns={"Flow Duration": " Flow Duration", "Label": " Label"})
        path = raw_dir / f"{day}-synthetic.csv"
        out.to_csv(path, index=False)
        written.append(path)
    logger.info("Wrote synthetic raw CSVs", extra={"files": len(written), "dir": str(raw_dir)})
    return written
