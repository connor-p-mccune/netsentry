"""A synthetic 'foreign' dataset (NetFlow schema) for cross-dataset generalization.

Real cross-dataset evaluation — train on CIC-IDS2017, test on UNSW-NB15 or the
NetFlow ``NF-*-v2`` datasets — is the strongest honesty test there is: it measures
whether the model learned *attack behaviour* or just one capture's idiosyncrasies.
Those datasets are not shipped here, so this generates a schema-faithful
NetFlow-style stand-in (different column names, a small feature subset) plus an
adapter that maps it into the CIC feature space.

The realistic difficulty this surfaces is **feature-space mismatch**: a NetFlow
record exposes a handful of counters, so most of CIC's 78 features have no
equivalent and must be imputed. Detection transfers only through the few shared,
behaviour-bearing quantities (packet/byte volumes and rates). That is exactly the
honest cross-dataset story — the methodology, not the absolute number, is the point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET
from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

# NetFlow-style columns (cf. the NF-*-v2 schema): deliberately *not* CIC names.
FOREIGN_COLUMNS: tuple[str, ...] = (
    "IN_PKTS",
    "OUT_PKTS",
    "IN_BYTES",
    "OUT_BYTES",
    "FLOW_DURATION_MILLISECONDS",
    "L4_DST_PORT",
    "PROTOCOL",
    "TCP_FLAGS",
    "Attack",
)


def generate_foreign(
    settings: Settings, *, rows: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Generate a labelled NetFlow-schema dataset with transferable attack signal.

    Attacks are high-volume, high-rate, short flows (DoS/DDoS-like) — behaviour
    that overlaps what the CIC-trained model learned, so detection partly (and
    honestly imperfectly) transfers across the schema gap.
    """
    cfg = settings.crossdata
    n = rows if rows is not None else cfg.rows
    rng = np.random.default_rng(seed if seed is not None else settings.seed + 1)
    attack = rng.random(n) < cfg.attack_fraction

    in_pkts = rng.lognormal(2.0, 0.8, n)
    out_pkts = rng.lognormal(1.8, 0.8, n)
    pkt_size = rng.lognormal(4.0, 0.6, n)
    duration_ms = rng.lognormal(4.6, 1.0, n)

    # Modest, overlapping elevation (cf. the main synthetic generator's "learnable
    # but overlapping" signal): attacks are higher-volume and a touch shorter, but the
    # distributions overlap benign so the foreign set is a realistic challenge, not a
    # giveaway. Aggressive separation here would make cross-eval look deceptively easy.
    in_pkts = np.where(attack, in_pkts * rng.uniform(1.3, 2.5, n), in_pkts)
    out_pkts = np.where(attack, out_pkts * rng.uniform(1.2, 2.2, n), out_pkts)
    duration_ms = np.where(attack, duration_ms * rng.uniform(0.5, 1.2, n), duration_ms)

    in_bytes = in_pkts * pkt_size
    out_bytes = out_pkts * pkt_size * rng.uniform(0.5, 1.5, n)
    dst_port = np.where(
        attack, rng.choice([80, 443, 53], n), rng.choice([80, 443, 22, 25, 8080], n)
    )

    frame = pd.DataFrame(
        {
            "IN_PKTS": np.round(in_pkts),
            "OUT_PKTS": np.round(out_pkts),
            "IN_BYTES": np.round(in_bytes),
            "OUT_BYTES": np.round(out_bytes),
            "FLOW_DURATION_MILLISECONDS": np.round(duration_ms),
            "L4_DST_PORT": dst_port.astype(int),
            "PROTOCOL": rng.choice([6, 17], n),
            "TCP_FLAGS": rng.integers(0, 255, n),
            "Attack": attack.astype(int),
        }
    )
    logger.info(
        "Generated SYNTHETIC foreign dataset (%s, NetFlow schema)",
        cfg.name,
        extra={"rows": n, "attacks": int(attack.sum())},
    )
    return frame


def adapt_foreign_to_cic(foreign: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Map a NetFlow-schema frame into CIC features plus a binary target.

    Overlapping quantities are renamed/unit-converted, a few CIC rates are derived,
    and every CIC feature with no NetFlow equivalent is left NaN for the fitted
    pipeline to impute (train median) — the honest reality of cross-schema serving.
    """
    in_pkts = foreign["IN_PKTS"].to_numpy(dtype=float)
    out_pkts = foreign["OUT_PKTS"].to_numpy(dtype=float)
    in_bytes = foreign["IN_BYTES"].to_numpy(dtype=float)
    out_bytes = foreign["OUT_BYTES"].to_numpy(dtype=float)
    duration_us = foreign["FLOW_DURATION_MILLISECONDS"].to_numpy(dtype=float) * 1000.0  # ms -> us
    dst_port = foreign["L4_DST_PORT"].to_numpy(dtype=float)

    total_pkts = in_pkts + out_pkts
    total_bytes = in_bytes + out_bytes
    # NaN-safe denominators so derived rates never divide by zero (-> NaN, imputed).
    dur_s = np.where(duration_us > 0.0, duration_us / 1e6, np.nan)
    safe_pkts = np.where(total_pkts > 0.0, total_pkts, np.nan)
    safe_in = np.where(in_pkts > 0.0, in_pkts, np.nan)

    mapped: dict[str, np.ndarray] = {
        "Flow Duration": duration_us,
        "Total Fwd Packets": in_pkts,
        "Total Backward Packets": out_pkts,
        "Total Length of Fwd Packets": in_bytes,
        "Total Length of Bwd Packets": out_bytes,
        "Destination Port": dst_port,
        "Flow Bytes/s": total_bytes / dur_s,
        "Flow Packets/s": total_pkts / dur_s,
        "Fwd Packets/s": in_pkts / dur_s,
        "Bwd Packets/s": out_pkts / dur_s,
        "Average Packet Size": total_bytes / safe_pkts,
        "Down/Up Ratio": out_pkts / safe_in,
    }

    out = pd.DataFrame(index=foreign.index)
    for col in schema.FEATURE_COLUMNS:
        out[col] = np.nan
    for col, values in mapped.items():
        out[col] = values
    out = out.replace([np.inf, -np.inf], np.nan)
    out[BINARY_TARGET] = foreign["Attack"].to_numpy(dtype=int)
    return out
