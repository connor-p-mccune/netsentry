"""Cross-dataset generalization: foreign generator + the schema adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings, load_settings
from netsentry.data import schema
from netsentry.data.clean import BINARY_TARGET
from netsentry.data.cross_dataset import (
    FOREIGN_COLUMNS,
    adapt_foreign_to_cic,
    generate_foreign,
)


@pytest.fixture
def settings(default_config_path: Path) -> Settings:
    s = load_settings(default_config_path)
    s.crossdata.rows = 3000
    return s


def test_generate_foreign_schema_and_determinism(settings: Settings) -> None:
    a = generate_foreign(settings, seed=7)
    b = generate_foreign(settings, seed=7)
    assert tuple(a.columns) == FOREIGN_COLUMNS
    assert set(a["Attack"].unique()) <= {0, 1}
    pd.testing.assert_frame_equal(a, b)  # same seed -> identical


def test_generate_foreign_attacks_are_higher_volume(settings: Settings) -> None:
    foreign = generate_foreign(settings, seed=1)
    attacks = foreign[foreign["Attack"] == 1]["IN_PKTS"].mean()
    benign = foreign[foreign["Attack"] == 0]["IN_PKTS"].mean()
    assert attacks > benign  # transferable DoS-like signal


def test_adapter_produces_all_cic_features_and_target(settings: Settings) -> None:
    adapted = adapt_foreign_to_cic(generate_foreign(settings, seed=2), settings)
    for col in schema.FEATURE_COLUMNS:
        assert col in adapted.columns
    assert BINARY_TARGET in adapted.columns
    assert len(adapted) == settings.crossdata.rows


def test_adapter_unit_conversion_and_rate_derivation(settings: Settings) -> None:
    foreign = pd.DataFrame(
        {
            "IN_PKTS": [10.0],
            "OUT_PKTS": [30.0],
            "IN_BYTES": [1000.0],
            "OUT_BYTES": [3000.0],
            "FLOW_DURATION_MILLISECONDS": [200.0],  # -> 200_000 us, 0.2 s
            "L4_DST_PORT": [80],
            "PROTOCOL": [6],
            "TCP_FLAGS": [24],
            "Attack": [1],
        }
    )
    adapted = adapt_foreign_to_cic(foreign, settings)
    assert adapted["Flow Duration"].iloc[0] == pytest.approx(200_000.0)  # ms -> us
    assert adapted["Flow Packets/s"].iloc[0] == pytest.approx(40 / 0.2)  # (10+30)/0.2s
    assert adapted["Down/Up Ratio"].iloc[0] == pytest.approx(3.0)  # 30/10


def test_adapter_unmapped_features_are_nan(settings: Settings) -> None:
    adapted = adapt_foreign_to_cic(generate_foreign(settings, seed=3), settings)
    # A CIC feature with no NetFlow equivalent must be left NaN (pipeline imputes it).
    assert adapted["Active Mean"].isna().all()
    assert adapted["Total Fwd Packets"].notna().any()  # a mapped feature is populated


def test_adapter_has_no_infinities(settings: Settings) -> None:
    adapted = adapt_foreign_to_cic(generate_foreign(settings, seed=4), settings)
    numeric = adapted[list(schema.FEATURE_COLUMNS)].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()  # inf was replaced with NaN for the imputer
