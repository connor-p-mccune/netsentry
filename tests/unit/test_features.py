"""The leakage-firewall tests: no identifier survives, and fits use train only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings
from netsentry.data import schema
from netsentry.features.feature_sets import numeric_features
from netsentry.features.pipeline import build_pipeline


def test_pipeline_drops_identifier_columns(clean_synth: pd.DataFrame, settings: Settings) -> None:
    # Even if an identifier leaks into X, the pipeline must discard it.
    contaminated = clean_synth.copy()
    contaminated["Source IP"] = "10.0.0.1"
    contaminated["Flow ID"] = "leak"

    pipe = build_pipeline(settings)
    pipe.fit(contaminated)
    names = list(pipe.named_steps["features"].get_feature_names_out())

    for leak in schema.identifier_columns():
        assert all(leak not in name for name in names), f"{leak} leaked into features"
    # Headline pipeline (no port) emits exactly the numeric features.
    assert len(names) == len(numeric_features())


def test_fit_uses_train_statistics_only(clean_synth: pd.DataFrame, settings: Settings) -> None:
    df = clean_synth.reset_index(drop=True).copy()
    half = len(df) // 2
    # Force a feature whose train median differs sharply from the combined median.
    df.loc[: half - 1, "Flow Duration"] = 10.0
    df.loc[half:, "Flow Duration"] = 1000.0
    train, test = df.iloc[:half], df.iloc[half:]

    pipe = build_pipeline(settings)
    pipe.fit(train)

    column_transformer = pipe.named_steps["features"]
    imputer = column_transformer.named_transformers_["numeric"].named_steps["impute"]
    idx = numeric_features().index("Flow Duration")
    # The imputer learned the TRAIN median (10), not the combined median (~505).
    assert imputer.statistics_[idx] == pytest.approx(10.0)

    # Transforming the test set must not error and must leave no NaNs.
    transformed = pipe.transform(test)
    assert not np.isnan(transformed).any()


def test_transform_output_shape_headline(clean_synth: pd.DataFrame, settings: Settings) -> None:
    pipe = build_pipeline(settings)
    out = pipe.fit_transform(clean_synth)
    assert out.shape[0] == len(clean_synth)
    assert out.shape[1] == len(numeric_features())


def test_port_variant_adds_categorical_features(
    clean_synth: pd.DataFrame, settings: Settings
) -> None:
    settings.features.encode_destination_port = True
    pipe = build_pipeline(settings)
    pipe.fit(clean_synth)
    names = list(pipe.named_steps["features"].get_feature_names_out())
    # The port variant is strictly wider than the numeric-only headline.
    assert len(names) > len(numeric_features())
    assert any("Destination Port" in name for name in names)
