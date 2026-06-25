"""The leakage firewall: one fitted sklearn pipeline applied at train and serve.

It drops identifier/leaky columns (via ``remainder="drop"`` — anything not in the
explicit feature lists is discarded), imputes with a train-fit median, scales,
and optionally one-hot-encodes ``Destination Port``. Fitting happens on the
**training split only**; the same fitted object is applied at serve time, so
train and serve preprocessing are guaranteed identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from netsentry.features.feature_sets import categorical_features, numeric_features

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings

NUMERIC_BRANCH = "numeric"
CATEGORICAL_BRANCH = "categorical"


def _scaler(name: str) -> object:
    scalers: dict[str, object] = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "none": "passthrough",
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler {name!r}")
    return scalers[name]


def build_pipeline(settings: Settings) -> Pipeline:
    """Construct the (unfitted) leakage-safe preprocessing pipeline."""
    numeric = numeric_features()
    categorical = categorical_features(settings)

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=settings.features.impute_strategy)),
    ]
    scaler = _scaler(settings.features.scaler)
    if scaler != "passthrough":
        numeric_steps.append(("scale", scaler))
    numeric_pipe = Pipeline(numeric_steps)

    transformers: list[tuple[str, object, list[str]]] = [(NUMERIC_BRANCH, numeric_pipe, numeric)]
    if categorical:
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        max_categories=settings.features.destination_port_top_k,
                    ),
                ),
            ]
        )
        transformers.append((CATEGORICAL_BRANCH, categorical_pipe, categorical))

    # remainder="drop" is the firewall: any column not explicitly listed above
    # (e.g. an identifier that slipped through) is discarded, never modelled.
    column_transformer = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("features", column_transformer)])


def feature_frame(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Select exactly the columns the pipeline consumes (defensive, explicit)."""
    columns = [
        col for col in (numeric_features() + categorical_features(settings)) if col in df.columns
    ]
    return df[columns]
