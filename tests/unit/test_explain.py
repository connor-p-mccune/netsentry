"""Explainer contract: ranked per-prediction contributions and global importance."""

from __future__ import annotations

import pandas as pd

from netsentry.config import Settings
from netsentry.data.clean import BINARY_TARGET
from netsentry.explain.shap_explainer import ShapExplainer, top_feature_contributions
from netsentry.features.pipeline import build_pipeline
from netsentry.models.registry import ModelBundle
from netsentry.models.supervised import SupervisedClassifier


def _fitted_bundle(clean_synth: pd.DataFrame, settings: Settings) -> ModelBundle:
    settings.supervised.n_estimators = 60
    pipeline = build_pipeline(settings)
    x = pipeline.fit_transform(clean_synth)
    y = clean_synth[BINARY_TARGET].to_numpy()
    model = SupervisedClassifier(settings).fit(x, y)
    return ModelBundle(pipeline=pipeline, model=model, metadata={})


def test_explain_row_returns_ranked_topk(clean_synth: pd.DataFrame, settings: Settings) -> None:
    bundle = _fitted_bundle(clean_synth, settings)
    explainer = ShapExplainer(bundle)
    contributions = explainer.explain_row(clean_synth.head(1), k=5)

    assert len(contributions) == 5
    names = bundle.feature_names()
    for feature, value in contributions:
        assert feature in names
        assert isinstance(value, float)
    # Ranked by absolute contribution (descending).
    magnitudes = [abs(v) for _, v in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_global_importance_shape(clean_synth: pd.DataFrame, settings: Settings) -> None:
    bundle = _fitted_bundle(clean_synth, settings)
    explainer = ShapExplainer(bundle)
    background = bundle.pipeline.transform(clean_synth.head(100))
    top = explainer.global_importance(background, top_n=8)

    assert len(top) == 8
    assert all(value >= 0 for _, value in top)


def test_top_feature_contributions_wrapper(clean_synth: pd.DataFrame, settings: Settings) -> None:
    bundle = _fitted_bundle(clean_synth, settings)
    contributions = top_feature_contributions(bundle, clean_synth.head(1), k=3)
    assert len(contributions) == 3
