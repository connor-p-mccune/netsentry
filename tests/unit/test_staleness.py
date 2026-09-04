"""Statistic extraction, the drift measure, and the arm-ordering argument.

The load-bearing claim is not that any single gap is small -- it is that the *oracle arm comes
last*, which is what turns "the effect is below the noise floor" into "the quantity does not
matter". That ordering test is the one that must not rot.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from netsentry.features.staleness import (
    Arm,
    StalenessStudy,
    StatisticDrift,
    compare_statistics,
    fitted_statistics,
)

# --------------------------------------------------------------------------------------
# Reading the fitted constants.
# --------------------------------------------------------------------------------------


class _Branch:
    """The shape `fitted_statistics` reaches through: a ColumnTransformer's numeric branch."""

    def __init__(self, numeric: object) -> None:
        self.named_transformers_ = {"numeric": numeric}


class _Outer:
    def __init__(self, branch: object) -> None:
        self.named_steps = {"features": branch}


def _fitted_numeric(values: np.ndarray) -> Pipeline:
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pipe.fit(values)
    return pipe


def test_the_imputer_and_scaler_constants_are_both_read() -> None:
    numeric = _fitted_numeric(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    found = fitted_statistics(_Outer(_Branch(numeric)), ["a", "b"])
    assert set(found) == {"impute", "centre", "scale"}
    assert found["impute"] == pytest.approx([2.0, 20.0])
    assert found["centre"] == pytest.approx([2.0, 20.0])


def test_a_pipeline_without_a_numeric_branch_yields_nothing() -> None:
    """Configurations vary; a missing branch must be empty rather than an exception."""
    assert fitted_statistics(_Outer(_Branch(None)), ["a"]) == {}


def test_an_object_that_is_not_a_pipeline_yields_nothing() -> None:
    assert fitted_statistics(object(), ["a"]) == {}


# --------------------------------------------------------------------------------------
# Comparing two fits.
# --------------------------------------------------------------------------------------


def test_every_constant_is_paired_between_the_fits() -> None:
    train = {"impute": np.array([1.0, 2.0]), "centre": np.array([1.0, 2.0])}
    later = {"impute": np.array([2.0, 4.0]), "centre": np.array([1.0, 2.0])}
    rows = compare_statistics(train, later, ["a", "b"])
    assert len(rows) == 4
    assert {row.statistic for row in rows} == {"impute", "centre"}


def test_a_statistic_missing_from_the_second_fit_is_skipped() -> None:
    rows = compare_statistics({"impute": np.array([1.0])}, {}, ["a"])
    assert rows == []


def test_mismatched_lengths_are_skipped_rather_than_zipped() -> None:
    """Silently truncating would pair a feature's statistic with another feature's."""
    rows = compare_statistics(
        {"impute": np.array([1.0, 2.0])}, {"impute": np.array([1.0])}, ["a", "b"]
    )
    assert rows == []


def test_movement_is_relative_to_the_training_value() -> None:
    assert StatisticDrift("a", "centre", 4.0, 5.0).relative == pytest.approx(0.25)


def test_movement_from_zero_does_not_divide_by_zero() -> None:
    """A constant that was zero on the training days is common and must not blow up."""
    assert StatisticDrift("a", "centre", 0.0, 3.0).relative == pytest.approx(3.0)


def test_movement_is_signed() -> None:
    assert StatisticDrift("a", "scale", 4.0, 2.0).relative < 0


def test_an_unnamed_column_is_still_reported() -> None:
    rows = compare_statistics(
        {"impute": np.array([1.0, 2.0])}, {"impute": np.array([1.0, 2.0])}, []
    )
    assert [row.feature for row in rows] == ["column 0", "column 1"]


# --------------------------------------------------------------------------------------
# The arms and the ordering argument.
# --------------------------------------------------------------------------------------


def _arm(name: str, pr_auc: float, legitimacy: str = "no labels needed") -> Arm:
    return Arm(
        name=name,
        describes="",
        legitimacy=legitimacy,
        pr_auc=pr_auc,
        detection=0.20,
        realised_fpr=0.0082,
        baseline_pr_auc=0.5276,
        baseline_detection=0.207,
    )


def _study(arms: list[Arm], detectable: float = 0.0168) -> StalenessStudy:
    return StalenessStudy(
        arms=arms,
        drifts=[StatisticDrift("a", "centre", 1.0, 2.0)],
        budget=0.01,
        n_train=28034,
        n_test=24957,
        refit_rows=4991,
        detectable=detectable,
        imputed_rows=0.025,
    )


def _four(deployed: float, periodic: float, transductive: float, oracle: float) -> list[Arm]:
    return [
        _arm("the deployed pipeline", deployed, "the shipped rule"),
        _arm("periodic refit", periodic),
        _arm("transductive (all later-day features)", transductive),
        _arm("oracle (fit on the later days alone)", oracle, "leaks labels"),
    ]


def test_the_oracle_arm_is_not_deployable() -> None:
    study = _study(_four(0.52, 0.52, 0.53, 0.55))
    assert not study.oracle().allowed
    assert len(study.legitimate()) == 3


def test_the_spread_is_the_whole_effect_whatever_its_sign() -> None:
    study = _study(_four(0.5276, 0.5276, 0.5287, 0.5264))
    assert study.spread() == pytest.approx(0.0023, abs=1e-4)


def test_an_oracle_that_comes_last_shows_the_quantity_does_not_matter() -> None:
    """The measured case: the arm that cheats is not the best, so the ordering is noise."""
    study = _study(_four(0.5276, 0.5276, 0.5287, 0.5264))
    assert not study.oracle_wins()
    assert not study.worth_doing()


def test_an_oracle_that_wins_is_reported_as_winning() -> None:
    study = _study(_four(0.50, 0.52, 0.54, 0.58))
    assert study.oracle_wins()
    assert study.worth_doing()


def test_the_recoverable_amount_is_measured_against_the_noise_floor() -> None:
    small = _study(_four(0.52, 0.52, 0.5205, 0.53))
    large = _study(_four(0.52, 0.53, 0.56, 0.58))
    assert not small.worth_doing()
    assert large.worth_doing()


def test_the_best_legitimate_arm_never_cheats() -> None:
    study = _study(_four(0.52, 0.52, 0.53, 0.99))
    assert study.best_legitimate().name.startswith("transductive")


def test_an_arm_reports_its_gain_against_the_deployed_pipeline() -> None:
    arm = _arm("transductive (all later-day features)", 0.5287)
    assert arm.pr_auc_gain == pytest.approx(0.0011, abs=1e-6)


def test_a_moved_statistic_is_counted_against_a_threshold() -> None:
    study = StalenessStudy(
        arms=_four(0.52, 0.52, 0.52, 0.52),
        drifts=[
            StatisticDrift("a", "centre", 1.0, 1.05),
            StatisticDrift("b", "centre", 1.0, 2.0),
        ],
        budget=0.01,
        n_train=1,
        n_test=1,
        refit_rows=1,
        detectable=0.0168,
    )
    assert study.moved_statistics(0.25) == 1


def test_the_worst_drifts_are_ranked_by_absolute_movement() -> None:
    study = StalenessStudy(
        arms=_four(0.52, 0.52, 0.52, 0.52),
        drifts=[
            StatisticDrift("small", "centre", 1.0, 1.1),
            StatisticDrift("big", "centre", 1.0, -2.0),
        ],
        budget=0.01,
        n_train=1,
        n_test=1,
        refit_rows=1,
        detectable=0.0168,
    )
    assert [row.feature for row in study.worst_drifts(1)] == ["big"]
