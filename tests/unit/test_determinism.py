"""The two digests, and the diff that turns a hash mismatch into a sentence.

The study's whole argument rests on a distinction between two questions -- *are these the same
bytes* and *is this the same function* -- so the two digests get tested against each other
directly: they must agree when nothing changed, disagree when a tree changes, and disagree in
exactly one direction when only the recorded environment changes.
"""

from __future__ import annotations

import pytest

from netsentry.training.determinism import (
    ENVIRONMENT_KEYS,
    DeterminismStudy,
    VariantRow,
    artifact_digest,
    behavioural_digest,
    first_difference,
)

MODEL = "\n".join(
    [
        "tree",
        "num_leaves=3",
        "split_feature=2 7",
        "threshold=0.5 1.25",
        "leaf_value=-0.1 0.4 0.9",
        "[num_threads: 14]",
        "[boosting: gbdt]",
        "end of parameters",
    ]
)


def _with(line: str, replacement: str) -> str:
    return MODEL.replace(line, replacement)


# --------------------------------------------------------------------------------------
# The digests.
# --------------------------------------------------------------------------------------


def test_both_digests_agree_with_themselves() -> None:
    assert artifact_digest(MODEL) == artifact_digest(MODEL)
    assert behavioural_digest(MODEL) == behavioural_digest(MODEL)


def test_the_thread_count_moves_the_file_and_not_the_function() -> None:
    """The finding, as an assertion: `n_jobs` is recorded, and recording is not computing."""
    other = _with("[num_threads: 14]", "[num_threads: 1]")
    assert artifact_digest(MODEL) != artifact_digest(other)
    assert behavioural_digest(MODEL) == behavioural_digest(other)


def test_a_changed_threshold_moves_both() -> None:
    """The digest has to stay sensitive to the thing it exists to detect."""
    other = _with("threshold=0.5 1.25", "threshold=0.6 1.25")
    assert artifact_digest(MODEL) != artifact_digest(other)
    assert behavioural_digest(MODEL) != behavioural_digest(other)


def test_a_changed_leaf_value_moves_both() -> None:
    other = _with("leaf_value=-0.1 0.4 0.9", "leaf_value=-0.1 0.4 0.95")
    assert behavioural_digest(MODEL) != behavioural_digest(other)


@pytest.mark.parametrize("key", ENVIRONMENT_KEYS)
def test_every_declared_environment_key_is_stripped(key: str) -> None:
    """A future alias must be added to the list, not silently included in the digest."""
    with_key = MODEL + f"\n[{key}: 99]"
    without = MODEL + f"\n[{key}: 4]"
    assert behavioural_digest(with_key) == behavioural_digest(without)


def test_a_parameter_that_is_not_environmental_still_counts() -> None:
    """Stripping is a whitelist: `boosting` is a model choice and must survive."""
    other = _with("[boosting: gbdt]", "[boosting: dart]")
    assert behavioural_digest(MODEL) != behavioural_digest(other)


# --------------------------------------------------------------------------------------
# Turning a mismatch into a sentence.
# --------------------------------------------------------------------------------------


def test_the_diff_names_the_line_that_changed() -> None:
    other = _with("[num_threads: 14]", "[num_threads: 1]")
    lines = first_difference(MODEL, other)
    assert any("num_threads: 14" in line for line in lines)
    assert any("num_threads: 1]" in line for line in lines)


def test_identical_models_produce_no_diff() -> None:
    assert first_difference(MODEL, MODEL) == []


def test_the_diff_is_bounded() -> None:
    """A hash mismatch on two unrelated models must not paste an entire model into a report."""
    other = "\n".join(f"line {index}" for index in range(500))
    assert len(first_difference(MODEL, other, limit=3)) == 3


# --------------------------------------------------------------------------------------
# The records the report reads from.
# --------------------------------------------------------------------------------------


def _row(artifact: bool, behavioural: bool, flips: int = 0) -> VariantRow:
    return VariantRow(
        variant="x",
        changes="y",
        artifact_stable=artifact,
        behavioural_stable=behavioural,
        margins_identical=behavioural,
        decision_flips=flips,
        pr_auc_delta=0.0,
        difference=[],
    )


def test_the_verdict_separates_the_three_outcomes() -> None:
    assert _row(True, True).verdict == "identical"
    assert _row(False, True).verdict == "same model, different bytes"
    assert "different model" in _row(False, False).verdict


def test_the_study_separates_byte_and_behavioural_failures() -> None:
    study = DeterminismStudy(
        variants=[_row(True, True), _row(False, True), _row(False, False, flips=3)],
        mechanisms=[],
        thread_counts=[1],
        reference_threads=-1,
        fit_seconds={},
        n_alerts=10,
        n_scored=100,
    )
    assert len(study.unstable()) == 2
    assert len(study.behaviourally_unstable()) == 1
