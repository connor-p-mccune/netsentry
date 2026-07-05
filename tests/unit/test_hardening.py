"""Adversarial hardening: augmentation mechanics + the end-to-end measure/fix loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netsentry.config import Settings, load_settings
from netsentry.robustness.hardening import adversarial_examples

# Post-transform feature names carry a ColumnTransformer branch prefix, which the
# controllable-index lookup strips — mirror that here so the test exercises the real
# name-matching path.
_NAMES = ["numeric__Flow Duration", "numeric__Total Fwd Packets", "numeric__SYN Flag Count"]


def _xy() -> tuple[np.ndarray, np.ndarray]:
    """Three benign rows and two attack rows in a 3-feature transformed space."""
    x = np.array(
        [
            [0.0, 0.0, 0.0],  # benign
            [2.0, 2.0, 2.0],  # benign
            [4.0, 4.0, 4.0],  # benign -> centroid (2,2,2)
            [10.0, 10.0, 10.0],  # attack
            [12.0, 12.0, 12.0],  # attack
        ]
    )
    y = np.array([0, 0, 0, 1, 1])
    return x, y


def test_adversarial_examples_are_all_attacks_moved_toward_benign(settings: Settings) -> None:
    # "SYN Flag Count" is not in the controllable set, so its column must stay fixed.
    settings.hardening.mimicry_train_fractions = [1.0]
    x, y = _xy()
    x_adv, y_adv = adversarial_examples(settings, x, y, _NAMES)

    assert len(x_adv) == 2  # one full-mimicry copy per attack row
    assert set(y_adv.tolist()) == {1}  # synthesized rows are still attacks
    centroid = x[y == 0].mean(axis=0)  # (2, 2, 2)
    # Controllable columns 0,1 collapse to the benign centroid at fraction 1.0…
    np.testing.assert_allclose(x_adv[:, 0], centroid[0])
    np.testing.assert_allclose(x_adv[:, 1], centroid[1])
    # …while the non-controllable column 2 is left at the original attack value.
    np.testing.assert_array_equal(x_adv[:, 2], x[y == 1][:, 2])


def test_adversarial_examples_stack_every_fraction(settings: Settings) -> None:
    settings.hardening.mimicry_train_fractions = [0.25, 0.5, 1.0]
    x, y = _xy()
    x_adv, _ = adversarial_examples(settings, x, y, _NAMES)
    assert len(x_adv) == 2 * 3  # attack rows x fractions


def test_adversarial_examples_respect_the_cap(settings: Settings) -> None:
    settings.hardening.mimicry_train_fractions = [0.5, 1.0]
    settings.hardening.max_augmented = 3
    x, y = _xy()
    x_adv, y_adv = adversarial_examples(settings, x, y, _NAMES)
    assert len(x_adv) == 3 and len(y_adv) == 3


def test_adversarial_examples_empty_without_controllable_features(settings: Settings) -> None:
    x, y = _xy()
    names = ["numeric__SYN Flag Count", "numeric__ACK Flag Count", "numeric__FIN Flag Count"]
    x_adv, y_adv = adversarial_examples(settings, x, y, names)
    assert x_adv.shape == (0, 3) and y_adv.shape == (0,)


def test_adversarial_examples_empty_without_attacks(settings: Settings) -> None:
    x, _ = _xy()
    y_all_benign = np.zeros(len(x), dtype=int)
    x_adv, _ = adversarial_examples(settings, x, y_all_benign, _NAMES)
    assert len(x_adv) == 0


@pytest.mark.slow
def test_hardening_runs_end_to_end_and_is_deterministic(
    repo_root: Path, tmp_path: Path, clean_synth: pd.DataFrame
) -> None:
    from netsentry.data.split import make_splits
    from netsentry.robustness.hardening import run_hardening

    settings: Settings = load_settings(repo_root / "configs" / "default.yaml")
    settings.paths.data_processed = tmp_path / "processed"
    settings.paths.models_dir = tmp_path / "models"
    settings.mlflow.enabled = False
    settings.supervised.n_estimators = 60
    settings.hardening.mimicry_train_fractions = [0.5, 1.0]
    settings.hardening.max_augmented = 800
    settings.robustness.mimicry_fractions = [0.0, 0.5, 1.0]
    settings.robustness.search_budgets = [0.0, 2.0]
    settings.robustness.search_iterations = 10
    settings.robustness.max_attack_samples = 300

    settings.paths.data_processed.mkdir(parents=True)
    clean_synth.to_parquet(settings.paths.data_processed / "clean.parquet", index=False)
    make_splits(settings)

    result = run_hardening(settings)
    assert result.n_augmented > 0
    assert result.baseline.mimicry_detection[0] == pytest.approx(result.baseline.baseline_detection)
    for detection in (*result.baseline.mimicry_detection, *result.hardened.mimicry_detection):
        assert 0.0 <= detection <= 1.0

    # Same seed + config -> identical numbers (the reproducibility invariant).
    again = run_hardening(settings)
    assert again.baseline_pr_auc == pytest.approx(result.baseline_pr_auc)
    assert again.hardened_pr_auc == pytest.approx(result.hardened_pr_auc)
    assert again.mimicry_gain == pytest.approx(result.mimicry_gain)
