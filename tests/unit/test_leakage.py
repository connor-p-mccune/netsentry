"""Leakage-attribution primitives: the injected session identifier and dense coercion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.evaluation.leakage import LadderRung, _dense, session_identifier


def _frame(days: list[str], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Day": days, MULTICLASS_TARGET: labels})


def test_session_id_is_constant_within_a_campaign() -> None:
    df = _frame(["Wednesday", "Wednesday", "Wednesday"], ["DoS Hulk", "DoS Hulk", "DoS Hulk"])
    ids = session_identifier(df)
    assert len(set(ids)) == 1  # one (day, class) campaign -> one id


def test_session_id_differs_across_campaigns() -> None:
    df = _frame(["Wednesday", "Friday"], ["DoS Hulk", "DDoS"])
    ids = session_identifier(df)
    assert ids[0] != ids[1]


def test_session_id_is_stable_across_frames() -> None:
    # Train and test must agree on a campaign's code for the leak to transfer.
    a = session_identifier(_frame(["Friday"], ["DDoS"]))
    b = session_identifier(_frame(["Friday", "Monday"], ["DDoS", "BENIGN"]))
    assert a[0] == b[0]


def test_dense_coerces_sparse_and_passes_dense_through() -> None:
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(_dense(dense), dense)
    assert np.array_equal(_dense(sp.csr_matrix(dense)), dense)


def test_ladder_rung_carries_delta() -> None:
    rung = LadderRung("+ port", "port memorisation", 0.85, 0.06)
    assert rung.delta == 0.06 and rung.pr_auc == 0.85
