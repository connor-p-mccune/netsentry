"""The two-party protocol: correctness, the pad, and the two ways it stops protecting anything.

A secure-computation implementation that returns the right answer is not evidence of anything
on its own -- the wrong protocol also returns the right answer. So the tests here check the
*security* properties directly: that a reused triple leaks the input difference exactly (which
is why the report carries a broken control), and that a client sending a vector the server
cannot inspect reads a table entry out in the clear.
"""

from __future__ import annotations

import numpy as np
import pytest

from netsentry.models.gam import AdditiveModel, Binner
from netsentry.serving.private_inference import (
    ELEMENT_BYTES,
    PRIME,
    PrivateAdditiveModel,
    deal_triples,
    decode,
    encode,
    private_score,
    reconstruct,
    secure_products,
    share,
)


def _model() -> AdditiveModel:
    return AdditiveModel(
        intercept=-0.75,
        shapes=[np.array([0.3, -1.2, 2.5, 0.1]), np.array([-0.4, 1.9])],
        binner=Binner(edges=[np.array([-1.0, 0.0, 1.0]), np.array([0.5])]),
    )


# --------------------------------------------------------------------------------------
# Field encoding.
# --------------------------------------------------------------------------------------


def test_encoding_round_trips_positive_and_negative_values() -> None:
    values = np.array([0.0, 1.5, -2.25, 1e-3, -1e-3])
    assert np.allclose(decode(encode(values, 20), 20), values, atol=1e-6)


def test_negative_values_live_in_the_upper_half_of_the_field() -> None:
    """The two's-complement-like convention the decoder depends on."""
    assert int(encode(np.array([-1.0]), 16)[0]) > PRIME // 2
    assert int(encode(np.array([1.0]), 16)[0]) < PRIME // 2


def test_more_fraction_bits_means_less_quantisation() -> None:
    value = np.array([0.123456789])
    coarse = abs(float(decode(encode(value, 4), 4)[0]) - float(value[0]))
    fine = abs(float(decode(encode(value, 24), 24)[0]) - float(value[0]))
    assert fine < coarse


# --------------------------------------------------------------------------------------
# Secret sharing.
# --------------------------------------------------------------------------------------


def test_shares_reconstruct_the_secret() -> None:
    rng = np.random.default_rng(0)
    secret = encode(np.array([1.0, -2.0, 3.5]), 16)
    assert np.array_equal(reconstruct(*share(secret, rng)), secret)


def test_one_share_alone_is_uniform_over_the_field() -> None:
    """The privacy property, checked statistically rather than asserted."""
    rng = np.random.default_rng(1)
    first, _ = share(np.zeros(20000, dtype=np.int64), rng)
    counts = np.bincount((first * 8) // PRIME, minlength=8)
    expected = counts.sum() / 8
    statistic = float(((counts - expected) ** 2 / expected).sum())
    assert statistic < 24.3  # chi-square, 7 df, p = 0.001


def test_a_triple_really_is_a_multiplication() -> None:
    rng = np.random.default_rng(2)
    triple = deal_triples(64, rng)
    a = reconstruct(triple.a_first, triple.a_second)
    b = reconstruct(triple.b_first, triple.b_second)
    c = reconstruct(triple.c_first, triple.c_second)
    assert np.array_equal(c, np.mod(a * b, PRIME))


# --------------------------------------------------------------------------------------
# The multiplication protocol.
# --------------------------------------------------------------------------------------


def test_secure_products_computes_the_products() -> None:
    rng = np.random.default_rng(3)
    x = np.array([1, 0, 1, 0, 1], dtype=np.int64)
    y = encode(np.array([0.5, -1.5, 2.0, 3.0, -0.25]), 16)
    first, second, _ = secure_products(share(x, rng), share(y, rng), deal_triples(5, rng))
    assert np.array_equal(reconstruct(first, second), np.mod(x * y, PRIME))


def test_the_whole_batch_costs_one_round() -> None:
    rng = np.random.default_rng(4)
    triple = deal_triples(500, rng)
    _, _, transcript = secure_products(
        share(np.ones(500, dtype=np.int64), rng), share(np.ones(500, dtype=np.int64), rng), triple
    )
    assert transcript.rounds == 1
    assert transcript.online_bytes == 4 * 500 * ELEMENT_BYTES
    assert transcript.preprocessing_bytes == 6 * 500 * ELEMENT_BYTES


def test_reusing_a_triple_leaks_the_difference_between_inputs_exactly() -> None:
    """Why the report carries a broken control: this is the failure, stated as an equation.

    With a fresh triple the opened mask is a one-time pad. Reuse it across two flows and the
    difference of the two opened vectors is the difference of the two secret inputs, in the
    clear -- no statistics required, just subtraction.
    """
    rng = np.random.default_rng(5)
    triple = deal_triples(6, rng)
    table = share(encode(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 16), rng)
    first_input = np.array([1, 0, 0, 0, 0, 0], dtype=np.int64)
    second_input = np.array([0, 0, 1, 0, 0, 0], dtype=np.int64)
    _, _, one = secure_products(share(first_input, rng), table, triple)
    _, _, two = secure_products(share(second_input, rng), table, triple)
    leaked = np.mod(one.opened[0] - two.opened[0], PRIME)
    assert np.array_equal(leaked, np.mod(first_input - second_input, PRIME))


# --------------------------------------------------------------------------------------
# Private evaluation of the model.
# --------------------------------------------------------------------------------------


def test_the_private_score_matches_the_plaintext_model() -> None:
    rng = np.random.default_rng(6)
    model = _model()
    private = PrivateAdditiveModel.of(model, 20)
    x = np.array([[0.4, 0.9]])
    bins = model.binner.transform(x)
    score, _ = private_score(private, private.selectors(bins[0]), rng)
    assert score == pytest.approx(float(model.margin(bins)[0]), abs=1e-5)


def test_the_selectors_are_one_hot_and_sized_to_the_tables() -> None:
    private = PrivateAdditiveModel.of(_model(), 16)
    selectors = private.selectors(np.array([2, 1]))
    assert [len(vector) for vector in selectors] == [4, 2]
    assert [int(vector.sum()) for vector in selectors] == [1, 1]
    assert selectors[0][2] == 1 and selectors[1][1] == 1


def test_the_cost_is_one_multiplication_per_table_entry() -> None:
    private = PrivateAdditiveModel.of(_model(), 16)
    assert private.multiplications == 6


def test_too_many_fraction_bits_wraps_rather_than_rounding() -> None:
    """The failure mode that does not announce itself: a different number, not a worse one."""
    rng = np.random.default_rng(7)
    model = _model()
    x = np.array([[0.4, 0.9]])
    bins = model.binner.transform(x)
    truth = float(model.margin(bins)[0])
    safe = PrivateAdditiveModel.of(model, 20)
    wrapping = PrivateAdditiveModel.of(model, 40)
    assert private_score(safe, safe.selectors(bins[0]), rng)[0] == pytest.approx(truth, abs=1e-5)
    assert abs(private_score(wrapping, wrapping.selectors(bins[0]), rng)[0] - truth) > 1.0


def test_a_malicious_client_reads_one_table_entry_per_query() -> None:
    """The attack the honest-but-curious assumption is hiding, executed.

    Secret sharing hides the client's vector so well that the server cannot check it is a
    *selector*. A unit vector on one feature and zeros on the rest returns that table entry
    plus the intercept, and the intercept comes from an all-zero query.
    """
    rng = np.random.default_rng(8)
    model = _model()
    private = PrivateAdditiveModel.of(model, 20)
    zeros = [np.zeros(len(table), dtype=np.int64) for table in private.tables]
    baseline, _ = private_score(private, zeros, rng)
    assert baseline == pytest.approx(model.intercept, abs=1e-5)
    for feature, shape in enumerate(model.shapes):
        for position, truth in enumerate(shape):
            crafted = [np.zeros(len(table), dtype=np.int64) for table in private.tables]
            crafted[feature][position] = 1
            score, _ = private_score(private, crafted, rng)
            assert score - baseline == pytest.approx(float(truth), abs=1e-5)
