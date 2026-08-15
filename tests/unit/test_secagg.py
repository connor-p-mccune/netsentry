"""The secure-aggregation protocol, tested against its guarantees rather than a golden output.

The valuable tests here are the ones that would fail if the protocol were subtly wrong in a
way that still produced a plausible model: masks that do not cancel exactly, a Shamir threshold
that reconstructs from too few shares, a group parameter that is not what the RFC says it is,
and — the one that motivates the whole double-masking design — a coordinator that unmasks a
live site by declaring it dropped.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from netsentry.training.federated import Weights
from netsentry.training.secagg import (
    DH_GENERATOR,
    DH_PRIME,
    FIELD_PRIME,
    SHARE_PRIME,
    build_site_keys,
    decode_aggregate,
    dequantize,
    dh_keypair,
    dh_shared,
    encode_contribution,
    encoding_headroom_bits,
    kdf,
    pairwise_masks,
    prg_bytes,
    prg_field,
    quantize,
    secure_round,
    shamir_recover,
    shamir_split,
)


def _is_probable_prime(n: int, rounds: int = 8, seed: int = 20260814) -> bool:
    """Miller-Rabin. Small enough to run in a unit test, strong enough to catch a typo."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    rng = random.Random(seed)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# --------------------------------------------------------------------------------------
# Parameters. Constants copied from a standard are constants that can be copied wrongly.
# --------------------------------------------------------------------------------------


def test_the_group_parameters_are_what_they_claim_to_be() -> None:
    # A composite modulus would still "work" -- exponentiation is exponentiation -- and would
    # silently void the security argument, which is exactly why this is checked rather than
    # trusted. RFC 3526 group 14 is a 2048-bit safe prime.
    assert DH_PRIME.bit_length() == 2048
    assert _is_probable_prime(DH_PRIME)
    assert _is_probable_prime((DH_PRIME - 1) // 2)  # safe prime: no small subgroups to fall into
    assert _is_probable_prime(FIELD_PRIME)
    assert _is_probable_prime(SHARE_PRIME)


def test_the_generator_sits_in_the_prime_order_subgroup() -> None:
    # 2 is a quadratic residue in the RFC 3526 groups, so it generates the subgroup of prime
    # order q rather than the whole group -- which is the property you want: every shared
    # secret lands in a prime-order subgroup, so there is no small-subgroup bit to leak.
    q = (DH_PRIME - 1) // 2
    assert pow(DH_GENERATOR, q, DH_PRIME) == 1
    assert pow(DH_GENERATOR, 2, DH_PRIME) != 1  # and the order is q, not 1 or 2


# --------------------------------------------------------------------------------------
# The PRG.
# --------------------------------------------------------------------------------------


def test_the_prg_is_deterministic_and_seed_dependent() -> None:
    assert prg_bytes(b"a" * 32, 64) == prg_bytes(b"a" * 32, 64)
    assert prg_bytes(b"a" * 32, 64) != prg_bytes(b"b" * 32, 64)


def test_prg_output_lands_in_the_field_and_looks_uniform() -> None:
    values = prg_field(b"seed" + b"\x00" * 28, 20000)
    assert values.dtype == np.int64
    assert (values >= 0).all() and (values < FIELD_PRIME).all()
    # A pad that is not uniform is not a one-time pad. Chi-square over 16 equal buckets:
    # the 5% critical value for 15 degrees of freedom is 25.0.
    counts = np.histogram(values, bins=16, range=(0, FIELD_PRIME))[0]
    expected = len(values) / 16
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < 25.0


def test_the_kdf_separates_rounds() -> None:
    # Reusing a pad across rounds would let a coordinator difference two rounds and read the
    # update difference in the clear, so the round index has to reach the key material.
    assert kdf(12345, b"pair", 0) != kdf(12345, b"pair", 1)
    assert kdf(12345, b"pair", 0) != kdf(12345, b"self", 0)


# --------------------------------------------------------------------------------------
# Fixed-point encoding.
# --------------------------------------------------------------------------------------


def test_quantization_round_trips_including_negatives() -> None:
    values = np.array([-3.5, -1e-4, 0.0, 1e-4, 7.25])
    scale = float(2**24)
    restored = dequantize(quantize(values, scale), scale)
    assert np.allclose(restored, values, atol=1.0 / scale)
    # Negatives live in the top half of the field, which is what makes modular addition work.
    assert quantize(np.array([-1.0]), scale)[0] > FIELD_PRIME // 2


def test_encoding_headroom_goes_negative_exactly_when_the_sum_wraps() -> None:
    scale = float(2**40)
    safe = float(FIELD_PRIME) / 2 / scale / 4
    assert encoding_headroom_bits(safe, scale) > 0
    over = 0.75 * float(FIELD_PRIME) / scale
    assert encoding_headroom_bits(over, scale) < 0
    # And the prediction is real: a value past the field's half-point decodes to the wrong
    # sign, silently, which is the failure mode the headroom column exists to make visible.
    assert dequantize(quantize(np.array([over]), scale), scale)[0] < 0


# --------------------------------------------------------------------------------------
# Shamir secret sharing.
# --------------------------------------------------------------------------------------


def test_shamir_recovers_at_the_threshold_and_not_below_it() -> None:
    rng = random.Random(7)
    secret = rng.randrange(SHARE_PRIME)
    shares = shamir_split(secret, n_shares=7, threshold=4, rng=rng)
    assert shamir_recover(shares[:4]) == secret
    assert shamir_recover(shares[2:6]) == secret  # any four, not just the first four
    assert shamir_recover(shares[:3]) != secret  # information-theoretically nothing


def test_shamir_rejects_an_impossible_threshold() -> None:
    with pytest.raises(ValueError):
        shamir_split(5, n_shares=3, threshold=4, rng=random.Random(0))


# --------------------------------------------------------------------------------------
# Key agreement and mask cancellation.
# --------------------------------------------------------------------------------------


def test_diffie_hellman_agrees_on_both_sides() -> None:
    rng = random.Random(11)
    a_private, a_public = dh_keypair(rng)
    b_private, b_public = dh_keypair(rng)
    assert dh_shared(a_private, b_public) == dh_shared(b_private, a_public)


def test_the_pairwise_masks_sum_to_zero_across_the_federation() -> None:
    # This is the mechanism in one assertion: antisymmetric pair masks cancel exactly when
    # every participant is included, so the coordinator's sum is the plaintext sum.
    keys = build_site_keys([f"s{i}" for i in range(5)], random.Random(2))
    masks = pairwise_masks(keys, round_index=0, length=6)
    total = np.zeros(6, dtype=np.int64)
    for mask in masks.values():
        total = np.mod(total + mask, FIELD_PRIME)
    assert np.all(total == 0)


# --------------------------------------------------------------------------------------
# The protocol.
# --------------------------------------------------------------------------------------


def _fixture(n_sites: int = 6, length: int = 5) -> tuple[list, dict[int, np.ndarray]]:
    keys = build_site_keys([f"s{i}" for i in range(n_sites)], random.Random(3))
    contributions = {
        i: quantize(np.linspace(-1.0, 1.0, length) * (i + 1), float(2**20)) for i in range(n_sites)
    }
    return keys, contributions


def _plain_sum(contributions: dict[int, np.ndarray], skip: set[int] | None = None) -> np.ndarray:
    skip = skip or set()
    total = np.zeros(len(next(iter(contributions.values()))), dtype=np.int64)
    for index, vector in contributions.items():
        if index not in skip:
            total = np.mod(total + vector, FIELD_PRIME)
    return total


def test_the_aggregate_is_exactly_the_plaintext_sum() -> None:
    keys, contributions = _fixture()
    transcript = secure_round(contributions, keys, round_index=0, threshold=3, rng=random.Random(4))
    assert transcript.recovered
    assert np.array_equal(transcript.total, _plain_sum(contributions))


def test_no_masked_vector_resembles_its_input() -> None:
    keys, contributions = _fixture()
    transcript = secure_round(contributions, keys, round_index=0, threshold=3, rng=random.Random(4))
    for index, masked in enumerate(transcript.masked):
        assert not np.array_equal(masked, contributions[index])


def test_dropouts_are_recovered_up_to_the_threshold() -> None:
    keys, contributions = _fixture()
    for dropped in (frozenset({1}), frozenset({0, 5}), frozenset({1, 2, 3})):
        transcript = secure_round(
            contributions,
            keys,
            round_index=0,
            threshold=3,
            dropped=dropped,
            rng=random.Random(4),
        )
        assert transcript.recovered
        assert np.array_equal(transcript.total, _plain_sum(contributions, set(dropped)))


def test_too_few_survivors_lose_the_round_rather_than_leaking_it() -> None:
    keys, contributions = _fixture()
    transcript = secure_round(
        contributions,
        keys,
        round_index=0,
        threshold=4,
        dropped=frozenset({0, 1, 2, 3}),
        rng=random.Random(4),
    )
    assert not transcript.recovered
    assert transcript.masked == []


def test_the_self_mask_is_what_stops_a_lying_coordinator() -> None:
    # The attack the double-masking exists for: the coordinator declares a *live* site dropped,
    # collects the shares that rebuild its pairwise masks, and subtracts them from the vector it
    # already holds. Without the self-mask that recovers the site's input exactly.
    keys, contributions = _fixture()
    without = secure_round(
        contributions,
        keys,
        round_index=0,
        threshold=3,
        use_self_mask=False,
        unmask_target=2,
        rng=random.Random(4),
    )
    assert np.array_equal(without.unmasked_target, contributions[2])

    with_mask = secure_round(
        contributions,
        keys,
        round_index=0,
        threshold=3,
        use_self_mask=True,
        unmask_target=2,
        rng=random.Random(4),
    )
    assert not np.array_equal(with_mask.unmasked_target, contributions[2])


def test_masks_differ_between_rounds() -> None:
    keys, contributions = _fixture()
    first = secure_round(contributions, keys, round_index=0, threshold=3, rng=random.Random(4))
    second = secure_round(contributions, keys, round_index=1, threshold=3, rng=random.Random(4))
    assert not np.array_equal(first.masked[0], second.masked[0])
    assert np.array_equal(first.total, second.total)  # ... and both open to the same sum


# --------------------------------------------------------------------------------------
# The FedAvg payload.
# --------------------------------------------------------------------------------------


def test_the_encoded_contributions_average_to_fedavg() -> None:
    # The sample count rides inside the secure sum as one more coordinate, so the coordinator
    # can form the size-weighted mean without any site announcing how much data it holds.
    scale = float(2**24)
    updates = [Weights(np.array([1.0, -2.0]), 0.5), Weights(np.array([3.0, 1.0]), -0.5)]
    sizes = [100, 300]
    total = np.zeros(4, dtype=np.int64)
    for update, size in zip(updates, sizes, strict=True):
        total = np.mod(total + encode_contribution(update, size, scale), FIELD_PRIME)
    recovered = decode_aggregate(total, scale)
    expected_coef = (100 * updates[0].coef + 300 * updates[1].coef) / 400
    expected_intercept = (100 * updates[0].intercept + 300 * updates[1].intercept) / 400
    assert np.allclose(recovered.coef, expected_coef, atol=1e-6)
    assert recovered.intercept == pytest.approx(expected_intercept, abs=1e-6)
