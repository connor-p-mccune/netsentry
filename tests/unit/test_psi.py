"""Private set intersection: correctness, and the two properties it is bought for.

A PSI implementation that returns the right answer while leaking the inputs is the failure
mode, and it looks identical from the outside. So the tests check the algebra the privacy rests
on (commutativity, subgroup membership), the ordering rule that separates PSI from
cardinality-only PSI, and the fact that a blinded element is unrecognisable without the
exponent.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from netsentry.intel.psi import (
    DH_PRIME,
    blind,
    enumerate_addresses,
    full_space_seconds,
    hash_to_group,
    inflation_attack,
    measure_hash_rate,
    private_set_intersection,
    recover_hashed_ports,
    secret_exponent,
)

# --------------------------------------------------------------------------------------
# The algebra the privacy rests on.
# --------------------------------------------------------------------------------------


def test_blinding_commutes() -> None:
    # M(x)^(ab) == M(x)^(ba) is the entire protocol; without it the double-blinded values of a
    # shared indicator would not match and the intersection would always be empty.
    element = hash_to_group("203.0.113.7")
    a, b = 7919, 104729
    assert pow(pow(element, a, DH_PRIME), b, DH_PRIME) == pow(
        pow(element, b, DH_PRIME), a, DH_PRIME
    )


def test_hashing_lands_in_the_prime_order_subgroup() -> None:
    # Squaring is what puts elements in the order-q subgroup; an element outside it leaks its
    # Legendre symbol, which is one bit of the plaintext per indicator.
    q = (DH_PRIME - 1) // 2
    for item in ("10.0.0.1", "evil.example", "a" * 64):
        element = hash_to_group(item)
        assert 1 < element < DH_PRIME
        assert pow(element, q, DH_PRIME) == 1


def test_hash_to_group_is_deterministic_and_collision_free_here() -> None:
    assert hash_to_group("1.2.3.4") == hash_to_group("1.2.3.4")
    values = {hash_to_group(f"10.0.0.{i}") for i in range(200)}
    assert len(values) == 200


def test_a_blinded_element_does_not_reveal_its_input() -> None:
    # The attacker knows the candidate and the group but not the exponent, so it cannot compute
    # the blinded form of a guess. This is exactly what a hash does not give you.
    rng = random.Random(0)
    exponent = secret_exponent(rng)
    blinded = blind([hash_to_group("198.51.100.9")], exponent)[0]
    assert blinded != hash_to_group("198.51.100.9")
    assert blinded not in {hash_to_group(f"198.51.100.{i}") for i in range(256)}


# --------------------------------------------------------------------------------------
# The protocol.
# --------------------------------------------------------------------------------------


def test_the_protocol_returns_exactly_the_intersection() -> None:
    a = ["1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"]
    b = ["3.3.3.3", "9.9.9.9", "1.1.1.1"]
    result = private_set_intersection(a, b, random.Random(1))
    assert result.intersection == ["1.1.1.1", "3.3.3.3"]
    assert result.exact


def test_disjoint_sets_return_nothing() -> None:
    result = private_set_intersection(["1.1.1.1"], ["2.2.2.2"], random.Random(2))
    assert result.intersection == []
    assert result.exact


def test_identical_sets_return_everything() -> None:
    items = [f"10.0.0.{i}" for i in range(20)]
    result = private_set_intersection(items, list(reversed(items)), random.Random(3))
    assert sorted(result.intersection) == sorted(items)


def test_duplicates_do_not_double_count() -> None:
    result = private_set_intersection(["1.1.1.1", "1.1.1.1"], ["1.1.1.1"], random.Random(4))
    assert result.intersection == ["1.1.1.1", "1.1.1.1"]  # A's own list is reported as given
    assert set(result.truth) == {"1.1.1.1"}


def test_the_answer_does_not_depend_on_the_secret_exponents() -> None:
    a = [f"172.16.0.{i}" for i in range(30)]
    b = [f"172.16.0.{i}" for i in range(15, 45)]
    first = private_set_intersection(a, b, random.Random(5))
    second = private_set_intersection(a, b, random.Random(6))
    assert first.intersection == second.intersection
    assert first.exact and second.exact


def test_wire_cost_is_linear_in_the_lists() -> None:
    small = private_set_intersection(["1.1.1.1"], ["2.2.2.2"], random.Random(7))
    large = private_set_intersection(
        [f"1.1.1.{i}" for i in range(10)], [f"2.2.2.{i}" for i in range(10)], random.Random(8)
    )
    assert large.wire_bytes == 10 * small.wire_bytes
    assert large.exponentiations == 10 * small.exponentiations


# --------------------------------------------------------------------------------------
# What hashing leaks.
# --------------------------------------------------------------------------------------


def test_hashed_indicators_fall_to_exhaustive_search() -> None:
    ports = [22, 445, 3389, 8080]
    hashed = {hashlib.sha256(str(port).encode()).hexdigest() for port in ports}
    attack = recover_hashed_ports(hashed)
    assert attack.recovered == len(ports)
    assert attack.share == 1.0


def test_salting_does_not_help_when_the_salt_is_shared() -> None:
    # Every participant in a sharing group must use the same salt or no two hashes would match,
    # so the salt is not a secret from anybody who can run the protocol.
    salt = "group-salt"
    ports = [22, 443, 9001]
    hashed = {hashlib.sha256(f"{salt}{port}".encode()).hexdigest() for port in ports}
    attack = recover_hashed_ports(hashed, salt=salt)
    assert attack.recovered == len(ports)
    assert attack.salted


def test_enumeration_finds_an_address_inside_the_slice_it_covers() -> None:
    target = "10.0.0.5"
    hashed = {hashlib.sha256(target.encode()).hexdigest()}
    start = (10 << 24) + 0  # 10.0.0.0
    attack = enumerate_addresses(hashed, 64, start)
    assert attack.recovered == 1
    assert attack.seconds > 0


def test_extrapolating_the_full_space_is_just_arithmetic() -> None:
    assert full_space_seconds(1_000_000, bits=32) == pytest.approx(4294.967296)
    assert full_space_seconds(2_000_000, bits=32) < full_space_seconds(1_000_000, bits=32)


def test_the_measured_hash_rate_is_positive() -> None:
    assert measure_hash_rate(2000) > 0


# --------------------------------------------------------------------------------------
# The attack on the protocol.
# --------------------------------------------------------------------------------------


def test_inflation_extracts_everything_the_submitted_universe_reaches() -> None:
    honest = ["10.0.0.1"]
    peer = [f"192.0.2.{i}" for i in range(20)]
    universe = [*peer[:8], *[f"203.0.113.{i}" for i in range(50)]]
    attack = inflation_attack(honest, universe, peer, random.Random(9))
    assert attack.universe_hits == 8
    assert attack.learned == 8  # the protocol reports every reachable indicator
    assert attack.yield_rate == 1.0
    assert attack.submitted > len(honest)


def test_an_honest_party_learns_only_its_own_overlap() -> None:
    honest = ["192.0.2.1", "192.0.2.2"]
    peer = [f"192.0.2.{i}" for i in range(20)]
    attack = inflation_attack(honest, [], peer, random.Random(10))
    assert attack.learned == 2
    assert attack.submitted == 2
