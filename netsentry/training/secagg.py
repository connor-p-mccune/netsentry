"""Secure aggregation: federate the model without the coordinator seeing anybody's update.

The [federated study](federated.md) makes one claim on which the whole arrangement rests:
raw flows never leave the site, only *weights* do. That claim is true and it is not the same
as privacy. An update is a function of the data that produced it, and the coordinator holds
one per site per round. The first thing this module does is measure what can be read off
that channel: a nearest-reference classifier over the raw update vector names **which attack
family a site is holding** — with no access to its flows, no model inversion, and no
training beyond one local pass. Federation moved the data-protection problem; it did not
solve it.

**Secure aggregation** (Bonawitz et al., CCS 2017) removes the channel rather than
regulating it. Every pair of sites agrees on a shared seed; site `i` adds `PRG(s_ij)` to its
update for every `j > i` and subtracts it for every `j < i`. Each masked vector is a
one-time pad — uniform in the field, independent of the input — and the masks cancel
*exactly* when the coordinator sums them. It learns the aggregate and provably nothing else.
Everything here is implemented from scratch on the standard library:

- **Key agreement** — Diffie-Hellman over RFC 3526 group 14 (the 2048-bit MODP group; the
  test suite runs Miller-Rabin on the modulus and on `(p-1)/2` rather than trusting the
  constant).
- **The PRG** — HMAC-SHA256 in counter mode, expanded into field elements by *rejection*
  sampling. Reducing 64 random bits modulo `2^61 - 1` would bias eight residues by one part
  in `2^61`; the rejection loop costs nothing (8 draws in `2^64` are discarded) and removes
  the footnote from the security argument.
- **Shamir secret sharing** over `2^521 - 1`, threshold `t`-of-`n`, for the dropout
  recovery that makes the protocol usable on a network where sites disappear mid-round.
- **Fixed-point encoding** into `Z_p`, because masking needs a finite group and floats are
  not one. The encoding has a floor (quantization error) and a ceiling (wraparound), and the
  report measures both rather than asserting a scale.

Four things get measured, in the order somebody deciding whether to deploy this needs them:

1. **Does it still train?** The recovered sum is compared against the plaintext one — exactly
   in the field, and after decoding, against the model FedAvg produces without any of this.
2. **What did it actually buy?** The family-identification attack is run on three views: the
   plaintext update, the masked vector, and — the one that matters — the aggregate, which is
   *not* protected by this protocol and never was. Secure aggregation hides the individual;
   differential privacy bounds what the sum reveals. They are complements, and the report
   refuses to let the first stand in for the second.
3. **What breaks it?** Sites drop; the protocol recovers while survivors exceed the
   threshold and fails cleanly below it. And the self-mask, which looks like belt-and-braces
   until you run the attack it exists to stop: a coordinator that *declares a live site
   dropped* collects the shares that reconstruct its pairwise masks and recovers that site's
   update exactly. The attack is executed here in both configurations.
4. **What does it cost?** Bandwidth and time, against training — and the cost nobody
   advertises, which is not bandwidth at all: **every Byzantine defence in the
   [byzantine study](byzantine.md) needs to see the individual updates this protocol exists
   to hide.** Median, trimmed mean and Krum are all functions of the per-site vectors. Under
   secure aggregation the coordinator holds one number: the sum. The study measures the
   damage a single liar does with the defence unavailable, then prices the two mitigations
   that keep some of each — an ideal range proof (the strongest attack an honest-range
   attacker can still play) and grouped aggregation, which trades anonymity-set size for
   partial visibility on a frontier rather than a slogan.

Reproducibility note: keys and mask seeds come from the seeded project RNG so the report is
reproducible run to run. A deployment must draw them from `secrets`/the OS CSPRNG — the
protocol's security rests on the seeds being unpredictable, and a study that could not be
re-run would be a different kind of failure.
"""

from __future__ import annotations

import hashlib
import hmac
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import operating_point
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.byzantine import coordinate_median, krum, sign_flip
from netsentry.training.federated import Weights, initial_weights, local_train
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SecAggConfig

logger = get_logger(__name__)

REPORT_NAME = "secagg.md"
FRONTIER_FIGURE_NAME = "secagg_frontier.png"
COST_FIGURE_NAME = "secagg_cost.png"

# --------------------------------------------------------------------------------------
# Field, group and hash parameters. Public constants: nothing here is a secret.
# --------------------------------------------------------------------------------------

#: The aggregation field. A Mersenne prime keeps the reduction cheap, and 61 bits leaves
#: every partial sum below 2^62 so the whole protocol runs in int64 without promotion.
FIELD_PRIME = (1 << 61) - 1

#: Shamir's field. Secrets shared here are 256-bit Diffie-Hellman exponents and mask seeds,
#: so the modulus has to exceed them comfortably; 2^521 - 1 is the smallest Mersenne prime
#: that does with room to spare.
SHARE_PRIME = (1 << 521) - 1

#: RFC 3526 group 14 — the 2048-bit MODP group. Standard, public, and a safe prime, with the
#: generator 2 sitting in the subgroup of prime order ``(p-1)/2``, so every shared secret lands
#: in a prime-order subgroup and there is no small-subgroup bit to leak. Both facts are checked
#: by Miller-Rabin in the test suite rather than trusted: a mistyped constant would still
#: *work* — exponentiation is exponentiation — and would silently void the security argument.
DH_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
DH_GENERATOR = 2
DH_BYTES = 256  # 2048 bits, the width every shared secret is encoded at before hashing
DH_EXPONENT_BITS = 256  # ephemeral exponent width (>= 2x the 128-bit target security level)

_SHARE_BYTES = 66  # 521 bits, rounded up — the wire width of one Shamir share
_SEED_BYTES = 32  # HMAC-SHA256 key width


# --------------------------------------------------------------------------------------
# Pseudorandom generator. Deterministic given a seed, uniform over the field.
# --------------------------------------------------------------------------------------


def prg_bytes(seed: bytes, n_bytes: int) -> bytes:
    """Expand a seed into ``n_bytes`` with HMAC-SHA256 in counter mode.

    The construction is the generate step of an HMAC_DRBG (NIST SP 800-90A) with the reseed
    machinery stripped out: the seed is the key, the counter is the message, and the outputs
    are concatenated. It is used rather than a hash of ``seed || counter`` because HMAC's
    security does not lean on the hash being a random oracle over concatenated inputs.
    """
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out += hmac.new(seed, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:n_bytes])


def prg_field(seed: bytes, count: int) -> np.ndarray:
    """``count`` field elements from a seed, exactly uniform over ``[0, FIELD_PRIME)``.

    Sampling 64 bits and reducing modulo ``2^61 - 1`` would leave the eight smallest
    residues more likely than the rest — a bias of one part in ``2^61``, invisible in any
    test and still a hole in the one-time-pad argument, which needs the mask to be *exactly*
    uniform. Rejection removes it: draws at or above the largest multiple of the prime are
    discarded, which happens for 8 values in ``2^64``, so the loop essentially never runs
    twice.
    """
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    limit = np.uint64((1 << 64) // FIELD_PRIME * FIELD_PRIME)
    out = np.empty(count, dtype=np.int64)
    filled = 0
    block = 0
    while filled < count:
        need = count - filled
        raw = np.frombuffer(
            prg_bytes(seed + block.to_bytes(4, "big"), 8 * (need + 4)), dtype=np.uint64
        )
        accepted = raw[raw < limit]
        take = min(len(accepted), need)
        out[filled : filled + take] = (accepted[:take] % np.uint64(FIELD_PRIME)).astype(np.int64)
        filled += take
        block += 1
    return out


# --------------------------------------------------------------------------------------
# Fixed-point encoding. Masking needs a finite group; floats are not one.
# --------------------------------------------------------------------------------------


def quantize(values: np.ndarray, scale: float) -> np.ndarray:
    """Encode reals as field elements at ``scale`` steps per unit; negatives wrap.

    A negative value ``-v`` is stored as ``p - v``, which is two's complement in ``Z_p``, so
    ordinary modular addition adds the signed quantities correctly and no separate sign
    channel is needed. Correctness has one precondition, checked in
    :func:`encoding_headroom_bits`: the *decoded* sum must stay inside ``(-p/2, p/2)``.
    """
    ints: np.ndarray = np.rint(np.asarray(values, dtype=float) * scale).astype(np.int64)
    encoded: np.ndarray = np.mod(ints, FIELD_PRIME)
    return encoded


def dequantize(values: np.ndarray, scale: float) -> np.ndarray:
    """Decode field elements back to signed reals (the top half of the field is negative)."""
    v: np.ndarray = np.asarray(values, dtype=np.int64)
    signed: np.ndarray = np.where(v > FIELD_PRIME // 2, v - FIELD_PRIME, v)
    return signed.astype(float) / scale


def encoding_headroom_bits(max_abs_sum: float, scale: float) -> float:
    """Bits left between the largest encoded magnitude and the wraparound point.

    Positive means the encoding is safe; zero or below means the sum has wrapped and the
    decoded model is not an approximation of the right answer, it is a different number.
    """
    magnitude = max(abs(max_abs_sum) * scale, 1.0)
    return float(np.log2(FIELD_PRIME / 2) - np.log2(magnitude))


# --------------------------------------------------------------------------------------
# Key agreement and secret sharing.
# --------------------------------------------------------------------------------------


def dh_keypair(rng: random.Random) -> tuple[int, int]:
    """An ephemeral Diffie-Hellman keypair in the RFC 3526 group."""
    private = rng.randrange(2, 1 << DH_EXPONENT_BITS)
    return private, pow(DH_GENERATOR, private, DH_PRIME)


def dh_shared(private: int, public: int) -> int:
    """The shared group element ``public^private``, identical on both sides of the pair."""
    return pow(public, private, DH_PRIME)


def kdf(material: int, label: bytes, round_index: int) -> bytes:
    """Derive a PRG seed from a group element, a label and the round.

    Deriving per round from one long-term agreement is what keeps the expensive modular
    exponentiations out of the per-round cost; the round index in the KDF input is what
    stops the same mask being reused across rounds, which would let a coordinator difference
    two rounds and recover the update difference in the clear.
    """
    return hashlib.sha256(
        material.to_bytes(DH_BYTES, "big") + b"|" + label + b"|" + round_index.to_bytes(4, "big")
    ).digest()


def shamir_split(
    secret: int, *, n_shares: int, threshold: int, rng: random.Random
) -> list[tuple[int, int]]:
    """Split ``secret`` into ``n_shares`` points on a degree ``threshold - 1`` polynomial.

    Any ``threshold`` shares determine the polynomial and therefore the constant term; any
    ``threshold - 1`` leave it uniform over the field — information-theoretic, not
    computational, which is the reason this rather than an encrypted backup copy.
    """
    if threshold < 1 or threshold > n_shares:
        raise ValueError(f"threshold {threshold} outside 1..{n_shares}")
    coefficients = [secret % SHARE_PRIME]
    coefficients += [rng.randrange(SHARE_PRIME) for _ in range(threshold - 1)]
    shares: list[tuple[int, int]] = []
    for x in range(1, n_shares + 1):
        y = 0
        for c in reversed(coefficients):  # Horner, so the evaluation is one pass
            y = (y * x + c) % SHARE_PRIME
        shares.append((x, y))
    return shares


def shamir_recover(shares: list[tuple[int, int]]) -> int:
    """Lagrange-interpolate the constant term from ``threshold`` or more shares."""
    if not shares:
        raise ValueError("no shares to interpolate")
    total = 0
    for i, (xi, yi) in enumerate(shares):
        numerator, denominator = 1, 1
        for j, (xj, _) in enumerate(shares):
            if i == j:
                continue
            numerator = (numerator * (-xj)) % SHARE_PRIME
            denominator = (denominator * (xi - xj)) % SHARE_PRIME
        total = (total + yi * numerator * pow(denominator, -1, SHARE_PRIME)) % SHARE_PRIME
    return total


# --------------------------------------------------------------------------------------
# The protocol.
# --------------------------------------------------------------------------------------


@dataclass
class SiteKeys:
    """One participant's key material for a federation run."""

    index: int
    name: str
    dh_private: int
    dh_public: int
    self_seed: int  # b_i: the second mask, and the reason a lying coordinator learns nothing


def build_site_keys(names: list[str], rng: random.Random) -> list[SiteKeys]:
    """Generate a keypair and a self-mask seed per site."""
    sites: list[SiteKeys] = []
    for index, name in enumerate(names):
        private, public = dh_keypair(rng)
        sites.append(
            SiteKeys(
                index=index,
                name=name,
                dh_private=private,
                dh_public=public,
                self_seed=rng.randrange(SHARE_PRIME),
            )
        )
    return sites


def pairwise_masks(sites: list[SiteKeys], round_index: int, length: int) -> dict[int, np.ndarray]:
    """Each site's total pairwise mask for one round, keyed by site index.

    Site ``i`` adds the pair mask for every ``j > i`` and subtracts it for every ``j < i``,
    so summing over any set of sites cancels every pair *inside* the set. That antisymmetry
    is the entire mechanism; everything else in the protocol exists to handle the pairs that
    straddle the boundary when somebody drops out.
    """
    masks = {site.index: np.zeros(length, dtype=np.int64) for site in sites}
    for a in range(len(sites)):
        for b in range(a + 1, len(sites)):
            shared = dh_shared(sites[a].dh_private, sites[b].dh_public)
            pad = prg_field(kdf(shared, b"pair", round_index), length)
            masks[sites[a].index] = np.mod(masks[sites[a].index] + pad, FIELD_PRIME)
            masks[sites[b].index] = np.mod(masks[sites[b].index] - pad, FIELD_PRIME)
    return masks


def self_mask(site: SiteKeys, round_index: int, length: int) -> np.ndarray:
    """The site's own one-time pad, removed only if the site is confirmed alive."""
    seed = hashlib.sha256(
        site.self_seed.to_bytes(_SHARE_BYTES, "big") + b"|self|" + round_index.to_bytes(4, "big")
    ).digest()
    return prg_field(seed, length)


@dataclass
class RoundTranscript:
    """Everything the coordinator holds after one aggregation round."""

    total: np.ndarray  # the recovered sum, in the field
    masked: list[np.ndarray]  # what each site put on the wire
    survivors: list[int]
    dropped: list[int]
    recovered: bool
    bytes_per_site: int
    mask_seconds: float
    recover_seconds: float
    unmasked_target: np.ndarray | None = None  # a malicious coordinator's recovered victim


def secure_round(
    contributions: dict[int, np.ndarray],
    sites: list[SiteKeys],
    *,
    round_index: int,
    threshold: int,
    dropped: frozenset[int] = frozenset(),
    use_self_mask: bool = True,
    unmask_target: int | None = None,
    rng: random.Random | None = None,
) -> RoundTranscript:
    """Run one masked aggregation round and return what the coordinator ends up with.

    ``dropped`` names sites that vanish after uploading nothing; the coordinator recovers
    their pairwise masks from ``threshold`` shares of their Diffie-Hellman exponent and
    subtracts the ones that straddle the survivor boundary. ``unmask_target`` executes the
    malicious-coordinator attack: a *live* site is declared dropped so its masks can be
    reconstructed, which recovers its individual update exactly — unless the self-mask is in
    place, which is precisely why it is there.
    """
    rng = rng or random.Random(0)
    length = len(next(iter(contributions.values())))
    survivors = [s.index for s in sites if s.index not in dropped]
    if len(survivors) < threshold:
        # Liveness failure, not a privacy failure: without `threshold` survivors nobody can
        # reconstruct the masks, so the round is lost and the coordinator learns nothing.
        return RoundTranscript(
            total=np.zeros(length, dtype=np.int64),
            masked=[],
            survivors=survivors,
            dropped=sorted(dropped),
            recovered=False,
            bytes_per_site=0,
            mask_seconds=0.0,
            recover_seconds=0.0,
        )

    start = time.perf_counter()
    pair = pairwise_masks(sites, round_index, length)
    masked: dict[int, np.ndarray] = {}
    for site in sites:
        vector = np.mod(contributions[site.index] + pair[site.index], FIELD_PRIME)
        if use_self_mask:
            vector = np.mod(vector + self_mask(site, round_index, length), FIELD_PRIME)
        masked[site.index] = vector
    mask_seconds = time.perf_counter() - start

    # Shares exist so the coordinator can reconstruct *one* secret per site: the self-mask
    # seed if the site is alive, the DH exponent if it is not. Never both -- a site that
    # answered the alive question refuses the dead one, which is what makes the protocol
    # robust to a coordinator that lies about who dropped.
    share_sets = {
        site.index: (
            shamir_split(site.self_seed, n_shares=len(sites), threshold=threshold, rng=rng),
            shamir_split(site.dh_private, n_shares=len(sites), threshold=threshold, rng=rng),
        )
        for site in sites
    }

    start = time.perf_counter()
    total = np.zeros(length, dtype=np.int64)
    for index in survivors:
        total = np.mod(total + masked[index], FIELD_PRIME)
    for index in survivors:  # remove the self-masks of the sites that are still here
        if use_self_mask:
            seed_shares = share_sets[index][0][:threshold]
            recovered_seed = shamir_recover(seed_shares)
            site = sites[index]
            total = np.mod(
                total
                - self_mask(SiteKeys(index, site.name, 0, 0, recovered_seed), round_index, length),
                FIELD_PRIME,
            )
    for index in sorted(dropped):  # remove the straddling pairs of the sites that left
        exponent = shamir_recover(share_sets[index][1][:threshold])
        for other in survivors:
            shared = dh_shared(exponent, sites[other].dh_public)
            pad = prg_field(kdf(shared, b"pair", round_index), length)
            # The residual is the term the *survivor* added for this pair, so the sign is
            # the survivor's: +1 when it holds the lower index, -1 when it holds the higher.
            sign = 1 if other < index else -1
            total = np.mod(total - sign * pad, FIELD_PRIME)
    recover_seconds = time.perf_counter() - start

    unmasked: np.ndarray | None = None
    if unmask_target is not None:
        # The attack the self-mask exists to stop. The coordinator claims `unmask_target`
        # dropped, collects `threshold` shares of its exponent, rebuilds every pairwise mask
        # it holds and subtracts them from the vector it already received.
        exponent = shamir_recover(share_sets[unmask_target][1][:threshold])
        residual = masked[unmask_target].copy()
        for other in (s.index for s in sites if s.index != unmask_target):
            shared = dh_shared(exponent, sites[other].dh_public)
            pad = prg_field(kdf(shared, b"pair", round_index), length)
            sign = 1 if unmask_target < other else -1
            residual = np.mod(residual - sign * pad, FIELD_PRIME)
        unmasked = residual

    # Wire cost per site: the masked vector, one public key, and two shares per peer.
    bytes_per_site = length * 8 + DH_BYTES + 2 * (len(sites) - 1) * (_SHARE_BYTES + 1)
    return RoundTranscript(
        total=total,
        masked=[masked[i] for i in sorted(masked)],
        survivors=survivors,
        dropped=sorted(dropped),
        recovered=True,
        bytes_per_site=bytes_per_site,
        mask_seconds=mask_seconds,
        recover_seconds=recover_seconds,
        unmasked_target=unmasked,
    )


def encode_contribution(weights: Weights, n_rows: int, scale: float) -> np.ndarray:
    """Encode one site's FedAvg contribution: ``n_i * w_i`` with ``n_i`` as the last element.

    The sample count rides inside the same secure sum rather than being announced, because
    announcing it hands the coordinator a per-site quantity for free — and dividing the
    aggregate by an aggregated count gives exactly FedAvg's size-weighted mean.
    """
    payload = np.append(weights.coef, weights.intercept) * float(n_rows)
    return quantize(np.append(payload, float(n_rows)), scale)


def decode_aggregate(total: np.ndarray, scale: float) -> Weights:
    """Invert :func:`encode_contribution` over the recovered sum."""
    decoded = dequantize(total, scale)
    count = decoded[-1]
    weighted = decoded[:-1] / (count if abs(count) > 1e-9 else 1.0)
    return Weights(np.asarray(weighted[:-1], dtype=float), float(weighted[-1]))


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class SiteData:
    """One federation participant: its rows, its label mix and its true attack family."""

    name: str
    index: int
    rows: np.ndarray
    n_rows: int
    attack_prior: float
    family: str


@dataclass
class ExactnessRow:
    """One fixed-point scale: what the encoding costs and where it breaks."""

    scale_bits: int
    headroom_bits: float
    max_weight_error: float
    pr_auc: float
    wrapped: bool


@dataclass
class PrivacyRow:
    """What a family-identification attack recovers from one view of the round."""

    view: str
    accuracy: float
    chance: float
    note: str
    display: str = ""

    def shown(self) -> str:
        """What the report prints -- an accuracy, unless the row carries its own text."""
        return self.display or f"**{self.accuracy:.0%}**"


@dataclass
class DropoutRow:
    """Recovery outcome at one dropout level."""

    n_dropped: int
    survivors: int
    threshold: int
    recovered: bool
    max_error: float
    recover_ms: float


@dataclass
class RobustnessRow:
    """One aggregation setting under one attacker, and whether the defence was available."""

    setting: str
    visible_to_coordinator: str
    attack: str
    pr_auc: float
    defence_available: bool


@dataclass
class RangeRow:
    """What an ideal range proof still permits at one certified coordinate bound."""

    bound: float
    bound_vs_honest: float
    pr_auc: float


@dataclass
class GroupRow:
    """Grouped secure aggregation: anonymity set against damage."""

    group_size: int
    n_groups: int
    pr_auc_clean: float
    pr_auc_attacked: float


@dataclass
class CostRow:
    """Wire and CPU cost at one federation size."""

    n_sites: int
    plaintext_bytes: int
    secure_bytes: int
    mask_ms: float
    recover_ms: float


@dataclass
class SecAggStudy:
    """Everything the report renders."""

    sites: list[SiteData]
    n_features: int
    rounds: int
    threshold: int
    scale_bits: int
    plaintext_pr_auc: float
    plaintext_tpr: float
    secure_pr_auc: float
    secure_tpr: float
    field_exact: bool
    max_weight_error: float
    exactness: list[ExactnessRow]
    privacy: list[PrivacyRow]
    unmask_error_without_self_mask: float
    unmask_error_with_self_mask: float
    dropout: list[DropoutRow]
    robustness: list[RobustnessRow]
    ranges: list[RangeRow]
    groups: list[GroupRow]
    cost: list[CostRow]
    train_seconds: float
    families: list[str]
    honest_max_coordinate: float
    target_fpr: float


def _site_partition(
    train_days: np.ndarray, labels: np.ndarray, benign: str, shards_per_day: int
) -> list[SiteData]:
    """Split each capture day into shards, so the federation has more than three members.

    The day is the natural silo (the [federated study](federated.md) uses it directly), but
    three participants cannot say anything about a dropout threshold or an anonymity set.
    Sharding *within* the day preserves the property that makes this data interesting — one
    site's traffic is nothing like another's, because Monday holds no attacks at all — while
    giving the protocol a realistic number of members.
    """
    sites: list[SiteData] = []
    for day in dict.fromkeys(train_days.tolist()):
        idx = np.flatnonzero(train_days == day)
        for shard, rows in enumerate(np.array_split(idx, shards_per_day)):
            attacks = labels[rows]
            attacks = attacks[attacks != benign]
            family = str(np.unique(attacks)[0]) if len(attacks) else "none (benign only)"
            if len(attacks):
                values, counts = np.unique(attacks, return_counts=True)
                family = str(values[int(np.argmax(counts))])
            sites.append(
                SiteData(
                    name=f"{day}-{shard + 1}",
                    index=len(sites),
                    rows=rows,
                    n_rows=len(rows),
                    attack_prior=float(np.mean(labels[rows] != benign)),
                    family=family,
                )
            )
    return sites


def _identify(update: Weights, references: dict[str, np.ndarray]) -> str:
    """Nearest reference by cosine similarity — the coordinator's cheapest possible attack."""
    vector = np.append(update.coef, update.intercept)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return next(iter(references))
    best, best_score = next(iter(references)), -np.inf
    for name, reference in references.items():
        score = float(vector @ reference / (norm * float(np.linalg.norm(reference)) + 1e-12))
        if score > best_score:
            best, best_score = name, score
    return best


def run_secagg_study(settings: Settings) -> SecAggStudy:
    """Train federated with and without masking, then attack, drop, and price both."""
    cfg: SecAggConfig = settings.secagg
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)
    key_rng = random.Random(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    multi = (
        train[MULTICLASS_TARGET].to_numpy().astype(str)
        if MULTICLASS_TARGET in train.columns
        else np.where(y_train == 1, "attack", settings.labels.benign_label)
    )

    pipeline = build_pipeline(variant)
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train))
    x_val: np.ndarray = np.asarray(pipeline.transform(val))
    x_test: np.ndarray = np.asarray(pipeline.transform(test))
    n_features = x_train.shape[1]
    target_fpr = variant.thresholds.primary_fpr
    flows_per_day = variant.thresholds.assumed_flows_per_day

    def _score(w: Weights) -> tuple[float, float]:
        s_val, s_test = w.scores(x_val), w.scores(x_test)
        op = operating_point(y_val, s_val, y_test, s_test, target_fpr, flows_per_day)
        return float(average_precision_score(y_test, s_test)), float(op["tpr"])

    days = (
        train[DAY_COLUMN].to_numpy().astype(str)
        if DAY_COLUMN in train.columns
        else np.zeros(len(train), dtype=str)
    )
    sites = _site_partition(days, multi, settings.labels.benign_label, cfg.shards_per_day)
    n_sites = len(sites)
    threshold = max(2, int(np.ceil(cfg.threshold_fraction * n_sites)))
    keys = build_site_keys([s.name for s in sites], key_rng)
    train_kwargs = {
        "epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "l2": cfg.l2,
    }

    def _local(w: Weights, site: SiteData, round_index: int) -> Weights:
        return local_train(
            w,
            x_train[site.rows],
            y_train[site.rows],
            seed=variant.seed + site.index + 1000 * round_index,
            **train_kwargs,  # type: ignore[arg-type]
        )

    # ---- Arm 1: plaintext FedAvg, the baseline this protocol must reproduce ------------
    start = time.perf_counter()
    plain = initial_weights(n_features)
    round_updates: list[list[Weights]] = []
    for r in range(cfg.rounds):
        updates = [_local(plain, site, r) for site in sites]
        round_updates.append(updates)
        total_rows = float(sum(s.n_rows for s in sites))
        coef = np.zeros(n_features, dtype=float)
        intercept = 0.0
        for site, update in zip(sites, updates, strict=True):
            share = site.n_rows / total_rows
            coef += share * update.coef
            intercept += share * update.intercept
        plain = Weights(coef, intercept)
    train_seconds = time.perf_counter() - start
    plaintext_pr_auc, plaintext_tpr = _score(plain)

    # ---- Arm 2: the same federation, aggregated under the protocol --------------------
    scale = float(2**cfg.scale_bits)
    secure = initial_weights(n_features)
    field_exact = True
    max_weight_error = 0.0
    for r in range(cfg.rounds):
        updates = [_local(secure, site, r) for site in sites]
        contributions = {
            site.index: encode_contribution(update, site.n_rows, scale)
            for site, update in zip(sites, updates, strict=True)
        }
        transcript = secure_round(
            contributions, keys, round_index=r, threshold=threshold, rng=key_rng
        )
        plain_total = np.zeros(n_features + 2, dtype=np.int64)
        for vector in contributions.values():
            plain_total = np.mod(plain_total + vector, FIELD_PRIME)
        field_exact = field_exact and bool(np.array_equal(transcript.total, plain_total))
        secure = decode_aggregate(transcript.total, scale)
    secure_pr_auc, secure_tpr = _score(secure)
    max_weight_error = float(
        np.max(
            np.abs(
                np.append(secure.coef, secure.intercept) - np.append(plain.coef, plain.intercept)
            )
        )
    )

    # ---- The encoding window: quantization floor at one end, wraparound at the other ---
    # Headroom is computed from the largest magnitude the run *actually* produced, not from an
    # estimate off the first round: the weights grow as training proceeds, and a scale that
    # fits round one and wraps in round five is exactly the failure this table exists to show.
    exactness: list[ExactnessRow] = []
    for bits in cfg.scale_bits_sweep:
        s = float(2**bits)
        w = initial_weights(n_features)
        largest = 0.0
        for r in range(cfg.rounds):
            updates = [_local(w, site, r) for site in sites]
            contributions = {
                site.index: encode_contribution(update, site.n_rows, s)
                for site, update in zip(sites, updates, strict=True)
            }
            true_sum = np.zeros(n_features + 2, dtype=float)
            for site, update in zip(sites, updates, strict=True):
                true_sum += np.append(
                    np.append(update.coef, update.intercept) * site.n_rows, float(site.n_rows)
                )
            largest = max(largest, float(np.max(np.abs(true_sum))))
            total = np.zeros(n_features + 2, dtype=np.int64)
            for vector in contributions.values():
                total = np.mod(total + vector, FIELD_PRIME)
            w = decode_aggregate(total, s)
        pr_auc, _ = _score(w)
        headroom = encoding_headroom_bits(largest, s)
        exactness.append(
            ExactnessRow(
                scale_bits=bits,
                headroom_bits=headroom,
                max_weight_error=float(
                    np.max(
                        np.abs(
                            np.append(w.coef, w.intercept) - np.append(plain.coef, plain.intercept)
                        )
                    )
                ),
                pr_auc=pr_auc,
                wrapped=headroom <= 0.0,
            )
        )

    # ---- What the coordinator can read off an update ----------------------------------
    families = sorted({s.family for s in sites if s.attack_prior > 0})
    benign_rows = np.flatnonzero(multi == settings.labels.benign_label)
    references: dict[str, np.ndarray] = {}
    for family in [*families, "none (benign only)"]:
        rows = (
            np.flatnonzero(multi == family)
            if family != "none (benign only)"
            else np.zeros(0, dtype=int)
        )
        pool = rng.choice(
            benign_rows, size=min(cfg.reference_benign_rows, len(benign_rows)), replace=False
        )
        idx = np.concatenate([pool, rows]) if len(rows) else pool
        reference = local_train(
            initial_weights(n_features),
            x_train[idx],
            y_train[idx],
            seed=variant.seed + 7,
            **train_kwargs,  # type: ignore[arg-type]
        )
        references[family] = np.append(reference.coef, reference.intercept)

    chance = 1.0 / len(references)
    correct = 0
    total_attempts = 0
    for updates in round_updates[: cfg.privacy_rounds]:
        for site, update in zip(sites, updates, strict=True):
            correct += int(_identify(update, references) == site.family)
            total_attempts += 1
    plaintext_accuracy = correct / max(total_attempts, 1)

    masked_correct = 0
    masked_attempts = 0
    contributions = {
        site.index: encode_contribution(update, site.n_rows, scale)
        for site, update in zip(sites, round_updates[0], strict=True)
    }
    transcript = secure_round(contributions, keys, round_index=0, threshold=threshold, rng=key_rng)
    for r, updates in enumerate(round_updates[: cfg.privacy_rounds]):
        round_contributions = {
            site.index: encode_contribution(update, site.n_rows, scale)
            for site, update in zip(sites, updates, strict=True)
        }
        masked_round = secure_round(
            round_contributions, keys, round_index=r, threshold=threshold, rng=key_rng
        )
        for site, vector in zip(sites, masked_round.masked, strict=True):
            decoded = dequantize(vector, scale)
            masked_update = Weights(decoded[:n_features], float(decoded[n_features]))
            masked_correct += int(_identify(masked_update, references) == site.family)
            masked_attempts += 1
    masked_accuracy = masked_correct / max(masked_attempts, 1)

    aggregate_weights = decode_aggregate(transcript.total, scale)
    aggregate_family = _identify(aggregate_weights, references)
    aggregate_hit = float(any(aggregate_family == s.family for s in sites if s.attack_prior > 0))

    privacy = [
        PrivacyRow(
            "plaintext update (what FedAvg sends today)",
            plaintext_accuracy,
            chance,
            "one local pass, cosine against per-family references",
        ),
        PrivacyRow(
            "masked vector (what the coordinator receives)",
            masked_accuracy,
            chance,
            "a one-time pad: uniform in the field, independent of the input",
        ),
        PrivacyRow(
            "the aggregate (what the protocol does release)",
            aggregate_hit,
            chance,
            "not protected by this protocol and never was -- that is what DP is for",
            display=f"resolves to `{aggregate_family}`",
        ),
    ]

    # ---- The attack the self-mask exists to stop -------------------------------------
    victim = next(s.index for s in sites if s.attack_prior > 0)
    truth = np.append(
        round_updates[0][victim].coef * sites[victim].n_rows,
        round_updates[0][victim].intercept * sites[victim].n_rows,
    )
    without = secure_round(
        contributions,
        keys,
        round_index=0,
        threshold=threshold,
        use_self_mask=False,
        unmask_target=victim,
        rng=key_rng,
    )
    with_mask = secure_round(
        contributions,
        keys,
        round_index=0,
        threshold=threshold,
        use_self_mask=True,
        unmask_target=victim,
        rng=key_rng,
    )

    def _victim_error(recovered: np.ndarray | None) -> float:
        if recovered is None:
            return float("inf")
        decoded = dequantize(recovered, scale)[: n_features + 1]
        return float(np.max(np.abs(decoded - truth)))

    unmask_error_without = _victim_error(without.unmasked_target)
    unmask_error_with = _victim_error(with_mask.unmasked_target)

    # ---- Dropout ----------------------------------------------------------------------
    dropout: list[DropoutRow] = []
    for n_dropped in cfg.dropout_counts:
        dropped = frozenset(range(min(n_dropped, n_sites)))
        transcript = secure_round(
            contributions,
            keys,
            round_index=0,
            threshold=threshold,
            dropped=dropped,
            rng=key_rng,
        )
        expected = np.zeros(n_features + 2, dtype=np.int64)
        for index, vector in contributions.items():
            if index not in dropped:
                expected = np.mod(expected + vector, FIELD_PRIME)
        max_error = (
            float(np.max(np.abs(dequantize(transcript.total, scale) - dequantize(expected, scale))))
            if transcript.recovered
            else float("inf")
        )
        dropout.append(
            DropoutRow(
                n_dropped=len(dropped),
                survivors=n_sites - len(dropped),
                threshold=threshold,
                recovered=transcript.recovered,
                max_error=max_error,
                recover_ms=transcript.recover_seconds * 1000.0,
            )
        )

    # ---- The tension: every robust rule needs what the protocol hides ------------------
    def _federate(
        rule: str, attacker: int | None, *, bound: float | None = None, groups: int = 1
    ) -> float:
        w = initial_weights(n_features)
        for r in range(cfg.rounds):
            updates = [_local(w, site, r) for site in sites]
            if attacker is not None:
                if bound is None:
                    updates[attacker] = sign_flip(updates[attacker], cfg.attack_scale)
                else:
                    # The strongest attack an ideal range proof still permits: every
                    # coordinate pinned to the far end of the admissible interval.
                    honest = np.append(updates[attacker].coef, updates[attacker].intercept)
                    pinned = -bound * np.sign(np.where(honest == 0.0, 1.0, honest))
                    updates[attacker] = Weights(pinned[:-1], float(pinned[-1]))
            sizes = [s.n_rows for s in sites]
            if groups > 1:
                partials: list[Weights] = []
                for chunk in np.array_split(np.arange(n_sites), groups):
                    rows = float(sum(sizes[i] for i in chunk))
                    coef = np.zeros(n_features, dtype=float)
                    intercept = 0.0
                    for i in chunk:
                        coef += (sizes[i] / rows) * updates[i].coef
                        intercept += (sizes[i] / rows) * updates[i].intercept
                    partials.append(Weights(coef, intercept))
                w = coordinate_median(partials)
            elif rule == "coordinate median":
                w = coordinate_median(updates)
            elif rule == "Krum":
                w = krum(updates, 1)
            else:
                total_rows = float(sum(sizes))
                coef = np.zeros(n_features, dtype=float)
                intercept = 0.0
                for size, update in zip(sizes, updates, strict=True):
                    coef += (size / total_rows) * update.coef
                    intercept += (size / total_rows) * update.intercept
                w = Weights(coef, intercept)
        return _score(w)[0]

    attacker_index = next(s.index for s in sites if s.attack_prior > 0)
    robustness = [
        RobustnessRow(
            "plaintext FedAvg (mean)",
            "every site's update",
            "none",
            plaintext_pr_auc,
            defence_available=True,
        ),
        RobustnessRow(
            "plaintext FedAvg (mean)",
            "every site's update",
            "sign flip",
            _federate("mean", attacker_index),
            defence_available=True,
        ),
        RobustnessRow(
            "plaintext + coordinate median",
            "every site's update",
            "sign flip",
            _federate("coordinate median", attacker_index),
            defence_available=True,
        ),
        RobustnessRow(
            "plaintext + Krum",
            "every site's update",
            "sign flip",
            _federate("Krum", attacker_index),
            defence_available=True,
        ),
        RobustnessRow(
            "secure aggregation (mean is the only option)",
            "the sum, and nothing else",
            "sign flip",
            _federate("mean", attacker_index),
            defence_available=False,
        ),
    ]

    honest_max_coordinate = float(
        np.max([np.max(np.abs(np.append(u.coef, u.intercept))) for u in round_updates[0]])
    )
    ranges = [
        RangeRow(
            bound=bound,
            bound_vs_honest=bound / max(honest_max_coordinate, 1e-12),
            pr_auc=_federate("mean", attacker_index, bound=bound),
        )
        for bound in cfg.range_bounds
    ]

    groups: list[GroupRow] = []
    for size in cfg.group_sizes:
        n_groups = max(1, n_sites // max(size, 1))
        groups.append(
            GroupRow(
                group_size=max(1, n_sites // n_groups),
                n_groups=n_groups,
                pr_auc_clean=_federate("mean", None, groups=n_groups),
                pr_auc_attacked=_federate("mean", attacker_index, groups=n_groups),
            )
        )

    # ---- Cost, as a function of how many sites are in the federation ------------------
    cost: list[CostRow] = []
    length = n_features + 2
    for size in cfg.cost_sites:
        synthetic_keys = build_site_keys([f"s{i}" for i in range(size)], random.Random(size))
        synthetic = {i: quantize(rng.normal(size=length), scale) for i in range(size)}
        transcript = secure_round(
            synthetic,
            synthetic_keys,
            round_index=0,
            threshold=max(2, int(np.ceil(cfg.threshold_fraction * size))),
            rng=random.Random(size),
        )
        cost.append(
            CostRow(
                n_sites=size,
                plaintext_bytes=length * 8,
                secure_bytes=transcript.bytes_per_site,
                mask_ms=transcript.mask_seconds * 1000.0,
                recover_ms=transcript.recover_seconds * 1000.0,
            )
        )

    return SecAggStudy(
        sites=sites,
        n_features=n_features,
        rounds=cfg.rounds,
        threshold=threshold,
        scale_bits=cfg.scale_bits,
        plaintext_pr_auc=plaintext_pr_auc,
        plaintext_tpr=plaintext_tpr,
        secure_pr_auc=secure_pr_auc,
        secure_tpr=secure_tpr,
        field_exact=field_exact,
        max_weight_error=max_weight_error,
        exactness=exactness,
        privacy=privacy,
        unmask_error_without_self_mask=unmask_error_without,
        unmask_error_with_self_mask=unmask_error_with,
        dropout=dropout,
        robustness=robustness,
        ranges=ranges,
        groups=groups,
        cost=cost,
        train_seconds=train_seconds,
        families=[*families, "none (benign only)"],
        honest_max_coordinate=honest_max_coordinate,
        target_fpr=target_fpr,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def run_secagg_report(settings: Settings) -> Path:
    """Run the secure-aggregation study and write the report + figures."""
    study = run_secagg_study(settings)

    sizes = np.array([row.group_size for row in study.groups], dtype=float)
    frontier_fig = plots.plot_lines(
        {
            "PR-AUC under a sign-flip attacker": (
                sizes,
                np.array([row.pr_auc_attacked for row in study.groups], dtype=float),
            ),
            "PR-AUC with nobody attacking": (
                sizes,
                np.array([row.pr_auc_clean for row in study.groups], dtype=float),
            ),
        },
        xlabel="anonymity set (sites aggregated before the coordinator sees a number)",
        ylabel="PR-AUC",
        title="Grouped secure aggregation: anonymity against robustness",
        out_path=settings.paths.figures_dir / FRONTIER_FIGURE_NAME,
    )
    site_counts = np.array([row.n_sites for row in study.cost], dtype=float)
    cost_fig = plots.plot_lines(
        {
            "masking": (site_counts, np.array([row.mask_ms for row in study.cost], dtype=float)),
            "recovery": (
                site_counts,
                np.array([row.recover_ms for row in study.cost], dtype=float),
            ),
        },
        xlabel="sites in the federation",
        ylabel="milliseconds per round",
        title="What the protocol costs as the federation grows",
        out_path=settings.paths.figures_dir / COST_FIGURE_NAME,
        yscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, frontier_fig, cost_fig), encoding="utf-8")
    logger.info("Wrote secure-aggregation report", extra={"path": str(out_path)})

    with track_run(settings, "secagg") as run:
        run.log_params(
            {
                "sites": len(study.sites),
                "threshold": study.threshold,
                "rounds": study.rounds,
                "scale_bits": study.scale_bits,
            }
        )
        run.log_metrics(
            {
                "plaintext_pr_auc": study.plaintext_pr_auc,
                "secure_pr_auc": study.secure_pr_auc,
                "max_weight_error": study.max_weight_error,
                "identification_plaintext": study.privacy[0].accuracy,
                "identification_masked": study.privacy[1].accuracy,
            }
        )
        run.log_artifact(frontier_fig)
        run.log_artifact(cost_fig)
        run.log_artifact(out_path)
    return out_path


def _site_table(study: SecAggStudy) -> str:
    rows = ["| site | flows | attack share | family it holds |", "|---|---|---|---|"]
    for site in study.sites:
        rows.append(
            f"| `{site.name}` | {site.n_rows:,} | {site.attack_prior:.1%} | {site.family} |"
        )
    return "\n".join(rows)


def _exactness_table(study: SecAggStudy) -> str:
    rows = [
        "| fixed-point scale | headroom to wraparound | max weight error vs plaintext | PR-AUC |",
        "|---|---|---|---|",
    ]
    for row in study.exactness:
        headroom = "**wrapped**" if row.wrapped else f"{row.headroom_bits:.1f} bits"
        rows.append(
            f"| 2^{row.scale_bits} | {headroom} | {row.max_weight_error:.2e} | {row.pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _privacy_table(study: SecAggStudy) -> str:
    rows = [
        "| what the coordinator holds | attack recovers | chance | how |",
        "|---|---|---|---|",
    ]
    for row in study.privacy:
        rows.append(f"| {row.view} | {row.shown()} | {row.chance:.0%} | {row.note} |")
    return "\n".join(rows)


def _dropout_table(study: SecAggStudy) -> str:
    rows = [
        "| sites dropped | survivors | threshold | aggregate recovered | max error | recovery |",
        "|---|---|---|---|---|---|",
    ]
    for row in study.dropout:
        verdict = "yes" if row.recovered else "**no (round lost)**"
        error = f"{row.max_error:.2e}" if row.recovered else "n/a"
        rows.append(
            f"| {row.n_dropped} | {row.survivors} | {row.threshold} | {verdict} | {error} | "
            f"{row.recover_ms:.0f} ms |"
        )
    return "\n".join(rows)


def _robustness_table(study: SecAggStudy) -> str:
    rows = [
        "| aggregation | coordinator sees | attack | PR-AUC | robust rule available |",
        "|---|---|---|---|---|",
    ]
    for row in study.robustness:
        available = "yes" if row.defence_available else "**no**"
        rows.append(
            f"| {row.setting} | {row.visible_to_coordinator} | {row.attack} | {row.pr_auc:.3f} | "
            f"{available} |"
        )
    return "\n".join(rows)


def _range_table(study: SecAggStudy) -> str:
    rows = [
        "| certified coordinate bound | as a multiple of the honest maximum | PR-AUC under the "
        "strongest in-bound attack |",
        "|---|---|---|",
    ]
    for row in study.ranges:
        rows.append(f"| {row.bound:g} | {row.bound_vs_honest:.2f}x | {row.pr_auc:.3f} |")
    return "\n".join(rows)


def _group_table(study: SecAggStudy) -> str:
    rows = [
        "| anonymity set | groups the median runs over | PR-AUC (no attacker) | "
        "PR-AUC (sign flip) |",
        "|---|---|---|---|",
    ]
    for row in study.groups:
        label = (
            "1 site (no aggregation privacy: today's FedAvg)"
            if row.group_size <= 1
            else f"{row.group_size} sites"
        )
        rows.append(
            f"| {label} | {row.n_groups} | {row.pr_auc_clean:.3f} | {row.pr_auc_attacked:.3f} |"
        )
    return "\n".join(rows)


def _cost_table(study: SecAggStudy) -> str:
    rows = [
        "| sites | plaintext upload | secure upload | overhead | masking | recovery |",
        "|---|---|---|---|---|---|",
    ]
    for row in study.cost:
        overhead = row.secure_bytes / max(row.plaintext_bytes, 1)
        rows.append(
            f"| {row.n_sites} | {row.plaintext_bytes:,} B | {row.secure_bytes:,} B | "
            f"{overhead:.1f}x | {row.mask_ms:.0f} ms | {row.recover_ms:.0f} ms |"
        )
    return "\n".join(rows)


def _exactness_read(study: SecAggStudy) -> str:
    exact = (
        "The recovered sum is **bit-identical** to the plaintext sum in the field at every "
        "round -- not close, equal, because the masks are group elements and they cancel."
        if study.field_exact
        else "The recovered sum did **not** match the plaintext sum, which is a bug in the "
        "protocol implementation rather than a property of it."
    )
    usable = [row for row in study.exactness if not row.wrapped]
    wrapped = [row for row in study.exactness if row.wrapped]
    floor = min(usable, key=lambda r: r.scale_bits) if usable else None
    ceiling = max(usable, key=lambda r: r.scale_bits) if usable else None
    cliff = min(wrapped, key=lambda r: r.scale_bits) if wrapped else None
    window = (
        f"The usable window runs from 2^{floor.scale_bits} to 2^{ceiling.scale_bits}"
        + (
            f", and at 2^{cliff.scale_bits} the encoded sum passes the field's half-point and "
            f"wraps: PR-AUC {cliff.pr_auc:.3f} against the plaintext model's "
            f"{study.plaintext_pr_auc:.3f}. Nothing warns you. The decoded weights are "
            "well-formed floats of the wrong sign and magnitude, the training loop continues, "
            "and the model that comes out is not an approximation of anything."
            if cliff
            else "."
        )
        if floor and ceiling
        else "No scale in the sweep left headroom, which means the sweep is mis-parameterised."
    )
    return (
        f"{exact} After decoding, the largest single-weight difference between the secure model "
        f"and the plaintext one is **{study.max_weight_error:.2e}** — one quantization step at "
        f"2^{study.scale_bits} — and PR-AUC agrees to "
        f"{abs(study.plaintext_pr_auc - study.secure_pr_auc):.4f}. {window}\n\n"
        "The *lower* end is the surprise: detection is unchanged even at a scale of 2^0, one "
        "step per unit, where the quantization step is larger than the weights being encoded. "
        "It survives because the payload is not the weight vector, it is the **size-weighted** "
        "one — FedAvg's numerator — so each coordinate arrives pre-multiplied by a site's few "
        "thousand rows, which is eleven bits of scale the encoding gets for free. That is a "
        "property of this aggregation, not of fixed-point arithmetic, and it flips the usual "
        "advice: the risk here is not too little precision, it is a scale chosen 'generously' "
        "and then meeting a larger federation."
    )


def _privacy_read(study: SecAggStudy) -> str:
    plaintext = study.privacy[0]
    masked = study.privacy[1]
    return (
        f"The attack is the cheapest one available -- cosine similarity against a reference "
        f"update per attack family, no model inversion, no auxiliary data beyond the family "
        f"labels a coordinator running a detection consortium already has. On the plaintext "
        f"update it names the family a site is holding **{plaintext.accuracy:.0%}** of the time "
        f"against a {plaintext.chance:.0%} chance rate. This is the channel FedAvg leaves open "
        f"in exchange for not moving the flows: it does not reveal a flow, it reveals what kind "
        f"of incident the site is having. On the masked vector the same attack lands at "
        f"{masked.accuracy:.0%} against the {masked.chance:.0%} chance rate — indistinguishable "
        "from guessing, and it has to be. Each masked vector is a "
        "one-time pad, so the coordinator's view is *independent of the input* and no attack, "
        "present or future, extracts anything from it. That is an information-theoretic "
        "statement about a single round, not a measured one; the measurement is here because a "
        "protocol nobody ran is a protocol nobody has debugged."
    )


def _aggregate_read(study: SecAggStudy) -> str:
    return (
        "The third row is the one that keeps this honest. Secure aggregation protects the "
        "**individual** update and does nothing whatsoever for the sum, which is released by "
        "design -- and the sum of a federation that happens to contain one attacked site still "
        "carries that site's signal. The identification attack run on the released aggregate "
        f"still lands on a family a site is genuinely holding. With {len(study.sites)} sites the "
        "aggregate is a weak channel; with three it is a strong one, and with two it is barely "
        "a channel at all in the sense that each participant can subtract itself and read the "
        "other. Secure aggregation buys anonymity within the cohort; **differential privacy** "
        "bounds what the cohort's output says about any member. The [DP-FedAvg arm of the "
        "federated study](federated.md) is the other half of this, and neither substitutes for "
        "the other."
    )


def _selfmask_read(study: SecAggStudy) -> str:
    return (
        "The self-mask looks like redundancy until the attack is run. A coordinator that wants "
        "one site's update in the clear does not have to break anything: it waits until the "
        "masked vector has arrived, **declares that site dropped**, and asks the others for the "
        "shares that reconstruct the victim's pairwise masks -- which the protocol is obliged "
        "to hand over, because that is exactly the recovery path a genuine dropout needs. "
        f"Executed here, with the self-mask removed, it recovers the victim's update to a "
        f"maximum error of **{study.unmask_error_without_self_mask:.2e}** -- the quantization "
        "step, i.e. exactly. With the self-mask in place the same attack leaves the coordinator "
        f"holding a residual whose error against the true update is "
        f"{study.unmask_error_with_self_mask:.2e}: uniform noise, because the shares it "
        "collected unmask the pairs and not the site's own pad. The invariant that makes this "
        "work is that a site answers **one** of the two questions -- self-seed if it is alive, "
        "exponent if it is not -- and never both."
    )


def _tension_read(study: SecAggStudy) -> str:
    plain_mean = next(
        r
        for r in study.robustness
        if r.setting.startswith("plaintext FedAvg") and r.attack == "sign flip"
    )
    median = next(r for r in study.robustness if "median" in r.setting)
    clean = next(r for r in study.robustness if r.attack == "none")
    return (
        f"The [byzantine study](byzantine.md) shows one lying site destroys a mean "
        f"({plain_mean.pr_auc:.3f} against {clean.pr_auc:.3f} clean) and that a coordinate "
        f"median recovers most of it ({median.pr_auc:.3f}). Every one of those defences is a "
        "function of the **individual** update vectors. Secure aggregation delivers the "
        "coordinator one vector: the sum. There is no median of one number, no Krum among one "
        "candidate, no norm to inspect -- the last row is the mean, because under this protocol "
        "the mean is the only rule that exists, and it therefore reproduces the undefended "
        "number exactly. The privacy property and the robustness property are not merely hard "
        "to have together; they ask for opposite things from the same channel, which is why the "
        "work that gets both (Prio, RoFL, secure aggregation with verifiable norm bounds) is a "
        "separate line of research rather than a configuration flag."
    )


def _range_read(study: SecAggStudy) -> str:
    if not study.ranges:
        return ""
    clean = next(r for r in study.robustness if r.attack == "none").pr_auc
    unbounded = next(
        r for r in study.robustness if r.setting.startswith("secure aggregation (mean")
    ).pr_auc
    worst = min(study.ranges, key=lambda r: r.pr_auc)
    best = max(study.ranges, key=lambda r: r.pr_auc)
    loose = [r for r in study.ranges if r.pr_auc <= unbounded]
    return (
        "The first escape keeps the privacy and buys back some robustness by assuming what the "
        "protocol cannot check: that each site can prove, in zero knowledge, that every "
        "coordinate of its input lies inside a certified interval. The attacker then plays the "
        "strongest move the proof still permits — every coordinate pinned to the far end of the "
        "interval — and the question becomes what the bound is worth.\n\n"
        f"{_range_table(study)}\n\n"
        f"The honest sites' largest coordinate in the first round is "
        f"{study.honest_max_coordinate:.3f}, and that is the number the bound has to be set "
        f"against. A bound of {best.bound:g} ({best.bound_vs_honest:.2f}x the honest maximum) "
        f"holds detection at {best.pr_auc:.3f} against the clean {clean:.3f}. A bound of "
        f"{worst.bound:g} ({worst.bound_vs_honest:.2f}x) gives back "
        f"{clean - worst.pr_auc:.3f} PR-AUC"
        + (
            " — **worse than the unbounded sign-flip attack it was meant to stop** "
            f"({unbounded:.3f}), because the honest updates are small and a 'reasonable-looking' "
            "per-coordinate limit is enormous relative to them."
            if loose and worst.pr_auc < unbounded
            else "."
        )
        + " A range proof is not a defence on its own; it is a defence *plus* a calibration "
        "problem, and the calibration has to be done against measured honest updates rather "
        "than against a round number that looks conservative."
    )


def _group_read(study: SecAggStudy) -> str:
    if not study.groups:
        return ""
    smallest = min(study.groups, key=lambda r: r.group_size)
    largest = max(study.groups, key=lambda r: r.group_size)
    middle = min(
        (r for r in study.groups if r.group_size not in (smallest.group_size, largest.group_size)),
        key=lambda r: abs(r.pr_auc_clean - r.pr_auc_attacked),
        default=smallest,
    )
    return (
        "Sites are aggregated in groups and the coordinator applies a robust rule *across the "
        "group sums*. The anonymity set is no longer the federation, it is the group. At the "
        f"top of the table the group is one site, which is not anonymity at all — that row is "
        "today's plaintext FedAvg with a median bolted on, and it is the most robust "
        f"({smallest.pr_auc_attacked:.3f} under attack) and the *least* accurate when nobody "
        f"attacks ({smallest.pr_auc_clean:.3f} against {largest.pr_auc_clean:.3f}), which is "
        "the price of a median that the byzantine study already charged. At the bottom the "
        f"group is the whole federation: maximum anonymity, no visibility, and "
        f"{largest.pr_auc_attacked:.3f} under the same attacker.\n\n"
        f"The interesting rows are the middle ones. Groups of {middle.group_size} give the "
        f"coordinator {middle.group_size}-anonymised partial sums and hold "
        f"{middle.pr_auc_attacked:.3f} under attack at {middle.pr_auc_clean:.3f} clean — "
        "strictly more privacy than plaintext and strictly more robustness than a single "
        "aggregate. Neither end of this table is the answer; the frontier is, and an operator "
        "picks a point on it by naming which adversary they actually fear — the coordinator, or "
        "a member."
    )


def _cost_read(study: SecAggStudy) -> str:
    small = study.cost[0]
    large = study.cost[-1]
    return (
        f"Upload grows because every site shares two secrets with every peer: "
        f"{small.secure_bytes:,} bytes at {small.n_sites} sites against "
        f"{large.secure_bytes:,} at {large.n_sites}, an overhead of "
        f"{small.secure_bytes / max(small.plaintext_bytes, 1):.1f}x rising to "
        f"{large.secure_bytes / max(large.plaintext_bytes, 1):.1f}x. Compute grows faster: the "
        f"pairwise masks are O(n^2) modular exponentiations across the federation, "
        f"{small.mask_ms:.0f} ms at {small.n_sites} sites and {large.mask_ms:.0f} ms at "
        f"{large.n_sites}. Against a round of local training that costs "
        f"{study.train_seconds / max(study.rounds, 1) * 1000:.0f} ms across all sites, the "
        "protocol is not free and is not the bottleneck either -- which is the honest summary "
        "at this scale. At a thousand sites it would be, and the standard answer is the one "
        "Bonawitz et al. give: mask against a sampled *subset* of peers rather than all of "
        "them, which turns the quadratic term linear at the cost of a probabilistic security "
        "argument."
    )


def _render(study: SecAggStudy, frontier_fig: Path, cost_fig: Path) -> str:
    return f"""# NetSentry — Secure Aggregation: Federating Without a Trusted Coordinator

_Bonawitz et al. (CCS 2017) implemented from scratch over RFC 3526 group 14 and the field
`2^61 - 1`, run across {len(study.sites)} sites for {study.rounds} rounds with a
{study.threshold}-of-{len(study.sites)} recovery threshold._

## Why this report exists

The [federated study](federated.md) rests on one claim: raw flows never leave the site, only
weights do. That claim is true, and it is not privacy. An update is a function of the data that
produced it, and the coordinator collects one per site per round. So the first measurement here
is not of a protocol, it is of the channel the current design leaves open.

{_privacy_table(study)}

{_privacy_read(study)}

{_aggregate_read(study)}

## The sites

Capture days are the natural silo — Monday holds no attacks at all, Tuesday holds the patators,
Wednesday the DoS family — but three participants cannot say anything about a recovery threshold
or an anonymity set, so each day is sharded. The skew that makes federation hard is preserved.

{_site_table(study)}

## Does it still train?

| | PR-AUC | TPR @ {study.target_fpr:.1%} FPR |
|---|---|---|
| plaintext FedAvg | {study.plaintext_pr_auc:.3f} | {study.plaintext_tpr:.1%} |
| secure aggregation | {study.secure_pr_auc:.3f} | {study.secure_tpr:.1%} |

{_exactness_table(study)}

{_exactness_read(study)}

## Dropout: the reason this is not just a one-time pad

A mask that only cancels when *everybody* arrives is useless on a real network. Each site
secret-shares two values with `t`-of-`n` recovery: its self-mask seed, released only for sites
confirmed **alive**, and its Diffie-Hellman exponent, released only for sites confirmed **gone**.

{_dropout_table(study)}

Below the threshold the round is lost — a liveness failure, and the right one: the coordinator
ends up with a masked sum it cannot open rather than a partial result it can.

## The attack the self-mask exists to stop

{_selfmask_read(study)}

## The cost nobody advertises: robustness

{_robustness_table(study)}

{_tension_read(study)}

### Escape 1: an ideal range proof

{_range_read(study)}

### Escape 2: grouped aggregation

![Grouped secure aggregation frontier](../figures/{frontier_fig.name})

{_group_table(study)}

{_group_read(study)}

## What it costs

{_cost_table(study)}

![Protocol cost by federation size](../figures/{cost_fig.name})

{_cost_read(study)}

## Scope and honest limits

- **This is the masking protocol, not the whole system.** Bonawitz et al. run key agreement
  over an authenticated channel and encrypt the shares to their recipients; here the shares are
  handed to the coordinator in the clear because the study measures what the *aggregation*
  reveals, not what a network attacker does. A deployment needs both, and needs the keys drawn
  from an OS CSPRNG rather than the seeded generator that makes this report reproducible.
- **The security argument is per round, and the rounds are not independent.** Masks are
  re-derived per round from the KDF, so no two rounds share a pad; what a coordinator learns
  across rounds is the *sequence of aggregates*, and that is a differential-privacy question
  (composition), which the DP-FedAvg arm of the federated study accounts for and this does not.
- **The model is linear** for the same reason it is in the federated study: FedAvg averages
  parameters, and a boosted forest does not have any to average. Secure aggregation is
  architecture-agnostic — it sums vectors — but the study inherits the linear arm's ceiling.
- **The ideal range proof is assumed, not implemented.** The sweep prices what an attacker
  could still do *given* such a proof; building one (Prio-style secret-shared range checks, or
  RoFL's norm bounds) is a substantially larger piece of engineering and is named here rather
  than hand-waved as a config option. Treat those rows as an upper bound on what the technique
  buys, since a real proof also costs bandwidth and verification time this does not model.
- **The anonymity set is the federation, not the world.** A coordinator that already knows
  eleven of twelve sites' data learns the twelfth exactly from the sum. That is not a flaw in
  the protocol; it is the definition of what the protocol promises, and it is why the frontier
  above is about group *size* rather than about a binary."""
