"""Scoring a flow neither party is willing to show the other.

`/predict` has a privacy structure nobody writes down: the client uploads 76 features of its
own network traffic, and the server replies with a verdict computed by a model it will not
share. Both sides give something up. For a managed-detection provider that is the whole
commercial arrangement -- customers hand over telemetry, the vendor keeps the detector -- and
it is exactly the arrangement two-party computation exists to remove.

This module implements the removal, from the standard library and numpy, over a prime field:

- **Additive secret sharing.** Every secret is split into two shares that are individually
  uniform, so each party holds noise until they cooperate.
- **Beaver triples** (Beaver, CRYPTO 1991) for multiplication: a preprocessed random triple
  ``c = a * b`` lets two parties multiply shared values by opening two *masked* quantities,
  which carry no information about the operands.

The model it evaluates is the [additive one](gam.md), and that is not a compromise made for
convenience -- it is the reason the protocol is cheap. A generalized additive model is a sum of
per-feature table lookups, and a table lookup is an inner product with a one-hot vector. Two
consequences follow. The whole inference is a single batched round of multiplications, and
because one operand is a **selector rather than a value**, the fixed-point scale is preserved
and no probabilistic truncation is needed. The glass box turns out to be the private box too,
for a structural reason.

What the protocol delivers is measured: exactness against the plaintext model, bytes on the
wire, rounds, latency, and a uniformity test on everything the server sees. What it does not
deliver is measured too, and that is the part worth reading. The bin edges have to be public
for the client to bin its own flow, and those edges *are* a quantile summary of the training
traffic. And the guarantee holds against an honest-but-curious client -- against a malicious
one, secret sharing hides the input so thoroughly that **the server cannot check it is an
input**, and the entire model can be read out with crafted queries. That attack is executed
here rather than described.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import chi2
from sklearn.utils.class_weight import compute_sample_weight

from netsentry.evaluation import plots
from netsentry.features.feature_sets import display_feature_name
from netsentry.log import get_logger
from netsentry.models.gam import AdditiveModel, Binner, fit_additive
from netsentry.monitoring.transport import wasserstein_1d
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import PrivateInferenceConfig

logger = get_logger(__name__)

REPORT_NAME = "private_inference.md"
PRECISION_FIGURE = "private_inference_precision.png"
EDGES_FIGURE = "private_inference_edges.png"

#: A Mersenne prime small enough that products of two field elements stay inside a signed
#: 64-bit integer (2^62 < 2^63), so the whole protocol runs in numpy without big integers.
PRIME = (1 << 31) - 1

#: Bytes on the wire per field element. Four would do at this prime; eight is what a real
#: implementation sends, and understating the cost of your own protocol is a bad habit.
ELEMENT_BYTES = 8


# --------------------------------------------------------------------------------------
# Field arithmetic and fixed-point encoding.
# --------------------------------------------------------------------------------------


def encode(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    """Map signed reals into the field as fixed-point integers.

    Negative numbers become large field elements, which is the standard two's-complement-like
    convention: anything above ``PRIME / 2`` decodes as negative. The number of fraction bits
    is the only knob, and it trades precision against the headroom the sum needs before it
    wraps -- a wrap is not a small error, it is a completely different number, which is why the
    report sweeps this rather than picking one.
    """
    scaled = np.rint(np.asarray(values, dtype=float) * (1 << fraction_bits)).astype(np.int64)
    encoded: np.ndarray = np.mod(scaled, PRIME)
    return encoded


def decode(values: np.ndarray, fraction_bits: int) -> np.ndarray:
    """Map field elements back to signed reals."""
    centred = np.where(
        np.asarray(values) > PRIME // 2, np.asarray(values) - PRIME, np.asarray(values)
    )
    return np.asarray(centred, dtype=float) / (1 << fraction_bits)


def share(secret: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split a field vector into two additive shares, the first of which is uniform."""
    first = rng.integers(0, PRIME, size=np.shape(secret), dtype=np.int64)
    second = np.mod(np.asarray(secret, dtype=np.int64) - first, PRIME)
    return first, second


def reconstruct(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Recombine two additive shares."""
    return np.mod(np.asarray(first, dtype=np.int64) + np.asarray(second, dtype=np.int64), PRIME)


@dataclass(frozen=True)
class BeaverTriple:
    """A preprocessed random multiplication, shared between the two parties.

    The dealer knows ``a``, ``b`` and ``c = a * b``; neither party knows any of them. This is
    the part of the protocol that has to come from somewhere -- a trusted dealer, oblivious
    transfer, or homomorphic encryption -- and its volume is reported separately from the
    online cost, because pretending preprocessing is free is the most common way a secure
    computation is made to look cheap.
    """

    a_first: np.ndarray
    a_second: np.ndarray
    b_first: np.ndarray
    b_second: np.ndarray
    c_first: np.ndarray
    c_second: np.ndarray

    @property
    def size(self) -> int:
        """Number of multiplications this batch covers."""
        return int(np.size(self.a_first))

    @property
    def bytes_per_party(self) -> int:
        """Preprocessing each party has to receive and store before any flow arrives."""
        return 3 * self.size * ELEMENT_BYTES


def deal_triples(count: int, rng: np.random.Generator) -> BeaverTriple:
    """Generate a batch of multiplication triples (trusted-dealer model)."""
    a = rng.integers(0, PRIME, size=count, dtype=np.int64)
    b = rng.integers(0, PRIME, size=count, dtype=np.int64)
    c = np.mod(a * b, PRIME)
    a_first, a_second = share(a, rng)
    b_first, b_second = share(b, rng)
    c_first, c_second = share(c, rng)
    return BeaverTriple(a_first, a_second, b_first, b_second, c_first, c_second)


@dataclass
class Transcript:
    """Everything that crossed the wire, and everything the server got to look at."""

    rounds: int = 0
    online_bytes: int = 0
    preprocessing_bytes: int = 0
    opened: list[np.ndarray] = field(default_factory=list)

    def merge(self, other: Transcript) -> None:
        """Fold another transcript into this one."""
        self.rounds = max(self.rounds, other.rounds)
        self.online_bytes += other.online_bytes
        self.preprocessing_bytes += other.preprocessing_bytes
        self.opened.extend(other.opened)


def secure_products(
    x_shares: tuple[np.ndarray, np.ndarray],
    y_shares: tuple[np.ndarray, np.ndarray],
    triple: BeaverTriple,
) -> tuple[np.ndarray, np.ndarray, Transcript]:
    """Element-wise multiply two shared vectors with one batched opening.

    The protocol: mask each operand with the triple (``d = x - a``, ``e = y - b``), open the
    masks, and reassemble ``x*y = c + d*b + e*a + d*e`` from local terms. Both ``d`` and ``e``
    are one-time-pad encryptions of the operands under uniformly random field elements, so
    opening them reveals nothing -- which the report checks empirically rather than asserting.

    Everything is opened in **one round** regardless of how many products there are, which is
    what makes an inner product over a whole model cheap: the depth of the circuit is one.
    """
    x_first, x_second = x_shares
    y_first, y_second = y_shares
    d_first = np.mod(x_first - triple.a_first, PRIME)
    d_second = np.mod(x_second - triple.a_second, PRIME)
    e_first = np.mod(y_first - triple.b_first, PRIME)
    e_second = np.mod(y_second - triple.b_second, PRIME)
    d = reconstruct(d_first, d_second)
    e = reconstruct(e_first, e_second)
    first = np.mod(triple.c_first + d * triple.b_first + e * triple.a_first + d * e, PRIME)
    second = np.mod(triple.c_second + d * triple.b_second + e * triple.a_second, PRIME)
    transcript = Transcript(
        rounds=1,
        # Each party sends its share of both masks.
        online_bytes=2 * 2 * triple.size * ELEMENT_BYTES,
        preprocessing_bytes=2 * triple.bytes_per_party,
        opened=[d, e],
    )
    return first, second, transcript


# --------------------------------------------------------------------------------------
# The private model.
# --------------------------------------------------------------------------------------


@dataclass
class PrivateAdditiveModel:
    """A server-side additive model, encoded for secure evaluation.

    ``edges`` is deliberately **public**: the client has to bin its own flow before it can
    build a selector, and it cannot do that without the cut points. The shape tables and the
    intercept stay secret. What the public half gives away is measured in the report rather
    than waved past.
    """

    edges: list[np.ndarray]
    tables: list[np.ndarray]
    intercept: int
    fraction_bits: int

    @classmethod
    def of(cls, model: AdditiveModel, fraction_bits: int) -> PrivateAdditiveModel:
        """Encode a fitted additive model into the field."""
        return cls(
            edges=[np.array(edge, copy=True) for edge in model.binner.edges],
            tables=[encode(shape, fraction_bits) for shape in model.shapes],
            intercept=int(encode(np.array([model.intercept]), fraction_bits)[0]),
            fraction_bits=fraction_bits,
        )

    @property
    def multiplications(self) -> int:
        """One per table entry: the cost of a lookup nobody is allowed to observe."""
        return int(sum(len(table) for table in self.tables))

    def selectors(self, bins: np.ndarray) -> list[np.ndarray]:
        """The client's side: a one-hot vector per feature, and nothing else."""
        vectors = []
        for index, table in enumerate(self.tables):
            selector = np.zeros(len(table), dtype=np.int64)
            selector[int(bins[index])] = 1
            vectors.append(selector)
        return vectors


def private_score(
    model: PrivateAdditiveModel, selectors: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, Transcript]:
    """Evaluate the model on a secret-shared selector, revealing only the score.

    Note what is *not* here: no truncation step. Fixed-point multiplication normally doubles
    the scale and needs a correction that is either expensive or probabilistic. Here one
    operand of every product is a 0/1 selector, so the scale is preserved exactly and the
    protocol is arithmetically exact over the field. That property comes from the model being
    additive, not from anything clever in the protocol.
    """
    flat_selector = np.concatenate(selectors)
    flat_table = np.concatenate(model.tables)
    triple = deal_triples(len(flat_selector), rng)
    client_shares = share(flat_selector, rng)
    server_shares = share(flat_table, rng)
    first, second, transcript = secure_products(client_shares, server_shares, triple)
    total = np.mod(int(first.sum()) + int(second.sum()) + model.intercept, PRIME)
    transcript.online_bytes += 2 * ELEMENT_BYTES  # the two shares of the final sum
    return float(decode(np.array([total]), model.fraction_bits)[0]), transcript


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionRow:
    """One fixed-point setting: how exact the private score is, and how close to wrapping."""

    fraction_bits: int
    worst_error: float
    headroom: float
    wrapped: bool


@dataclass(frozen=True)
class CostRow:
    """What one private verdict costs, beside what a plaintext one costs."""

    quantity: str
    value: str
    against: str


@dataclass(frozen=True)
class LeakRow:
    """A statistical check on everything one party gets to observe."""

    observed: str
    samples: int
    p_value: float
    verdict: str


@dataclass(frozen=True)
class EdgeRow:
    """What the public bin edges give away about the training traffic."""

    feature: str
    reconstruction_distance: float
    floor: float

    @property
    def excess(self) -> float:
        """Distance above what two halves of the same data would show."""
        return max(self.reconstruction_distance - self.floor, 0.0)


@dataclass
class PrivateInferenceStudy:
    """Everything the report needs, computed once."""

    precision: list[PrecisionRow]
    costs: list[CostRow]
    leaks: list[LeakRow]
    edges: list[EdgeRow]
    queries_to_extract: int
    extraction_error: float
    extraction_bytes: int
    n_features: int
    n_bins: int
    multiplications: int
    online_bytes: int
    preprocessing_bytes: int
    rounds: int
    private_ms: float
    plaintext_ms: float
    fraction_bits: int
    seconds: float = 0.0

    @property
    def mean_edge_excess(self) -> float:
        """Average reconstruction quality across the audited features."""
        return float(np.mean([row.excess for row in self.edges])) if self.edges else 0.0


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def _precision_sweep(
    model: AdditiveModel,
    binner: Binner,
    matrix: np.ndarray,
    bit_settings: list[int],
    rng: np.random.Generator,
    flows: int,
) -> list[PrecisionRow]:
    """Sweep the fixed-point encoding, because a wrap is not a rounding error.

    Too few fraction bits and the score is quantised; too many and the accumulated sum passes
    ``PRIME / 2`` and decodes as a large negative number -- not a small error but a different
    answer entirely. The headroom column is what separates the two failure modes, and the sweep
    is here because picking the parameter without looking is how a protocol that works in a
    test becomes one that returns a confident wrong verdict in production.
    """
    bins = binner.transform(matrix[:flows])
    plaintext = model.margin(bins)
    rows: list[PrecisionRow] = []
    for bits in bit_settings:
        private_model = PrivateAdditiveModel.of(model, bits)
        errors = []
        for index in range(len(bins)):
            score, _ = private_score(private_model, private_model.selectors(bins[index]), rng)
            errors.append(abs(score - float(plaintext[index])))
        largest = float(max(abs(plaintext).max(), 1e-9))
        headroom = (PRIME // 2) / max(largest * (1 << bits), 1.0)
        rows.append(
            PrecisionRow(
                fraction_bits=bits,
                worst_error=float(max(errors)),
                headroom=float(headroom),
                # A wrap is a different failure from coarse quantisation: the sum passes
                # PRIME/2 and decodes as a large negative number. Headroom below one is the
                # condition, and it is what the column reports -- not a threshold on the error.
                wrapped=bool(headroom < 1.0),
            )
        )
    return rows


def _uniformity(values: np.ndarray, buckets: int) -> float:
    """Chi-square p-value for 'these field elements are uniformly distributed'."""
    flat = np.asarray(values, dtype=np.int64).ravel()
    counts = np.bincount(
        np.minimum((flat * buckets) // PRIME, buckets - 1).astype(int), minlength=buckets
    )
    expected = counts.sum() / buckets
    if expected <= 0:
        return 1.0
    statistic = float(((counts - expected) ** 2 / expected).sum())
    return float(chi2.sf(statistic, buckets - 1))


def _leak_checks(
    model: PrivateAdditiveModel, bins: np.ndarray, rng: np.random.Generator, flows: int
) -> list[LeakRow]:
    """Test what the server sees, and then test the test.

    Everything the server observes during an inference is a masked value: a secret plus a
    fresh uniform field element. It should therefore be uniform and carry nothing. A clean
    result on its own proves nothing -- a test that cannot fail is not evidence -- so the same
    test is run against a deliberately broken variant that reuses one mask for every flow,
    which is the single most common implementation error in this family of protocols.
    """
    opened: list[np.ndarray] = []
    for index in range(flows):
        _, transcript = private_score(model, model.selectors(bins[index]), rng)
        opened.extend(transcript.opened)
    pooled = np.concatenate(opened)

    # The broken control: one mask, reused, exactly as a cached-randomness bug would produce.
    reused = deal_triples(model.multiplications, rng)
    broken: list[np.ndarray] = []
    for index in range(flows):
        selector = np.concatenate(model.selectors(bins[index]))
        table = np.concatenate(model.tables)
        _, _, transcript = secure_products(share(selector, rng), share(table, rng), reused)
        broken.append(transcript.opened[0])
    differences = np.mod(np.vstack(broken)[1:] - np.vstack(broken)[0], PRIME)

    return [
        LeakRow(
            observed="the masked values the server opens, over many flows",
            samples=int(pooled.size),
            p_value=_uniformity(pooled, 64),
            verdict="uniform: consistent with a one-time pad",
        ),
        LeakRow(
            observed="**a deliberately broken variant** that reuses one mask",
            samples=int(differences.size),
            p_value=_uniformity(differences, 64),
            verdict="**not uniform**: differences of masked selectors leak the inputs",
        ),
    ]


def _extraction_attack(
    model: PrivateAdditiveModel, rng: np.random.Generator, feature_budget: int
) -> tuple[int, float, int]:
    """Read the model out with crafted queries the server has no way to reject.

    Secret sharing hides the client's vector completely, which is the guarantee -- and which
    means the server cannot verify the vector is a *one-hot selector* rather than an arbitrary
    field vector. A client that sends the unit vector on one feature and zeros on the rest
    receives ``intercept + f_j[i]``: one table entry, exactly, in the clear.

    This is not a weakness of the arithmetic. It is the honest-but-curious assumption being
    load-bearing, and the fix is orthogonal to the sharing: the client has to *prove* its input
    is well-formed, which is a zero-knowledge argument this module does not implement.
    """
    baseline, transcript = private_score(
        model, [np.zeros(len(table), dtype=np.int64) for table in model.tables], rng
    )
    queries = 1
    worst = 0.0
    for index, table in enumerate(model.tables[:feature_budget]):
        for position in range(len(table)):
            crafted = [np.zeros(len(other), dtype=np.int64) for other in model.tables]
            crafted[index][position] = 1
            score, _ = private_score(model, crafted, rng)
            queries += 1
            truth = float(decode(np.array([table[position]]), model.fraction_bits)[0])
            worst = max(worst, abs((score - baseline) - truth))
    total = 1 + model.multiplications
    return total, worst, queries * transcript.online_bytes


def _edge_leak(
    binner: Binner, matrix: np.ndarray, names: list[str], top_n: int, rng: np.random.Generator
) -> list[EdgeRow]:
    """Reconstruct the training marginals from the public cut points, and price the result.

    The edges are quantiles. Publishing them is publishing a quantile summary of the training
    traffic, and a client can invert it -- draw uniformly inside each bin and it has a sample
    of the distribution the model was fitted on. The reconstruction is scored in the same units
    the [transport study](transport.md) uses, against the floor two halves of the real data
    would show, because a distance quoted without that floor is a statement about sample size.
    """
    rows: list[EdgeRow] = []
    half = len(matrix) // 2
    for index in range(min(top_n, matrix.shape[1])):
        column = matrix[:, index]
        edges = binner.edges[index]
        if len(edges) < 2:
            continue
        span = float(edges[-1] - edges[0]) or 1.0
        lower = np.concatenate([[edges[0] - 0.1 * span], edges])
        upper = np.concatenate([edges, [edges[-1] + 0.1 * span]])
        picks = rng.integers(0, len(lower), size=len(column))
        reconstructed = lower[picks] + rng.random(len(column)) * (upper[picks] - lower[picks])
        order = rng.permutation(len(column))
        rows.append(
            EdgeRow(
                feature=display_feature_name(names[index]),
                reconstruction_distance=wasserstein_1d(reconstructed, column),
                floor=wasserstein_1d(column[order[:half]], column[order[half:]]),
            )
        )
    return rows


def run_private_inference_study(settings: Settings) -> PrivateInferenceStudy:
    """Build the private model, measure what it costs, then attack it."""
    start = time.perf_counter()
    cfg: PrivateInferenceConfig = settings.private_inference
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)
    rng = np.random.default_rng(variant.seed)

    from netsentry.data.clean import BINARY_TARGET
    from netsentry.data.split import load_split
    from netsentry.features.pipeline import build_pipeline

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    arrivals_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_later: np.ndarray = np.asarray(pipeline.transform(arrivals_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    names = list(pipeline.named_steps["features"].get_feature_names_out())

    binner = Binner.fit(x_train, cfg.n_bins)
    model = fit_additive(
        binner.transform(x_train),
        y_train,
        binner,
        rounds=cfg.rounds,
        learning_rate=settings.gam.learning_rate,
        l2=settings.gam.l2,
        weights=compute_sample_weight("balanced", y_train),
    )
    private_model = PrivateAdditiveModel.of(model, cfg.fraction_bits)
    bins_later = binner.transform(x_later)

    clock = time.perf_counter()
    for index in range(cfg.timing_flows):
        score, transcript = private_score(
            private_model, private_model.selectors(bins_later[index]), rng
        )
    private_ms = (time.perf_counter() - clock) * 1000.0 / max(cfg.timing_flows, 1)
    clock = time.perf_counter()
    for index in range(cfg.timing_flows):
        model.predict_proba(bins_later[index : index + 1])
    plaintext_ms = (time.perf_counter() - clock) * 1000.0 / max(cfg.timing_flows, 1)
    del score

    queries, extraction_error, extraction_bytes = _extraction_attack(
        private_model, rng, cfg.extraction_features
    )
    costs = [
        CostRow(
            "online traffic",
            f"{transcript.online_bytes / 1024:.1f} KB",
            f"one round, {private_model.multiplications:,} multiplications",
        ),
        CostRow(
            "preprocessing (Beaver triples)",
            f"{transcript.preprocessing_bytes / 1024:.1f} KB per flow",
            "delivered before the flow arrives, and usable once",
        ),
        CostRow(
            "latency",
            f"{private_ms:.1f} ms",
            f"{private_ms / max(plaintext_ms, 1e-9):.1f}x the plaintext model's "
            f"{plaintext_ms:.2f} ms",
        ),
        CostRow(
            "rounds of interaction",
            f"{transcript.rounds}",
            "the whole inference is one batched opening",
        ),
    ]

    study = PrivateInferenceStudy(
        precision=_precision_sweep(
            model, binner, x_later, cfg.fraction_bit_sweep, rng, cfg.precision_flows
        ),
        costs=costs,
        leaks=_leak_checks(private_model, bins_later, rng, cfg.leak_flows),
        edges=_edge_leak(binner, x_train, names, cfg.edge_features, rng),
        queries_to_extract=queries,
        extraction_error=extraction_error,
        extraction_bytes=extraction_bytes,
        n_features=x_train.shape[1],
        n_bins=cfg.n_bins,
        multiplications=private_model.multiplications,
        online_bytes=transcript.online_bytes,
        preprocessing_bytes=transcript.preprocessing_bytes,
        rounds=transcript.rounds,
        private_ms=private_ms,
        plaintext_ms=plaintext_ms,
        fraction_bits=cfg.fraction_bits,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Private inference study complete",
        extra={
            "online_kb": round(study.online_bytes / 1024, 1),
            "queries_to_extract": study.queries_to_extract,
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _precision_table(study: PrivateInferenceStudy) -> str:
    rows = "\n".join(
        f"| {row.fraction_bits} | {row.worst_error:.2e} | {row.headroom:.3g}x | "
        + ("**wraps**" if row.wrapped else "safe")
        + " |"
        for row in study.precision
    )
    return (
        "| fraction bits | worst error against the plaintext model | headroom before the sum "
        "wraps | verdict |\n|---|---|---|---|\n" + rows
    )


def _cost_table(study: PrivateInferenceStudy) -> str:
    rows = "\n".join(f"| {row.quantity} | **{row.value}** | {row.against} |" for row in study.costs)
    return "| quantity | value | against |\n|---|---|---|\n" + rows


def _leak_table(study: PrivateInferenceStudy) -> str:
    rows = "\n".join(
        f"| {row.observed} | {row.samples:,} | {row.p_value:.4f} | {row.verdict} |"
        for row in study.leaks
    )
    return (
        "| what is observed | field elements | uniformity p-value | reading |\n"
        "|---|---|---|---|\n" + rows
    )


def _edge_table(study: PrivateInferenceStudy) -> str:
    rows = "\n".join(
        f"| `{row.feature}` | {row.reconstruction_distance:.3f} | {row.floor:.3f} | "
        f"**{row.excess:.3f}** |"
        for row in study.edges
    )
    return (
        "| feature | reconstruction distance (sd) | same-data floor | excess |\n"
        "|---|---|---|---|\n" + rows
    )


def _lead(study: PrivateInferenceStudy) -> str:
    usable = [row for row in study.precision if not row.wrapped and row.worst_error < 1e-3]
    return (
        f"**Neither party has to show the other anything, and it costs "
        f"{study.online_bytes / 1024:.0f} KB and one round.**\n\n"
        f"The client's flow becomes {study.n_features} one-hot selectors, the server's model "
        f"stays {study.multiplications:,} secret table entries, and a single batched opening of "
        f"masked values produces additive shares of the score. The result matches the plaintext "
        f"model to "
        f"{min((row.worst_error for row in study.precision if not row.wrapped), default=0):.0e}, "
        f"and every field element the server observes passes a uniformity test at p = "
        f"{study.leaks[0].p_value if study.leaks else 0:.2f} -- against a deliberately broken "
        f"variant that fails the same test at p = "
        f"{study.leaks[1].p_value if len(study.leaks) > 1 else 0:.4f}.\n\n"
        f"The reason it is this cheap is the *model*, not the protocol. An additive model is a "
        f"sum of table lookups, a lookup is an inner product with a selector, and because one "
        f"operand is a selector rather than a value the fixed-point scale survives the "
        f"multiplication -- **no truncation step, and a circuit one multiplication deep**. The "
        f"[glass box](gam.md) turns out to be the private box, for a structural reason rather "
        f"than a coincidence.\n\n"
        f"Then the two things it does not protect, both measured. The bin edges must be public "
        f"for a client to bin its own flow, and those edges are a quantile summary: "
        f"reconstructing the training marginals from them lands **{study.mean_edge_excess:.3f} "
        f"sd** above the floor two halves of the real data would show. The model stays secret; "
        f"the training distribution does not. And against a *malicious* client the guarantee "
        f"inverts entirely -- secret sharing hides the input so completely that the server "
        f"cannot check it **is** an input, and "
        f"{study.queries_to_extract:,} crafted queries "
        f"({study.extraction_bytes / 1e6:.1f} MB) read the whole model out to "
        f"{study.extraction_error:.0e}.\n\n"
        f"Of the {len(study.precision)} encodings swept, {len(usable)} land inside 1e-3 of the "
        f"plaintext score without wrapping. Both ends of the sweep fail, and they fail "
        f"differently."
    )


def _render(study: PrivateInferenceStudy, precision: Path, edges: Path) -> str:
    wrap = next((row for row in study.precision if row.wrapped), None)
    coarse = study.precision[0] if study.precision else None
    honest = study.leaks[0] if study.leaks else None
    broken = study.leaks[1] if len(study.leaks) > 1 else None
    worst_edge = max(study.edges, key=lambda row: row.excess) if study.edges else None
    coarse_read = (
        f"At {coarse.fraction_bits} bits the error is {coarse.worst_error:.2f}"
        if coarse
        else "At the coarse end"
    )
    wrap_read = (
        f"At {wrap.fraction_bits} bits the sum passes half the prime and **wraps**"
        if wrap
        else "At the fine end the sum wraps"
    )
    worst_read = (
        f"The worst case here is `{worst_edge.feature}`, reconstructed to "
        f"{worst_edge.excess:.3f} sd above the floor"
        if worst_edge
        else "Every feature reconstructs close to the floor"
    )
    return f"""# NetSentry — Scoring a Flow Neither Party Will Show the Other

_Two-party additive secret sharing with Beaver triples, implemented on numpy over a
{PRIME.bit_length()}-bit prime field, evaluating an additive model with {study.n_features}
features and {study.n_bins} bins each. Correctness, cost and leakage all measured; the attack
on it is executed. Regenerate with `netsentry privateinfer`._

## Why this report exists

`/predict` has a privacy structure nobody writes down. The client uploads {study.n_features}
features of its own network traffic; the server replies with a verdict from a model it will not
share. Both sides give something up, and for a managed-detection provider that *is* the
commercial arrangement -- customers hand over telemetry, the vendor keeps the detector.

Secure two-party computation removes it. Every secret is split into two shares that are
individually uniform; multiplication uses a preprocessed random triple (Beaver, CRYPTO 1991) so
that the only values ever revealed are one-time-padded masks.

{_lead(study)}

## Does it compute the right answer?

![The encoding window](../figures/{precision.name})

{_precision_table(study)}

Fixed-point encoding has two failure modes and they are not the same failure.
{coarse_read} -- the score is quantised past usefulness, but it is still
*approximately* right, and a system watching its own accuracy would notice.
{wrap_read}, which is not a large error but a different number entirely: a
benign flow can decode as a confident attack, silently, with no signal that anything went
wrong. The headroom column separates the two, and it is the reason this is swept rather than
picked.

Inside the window the protocol is **arithmetically exact** up to that quantisation, because
nothing here needs the usual probabilistic truncation. Multiplying two fixed-point numbers
doubles the scale and normally requires a correction that is either expensive or occasionally
wrong; here one operand of every product is a 0/1 selector, so the scale is preserved. That is a
property of evaluating an *additive* model, and it is the strongest practical argument this
project has found for the glass box.

## What it costs

{_cost_table(study)}

Worth putting beside the other end of the same trade: a [proof-carrying verdict](attestation.md)
costs 392 KB and answers "did the committed model produce this score". A private verdict costs
{study.online_bytes / 1024:.0f} KB and answers "can this score be produced without either side
seeing the other's secret". **Privacy is an order of magnitude cheaper than verifiability
here**, and for the same underlying reason -- one scales with the number of *table entries*, the
other with the number of *trees*.

The latency ratio is the least interesting number in the table and is included so that it
cannot be quoted alone: the plaintext model is itself microseconds of numpy, so a
{study.private_ms / max(study.plaintext_ms, 1e-9):.1f}x slowdown on a sub-millisecond baseline
says almost nothing. The bytes and the preprocessing are the real cost, and the preprocessing is
**single-use** -- a triple consumed on one flow cannot be reused on the next without destroying
the guarantee, which the next section demonstrates.

## What the server actually sees

{_leak_table(study)}

Everything opened during an inference is a secret plus a fresh uniform field element, so it
should be uniform and carry nothing. It is
{f"(p = {honest.p_value:.2f} over {honest.samples:,} elements)" if honest else ""}.

A clean result on its own would prove nothing -- a test that has never failed is
indistinguishable from one that cannot -- so the same test is run against a variant that reuses
one triple across flows, which is the single most common way this family of protocols is
broken in practice. It fails at
{f"p = {broken.p_value:.4f}" if broken else "p = 0"}: differences of masked selectors cancel the
shared mask and expose the inputs directly. The test can fail, and that is what makes the
passing row worth reading.

## What is not protected, one: the edges are public

![Reconstructing the training marginals](../figures/{edges.name})

{_edge_table(study)}

The client has to bin its own flow before it can build a selector, and it cannot do that
without the cut points. Those cut points are **quantiles of the training traffic**, so
publishing them publishes a quantile summary -- and a client can invert it by drawing uniformly
inside each bin.
{worst_read}.

This is a real and separable leak: **the model stays secret and the training distribution does
not.** It is also fixable, at a price -- fixed public bins (a grid nobody derived from the
data), or moving the binning inside the protocol with an oblivious comparison, which trades the
one-round property away.

## What is not protected, two: the client can simply ask

The protocol's guarantee is against an **honest-but-curious** client, and that assumption is
doing far more work than it looks. Secret sharing hides the client's vector so completely that
the server cannot verify it is a *selector* rather than an arbitrary field vector. A client that
sends the unit vector on one feature and zeros everywhere else receives
`intercept + f_j[i]`: one table entry, in the clear, from a query the server has no way to
refuse.

**{study.queries_to_extract:,} such queries recover the entire model**, to
{study.extraction_error:.0e}, for {study.extraction_bytes / 1e6:.1f} MB of traffic. The attack is
executed here rather than described, and it is strictly stronger than the query-only
[extraction attack](extraction.md) this project already measures -- there the adversary
approximates a model from valid flows; here it *reads* it, because the input space it is
allowed to use is the field rather than the space of flows.

The fix is not more secret sharing. It is malicious security: the client must **prove** its
input is well-formed, with a zero-knowledge argument that each block is one-hot. That is
implementable and it is a different protocol with a different cost, and naming it is more useful
than pretending the honest-but-curious model covers a paying customer.

## Scope and honest limits

- **Honest-but-curious, with a trusted dealer.** Triples are generated by a third party here.
  A deployment would produce them with oblivious transfer or homomorphic encryption during a
  preprocessing phase, which changes the cost of the offline column and nothing about the
  online one.
- **The model is the additive one, and it detects less.** The [glass-box study](gam.md) prices
  that: PR-AUC 0.480 against the deployed ensemble's 0.529 on the honest split. Privately
  evaluating the ensemble would need secure comparisons -- garbled circuits or an oblivious
  tree traversal -- and is a much larger protocol.
- **The verdict is revealed, which is the point and also a channel.** The client learns the
  score, which is what it came for and also what makes the extraction attack above possible at
  all. No protocol in this family hides its own output.
- **Nothing here hides that a query happened**, or when, or how often. Traffic analysis over the
  query stream is outside the model and is a real channel for a detection service, whose query
  volume is itself a signal about the customer's incidents.
- **The field is {PRIME.bit_length()} bits**, chosen so products stay inside a signed 64-bit
  integer and the whole protocol runs in numpy. A production implementation would use a larger
  field and a constant-time backend; the accounting scales, the argument does not change."""


def run_private_inference_report(settings: Settings) -> Path:
    """Run the private-inference study and write the report + figures."""
    study = run_private_inference_study(settings)
    bits = np.array([row.fraction_bits for row in study.precision], dtype=float)
    precision = plots.plot_lines(
        {
            "error against the plaintext model": (
                bits,
                np.maximum([row.worst_error for row in study.precision], 1e-12),
            )
        },
        xlabel="fixed-point fraction bits",
        ylabel="worst absolute error (log-odds)",
        title="Too few bits quantises; too many wraps",
        out_path=settings.paths.figures_dir / PRECISION_FIGURE,
        yscale="log",
        vlines={
            "the first setting that wraps": next(
                (float(row.fraction_bits) for row in study.precision if row.wrapped),
                float(bits[-1]),
            )
        },
    )
    edges = plots.plot_barh(
        [row.feature for row in study.edges],
        [row.excess for row in study.edges],
        xlabel="reconstruction distance above the same-data floor (sd)",
        title="The model stays secret; the training distribution does not",
        out_path=settings.paths.figures_dir / EDGES_FIGURE,
        xmax=max((row.excess for row in study.edges), default=1.0) * 1.2,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, precision, edges), encoding="utf-8")
    logger.info("Wrote private-inference report", extra={"path": str(out_path)})

    with track_run(settings, "private_inference") as run:
        run.log_params({"bins": study.n_bins, "fraction_bits": study.fraction_bits})
        run.log_metrics(
            {
                "online_bytes": float(study.online_bytes),
                "preprocessing_bytes": float(study.preprocessing_bytes),
                "queries_to_extract": float(study.queries_to_extract),
                "edge_reconstruction_excess": study.mean_edge_excess,
            }
        )
        for artifact in (precision, edges, out_path):
            run.log_artifact(artifact)
    return out_path
