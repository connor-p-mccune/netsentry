"""A streaming decision tree with a statistical split guarantee, and the window that resets it.

Everything else in this project learns in batches: fit on a training split, evaluate, redeploy.
That is the right default, and it has a cost the batch framing hides — between two retrains the
model is frozen, and every flow that arrives in the gap is scored by a model that has already
seen its last new example. The streaming literature answers with learners that update per
example in bounded memory, and the founding one is the **Hoeffding tree** (Domingos & Hulten,
KDD 2000).

Its idea is small and genuinely clever. A batch tree looks at every example to choose a split.
A streaming tree cannot, so it asks a different question: *how many examples do I need before I
can be confident the best split really is the best?* The Hoeffding bound answers it without
assuming anything about the distribution — after ``n`` observations, a mean estimated from a
variable of range ``R`` is within

    eps = sqrt(R^2 ln(1/delta) / (2n))

of its true value with probability ``1 - delta``. So if the best candidate split beats the
runner-up by more than ``eps`` on the split criterion, the ranking will not change with more
data, and the split can be taken *now* and never revisited. The tree grows incrementally, one
pass, no stored dataset, and the resulting model is provably close to the one a batch learner
would have built from the same stream.

Two implementations live here:

- ``HoeffdingTree`` — the VFDT itself. Numeric features are summarised per class by Gaussian
  sufficient statistics (the standard MOA observer), candidate thresholds are proposed across
  each feature's observed range, and split decisions use information gain plus the bound above,
  with a tie-break for the case where two splits really are equally good. Leaves predict either
  by majority class or by a Gaussian naive-Bayes model fitted from the same statistics -- which
  costs nothing extra, because the statistics were already being kept for the split test.
- ``ADWIN`` — adaptive windowing (Bifet & Gavalda, SDM 2007), which keeps a window of recent
  values compressed into exponential-histogram buckets and drops the old half whenever *any*
  split of the window shows a statistically significant difference in mean. It needs no window
  size, no threshold, and no change-magnitude assumption -- only a confidence level -- and it
  is what lets the tree notice that its own error rate has moved and start over.

Both are pure NumPy, seeded, and tested against their own guarantees rather than against a
reference implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

_EPS = 1e-12


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a class-count vector, in bits (zero for a pure node)."""
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = np.asarray(counts, dtype=float) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Entropy reduction of a binary split, weighted by the mass each side receives."""
    total = float(np.sum(parent))
    if total <= 0:
        return 0.0
    n_left, n_right = float(np.sum(left)), float(np.sum(right))
    return entropy(parent) - (n_left / total) * entropy(left) - (n_right / total) * entropy(right)


def hoeffding_bound(value_range: float, delta: float, n: int) -> float:
    """``sqrt(R^2 ln(1/delta) / (2n))`` -- how far a mean of ``n`` observations can be off.

    Distribution-free, which is the whole point: a streaming learner cannot afford an assumption
    about the data it has not seen yet. For information gain the range is ``log2(#classes)``.
    """
    if n <= 0:
        return float("inf")
    return float(math.sqrt((value_range**2) * math.log(1.0 / delta) / (2.0 * n)))


@dataclass
class _Leaf:
    """Sufficient statistics for one leaf: enough to split it, and to predict from it.

    The per-class Gaussian summary is the memory trick that makes this bounded: whatever the
    stream's length, a leaf costs ``O(features x classes)`` numbers, never ``O(examples)``.
    """

    n_classes: int
    n_features: int
    counts: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)
    squares: np.ndarray = field(init=False)
    minimum: np.ndarray = field(init=False)
    maximum: np.ndarray = field(init=False)
    seen_since_attempt: int = 0

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.n_classes, dtype=float)
        self.sums = np.zeros((self.n_features, self.n_classes), dtype=float)
        self.squares = np.zeros((self.n_features, self.n_classes), dtype=float)
        self.minimum = np.full(self.n_features, np.inf)
        self.maximum = np.full(self.n_features, -np.inf)

    @property
    def total(self) -> float:
        return float(np.sum(self.counts))

    def observe(self, x: np.ndarray, label: int, weight: float = 1.0) -> None:
        """Fold one example into the leaf's statistics."""
        self.counts[label] += weight
        self.sums[:, label] += weight * x
        self.squares[:, label] += weight * x * x
        np.minimum(self.minimum, x, out=self.minimum)
        np.maximum(self.maximum, x, out=self.maximum)
        self.seen_since_attempt += 1

    def means(self) -> np.ndarray:
        """Per-feature, per-class means."""
        denom = np.maximum(self.counts, _EPS)
        mean: np.ndarray = self.sums / denom
        return mean

    def variances(self) -> np.ndarray:
        """Per-feature, per-class variances, floored so a constant feature stays usable."""
        denom = np.maximum(self.counts, _EPS)
        mean = self.sums / denom
        var = self.squares / denom - mean * mean
        floored: np.ndarray = np.maximum(var, 1e-9)
        return floored

    def split_counts(self, feature: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Estimated class counts either side of a threshold, from the Gaussian summaries.

        ``left_c = n_c * Phi((t - mu_c) / sigma_c)``: the fraction of class ``c``'s mass the
        fitted normal places below the threshold. It is an approximation -- the true conditional
        is not Gaussian -- and it is the standard one, because the alternative is storing the
        values themselves and giving up the bounded-memory property that motivates the tree.
        """
        mean = self.means()[feature]
        sigma = np.sqrt(self.variances()[feature])
        left = self.counts * ndtr((threshold - mean) / sigma)
        return left, self.counts - left

    def candidate_thresholds(self, feature: int, n_thresholds: int) -> np.ndarray:
        """Evenly spaced interior points of the feature's observed range."""
        low, high = float(self.minimum[feature]), float(self.maximum[feature])
        if not np.isfinite(low) or not np.isfinite(high) or high - low <= _EPS:
            return np.empty(0)
        return np.linspace(low, high, n_thresholds + 2)[1:-1]


@dataclass
class _Node:
    """A tree node: a leaf holds statistics, an internal node holds a test and two children."""

    leaf: _Leaf | None
    feature: int = -1
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None


@dataclass
class SplitCandidate:
    """The best split found for one feature, and what it would buy."""

    feature: int
    threshold: float
    gain: float


class HoeffdingTree:
    """A very fast decision tree (VFDT) for streams: one pass, bounded memory, per-example update.

    ``grace_period`` is how many examples a leaf collects between split attempts (the test is the
    expensive part, so it is amortised), ``delta`` the confidence of the Hoeffding bound, and
    ``tie_threshold`` the escape hatch for the case the bound cannot resolve: when two candidate
    splits are genuinely equally good, their difference never exceeds ``eps``, so the tree would
    wait forever. Splitting once ``eps`` itself falls below the tie threshold is Domingos and
    Hulten's answer, and it is why a VFDT terminates on real data.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        *,
        grace_period: int = 200,
        delta: float = 1e-6,
        tie_threshold: float = 0.05,
        n_thresholds: int = 10,
        max_depth: int = 12,
        min_leaf_samples: float = 20.0,
        leaf_prediction: str = "nb",
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.grace_period = grace_period
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.n_thresholds = n_thresholds
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.leaf_prediction = leaf_prediction
        self.n_seen = 0
        self.n_splits = 0
        self.root = _Node(leaf=_Leaf(n_classes, n_features))

    # -- structure ---------------------------------------------------------------------

    def _route(self, x: np.ndarray) -> tuple[_Node, int]:
        """Walk to the leaf that owns this example, returning it and its depth."""
        node = self.root
        depth = 0
        while not node.is_leaf:
            assert node.left is not None and node.right is not None
            node = node.left if x[node.feature] <= node.threshold else node.right
            depth += 1
        return node, depth

    def n_nodes(self) -> int:
        """Total nodes -- the memory the model occupies, which a stream learner must bound."""
        return 2 * self.n_splits + 1

    def n_leaves(self) -> int:
        return self.n_splits + 1

    def memory_bytes(self) -> int:
        """Approximate footprint: each leaf carries ``2 x features x classes`` floats + bounds."""
        per_leaf = 8 * (2 * self.n_features * self.n_classes + 2 * self.n_features + self.n_classes)
        return self.n_leaves() * per_leaf

    # -- learning ----------------------------------------------------------------------

    def best_splits(self, leaf: _Leaf) -> list[SplitCandidate]:
        """The best candidate split per feature, ranked by information gain."""
        candidates: list[SplitCandidate] = []
        for feature in range(self.n_features):
            thresholds = leaf.candidate_thresholds(feature, self.n_thresholds)
            best: SplitCandidate | None = None
            for threshold in thresholds:
                left, right = leaf.split_counts(feature, float(threshold))
                if min(left.sum(), right.sum()) < self.min_leaf_samples:
                    continue
                gain = information_gain(leaf.counts, left, right)
                if best is None or gain > best.gain:
                    best = SplitCandidate(feature, float(threshold), gain)
            if best is not None:
                candidates.append(best)
        candidates.sort(key=lambda c: c.gain, reverse=True)
        return candidates

    def _attempt_split(self, node: _Node, depth: int) -> None:
        """Split the leaf if the Hoeffding bound says the ranking of candidates is settled."""
        leaf = node.leaf
        assert leaf is not None
        leaf.seen_since_attempt = 0
        if depth >= self.max_depth or np.count_nonzero(leaf.counts) < 2:
            return
        candidates = self.best_splits(leaf)
        if not candidates:
            return
        best = candidates[0]
        runner_up = candidates[1].gain if len(candidates) > 1 else 0.0
        eps = hoeffding_bound(math.log2(max(self.n_classes, 2)), self.delta, int(leaf.total))
        if best.gain <= 0:
            return
        if (best.gain - runner_up) <= eps and eps >= self.tie_threshold:
            return  # not yet resolved, and not a tie either: collect more examples

        left_counts, right_counts = leaf.split_counts(best.feature, best.threshold)
        node.feature, node.threshold = best.feature, best.threshold
        node.left = _Node(leaf=_Leaf(self.n_classes, self.n_features))
        node.right = _Node(leaf=_Leaf(self.n_classes, self.n_features))
        # Seed the children with the split's own estimate of their class distribution. A VFDT
        # that starts its children empty predicts uniformly for the next `grace_period`
        # examples, which on imbalanced traffic is a real and avoidable hole in the score.
        assert node.left.leaf is not None and node.right.leaf is not None
        node.left.leaf.counts = np.maximum(left_counts, 0.0)
        node.right.leaf.counts = np.maximum(right_counts, 0.0)
        node.leaf = None
        self.n_splits += 1

    def learn_one(self, x: np.ndarray, label: int, weight: float = 1.0) -> None:
        """Fold one example into the tree, splitting its leaf when the bound allows."""
        node, depth = self._route(x)
        assert node.leaf is not None
        node.leaf.observe(x, int(label), weight)
        self.n_seen += 1
        if node.leaf.seen_since_attempt >= self.grace_period:
            self._attempt_split(node, depth)

    def learn_many(self, X: np.ndarray, y: np.ndarray) -> HoeffdingTree:
        """Fold a batch in, example by example -- the order is part of the algorithm."""
        for i in range(len(X)):
            self.learn_one(X[i], int(y[i]))
        return self

    # -- prediction --------------------------------------------------------------------

    def _leaf_proba(self, leaf: _Leaf, x: np.ndarray) -> np.ndarray:
        """Class posterior at a leaf: Laplace-smoothed frequencies, or naive Bayes.

        The naive-Bayes option costs nothing to fit -- the Gaussian statistics are already there
        for the split test -- and it is what makes a young leaf useful: majority voting throws
        away every feature value the example carries, so a leaf that has just been created can
        only repeat its parent's prior until the next split.
        """
        prior = (leaf.counts + 1.0) / (leaf.total + self.n_classes)
        if self.leaf_prediction != "nb" or leaf.total < 2:
            return prior
        mean = leaf.means()
        var = leaf.variances()
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((x[:, None] - mean) ** 2) / var, axis=0
        )
        log_post = np.log(prior) + log_likelihood
        log_post -= log_post.max()
        posterior = np.exp(log_post)
        total = float(posterior.sum())
        return posterior / total if total > 0 else prior

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Class posterior for one example."""
        node, _ = self._route(x)
        assert node.leaf is not None
        return self._leaf_proba(node.leaf, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class posteriors for a batch (a convenience -- the tree itself is per-example)."""
        return np.vstack([self.predict_proba_one(X[i]) for i in range(len(X))])

    def score_one(self, x: np.ndarray, positive: int = 1) -> float:
        """``P(attack | x)`` from the leaf -- the score the operating threshold is applied to."""
        return float(self.predict_proba_one(x)[positive])


class ADWIN:
    """Adaptive windowing: a drift detector with no window size and no magnitude assumption.

    ADWIN keeps a window of the values it has been shown and asks, after each one, whether
    *any* way of cutting that window in two produces a statistically significant difference in
    mean. If one does, the older part is discarded: the window is exactly as long as the data
    has been stationary, which is the property that makes it usable as a learner's memory
    controller rather than merely as an alarm.

    Storing the window would defeat the purpose, so it is compressed into an exponential
    histogram -- buckets of size 1, 2, 4, ... with at most ``max_buckets`` per size -- giving
    ``O(log n)`` memory and ``O(log n)`` work per update while keeping the cut test accurate to
    within the granularity of the largest bucket.
    """

    def __init__(
        self,
        delta: float = 0.002,
        *,
        max_buckets: int = 5,
        min_window_length: int = 5,
        min_clock: int = 32,
    ) -> None:
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window_length = min_window_length
        self.min_clock = min_clock
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        self._clock = 0
        self._detections = 0
        # Each level holds buckets of the same size; a bucket is (sum, variance).
        self._levels: list[list[tuple[float, float]]] = [[]]

    @property
    def estimation(self) -> float:
        """Mean over the current window -- the learner's current view of its own error rate."""
        return self.total / self.width if self.width else 0.0

    @property
    def detections(self) -> int:
        return self._detections

    def n_buckets(self) -> int:
        """Buckets held: the memory, which grows logarithmically rather than linearly."""
        return sum(len(level) for level in self._levels)

    def _insert(self, value: float) -> None:
        if self.width > 0:
            mean = self.total / self.width
            self.variance += (value - mean) * (value - (self.total + value) / (self.width + 1))
        self.total += value
        self.width += 1
        self._levels[0].insert(0, (value, 0.0))
        self._compress()

    def _compress(self) -> None:
        """Merge overflowing buckets upward, doubling their size -- the exponential histogram."""
        level = 0
        while level < len(self._levels):
            if len(self._levels[level]) <= self.max_buckets:
                break
            oldest = self._levels[level][-2:]
            self._levels[level] = self._levels[level][:-2]
            (sum_a, var_a), (sum_b, var_b) = oldest
            size = float(2**level)
            mean_a, mean_b = sum_a / size, sum_b / size
            merged_var = var_a + var_b + size * size / (2.0 * size) * (mean_a - mean_b) ** 2
            if level + 1 == len(self._levels):
                self._levels.append([])
            self._levels[level + 1].insert(0, (sum_a + sum_b, merged_var))
            level += 1

    def _cut_threshold(self, n0: int, n1: int) -> float:
        """The significance threshold for one candidate cut (Bifet & Gavalda, eq. for eps_cut).

        The ``2 ln(width)`` inflation is a union bound over the cut points being tested: without
        it the detector would fire simply because it is asking the same question many times, and
        the promised false-alarm rate would not hold.
        """
        m = 1.0 / max(n0 - self.min_window_length + 1, 1) + 1.0 / max(
            n1 - self.min_window_length + 1, 1
        )
        dd = math.log(2.0 * math.log(max(self.width, 2)) / self.delta)
        v = self.variance / max(self.width, 1)
        return math.sqrt(2.0 * m * v * dd) + 2.0 / 3.0 * dd * m

    def update(self, value: float) -> bool:
        """Show the detector one value; returns ``True`` when the window has just been cut."""
        self._insert(value)
        self._clock += 1
        if self._clock % self.min_clock != 0 or self.width < 2 * self.min_window_length:
            return False

        changed = False
        shrink = True
        while shrink:
            shrink = False
            n0, sum0 = 0, 0.0
            n1, sum1 = self.width, self.total
            # Walk cut points from the oldest end: level order is oldest-last within a level,
            # and levels are ordered smallest-first, so the outer loop goes newest to oldest.
            for level in range(len(self._levels) - 1, -1, -1):
                size = float(2**level)
                for index in range(len(self._levels[level]) - 1, -1, -1):
                    bucket_sum, _ = self._levels[level][index]
                    n0 += int(size)
                    n1 -= int(size)
                    sum0 += bucket_sum
                    sum1 -= bucket_sum
                    if n1 <= 0:
                        break
                    if n0 < self.min_window_length or n1 < self.min_window_length:
                        continue
                    if abs(sum0 / n0 - sum1 / n1) > self._cut_threshold(n0, n1):
                        self._drop_oldest()
                        changed = True
                        shrink = self.width >= 2 * self.min_window_length
                        break
                if shrink:
                    break
        if changed:
            self._detections += 1
        return changed

    def _drop_oldest(self) -> None:
        """Discard the oldest bucket: the window keeps only the stationary tail."""
        for level in range(len(self._levels) - 1, -1, -1):
            if not self._levels[level]:
                continue
            bucket_sum, bucket_var = self._levels[level].pop()
            size = float(2**level)
            mean_dropped = bucket_sum / size
            remaining = max(self.width - size, 1.0)
            mean_rest = (self.total - bucket_sum) / remaining
            self.variance = max(
                self.variance
                - bucket_var
                - size * remaining / (size + remaining) * (mean_dropped - mean_rest) ** 2,
                0.0,
            )
            self.total -= bucket_sum
            self.width -= int(size)
            return

    def reset(self) -> None:
        """Forget everything -- used when the learner it monitors is itself rebuilt."""
        self.width = 0
        self.total = 0.0
        self.variance = 0.0
        self._clock = 0
        self._levels = [[]]
