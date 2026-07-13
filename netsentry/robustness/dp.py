"""Differentially-private training — the formal privacy control, priced.

The [membership-inference audit](membership.py) *measures* how much the model
memorises its training data; it ends by naming the mitigation with a formal
guarantee: **differentially-private training**, which buys an (ε, δ) bound at a
measured detection cost. This module is that named next study. It has three parts:

- **A Rényi differential-privacy (RDP) accountant** for the subsampled Gaussian
  mechanism (Abadi et al. 2016; Mironov 2017), implemented in **pure stdlib**
  (``math`` only — no scipy) at integer orders. Integer-order accounting is a
  *sound upper bound* on ε (each order gives a valid RDP->(ε, δ) conversion, and
  scanning more orders only tightens it); fractional orders would sharpen it
  marginally. Auditable, dependency-free, and unit-tested against the closed
  forms — the same posture as the from-scratch pcap reader.
- **A DP-SGD logistic classifier** (Abadi et al. 2016): per-example gradient
  clipping to an L2 norm ``l2_clip`` bounds any one flow's influence, then
  Gaussian noise ``N(0, (noise_multiplier * l2_clip)^2)`` is added to the summed
  gradient. The privacy cost is a function only of the noise multiplier, the
  minibatch sampling rate, and the number of steps — not of the data — so the
  same ``fit`` yields a certified ε for *any* dataset.
- **The frontier study** (``run_dp_report``): a non-private reference and DP
  models at a sweep of noise multipliers, each priced on the same axis — the ε it
  spends, the detection it keeps (PR-AUC + TPR@FPR), and the membership leakage it
  closes (the same Yeom attack the membership audit runs). That is the honest arc
  one axis further: the membership study *measured* the leak, DP *fixes* it, and
  this *re-measures* both the residual leak and what the guarantee costs.

A linear model is the deliberate choice: DP-SGD is standard and well-understood on
linear/deep models (DP for tree ensembles is a different, messier mechanism), and
on this data the leaderboard already shows logistic regression is competitive on
the honest split — so the utility ceiling is real, not a straw man.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from netsentry.log import get_logger

logger = get_logger(__name__)

# Integer RDP orders scanned by default. A wide ladder so the ε-minimising order is
# available across the (q, sigma, steps) regimes the study visits; integer-only keeps
# the accountant exact and pure-stdlib (a sound upper bound on ε).
DEFAULT_ORDERS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256)

_NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# Rényi differential-privacy accountant for the subsampled Gaussian mechanism.
# Pure stdlib (math): log-space arithmetic, integer orders only.
# --------------------------------------------------------------------------- #
def _log_add(log_x: float, log_y: float) -> float:
    """log(exp(log_x) + exp(log_y)), stable for very negative logs."""
    lo, hi = min(log_x, log_y), max(log_x, log_y)
    if lo == _NEG_INF:
        return hi
    return math.log1p(math.exp(lo - hi)) + hi


def _log_comb(n: int, k: int) -> float:
    """log of the binomial coefficient C(n, k) via lgamma (exact, no overflow)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_a_int(q: float, sigma: float, alpha: int) -> float:
    """log A_alpha for the subsampled Gaussian at an integer order (Mironov 2017).

    A_alpha = sum_{i=0..alpha} C(alpha, i) (1-q)^{alpha-i} q^i exp((i^2-i)/(2sigma^2));
    RDP(alpha) = log A_alpha / (alpha - 1). Computed in log space so the largest
    term never overflows.
    """
    log_a = _NEG_INF
    for i in range(alpha + 1):
        log_term = (
            _log_comb(alpha, i)
            + i * math.log(q)
            + (alpha - i) * math.log1p(-q)
            + (i * i - i) / (2.0 * sigma * sigma)
        )
        log_a = _log_add(log_a, log_term)
    return log_a


def rdp_of_step(q: float, noise_multiplier: float, alpha: int) -> float:
    """RDP at order ``alpha`` of one subsampled-Gaussian step (sampling rate ``q``).

    ``q == 1`` (no subsampling) is the Gaussian-mechanism closed form alpha/(2sigma^2);
    ``q == 0`` spends nothing. Otherwise the tight integer-order sum is used.
    """
    if noise_multiplier <= 0.0:
        return math.inf
    if q <= 0.0:
        return 0.0
    sigma = noise_multiplier
    if q >= 1.0:
        return alpha / (2.0 * sigma * sigma)
    return _log_a_int(q, sigma, alpha) / (alpha - 1)


def compute_rdp(
    q: float, noise_multiplier: float, steps: int, orders: tuple[int, ...] = DEFAULT_ORDERS
) -> dict[int, float]:
    """Total RDP after ``steps`` composed subsampled-Gaussian steps, per order.

    RDP composes by addition, so ``steps`` steps cost ``steps x`` the per-step RDP
    at every order (the composition theorem — the reason RDP is the convenient
    accountant for iterated DP-SGD).
    """
    return {a: steps * rdp_of_step(q, noise_multiplier, a) for a in orders}


def rdp_to_epsilon(rdp: dict[int, float], target_delta: float) -> tuple[float, int]:
    """Convert per-order RDP to the tightest (ε, δ) DP guarantee; return (ε, order).

    Uses the sharpened RDP->DP conversion (Canonne-Kamath-Steinke 2020, as in
    TF-Privacy): per order a, eps_a = r_a + log((a-1)/a) - (log(delta) + log(a))/(a-1),
    and the reported ε is the minimum over the scanned orders.
    """
    if not 0.0 < target_delta < 1.0:
        raise ValueError("target_delta must lie in (0, 1).")
    best_eps, best_order = math.inf, 0
    for alpha, r in rdp.items():
        if alpha <= 1 or not math.isfinite(r):
            continue
        eps = (
            r
            + math.log((alpha - 1) / alpha)
            - (math.log(target_delta) + math.log(alpha)) / (alpha - 1)
        )
        if eps < best_eps:
            best_eps, best_order = eps, alpha
    return best_eps, best_order


def dp_sgd_epsilon(
    *,
    sampling_rate: float,
    noise_multiplier: float,
    steps: int,
    target_delta: float,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
) -> float:
    """End-to-end ε for a DP-SGD run: (q, sigma, steps) -> ε at ``target_delta``."""
    if noise_multiplier <= 0.0:
        return math.inf
    rdp = compute_rdp(sampling_rate, noise_multiplier, steps, orders)
    eps, _ = rdp_to_epsilon(rdp, target_delta)
    return eps


# --------------------------------------------------------------------------- #
# DP-SGD logistic-regression classifier (binary).
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class DPClassifier:
    """A binary logistic classifier trained with DP-SGD (Abadi et al. 2016).

    The privacy guarantee is a property of the *training procedure* (clip norm,
    noise multiplier, sampling rate, step count), so ``epsilon`` returns a certified
    (ε, δ) that holds for any dataset. Setting ``noise_multiplier == 0`` disables
    both clipping and noise, giving an ordinary (ε = inf) SGD reference on the same
    model family — the non-private end of the frontier.

    Exposes ``predict_proba`` / ``classes_`` so it drops straight into the existing
    metric and membership-attack helpers.
    """

    noise_multiplier: float = 1.0
    l2_clip: float = 1.0
    epochs: int = 60
    lr: float = 0.5
    batch_size: int = 256
    l2_reg: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        self.classes_: np.ndarray = np.array([0, 1])
        self._w: np.ndarray = np.zeros(0)
        self._mean: np.ndarray = np.zeros(0)
        self._std: np.ndarray = np.zeros(0)
        self.sampling_rate_: float = 0.0
        self.steps_: int = 0

    @property
    def private(self) -> bool:
        """Whether this run actually spends a finite privacy budget."""
        return self.noise_multiplier > 0.0

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        scaled: np.ndarray = (np.asarray(x, dtype=float) - self._mean) / self._std
        return scaled

    def fit(self, x: np.ndarray, y: np.ndarray) -> DPClassifier:
        """DP-SGD fit. Per-example gradients are clipped, summed, and noised."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y).astype(float)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0) + 1e-8
        xs = self._standardize(x)
        n, d = xs.shape
        # Augment with a bias column (unregularised, but clipped/noised like any coord).
        xb = np.hstack([xs, np.ones((n, 1))])
        rng = np.random.default_rng(self.seed)
        w = np.zeros(d + 1)

        batch = min(self.batch_size, n)
        self.sampling_rate_ = batch / n
        steps_per_epoch = max(1, n // batch)
        self.steps_ = self.epochs * steps_per_epoch
        clip = self.l2_clip

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for s in range(steps_per_epoch):
                idx = order[s * batch : (s + 1) * batch]
                xb_i = xb[idx]
                # Per-example logistic gradient: (sigmoid(x*w) - y) * x.
                err = _sigmoid(xb_i @ w) - y[idx]
                grads = err[:, None] * xb_i  # (batch, d+1)
                if self.private:
                    norms = np.linalg.norm(grads, axis=1)
                    scale = np.minimum(1.0, clip / (norms + 1e-12))
                    grads = grads * scale[:, None]
                    summed = grads.sum(axis=0)
                    noise = rng.normal(0.0, self.noise_multiplier * clip, size=summed.shape)
                    grad = (summed + noise) / len(idx)
                else:
                    grad = grads.mean(axis=0)
                # L2 regularisation on the weights (not the bias) — a private prior.
                reg = self.l2_reg * w
                reg[-1] = 0.0
                w = w - self.lr * (grad + reg)

        self._w = w
        logger.info(
            "DP-SGD fit complete",
            extra={
                "noise_multiplier": self.noise_multiplier,
                "steps": self.steps_,
                "sampling_rate": round(self.sampling_rate_, 4),
                "private": self.private,
            },
        )
        return self

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        """P(attack) per row."""
        xs = self._standardize(x)
        xb = np.hstack([xs, np.ones((len(xs), 1))])
        return _sigmoid(xb @ self._w)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """[P(benign), P(attack)] per row — the sklearn-style proba matrix."""
        p1 = self.decision_scores(x)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.decision_scores(x) >= 0.5).astype(int)

    def epsilon(self, target_delta: float, orders: tuple[int, ...] = DEFAULT_ORDERS) -> float:
        """The certified ε at ``target_delta`` this fit spent (inf if non-private)."""
        return dp_sgd_epsilon(
            sampling_rate=self.sampling_rate_,
            noise_multiplier=self.noise_multiplier,
            steps=self.steps_,
            target_delta=target_delta,
            orders=orders,
        )
