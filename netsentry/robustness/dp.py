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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config import Settings

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


# --------------------------------------------------------------------------- #
# The privacy-utility-leakage frontier study.
# --------------------------------------------------------------------------- #
REPORT_NAME = "dp.md"


@dataclass
class DPPoint:
    """One model on the frontier: its privacy cost, detection, and residual leak."""

    label: str
    noise_multiplier: float
    epsilon: float  # inf for the non-private reference
    pr_auc: float
    tpr_at_fpr: float
    fpr_budget: float
    membership_auc: float  # Yeom threshold attack (0.5 == no leakage)
    membership_advantage: float

    @property
    def private(self) -> bool:
        return math.isfinite(self.epsilon)


@dataclass
class _AttackPools:
    """The member / non-member pools the Yeom attack is scored on."""

    x_mem: np.ndarray
    y_mem: np.ndarray
    x_non: np.ndarray
    y_non: np.ndarray


def _fit_and_score(
    settings: Settings,
    noise_multiplier: float,
    x_mem: np.ndarray,
    y_mem: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack: _AttackPools,
) -> DPPoint:
    """Train one (non-)private model and measure privacy, utility, and leakage."""
    from netsentry.evaluation.metrics import operating_point
    from netsentry.robustness.membership import attack_scores, true_class_probability

    cfg = settings.dp
    clf = DPClassifier(
        noise_multiplier=noise_multiplier,
        l2_clip=cfg.l2_clip,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        l2_reg=cfg.l2_reg,
        seed=settings.seed,
    ).fit(x_mem, y_mem)

    scores_val = clf.decision_scores(x_val)
    scores_test = clf.decision_scores(x_test)
    pr_auc = float(average_precision_score(y_test, scores_test))
    op = operating_point(
        y_val,
        scores_val,
        y_test,
        scores_test,
        cfg.primary_fpr,
        settings.thresholds.assumed_flows_per_day,
    )

    # Yeom membership attack: members are over-confident on their true class.
    classes = clf.classes_
    s_mem = true_class_probability(clf.predict_proba(attack.x_mem), classes, attack.y_mem)
    s_non = true_class_probability(clf.predict_proba(attack.x_non), classes, attack.y_non)
    is_member = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    scores = np.concatenate([s_mem, s_non])
    auc, adv, _, _, _ = attack_scores(is_member, scores, cfg.attack_fpr)

    eps = clf.epsilon(cfg.delta)
    label = "non-private" if not clf.private else f"sigma={noise_multiplier:g}"
    point = DPPoint(
        label=label,
        noise_multiplier=noise_multiplier,
        epsilon=eps,
        pr_auc=pr_auc,
        tpr_at_fpr=float(op["tpr"]),
        fpr_budget=cfg.primary_fpr,
        membership_auc=auc,
        membership_advantage=adv,
    )
    logger.info(
        "DP frontier point",
        extra={
            "label": label,
            "epsilon": None if math.isinf(eps) else round(eps, 3),
            "pr_auc": round(pr_auc, 4),
            "membership_auc": round(auc, 4),
        },
    )
    return point


def run_dp(settings: Settings) -> list[DPPoint]:
    """Train the non-private reference and DP models; price the frontier."""
    from netsentry.data.clean import BINARY_TARGET
    from netsentry.data.split import load_split
    from netsentry.features.pipeline import build_pipeline
    from netsentry.seed import seed_everything

    cfg = settings.dp
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"  # the exchangeable split the leakage measure needs
    seed_everything(variant.seed)

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")

    pipeline = build_pipeline(variant)
    pipeline.fit(train)  # unsupervised fit on the full train split (leakage-safe)

    rng = np.random.default_rng(variant.seed)
    train = train.reset_index(drop=True)
    n_mem = min(cfg.target_train_rows, len(train))
    members = train.iloc[rng.choice(len(train), size=n_mem, replace=False)]

    x_mem = np.asarray(pipeline.transform(members))
    y_mem = members[BINARY_TARGET].to_numpy().astype(int)
    x_val = np.asarray(pipeline.transform(val))
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    x_test = np.asarray(pipeline.transform(test))
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    # Attack pools: a capped slice of members vs fresh (non-member) test rows.
    non = test.sample(n=min(cfg.eval_rows, len(test)), random_state=variant.seed)
    mem_eval = members.sample(n=min(cfg.eval_rows, len(members)), random_state=variant.seed)
    attack = _AttackPools(
        x_mem=np.asarray(pipeline.transform(mem_eval)),
        y_mem=mem_eval[BINARY_TARGET].to_numpy().astype(int),
        x_non=np.asarray(pipeline.transform(non)),
        y_non=non[BINARY_TARGET].to_numpy().astype(int),
    )

    points: list[DPPoint] = []
    for sigma in cfg.noise_multipliers:
        points.append(
            _fit_and_score(variant, sigma, x_mem, y_mem, x_val, y_val, x_test, y_test, attack)
        )
    return points


def _eps_str(eps: float) -> str:
    return "inf (non-private)" if math.isinf(eps) else f"{eps:.2f}"


def _table(points: list[DPPoint]) -> str:
    budget = f"{points[0].fpr_budget:.1%}"
    rows = [
        f"| model | noise (sigma) | epsilon (delta fixed) | PR-AUC | TPR @ {budget} FPR "
        "| membership AUC | advantage |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in points:
        rows.append(
            f"| {p.label} | {p.noise_multiplier:g} | {_eps_str(p.epsilon)} | {p.pr_auc:.3f} "
            f"| {p.tpr_at_fpr * 100:.1f}% | {p.membership_auc:.3f} | {p.membership_advantage:.3f} |"
        )
    return "\n".join(rows)


def _read(points: list[DPPoint]) -> str:
    """Sign-aware prose so the narrative tracks whatever the numbers actually do."""
    ref = points[0]
    private = [p for p in points if p.private]
    if not private:
        return "No private models were configured; add noise multipliers > 0 to price the frontier."
    tightest = min(private, key=lambda p: p.epsilon)  # smallest epsilon = strongest privacy
    util_cost = ref.pr_auc - tightest.pr_auc
    tpr_cost = ref.tpr_at_fpr - tightest.tpr_at_fpr
    leak_drop = ref.membership_auc - tightest.membership_auc

    head = (
        f"The non-private reference detects at PR-AUC **{ref.pr_auc:.3f}** "
        f"(TPR@{ref.fpr_budget:.1%}FPR {ref.tpr_at_fpr * 100:.1f}%) and its membership leakage "
        f"sits at AUC {ref.membership_auc:.3f}. Tightening to a formal **epsilon = "
        f"{tightest.epsilon:.2f}** guarantee costs **{util_cost:+.3f} PR-AUC** "
        f"({tpr_cost * 100:+.1f} pts of detection at the operating point) and moves membership "
        f"leakage by {(-leak_drop):+.3f} AUC."
    )

    if leak_drop > 0.01:
        leak_note = (
            "The leak closes as the budget tightens — the expected direction: noise added to the "
            "gradient is exactly what stops the model over-fitting the rows it saw."
        )
    else:
        leak_note = (
            "The measured leak barely moves here, and honestly so: a regularised **linear** model "
            "memorises little to begin with (the membership audit's thesis — leakage tracks "
            "memorisation, which linear models and early stopping already suppress), so the "
            "empirical attack has little to close. That does not make DP pointless: its value is "
            "the *formal* (epsilon, delta) bound, which holds against **every** attacker and "
            "dataset, not just the Yeom attack measured here. The frontier prices its cost."
        )

    tail = (
        "This is the project's measure -> fix -> re-measure arc one axis over: the membership "
        "audit *measured* the leak, DP-SGD *applies* the control with a certificate, and the table "
        "*re-measures* both the residual empirical leak and the detection the guarantee costs."
    )
    return f"{head}\n\n{leak_note}\n\n{tail}"


def _render(settings: Settings, points: list[DPPoint], fig: Path) -> str:
    cfg = settings.dp
    return f"""# NetSentry - Differential Privacy: the Utility-Leakage Frontier

_Synthetic stand-in; the methodology is the point. DP-SGD **logistic** models on the
exchangeable **stratified**, binary split (the split the membership audit uses), at a
fixed delta = {cfg.delta:g}. epsilon is spent by a pure-stdlib integer-order Renyi-DP
accountant; utility is binary attack-vs-benign detection, leakage is the same Yeom
confidence-threshold attack the [membership audit](membership.md) runs._

The [membership-inference audit](membership.md) measures how much the model memorises
its training data and ends by naming the mitigation with a formal guarantee -
**differentially-private training** - which buys an (epsilon, delta) bound at a
measured detection cost. This is that study. DP-SGD clips each flow's gradient to a
fixed L2 norm (bounding any one flow's influence) and adds Gaussian noise, so the
spent epsilon is a function of the noise multiplier, the minibatch sampling rate, and
the number of steps **only** - a certificate that holds for any dataset and any
attacker, not just the one measured below.

## The frontier

{_table(points)}

Smaller **epsilon** is a stronger privacy guarantee. **PR-AUC** and **TPR @ FPR** are
the detection kept; **membership AUC** (0.5 = no leakage) is the empirical leak that
remains against the Yeom attack. The **advantage** is Yeom's max(TPR - FPR).

![Privacy-utility-leakage frontier]({fig.as_posix()})

## Read

{_read(points)}

## Scope

- The guarantee is **formal**: DP bounds the influence of any single training flow on
  the released model, so it defends against attacks not enumerated here (the shadow
  attack, reconstruction, future attacks) - which is the whole point of a certificate
  over an empirical patch.
- The mechanism is **DP-SGD on a linear model**. DP for gradient-boosted trees is a
  different, messier mechanism; a linear model keeps the accountant honest and the
  utility ceiling real (the leaderboard shows logistic regression is competitive on
  the honest split). The deployed GBDT is unchanged.
- The accountant scans **integer** RDP orders, a sound upper bound on epsilon;
  fractional orders (Mironov 2019) would tighten the reported epsilon marginally.
"""


def run_dp_report(settings: Settings) -> Path:
    """Run the DP frontier study and write the report + figure."""
    from netsentry.evaluation import plots
    from netsentry.training.tracking import track_run

    points = run_dp(settings)
    private = [p for p in points if p.private]

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if private:
        eps = np.array([p.epsilon for p in private])
        series["detection (TPR @ FPR budget)"] = (eps, np.array([p.tpr_at_fpr for p in private]))
        series["membership advantage"] = (eps, np.array([p.membership_advantage for p in private]))
    fig = plots.plot_lines(
        series or {"detection": (np.array([1.0]), np.array([0.0]))},
        xlabel="Privacy budget epsilon (log scale; smaller = more private)",
        ylabel="Rate",
        title="Differential privacy: detection and leakage vs epsilon",
        out_path=settings.paths.figures_dir / "dp_frontier.png",
        xscale="log",
    )

    report = _render(settings, points, Path("..") / "figures" / fig.name)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote DP report", extra={"path": str(out_path)})

    with track_run(settings, "dp") as run:
        for p in points:
            tag = "nonpriv" if not p.private else f"sig{p.noise_multiplier:g}"
            run.log_metrics(
                {
                    f"{tag}_pr_auc": p.pr_auc,
                    f"{tag}_tpr": p.tpr_at_fpr,
                    f"{tag}_membership_auc": p.membership_auc,
                    f"{tag}_epsilon": p.epsilon if math.isfinite(p.epsilon) else -1.0,
                }
            )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path
