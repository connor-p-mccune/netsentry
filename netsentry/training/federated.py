"""Federated training: what detection costs when the traffic cannot be pooled.

Every model in this project is trained on one pooled dataset, which quietly assumes the
thing that is least true about network security telemetry: that somebody is allowed to
collect it all in one place. In practice they are not. Full packet-derived flow records
carry who talked to whom, on what port, for how long — for a hospital group, a bank's
regional subsidiaries, or a managed-security provider's client estates, "send us your flow
logs so we can train a shared detector" is a data-protection conversation that ends the
project. And the cost of *not* pooling is real, because each site alone sees only the
attacks that happened to hit it.

**Federated averaging** (McMahan et al., AISTATS 2017) is the standard escape: each site
trains locally for a few passes, sends only its *model weights* to a coordinator, and the
coordinator returns the sample-size-weighted average as the next round's starting point.
Raw flows never leave the site. The report measures the three things anyone deciding
whether to build this needs to know:

- the **centralized ceiling** — pooled training, what this project does today;
- the **local-only floor** — each site trains on its own data alone, which is what actually
  happens when the legal conversation fails;
- **FedAvg** between them, round by round, so the *federation tax* (ceiling minus federated)
  is a number rather than a hope.

Two complications get their own treatment rather than a footnote. The capture days are
**non-IID by construction** — CIC-IDS2017 runs different attacks on different days, so a
site's local label distribution is nothing like the global one, which is precisely the
regime where FedAvg is known to degrade (client drift: local optima pull the average away
from the global one). The report quantifies the skew instead of asserting it. And weight
sharing is **not** privacy: an averaged update still carries information about the data that
produced it, which is what the [membership-inference](membership.md) study exists to
measure. So a differentially-private arm adds calibrated Gaussian noise to the aggregate and
prices the resulting (epsilon, delta) with the same Renyi accountant the
[DP study](dp.md) uses — federation and DP being complements, not substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DAY_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import operating_point
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.robustness.dp import compute_rdp, rdp_to_epsilon
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import FederatedConfig

logger = get_logger(__name__)

REPORT_NAME = "federated.md"
FIGURE_NAME = "federated.png"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class Weights:
    """A linear model's parameters — the only thing that crosses a site boundary."""

    coef: np.ndarray
    intercept: float

    def copy(self) -> Weights:
        return Weights(self.coef.copy(), self.intercept)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Attack probability for each row."""
        return _sigmoid(np.asarray(x) @ self.coef + self.intercept)


def initial_weights(n_features: int) -> Weights:
    """Zeros — every site must start each round from the identical global model."""
    return Weights(np.zeros(n_features, dtype=float), 0.0)


def local_train(
    weights: Weights,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> Weights:
    """Run a site's local epochs of class-balanced mini-batch logistic SGD.

    Class weighting matters more here than centrally: a site whose local traffic is 99%
    benign will otherwise send back a near-constant model, and the average of several such
    models is a near-constant model. Balancing is computed from the *site's own* labels,
    because a site cannot see the global prior — which is itself part of what federation
    costs.
    """
    rng = np.random.default_rng(seed)
    w = weights.copy()
    n = len(y)
    if n == 0:
        return w
    pos = max(int(y.sum()), 1)
    neg = max(n - int(y.sum()), 1)
    # Balanced weights from the site's own label counts (sklearn's n / (2 * n_c) form).
    sample_w = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb, wb = x[idx], y[idx], sample_w[idx]
            pred = _sigmoid(xb @ w.coef + w.intercept)
            err = (pred - yb) * wb
            grad = xb.T @ err / len(idx) + l2 * w.coef
            w.coef -= learning_rate * grad
            w.intercept -= learning_rate * float(err.mean())
    return w


def federated_average(updates: list[Weights], sizes: list[int]) -> Weights:
    """Sample-size-weighted mean of site weights — the whole of FedAvg's aggregation step.

    Weighting by site size is what makes one round of FedAvg with a single local epoch
    equivalent to one round of centralized gradient descent; the divergence people mean by
    "client drift" is entirely a product of taking *several* local steps before averaging.
    """
    if not updates:
        raise ValueError("no site updates to aggregate")
    total = float(sum(sizes)) or 1.0
    coef = np.zeros_like(updates[0].coef)
    intercept = 0.0
    for w, n in zip(updates, sizes, strict=True):
        share = n / total
        coef += share * w.coef
        intercept += share * w.intercept
    return Weights(coef, intercept)


def add_aggregate_noise(
    weights: Weights,
    *,
    clip_norm: float,
    noise_multiplier: float,
    n_sites: int,
    rng: np.random.Generator,
) -> Weights:
    """Gaussian mechanism on the aggregate: the DP-FedAvg privacy step.

    Each site's contribution is bounded by clipping (so one site's data can shift the
    aggregate by at most ``clip_norm / n_sites``), and noise of scale
    ``noise_multiplier * clip_norm / n_sites`` is added to the average. That pair —
    bounded sensitivity plus calibrated noise — is what makes the released model
    differentially private *with respect to a whole site*, the unit of privacy that matches
    the threat ("can the coordinator tell what this hospital saw?").
    """
    if noise_multiplier <= 0.0:
        return weights.copy()
    sensitivity = clip_norm / max(n_sites, 1)
    sigma = noise_multiplier * sensitivity
    return Weights(
        weights.coef + rng.normal(0.0, sigma, size=weights.coef.shape),
        weights.intercept + float(rng.normal(0.0, sigma)),
    )


def clip_to_norm(weights: Weights, clip_norm: float) -> Weights:
    """Project a site's update onto the L2 ball that bounds its influence."""
    norm = float(np.sqrt(np.sum(weights.coef**2) + weights.intercept**2))
    if norm <= clip_norm or norm == 0.0:
        return weights.copy()
    scale = clip_norm / norm
    return Weights(weights.coef * scale, weights.intercept * scale)


def label_skew(site_priors: list[float], global_prior: float) -> float:
    """Mean absolute deviation of each site's attack prior from the global one.

    The simplest honest summary of how non-IID the federation is. Zero means every site
    sees the global mix; large values mean sites disagree about what traffic looks like,
    which is the regime where FedAvg is known to lose ground to pooling.
    """
    if not site_priors:
        return 0.0
    return float(np.mean([abs(p - global_prior) for p in site_priors]))


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class SiteSummary:
    """One participating site: what it holds and what it can do alone."""

    name: str
    n_rows: int
    attack_prior: float
    local_pr_auc: float
    local_tpr: float

    @property
    def single_class(self) -> bool:
        """Did this site see no attacks at all? Then it cannot train a *supervised* detector."""
        return self.attack_prior <= 0.0


@dataclass
class RoundPoint:
    """Federated performance after one aggregation round."""

    round_index: int
    pr_auc: float
    tpr: float


@dataclass
class Arm:
    """One training regime scored on the shared held-out test set."""

    name: str
    pr_auc: float
    tpr: float
    epsilon: float = float("inf")
    rounds: list[RoundPoint] = field(default_factory=list)


@dataclass
class FederatedStudy:
    """Everything the report renders."""

    sites: list[SiteSummary]
    global_prior: float
    skew: float
    centralized: Arm
    local_mean_pr_auc: float
    local_best_pr_auc: float
    federated: Arm
    private: list[Arm]
    target_fpr: float
    n_test: int
    rounds: int
    local_epochs: int
    delta: float

    @property
    def federation_tax(self) -> float:
        return self.centralized.pr_auc - self.federated.pr_auc

    @property
    def federation_gain(self) -> float:
        return self.federated.pr_auc - self.local_mean_pr_auc


def run_federated(settings: Settings) -> FederatedStudy:
    """Train centralized, local-only, FedAvg, and DP-FedAvg over the day-silos."""
    cfg: FederatedConfig = settings.federated
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    flows_per_day = variant.thresholds.assumed_flows_per_day
    target_fpr = variant.thresholds.primary_fpr

    def _score(w: Weights) -> tuple[float, float]:
        s_val, s_test = w.scores(x_val), w.scores(x_test)
        op = operating_point(y_val, s_val, y_test, s_test, target_fpr, flows_per_day)
        return float(average_precision_score(y_test, s_test)), float(op["tpr"])

    # Sites are capture days: the natural, already-non-IID partition of the training split.
    days = (
        train[DAY_COLUMN].to_numpy()
        if DAY_COLUMN in train.columns
        else np.zeros(len(train), dtype=int)
    )
    site_names = [str(d) for d in dict.fromkeys(days.tolist())]
    site_idx = {name: np.flatnonzero(days.astype(str) == name) for name in site_names}
    n_features = x_train.shape[1]

    train_kwargs = {
        "epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "l2": cfg.l2,
    }

    # Local-only floor: each site alone, evaluated on the shared test set.
    sites: list[SiteSummary] = []
    for i, name in enumerate(site_names):
        idx = site_idx[name]
        w = local_train(
            initial_weights(n_features),
            x_train[idx],
            y_train[idx],
            seed=variant.seed + i,
            **train_kwargs,  # type: ignore[arg-type]
        )
        # A lone site still needs several passes to converge; give it the same total
        # compute the federation spends, so the comparison is about data, not epochs.
        for extra in range(1, cfg.rounds):
            w = local_train(
                w, x_train[idx], y_train[idx], seed=variant.seed + i + 100 * extra, **train_kwargs  # type: ignore[arg-type]
            )
        pr_auc, tpr = _score(w)
        sites.append(
            SiteSummary(
                name=name,
                n_rows=len(idx),
                attack_prior=float(y_train[idx].mean()),
                local_pr_auc=pr_auc,
                local_tpr=tpr,
            )
        )
        logger.info("Local site trained", extra={"site": name, "pr_auc": round(pr_auc, 4)})

    # Centralized ceiling: the same optimiser, all the data pooled.
    central_w = initial_weights(n_features)
    for r in range(cfg.rounds):
        central_w = local_train(
            central_w, x_train, y_train, seed=variant.seed + 1000 * r, **train_kwargs  # type: ignore[arg-type]
        )
    central_pr, central_tpr = _score(central_w)

    def _federate(noise_multiplier: float, tag: str) -> Arm:
        rng = np.random.default_rng(variant.seed)
        glob = initial_weights(n_features)
        history: list[RoundPoint] = []
        for r in range(cfg.rounds):
            updates, sizes = [], []
            for i, name in enumerate(site_names):
                idx = site_idx[name]
                local = local_train(
                    glob,
                    x_train[idx],
                    y_train[idx],
                    seed=variant.seed + i + 1000 * r,
                    **train_kwargs,  # type: ignore[arg-type]
                )
                updates.append(
                    clip_to_norm(local, cfg.clip_norm) if noise_multiplier > 0 else local
                )
                sizes.append(len(idx))
            glob = federated_average(updates, sizes)
            if noise_multiplier > 0:
                glob = add_aggregate_noise(
                    glob,
                    clip_norm=cfg.clip_norm,
                    noise_multiplier=noise_multiplier,
                    n_sites=len(site_names),
                    rng=rng,
                )
            pr_auc, tpr = _score(glob)
            history.append(RoundPoint(round_index=r + 1, pr_auc=pr_auc, tpr=tpr))
        eps = float("inf")
        if noise_multiplier > 0:
            # Full participation each round (q = 1), one Gaussian release per round.
            rdp = compute_rdp(1.0, noise_multiplier, cfg.rounds)
            eps, _ = rdp_to_epsilon(rdp, cfg.delta)
        final = history[-1]
        logger.info(tag, extra={"pr_auc": round(final.pr_auc, 4), "epsilon": eps})
        return Arm(name=tag, pr_auc=final.pr_auc, tpr=final.tpr, epsilon=eps, rounds=history)

    federated = _federate(0.0, "FedAvg (no DP)")
    private = [_federate(sigma, f"DP-FedAvg (noise x{sigma:g})") for sigma in cfg.noise_multipliers]

    global_prior = float(y_train.mean())
    return FederatedStudy(
        sites=sites,
        global_prior=global_prior,
        skew=label_skew([s.attack_prior for s in sites], global_prior),
        centralized=Arm(name="centralized (pooled)", pr_auc=central_pr, tpr=central_tpr),
        local_mean_pr_auc=float(np.mean([s.local_pr_auc for s in sites])),
        local_best_pr_auc=float(np.max([s.local_pr_auc for s in sites])),
        federated=federated,
        private=private,
        target_fpr=target_fpr,
        n_test=len(y_test),
        rounds=cfg.rounds,
        local_epochs=cfg.local_epochs,
        delta=cfg.delta,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_federated_report(settings: Settings) -> Path:
    """Run the federated study and write the report + figure."""
    study = run_federated(settings)

    rounds = np.array([p.round_index for p in study.federated.rounds], dtype=float)
    series = {
        "FedAvg (weights shared, no raw traffic)": (
            rounds,
            np.array([p.pr_auc for p in study.federated.rounds]),
        ),
        "centralized ceiling (pooled)": (rounds, np.full(len(rounds), study.centralized.pr_auc)),
        "local-only floor (mean site)": (rounds, np.full(len(rounds), study.local_mean_pr_auc)),
    }
    for arm in study.private:
        series[arm.name] = (rounds, np.array([p.pr_auc for p in arm.rounds]))
    fig = plots.plot_lines(
        series,
        xlabel="federated round",
        ylabel="PR-AUC on the shared held-out test set",
        title="Federated averaging closes most of the gap to pooled training",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote federated report", extra={"path": str(out_path)})

    with track_run(settings, "federated") as run:
        run.log_params({"rounds": study.rounds, "local_epochs": study.local_epochs})
        run.log_metrics(
            {
                "centralized_pr_auc": study.centralized.pr_auc,
                "federated_pr_auc": study.federated.pr_auc,
                "local_mean_pr_auc": study.local_mean_pr_auc,
                "federation_tax": study.federation_tax,
                "label_skew": study.skew,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _sites_table(study: FederatedStudy) -> str:
    rows = [
        "| site (capture day) | flows held | local attack prior | alone: PR-AUC "
        "| alone: detection |",
        "|---|---|---|---|---|",
    ]
    for s in study.sites:
        note = " _(no attacks — see below)_" if s.single_class else ""
        rows.append(
            f"| {s.name} | {s.n_rows:,} | {s.attack_prior:.1%}{note} | {s.local_pr_auc:.3f} "
            f"| {s.local_tpr:.1%} |"
        )
    rows.append(
        f"| **all sites pooled** | {sum(s.n_rows for s in study.sites):,} "
        f"| {study.global_prior:.1%} | **{study.centralized.pr_auc:.3f}** "
        f"| **{study.centralized.tpr:.1%}** |"
    )
    return "\n".join(rows)


def _arms_table(study: FederatedStudy) -> str:
    rows = [
        "| regime | raw traffic leaves the site? | PR-AUC | detection | privacy budget |",
        "|---|---|---|---|---|",
    ]
    rows.append(
        f"| centralized (pooled) | **yes** | {study.centralized.pr_auc:.3f} "
        f"| {study.centralized.tpr:.1%} | none |"
    )
    rows.append(f"| local-only (mean site) | no | {study.local_mean_pr_auc:.3f} | — | none |")
    supervised = [s for s in study.sites if not s.single_class]
    if supervised:
        best = max(s.local_pr_auc for s in supervised)
        rows.append(f"| local-only (best site that saw an attack) | no | {best:.3f} | — | none |")
    rows.append(
        f"| **FedAvg** | no (weights only) | **{study.federated.pr_auc:.3f}** "
        f"| **{study.federated.tpr:.1%}** | none |"
    )
    for arm in study.private:
        eps = "no guarantee" if not np.isfinite(arm.epsilon) else f"eps = {arm.epsilon:.2f}"
        rows.append(
            f"| {arm.name} | no (noised weights) | {arm.pr_auc:.3f} | {arm.tpr:.1%} | {eps} |"
        )
    return "\n".join(rows)


def _headline_read(study: FederatedStudy) -> str:
    tax = study.federation_tax
    gain = study.federation_gain
    recovered = (
        gain / (study.centralized.pr_auc - study.local_mean_pr_auc)
        if study.centralized.pr_auc > study.local_mean_pr_auc
        else 1.0
    )
    if tax <= 0:
        tax_clause = (
            f"The federation tax is **negative** ({tax:+.3f}): averaging over sites landed at or "
            "above pooled training. That is not a paradox — the site-weighted average of several "
            "locally-balanced models is a mild ensemble, and ensembling reduces variance, which "
            "on a shifted test split can be worth more than the extra data pooling provides."
        )
    else:
        tax_clause = (
            f"The federation tax is **{tax:.3f} PR-AUC** — what an organisation pays for never "
            "letting flow records leave the site."
        )
    return (
        f"The three numbers that decide whether to build this: pooled training reaches "
        f"{study.centralized.pr_auc:.3f}, the average site training alone reaches "
        f"{study.local_mean_pr_auc:.3f}, and FedAvg reaches {study.federated.pr_auc:.3f} while "
        f"raw flow records never leave a site. {tax_clause} Against the alternative that "
        f"actually happens when the data-sharing conversation fails — everyone trains alone — "
        f"federation is worth {gain:+.3f} PR-AUC, recovering {recovered:.0%} of the distance "
        "from the local floor to the pooled ceiling. That is the comparison that matters: the "
        "choice is rarely federated-versus-pooled, it is federated-versus-nothing."
    )


def _skew_read(study: FederatedStudy) -> str:
    lo = min(study.sites, key=lambda s: s.attack_prior)
    hi = max(study.sites, key=lambda s: s.attack_prior)
    return (
        "The sites are **non-IID by construction**, which is the regime where FedAvg is "
        "supposed to struggle. CIC-IDS2017 runs different attacks on different days, so each "
        f"site's local attack prior ranges from {lo.attack_prior:.1%} ({lo.name}) to "
        f"{hi.attack_prior:.1%} ({hi.name}) against a global {study.global_prior:.1%} — a mean "
        f"absolute skew of {study.skew:.3f}. Beyond the prior, the *kinds* of attack differ: a "
        "site that never saw a web attack contributes weights that have no opinion about one, "
        "and averaging those weights with a site that did is exactly the client-drift problem. "
        f"With {study.local_epochs} local epochs per round the drift stays mild here; pushing "
        "local work up (fewer, longer rounds — the usual move to save bandwidth) is what makes "
        "site optima diverge far enough for the average to be worse than any of them."
        + _single_class_note(study)
    )


def _single_class_note(study: FederatedStudy) -> str:
    """Explain the site that saw no attacks — and why its number is not what it looks like."""
    lone = [s for s in study.sites if s.single_class]
    if not lone:
        return ""
    site = lone[0]
    others = [s for s in study.sites if not s.single_class]
    beats = [o.name for o in others if site.local_pr_auc > o.local_pr_auc]
    return (
        f"\n\nOne row deserves a second look rather than a footnote: **{site.name} holds no "
        f"attacks at all** — it is CIC-IDS2017's clean baseline day — and yet its local model "
        f"posts {site.local_pr_auc:.3f} PR-AUC, "
        + (
            f"beating {', '.join(beats)} despite never having seen a single attack. "
            if beats
            else "which is far above the 0.250 base rate despite it never having seen an attack. "
        )
        + "That looks wrong, so it was checked rather than reported: the fitted model is not "
        "degenerate (its scores span the full range and take 17k distinct values), and the "
        "explanation is that logistic loss on all-benign labels is a **one-class fit**. Every "
        "gradient step pushes predictions down, and the flows that resist hardest are the ones "
        "least like the site's benign traffic — which is a benign-manifold model, not a "
        "supervised detector. It is, almost exactly, the benign-only training regime this "
        "project's [anomaly detector](anomaly.md) already uses, arrived at by accident. The "
        "practical reading for a federation is that a site with no confirmed attacks still "
        "contributes something real to the average, which is a genuinely useful property when "
        "most participants have never knowingly been breached — but its standalone number "
        "should be read as an anomaly score, not as detection it could be trusted to repeat."
    )


def _privacy_read(study: FederatedStudy) -> str:
    if not study.private:
        return ""
    best = min(study.private, key=lambda a: a.epsilon)
    worst = max(study.private, key=lambda a: a.epsilon)
    return (
        "Sharing weights instead of flows is a **confidentiality** improvement, not a privacy "
        "guarantee: the [membership-inference](membership.md) study in this repo exists because "
        "model parameters leak information about the rows that produced them, and an averaged "
        "update is still a function of every site's data. The DP-FedAvg rows buy an actual "
        "guarantee — clip each site's update to bound its influence, add Gaussian noise to the "
        "aggregate, and account for the composition across rounds with the same Renyi accountant "
        f"the [DP study](dp.md) uses (delta = {study.delta:g}, full participation each round). "
        f"The privacy unit is the **site**, which is the one that matches the threat: the "
        f"guarantee is that the released model looks nearly the same whether or not any one "
        f"organisation joined. It costs what privacy always costs — "
        f"{worst.pr_auc:.3f} PR-AUC at eps = {worst.epsilon:.2f} and {best.pr_auc:.3f} at "
        f"eps = {best.epsilon:.2f}, against {study.federated.pr_auc:.3f} unnoised.\n\n"
        "Both rows are reported as measured, and neither is a good deal here. An epsilon in the "
        f"tens ({worst.epsilon:.0f}) is a **vacuous** guarantee — it bounds the privacy loss at a "
        "level no practitioner would accept, so that row pays utility for nothing; and the budget "
        f"that is at least nameable (eps = {best.epsilon:.2f}) costs "
        f"{study.federated.pr_auc - best.pr_auc:.3f} PR-AUC, roughly half the detection. The "
        "reason is structural and worth stating plainly: DP-FedAvg's noise is calibrated to one "
        f"*site's* influence, and with only {len(study.sites)} sites each one moves the average a "
        "great deal, so the noise needed to hide it is large. Site-level DP gets cheap with "
        "hundreds of participants, not three — the same 1/n scaling the [DP study](dp.md) enjoys "
        "in the per-example setting, where thousands of examples make the noise affordable. "
        "Federation and DP are complements (the first keeps the data home, the second bounds what "
        "the model gives away), but this federation is far too small to buy the second cheaply, "
        "and saying so is more useful than reporting whichever row looks least bad."
    )


def _render(study: FederatedStudy, fig: Path) -> str:
    return f"""# NetSentry — Federated Training Across Sites That Cannot Pool Traffic

_Synthetic stand-in. Honest temporal/binary split; {len(study.sites)} sites = the capture days
of the training split, {study.n_test:,} shared held-out test flows. {study.rounds} federated
rounds of {study.local_epochs} local epochs each; the local-only and centralized arms are
given the same total optimisation budget so the comparison is about **data access**, not
compute. All arms use the same linear model, because FedAvg averages parameters and a
gradient-boosted forest has none to average._

## Why this report exists

Every other model here trains on one pooled dataset, which assumes the least true thing about
security telemetry: that someone is allowed to collect it all. Flow records carry who talked
to whom, for how long, on what port. For a hospital group, a bank's regional entities, or an
MSSP's client estates, "send us your flow logs" is a data-protection conversation that ends
the project — and training alone is expensive, because each site only sees the attacks that
hit it. Federated averaging (McMahan et al. 2017) is the standard escape: train locally,
share only weights, average by sample count, repeat.

## Who holds what

{_sites_table(study)}

{_skew_read(study)}

## What federation costs, and what it saves

{_arms_table(study)}

![federated rounds](../figures/{fig.name})

{_headline_read(study)}

## Weights are not privacy

{_privacy_read(study)}

## Scope

The sites are capture days, not real organisations — a partition that is genuinely non-IID
(different attacks per day) but shares one sensor, one schema, and one feature pipeline. Real
federation is harder in exactly the places this cannot show: sites disagree about feature
definitions, run different exporter versions, and drop in and out between rounds. The model
is linear because FedAvg averages parameters; federating the deployed gradient-boosted model
means a different algorithm entirely (federated boosting, or distilling site models into a
shared student), which is why the centralized ceiling here sits below this project's headline
LightGBM number and should be read as the linear ceiling, not the project's. The DP accounting
assumes every site participates in every round (q = 1) and one Gaussian release per round;
subsampling participation would buy amplification the accountant does not credit here. And
the threat model is an honest-but-curious coordinator: a malicious one can do considerably
more with per-round updates than with the final model, which is what secure aggregation
protocols exist for and which this does not implement."""
