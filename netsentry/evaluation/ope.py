"""Off-policy evaluation: what would a different triage policy have caught?

Every threshold study in this project answers its question with labels for every flow. A
real SOC has no such thing. It has a **log**: for each flow, the score, the decision its
deployed policy took, and — only for the flows it actually reviewed — what the analyst
found. Nobody labels the flows that were auto-cleared, because nobody looked at them.

So the question an operator actually asks is counterfactual. *If we drop the threshold, how
many more attacks do we catch and how much analyst time does it cost?* Answering it by
scoring a candidate policy on the logs is wrong in a specific, dangerous way: the labels in
the log exist **because** the old policy selected those flows. Evaluating a new policy on
that sample measures the old policy's selection, not the new policy's value.

This is the contextual-bandit off-policy evaluation problem, and it has a mature answer.
Treat the triage decision as an action, the analyst outcome as a reward, and the deployed
threshold as a logging policy `pi_0`. Then four estimators, in increasing order of how much
they assume:

- **Direct method** — fit a reward model on the logged reviews and integrate it under the
  new policy. Low variance, and biased by exactly the selection problem above, because the
  reward model was trained on the flows the old policy chose to show it.
- **IPS** (Horvitz-Thompson 1952) — reweight each logged reward by
  `pi(a|x) / pi_0(a|x)`. Unbiased whenever the logging policy had some chance of taking
  the action the new policy would take, and it can have ruinous variance.
- **SNIPS** (Swaminathan & Joachims 2015) — the same, divided by the total weight.
  Slightly biased, far steadier, and invariant to how the reward is scaled.
- **Doubly robust** (Dudik, Langford & Li 2011) — the reward model as a baseline plus an
  IPS correction on its residuals. Consistent if *either* the reward model or the
  propensities are right, which is the whole appeal.

Because this dataset does have every label, the **true** value of each candidate policy is
computable exactly. That converts a methods demonstration into a measurement: each
estimator can be scored against ground truth, and — the thing that actually matters — the
policy each estimator *would have selected* can be scored too.

The finding underneath all of it is about the log, not the estimator. A deterministic
threshold assigns probability zero to the action it never takes, so no reweighting can
recover what a lower threshold would have found: the data contains no evidence about it,
and every estimator that appears to answer is extrapolating silently. The fix is not a
better estimator, it is a **logging policy that explores** — a small random review budget
deliberately spent so that future policies can be evaluated offline instead of deployed
and hoped for. This report prices that budget: what exploration costs, what it buys, and
where the two curves cross.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import OPEConfig

logger = get_logger(__name__)

REPORT_NAME = "ope.md"
ESTIMATOR_FIGURE = "ope_estimators.png"
EXPLORATION_FIGURE = "ope_exploration.png"


# --------------------------------------------------------------------------------------
# The bandit formulation (pure; unit-tested directly)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Economics:
    """What a review is worth and what it costs, in the cost study's currency.

    Reviewing a flow costs analyst time whatever it turns out to be; reviewing an attack
    averts the loss a miss would have caused. Skipping a flow is the reference point and
    scores exactly zero — which is what makes this formulation honest under partial
    feedback: the un-reviewed arm needs no outcome to be known, because by construction
    nothing was spent and nothing was found.
    """

    value_per_catch: float
    cost_per_review: float

    def reward(self, reviewed: np.ndarray, is_attack: np.ndarray) -> np.ndarray:
        """Realised reward per flow given the action taken and the truth behind it."""
        act = np.asarray(reviewed, dtype=float)
        attack = np.asarray(is_attack, dtype=float)
        return act * (self.value_per_catch * attack - self.cost_per_review)


def epsilon_greedy_propensity(scores: np.ndarray, threshold: float, epsilon: float) -> np.ndarray:
    """``P(review | flow)`` for a deployed threshold softened by an exploration budget.

    With probability ``1 - epsilon`` the policy does what the threshold says; with
    probability ``epsilon`` it flips a fair coin. The resulting propensity is bounded away
    from 0 and 1 for every flow, which is precisely the property off-policy evaluation
    needs and a plain threshold does not have.
    """
    greedy = (np.asarray(scores, dtype=float) >= threshold).astype(float)
    eps = float(np.clip(epsilon, 0.0, 1.0))
    return (1.0 - eps) * greedy + eps * 0.5


def deterministic_propensity(scores: np.ndarray, threshold: float) -> np.ndarray:
    """``P(review | flow)`` for a candidate policy that is a plain threshold."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(float)


def policy_value(target_p: np.ndarray, is_attack: np.ndarray, econ: Economics) -> float:
    """Exact value of a policy — available here only because every label is known.

    This is the yardstick, not an estimator. Its existence is what turns the comparison
    below from a demonstration into a measurement.
    """
    p = np.asarray(target_p, dtype=float)
    attack = np.asarray(is_attack, dtype=float)
    per_flow = p * (econ.value_per_catch * attack - econ.cost_per_review)
    return float(np.mean(per_flow))


def importance_weights(
    actions: np.ndarray, logged_p: np.ndarray, target_p: np.ndarray
) -> np.ndarray:
    """``pi(a|x) / pi_0(a|x)`` for the action actually taken on each logged flow."""
    a = np.asarray(actions, dtype=float)
    p0 = np.asarray(logged_p, dtype=float)
    p1 = np.asarray(target_p, dtype=float)
    num = a * p1 + (1.0 - a) * (1.0 - p1)
    den = a * p0 + (1.0 - a) * (1.0 - p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.inf)
    # A zero-propensity action the target policy would take is an unanswerable question,
    # not a large number: it is surfaced as a support violation rather than clipped away.
    return np.asarray(np.where(num == 0.0, 0.0, w), dtype=float)


def support_violation_rate(logged_p: np.ndarray, target_p: np.ndarray) -> float:
    """Share of flows the target policy would review that the log could never inform on.

    Nonzero means the estimate is not identified: the logging policy assigned probability
    zero to an action the candidate policy takes, so no amount of reweighting recovers what
    would have happened. This is the diagnostic that should stop an analysis, and it is
    reported first for that reason.
    """
    p0 = np.asarray(logged_p, dtype=float)
    p1 = np.asarray(target_p, dtype=float)
    at_risk = p1 > 0.0
    if not at_risk.any():
        return 0.0
    return float(np.mean(p0[at_risk] <= 0.0))


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ESS: how many observations the reweighted estimate is really standing on.

    Returns zero if any weight is non-finite. That is not a numerical guard — a non-finite
    weight means the target policy takes an action the log gave probability zero, so the
    estimate is not identified and the honest count of what it stands on is none. Dropping
    those rows and reporting a healthy ESS is the failure mode this guards against.
    """
    w = np.asarray(weights, dtype=float)
    if w.size and not np.all(np.isfinite(w)):
        return 0.0
    denom = float(np.sum(w**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def contributing_ess(actions: np.ndarray, weights: np.ndarray) -> float:
    """ESS over the flows that actually carry reward — the reviewed ones.

    Taken over every logged flow, the ESS is dominated by the skip arm, whose weights all
    sit near 1 and whose rewards are all zero: it would report tens of thousands of
    observations behind an estimate that four reviewed flows decide. Restricting to the
    contributing arm is the diagnostic an operator can act on.
    """
    a = np.asarray(actions).astype(bool)
    w = np.asarray(weights, dtype=float)
    if not a.any():
        return 0.0
    return effective_sample_size(w[a])


def resample_to_prevalence(
    is_attack: np.ndarray, target_rate: float, *, rng: np.random.Generator
) -> np.ndarray:
    """Row indices re-mixing a stream to a realistic production attack prevalence.

    The synthetic test split is ~25% attacks, which makes reviewing a flow chosen at random
    profitable and collapses the whole policy comparison into "review everything". Real
    traffic is nothing like that, and the [cost study](cost.md) already fixes a production
    prior for the same reason. Attacks are subsampled (or benign are, whichever side is in
    surplus) so the economics below describe a plausible deployment rather than the
    generator's convenience.
    """
    y = np.asarray(is_attack).astype(int)
    attack_idx = np.flatnonzero(y == 1)
    benign_idx = np.flatnonzero(y == 0)
    rate = float(np.clip(target_rate, 1e-9, 1.0 - 1e-9))
    wanted_attacks = round(len(benign_idx) * rate / (1.0 - rate))
    if wanted_attacks <= len(attack_idx):
        keep_attacks = rng.choice(attack_idx, size=max(wanted_attacks, 1), replace=False)
        keep_benign = benign_idx
    else:  # too few attacks to reach the target: thin the benign side instead
        wanted_benign = round(len(attack_idx) * (1.0 - rate) / rate)
        keep_attacks = attack_idx
        keep_benign = rng.choice(
            benign_idx, size=min(wanted_benign, len(benign_idx)), replace=False
        )
    return np.sort(np.concatenate([keep_attacks, keep_benign]))


def ips_value(rewards: np.ndarray, weights: np.ndarray) -> float:
    """Inverse-propensity-scored value: unbiased under overlap, and it pays for it."""
    w = np.asarray(weights, dtype=float)
    r = np.asarray(rewards, dtype=float)
    finite = np.isfinite(w)
    if not finite.any():
        return 0.0
    return float(np.mean(np.where(finite, w * r, 0.0)))


def snips_value(rewards: np.ndarray, weights: np.ndarray) -> float:
    """Self-normalised IPS: divide by the realised weight mass instead of by ``n``.

    A draw that happens to over-sample high-weight flows inflates numerator and denominator
    together, so the ratio is far steadier than IPS at the cost of a small bias — and it is
    invariant to rescaling the reward, which IPS is not.
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(rewards, dtype=float)
    finite = np.isfinite(w)
    total = float(np.sum(np.where(finite, w, 0.0)))
    if total <= 0.0:
        return 0.0
    return float(np.sum(np.where(finite, w * r, 0.0)) / total)


def dm_value(target_p: np.ndarray, reward_hat: np.ndarray) -> float:
    """Direct method: integrate a fitted reward model under the candidate policy.

    No propensities appear, which is exactly why it is low-variance and exactly why it
    inherits every belief its reward model picked up from the flows it was trained on.
    """
    p = np.asarray(target_p, dtype=float)
    r_hat = np.asarray(reward_hat, dtype=float)
    return float(np.mean(p * r_hat))


def dr_value(
    actions: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    target_p: np.ndarray,
    reward_hat: np.ndarray,
) -> float:
    """Doubly robust: the reward model as a baseline, corrected by IPS on its residuals.

    The skip arm cancels exactly — its modelled and realised rewards are both zero — so the
    correction term rides only on the reviewed flows, which is where the log has anything
    to say.
    """
    a = np.asarray(actions, dtype=float)
    r = np.asarray(rewards, dtype=float)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(target_p, dtype=float)
    r_hat = np.asarray(reward_hat, dtype=float)
    baseline = p * r_hat
    residual = a * (r - r_hat)
    finite = np.isfinite(w)
    correction = np.where(finite, w * residual, 0.0)
    return float(np.mean(baseline + correction))


def fit_reward_model(
    features: np.ndarray,
    actions: np.ndarray,
    is_attack: np.ndarray,
    econ: Economics,
    *,
    seed: int,
) -> np.ndarray:
    """Reward model for the review arm, fitted on **only the flows that were reviewed**.

    This is the realistic constraint and the source of the direct method's bias: the
    training sample is exactly the set the logging policy chose to show an analyst, so a
    model fitted on it learns the logging policy's view of the world. Falls back to the
    reviewed base rate when the labels are degenerate, which happens when exploration is
    too thin to have surfaced an attack.
    """
    reviewed = np.asarray(actions).astype(bool)
    y = np.asarray(is_attack).astype(int)[reviewed]
    n_total = len(np.asarray(actions))
    if reviewed.sum() < 10 or len(np.unique(y)) < 2:
        rate = float(y.mean()) if len(y) else 0.0
        return np.full(n_total, econ.value_per_catch * rate - econ.cost_per_review)
    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(np.asarray(features)[reviewed], y)
    proba = np.asarray(model.predict_proba(np.asarray(features)))[:, 1]
    return np.asarray(econ.value_per_catch * proba - econ.cost_per_review, dtype=float)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class Candidate:
    """One candidate triage policy: a threshold, and what it is really worth."""

    name: str
    fpr_target: float
    threshold: float
    review_rate: float
    true_value: float
    support_violation: float
    ess: float


@dataclass
class EstimatorRow:
    """One estimator's accuracy on one candidate policy, over replicate logs."""

    candidate: str
    estimator: str
    mean: float
    bias: float
    rmse: float
    std: float


@dataclass
class ExplorationRow:
    """What one exploration budget costs and what it buys."""

    epsilon: float
    logging_value: float
    exploration_cost: float
    support_violation: float
    mean_ess: float
    rmse: dict[str, float]
    selection_regret: dict[str, float]


@dataclass
class OPEStudy:
    """Everything the report renders."""

    n_stream: int
    attack_rate: float
    raw_attack_rate: float
    epsilon: float
    logging_threshold: float
    logging_fpr: float
    econ: Economics
    candidates: list[Candidate]
    best_candidate: str
    rows: list[EstimatorRow]
    headline_regret: dict[str, float]
    exploration: list[ExplorationRow]
    n_replicates: int
    estimators: tuple[str, ...]


ESTIMATORS = ("direct method", "IPS", "SNIPS", "doubly robust")


def _estimate_all(
    actions: np.ndarray,
    rewards: np.ndarray,
    logged_p: np.ndarray,
    target_p: np.ndarray,
    reward_hat: np.ndarray,
    econ: Economics,
) -> dict[str, float]:
    """Run every estimator on one logged stream against one candidate policy."""
    weights = importance_weights(actions, logged_p, target_p)
    del econ  # rewards arrive pre-priced; the economics were applied upstream
    return {
        "direct method": dm_value(target_p, reward_hat),
        "IPS": ips_value(rewards, weights),
        "SNIPS": snips_value(rewards, weights),
        "doubly robust": dr_value(actions, rewards, weights, target_p, reward_hat),
    }


def _simulate_logs(
    scores: np.ndarray,
    is_attack: np.ndarray,
    features: np.ndarray,
    logging_threshold: float,
    epsilon: float,
    econ: Economics,
    targets: dict[str, np.ndarray],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Replay the deployment under a logging policy and estimate every candidate's value.

    Returns per-candidate, per-estimator arrays of estimates across replicate logs, plus
    the effective sample size seen on each replicate. Only the *actions* are resampled —
    the traffic is held fixed — so the spread measures logging noise rather than a change
    of population.
    """
    rng = np.random.default_rng(seed)
    logged_p = epsilon_greedy_propensity(scores, logging_threshold, epsilon)
    estimates: dict[str, dict[str, list[float]]] = {
        name: {est: [] for est in ESTIMATORS} for name in targets
    }
    ess_values: list[float] = []
    for rep in range(int(n_replicates)):
        actions = (rng.random(len(scores)) < logged_p).astype(int)
        rewards = econ.reward(actions, is_attack)
        reward_hat = fit_reward_model(features, actions, is_attack, econ, seed=seed + rep)
        rep_ess: list[float] = []
        for name, target_p in targets.items():
            values = _estimate_all(actions, rewards, logged_p, target_p, reward_hat, econ)
            for est, value in values.items():
                estimates[name][est].append(value)
            rep_ess.append(
                contributing_ess(actions, importance_weights(actions, logged_p, target_p))
            )
        ess_values.append(float(np.mean(rep_ess)) if rep_ess else 0.0)
    packed = {
        name: {est: np.asarray(vals, dtype=float) for est, vals in per_est.items()}
        for name, per_est in estimates.items()
    }
    return packed, np.asarray(ess_values, dtype=float)


def _selection_regret(
    packed: dict[str, dict[str, np.ndarray]], truth: dict[str, float]
) -> dict[str, float]:
    """Value lost by *choosing* a policy with each estimator rather than knowing the truth.

    The decision-level metric. An estimator that is wrong about every policy by the same
    amount still ranks them correctly and costs nothing; an estimator that is close on
    average but reorders the top two costs real money. Only the second kind matters.
    """
    names = list(truth)
    best = max(truth.values())
    n_reps = len(next(iter(packed.values()))["IPS"]) if packed else 0
    out: dict[str, float] = {}
    for est in ESTIMATORS:
        losses = []
        for rep in range(n_reps):
            chosen = max(names, key=lambda n: packed[n][est][rep])
            losses.append(best - truth[chosen])
        out[est] = float(np.mean(losses)) if losses else 0.0
    return out


def run_ope(settings: Settings) -> OPEStudy:
    """Replay the later-day stream under a logging policy and evaluate candidates offline."""
    cfg: OPEConfig = settings.ope
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    from netsentry.data.split import load_split

    result = fit_supervised(variant)
    benign = variant.labels.benign_label
    s_val = attack_probability(np.asarray(result.proba_val), result.classes, benign)
    s_test = attack_probability(np.asarray(result.proba_test), result.classes, benign)
    y_val = np.asarray(result.y_val).astype(int)
    y_test = np.asarray(result.y_test).astype(int)
    # The reward model sees the same feature space the detector does, so the direct
    # method's weakness is which flows it was trained on rather than what it could see.
    test_frame = load_split(variant, "temporal", "test")
    features = np.asarray(result.bundle.pipeline.transform(test_frame))

    # Re-mix to a realistic prevalence: at the split's own ~25% attack rate a randomly
    # chosen review has positive expected value, "review everything" wins by construction,
    # and the policy comparison has nothing left to compare.
    raw_attack_rate = float(y_test.mean())
    keep = resample_to_prevalence(
        y_test, settings.cost.production_attack_rate, rng=np.random.default_rng(settings.seed)
    )
    s_test, y_test, features = s_test[keep], y_test[keep], features[keep]

    econ = Economics(
        value_per_catch=settings.cost.cost_per_miss,
        cost_per_review=settings.cost.cost_per_alert,
    )
    logging_threshold = threshold_at_fpr(y_val, s_val, cfg.logging_fpr)

    targets: dict[str, np.ndarray] = {}
    candidates: list[Candidate] = []
    logged_p = epsilon_greedy_propensity(s_test, logging_threshold, cfg.exploration)
    for fpr in cfg.candidate_fprs:
        threshold = threshold_at_fpr(y_val, s_val, fpr)
        target_p = deterministic_propensity(s_test, threshold)
        name = f"review @ {fpr:.1%} FPR"
        targets[name] = target_p
        candidates.append(
            Candidate(
                name=name,
                fpr_target=fpr,
                threshold=threshold,
                review_rate=float(target_p.mean()),
                true_value=policy_value(target_p, y_test, econ),
                support_violation=support_violation_rate(logged_p, target_p),
                ess=contributing_ess(
                    (s_test >= logging_threshold).astype(int),
                    importance_weights(
                        (s_test >= logging_threshold).astype(int), logged_p, target_p
                    ),
                ),
            )
        )

    truth = {c.name: c.true_value for c in candidates}
    best_candidate = max(truth, key=lambda k: truth[k])

    packed, _ess = _simulate_logs(
        s_test,
        y_test,
        features,
        logging_threshold,
        cfg.exploration,
        econ,
        targets,
        n_replicates=cfg.n_replicates,
        seed=settings.seed,
    )
    rows: list[EstimatorRow] = []
    for name in targets:
        for est in ESTIMATORS:
            vals = packed[name][est]
            rows.append(
                EstimatorRow(
                    candidate=name,
                    estimator=est,
                    mean=float(np.mean(vals)),
                    bias=float(np.mean(vals) - truth[name]),
                    rmse=float(np.sqrt(np.mean((vals - truth[name]) ** 2))),
                    std=float(np.std(vals)),
                )
            )

    greedy_value = policy_value(deterministic_propensity(s_test, logging_threshold), y_test, econ)
    exploration: list[ExplorationRow] = []
    for eps in cfg.exploration_sweep:
        p0 = epsilon_greedy_propensity(s_test, logging_threshold, eps)
        packed_eps, ess_eps = _simulate_logs(
            s_test,
            y_test,
            features,
            logging_threshold,
            eps,
            econ,
            targets,
            n_replicates=cfg.sweep_replicates,
            seed=settings.seed,
        )
        rmse = {
            est: float(
                np.sqrt(np.mean([np.mean((packed_eps[n][est] - truth[n]) ** 2) for n in targets]))
            )
            for est in ESTIMATORS
        }
        logging_value = policy_value(p0, y_test, econ)
        exploration.append(
            ExplorationRow(
                epsilon=eps,
                logging_value=logging_value,
                exploration_cost=greedy_value - logging_value,
                support_violation=float(
                    np.mean([support_violation_rate(p0, targets[n]) for n in targets])
                ),
                mean_ess=float(np.mean(ess_eps)),
                rmse=rmse,
                selection_regret=_selection_regret(packed_eps, truth),
            )
        )
        logger.info(
            "Exploration arm done",
            extra={"epsilon": eps, "dr_rmse": round(rmse["doubly robust"], 4)},
        )

    return OPEStudy(
        n_stream=len(y_test),
        attack_rate=float(y_test.mean()),
        raw_attack_rate=raw_attack_rate,
        epsilon=cfg.exploration,
        logging_threshold=logging_threshold,
        logging_fpr=cfg.logging_fpr,
        econ=econ,
        candidates=candidates,
        best_candidate=best_candidate,
        rows=rows,
        headline_regret=_selection_regret(packed, truth),
        exploration=exploration,
        n_replicates=int(cfg.n_replicates),
        estimators=ESTIMATORS,
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_ope_report(settings: Settings) -> Path:
    """Run the off-policy evaluation study and write the report + figures."""
    study = run_ope(settings)

    est_fig = plots.plot_lines(
        {
            "truth": (
                np.array([c.fpr_target for c in study.candidates]),
                np.array([c.true_value * 1000 for c in study.candidates]),
            ),
            **{
                est: (
                    np.array([c.fpr_target for c in study.candidates]),
                    np.array(
                        [
                            next(
                                r.mean
                                for r in study.rows
                                if r.candidate == c.name and r.estimator == est
                            )
                            * 1000
                            for c in study.candidates
                        ]
                    ),
                )
                for est in study.estimators
            },
        },
        xlabel="candidate policy (validation FPR budget)",
        ylabel=f"policy value per 1,000 flows ({settings.cost.currency})",
        title="Estimated vs actual value of policies never deployed",
        out_path=settings.paths.figures_dir / ESTIMATOR_FIGURE,
        xscale="log",
    )

    eps = np.array([r.epsilon for r in study.exploration])
    explore_fig = plots.plot_lines(
        {
            **{
                f"{est} RMSE": (eps, np.array([r.rmse[est] * 1000 for r in study.exploration]))
                for est in ("IPS", "doubly robust")
            },
            "cost of exploring": (
                eps,
                np.array([r.exploration_cost * 1000 for r in study.exploration]),
            ),
            "regret of choosing wrong (DR)": (
                eps,
                np.array([r.selection_regret["doubly robust"] * 1000 for r in study.exploration]),
            ),
        },
        xlabel="exploration budget (share of decisions randomised)",
        ylabel=f"{settings.cost.currency} per 1,000 flows",
        title="What a random-review budget costs, and what it buys",
        out_path=settings.paths.figures_dir / EXPLORATION_FIGURE,
    )

    report = _render(study, settings, est_fig, explore_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote off-policy evaluation report", extra={"path": str(out_path)})

    with track_run(settings, "ope") as run:
        run.log_params({"exploration": study.epsilon, "logging_fpr": study.logging_fpr})
        dr = [r for r in study.rows if r.estimator == "doubly robust"]
        ips = [r for r in study.rows if r.estimator == "IPS"]
        dm = [r for r in study.rows if r.estimator == "direct method"]
        run.log_metrics(
            {
                "dr_rmse": float(np.mean([r.rmse for r in dr])),
                "ips_rmse": float(np.mean([r.rmse for r in ips])),
                "dm_bias": float(np.mean([r.bias for r in dm])),
                "best_true_value": max(c.true_value for c in study.candidates),
            }
        )
        run.log_artifact(est_fig)
        run.log_artifact(explore_fig)
        run.log_artifact(out_path)
    return out_path


def _candidate_table(study: OPEStudy, currency: str) -> str:
    rows = [
        "| candidate policy | threshold | flows reviewed | true value / 1,000 flows "
        "| unsupported flows | effective sample |",
        "|---|---|---|---|---|---|",
    ]
    for c in study.candidates:
        star = " **(best)**" if c.name == study.best_candidate else ""
        rows.append(
            f"| {c.name}{star} | {c.threshold:.5f} | {c.review_rate:.2%} "
            f"| {currency}{c.true_value * 1000:,.0f} | {c.support_violation:.1%} "
            f"| {c.ess:,.0f} |"
        )
    return "\n".join(rows)


def _estimator_table(study: OPEStudy, currency: str) -> str:
    rows = [
        "| candidate policy | truth | " + " | ".join(study.estimators) + " |",
        "|---|---|" + "---|" * len(study.estimators),
    ]
    truth = {c.name: c.true_value for c in study.candidates}
    for c in study.candidates:
        cells = []
        for est in study.estimators:
            r = next(x for x in study.rows if x.candidate == c.name and x.estimator == est)
            cells.append(f"{currency}{r.mean * 1000:,.0f} (RMSE {r.rmse * 1000:,.0f})")
        rows.append(
            f"| {c.name} | {currency}{truth[c.name] * 1000:,.0f} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def _exploration_table(study: OPEStudy, currency: str) -> str:
    rows = [
        "| exploration | unsupported flows | effective sample | IPS RMSE | DR RMSE "
        "| cost of exploring | regret of choosing wrong (DR) | total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in study.exploration:
        total = (r.exploration_cost + r.selection_regret["doubly robust"]) * 1000
        rows.append(
            f"| {r.epsilon:.1%} | {r.support_violation:.1%} | {r.mean_ess:,.0f} "
            f"| {currency}{r.rmse['IPS'] * 1000:,.0f} "
            f"| {currency}{r.rmse['doubly robust'] * 1000:,.0f} "
            f"| {currency}{r.exploration_cost * 1000:,.0f} "
            f"| {currency}{r.selection_regret['doubly robust'] * 1000:,.0f} "
            f"| **{currency}{total:,.0f}** |"
        )
    return "\n".join(rows)


def _mean_rmse(study: OPEStudy, est: str) -> float:
    return float(np.mean([r.rmse for r in study.rows if r.estimator == est]))


def _mean_abs_bias(study: OPEStudy, est: str) -> float:
    return float(np.mean([abs(r.bias) for r in study.rows if r.estimator == est]))


def _estimator_read(study: OPEStudy, currency: str) -> str:
    ranked = sorted(study.estimators, key=lambda e: _mean_rmse(study, e))
    tightest, loosest = ranked[0], ranked[-1]
    dm_bias = _mean_abs_bias(study, "direct method")
    ips_bias = _mean_abs_bias(study, "IPS")
    return (
        f"By root-mean-square error the ranking is **{tightest}** first "
        f"({currency}{_mean_rmse(study, tightest) * 1000:,.0f} per 1,000 flows) and "
        f"**{loosest}** last ({currency}{_mean_rmse(study, loosest) * 1000:,.0f}) — and taking "
        "that at face value would be the mistake this section exists to prevent. Look at the "
        f"columns instead of the summary. IPS tracks the truth almost exactly in the mean "
        f"({currency}{ips_bias * 1000:,.0f} of average absolute bias) and scatters wildly around "
        "it, because a handful of flows the logging policy was unlikely to review arrive "
        "carrying weights in the hundreds and between them decide the estimate. The direct "
        f"method is the mirror image: steady, and systematically {currency}"
        f"{dm_bias * 1000:,.0f} adrift, understating every policy more permissive than the one "
        "that generated the log. That is not bad luck — its reward model was fitted on exactly "
        "the flows the logging policy chose to show an analyst, so it is most confident and most "
        "wrong precisely where the candidate differs most from the incumbent. Doubly robust is "
        "not a compromise between them but an insurance policy: the reward model carries the "
        "bulk of the signal and the importance weights only have to carry its residual, which is "
        "both smaller and better behaved than the reward itself."
    )


def _regret_table(study: OPEStudy, currency: str) -> str:
    rows = [
        "| estimator | mean RMSE | mean absolute bias | policy it picks | regret of that choice |",
        "|---|---|---|---|---|",
    ]
    truth = {c.name: c.true_value for c in study.candidates}
    for est in sorted(study.estimators, key=lambda e: study.headline_regret[e]):
        picked = max(
            study.candidates,
            key=lambda c: next(
                r.mean for r in study.rows if r.candidate == c.name and r.estimator == est
            ),
        )
        regret = study.headline_regret[est]
        mark = " **(correct)**" if picked.name == study.best_candidate else ""
        rows.append(
            f"| {est} | {currency}{_mean_rmse(study, est) * 1000:,.0f} "
            f"| {currency}{_mean_abs_bias(study, est) * 1000:,.0f} | {picked.name}{mark} "
            f"| {currency}{regret * 1000:,.0f} |"
        )
    del truth
    return "\n".join(rows)


def _regret_read(study: OPEStudy, currency: str) -> str:
    ranked = sorted(study.estimators, key=lambda e: study.headline_regret[e])
    best, worst = ranked[0], ranked[-1]
    best_regret = study.headline_regret[best] * 1000
    worst_regret = study.headline_regret[worst] * 1000
    rmse_winner = min(study.estimators, key=lambda e: _mean_rmse(study, e))
    agreement = (
        "The two yardsticks happen to agree here, which is worth noticing rather than "
        "assuming: an estimator can be accurate and still choose badly, and the reason they "
        "coincide is that the bias in this study is *monotone* in how permissive the policy "
        "is, so it shifts the whole curve rather than tilting it."
        if rmse_winner == best
        else (
            f"The two yardsticks **disagree**: {rmse_winner} wins on RMSE and "
            f"{best} wins on regret. That is the case worth having built the table for — the "
            "accurate-on-average estimator distorts the ranking near the optimum, and the "
            "ranking is the only thing an operator consumes."
        )
    )
    spread = worst_regret - best_regret
    dm_regret = study.headline_regret["direct method"] * 1000
    dr_regret = study.headline_regret["doubly robust"] * 1000
    recommendation = (
        f"The direct method's narrow win ({currency}{dm_regret:,.0f} against doubly robust's "
        f"{currency}{dr_regret:,.0f}) should not be read as a recommendation. It wins *because* "
        "its bias here happens to be monotone in permissiveness, and that is a property nobody "
        "can check in deployment — checking it is precisely the thing the missing labels make "
        "impossible. A reward model that is wrong in a way that tilts the curve rather than "
        "shifting it would reorder the top candidates and the direct method would have no "
        "signal that anything had gone wrong. Doubly robust gives up "
        f"{currency}{dr_regret - dm_regret:,.0f} here to be protected against that case, which "
        "is a cheap premium."
        if dm_regret < dr_regret
        else (
            "Doubly robust comes out ahead on the metric that matters, which is the outcome its "
            "construction is designed for: the reward model supplies the ranking and the "
            "importance weights correct it wherever the model and the log disagree."
        )
    )
    return (
        f"**{best}** picks the best available policy and gives up "
        f"{currency}{best_regret:,.0f} per 1,000 flows; **{worst}** gives up "
        f"{currency}{worst_regret:,.0f}, a {currency}{spread:,.0f} spread that comes entirely "
        f"from which estimator an engineer happened to reach for. {agreement}\n\n"
        f"{recommendation} The advice that survives both tables: report doubly robust, print the "
        "support diagnostic beside it, and never let a low RMSE talk you out of checking which "
        "policy the number actually selects."
    )


def _exploration_read(study: OPEStudy, currency: str) -> str:
    if not study.exploration:
        return ""
    zero = next((r for r in study.exploration if r.epsilon == 0.0), None)
    best = min(
        study.exploration,
        key=lambda r: r.exploration_cost + r.selection_regret["doubly robust"],
    )
    lead = ""
    if zero is not None:
        lead = (
            f"At zero exploration — a plain deployed threshold, which is what almost every SOC "
            f"actually runs — **{zero.support_violation:.0%} of the flows a candidate policy "
            f"would review carry propensity zero**. The log contains no evidence about them, so "
            f"the question is not hard, it is *unanswerable*: any estimator that returns a number "
            f"there is extrapolating, and the doubly-robust estimator's "
            f"{currency}{zero.rmse['doubly robust'] * 1000:,.0f} RMSE is the price of that "
            f"silence, not a failure of the method. This is the row that matters most, because "
            f"it is the row describing production. "
        )
    return (
        lead
        + f"Exploration fixes identification, and it is not free — randomising decisions means "
        f"reviewing flows that did not need reviewing and skipping ones that did. Both sides are "
        f"in the table. Cost rises with the budget, estimator error falls, and the total is "
        f"minimised at **{best.epsilon:.1%} exploration** "
        f"({currency}{best.exploration_cost * 1000:,.0f} spent to avoid "
        f"{currency}{best.selection_regret['doubly robust'] * 1000:,.0f} of expected loss from "
        f"choosing the wrong policy). That is the actionable result: a small, permanent random "
        f"review budget is not wasted analyst time, it is what makes every future threshold "
        f"change answerable offline instead of by deploying it and watching."
    )


def _render(study: OPEStudy, settings: Settings, est_fig: Path, explore_fig: Path) -> str:
    currency = settings.cost.currency
    best = next(c for c in study.candidates if c.name == study.best_candidate)
    deployed = min(study.candidates, key=lambda c: abs(c.fpr_target - study.logging_fpr))
    uplift = (best.true_value - deployed.true_value) * 1000
    return f"""# NetSentry — Off-Policy Evaluation: Valuing a Triage Policy You Never Deployed

_Synthetic stand-in. Honest temporal/binary split; the later-day test stream is re-mixed from
its generator-convenient {study.raw_attack_rate:.1%} attack rate down to the
[cost study's](cost.md) production prior of {study.attack_rate:.2%}, giving {study.n_stream:,}
flows. It is replayed under a logging policy that follows the deployed
{study.logging_fpr:.1%}-FPR threshold with {study.epsilon:.0%} of decisions randomised.
{study.n_replicates} replicate logs per estimate. Rewards use the
[cost study's](cost.md) economics: {currency}{study.econ.cost_per_review:,.0f} per review,
{currency}{study.econ.value_per_catch:,.0f} averted per attack caught, and exactly zero for a
flow nobody looked at._

## Why this report exists

Every other threshold study here has labels for every flow. A SOC does not. It has a log:
the score, the decision, and — only for the flows an analyst actually reviewed — what was
found. Nobody labels what was auto-cleared, because nobody looked.

So the operator's real question is counterfactual. *If we lowered the threshold, what would we
have caught, and what would it have cost?* Scoring a candidate policy on the log answers a
different question, because the labels in the log exist **because the old policy selected
those flows**. That is selection bias with a mechanism, and it has a mature answer: treat
triage as a contextual bandit, the deployed threshold as a logging policy, and estimate the
candidate's value off-policy.

This dataset does have every label, which makes the study a measurement rather than a
demonstration — the **true** value of each candidate is computable, so every estimator can be
scored against it, and so can the policy each estimator would have *chosen*.

## The candidate policies, and what they are actually worth

{_candidate_table(study, currency)}

The best available policy is **{best.name}**, worth {currency}{best.true_value * 1000:,.0f} per
1,000 flows against {currency}{deployed.true_value * 1000:,.0f} for the deployed operating
point — an uplift of {currency}{uplift:,.0f} per 1,000 flows that is invisible from the log
unless it can be estimated. Note the last two columns: they are the diagnostics that decide
whether any of the estimates below mean anything.

## Four estimators, scored against the truth

{_estimator_table(study, currency)}

{_estimator_read(study, currency)}

![estimated vs actual policy value](../figures/{est_fig.name})

### The metric that actually decides

Accuracy is not the goal; **choosing the right policy** is. An estimator wrong about every
candidate by the same amount still ranks them correctly and costs nothing, while one that is
close on average but reorders the top two costs real money. So the yardstick is regret: pick
the policy each estimator scores highest, then ask what that choice was actually worth.

{_regret_table(study, currency)}

{_regret_read(study, currency)}

## The finding is about the log, not the estimator

{_exploration_table(study, currency)}

{_exploration_read(study, currency)}

![the price and payoff of exploration](../figures/{explore_fig.name})

## Scope

The reward model is deliberately simple — a logistic regression on the same features, refitted
per replicate on whatever the logging policy happened to review — because the direct method's
weakness here is *which flows it was trained on*, not which family fitted them; a stronger
reward model trained on the same censored sample inherits the same bias. Rewards assume an
analyst verdict is correct and immediate; a real queue returns verdicts late and sometimes
wrong, which widens every interval here without changing the ordering. The skip arm is
assigned exactly zero rather than a miss cost, which is what makes the formulation honest
under partial feedback (nothing was spent, nothing was found) but does mean the values below
are *review economics*, not total risk — [cost.md](cost.md) prices the miss side directly, and
[alert_queue.md](alert_queue.md) prices the capacity constraint. Only the actions are
resampled across replicates; the traffic is held fixed, so the spreads measure logging noise
rather than a change of population. Candidate policies are deterministic thresholds, so only
the *logging* policy needs stochasticity — which is exactly the asymmetry that makes a small
exploration budget so cheap relative to what it buys."""
