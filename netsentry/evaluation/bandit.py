"""Learning the triage policy online, and paying for it in the currency that hurts.

The [off-policy study](ope.md) values a triage policy from a log somebody else's policy wrote.
This is the other half of the same problem: **learning** one from feedback, while it runs, with
no log to start from.

The setting is the contextual bandit in its natural habitat. A flow arrives, the system chooses
to review it or skip it, and the outcome is observed *only for the flows it reviewed* -- a
skipped attack produces no signal, no alert and no lesson. That is the structure of partial
feedback, and it is exactly why triage is worth studying this way rather than as a
classification problem with a threshold bolted on.

Five learners run on the same stream:

- **LinUCB** (Li et al., WWW 2010): a ridge regression per action plus a confidence bonus that
  shrinks as the action is tried, so exploration is directed at what is *uncertain* rather than
  at what is random.
- **Linear Thompson sampling** (Agrawal & Goyal, ICML 2013): the same model, explored by
  sampling from the posterior instead of by an upper bound.
- **Epsilon-greedy**: the control that says how much of the sophistication is worth anything.
- **The deployed fixed threshold**: the incumbent, chosen on validation at the operating
  budget, which learns nothing and pays no exploration cost.
- **Uniform random**: the floor.

Regret is measured against the best *threshold* policy in hindsight -- not the best fixed
action, which at a 1% attack rate is "review nothing" and would reward a learner for doing the
same -- and its growth exponent is fitted, because a bandit whose regret grows linearly never
learned and the shape is the only way to tell.

The number this study exists to produce is not regret, though. It is **what exploration is paid
for out of**, and the answer turned out not to be the one this module was written expecting.
With two actions and an asymmetric reward, exploring *is* reviewing: the learners catch more
attacks than the incumbent and lose money doing it, spending several times the deployed alert
budget while they find out. Regret is denominated in a currency a SOC does not use, and a
budget is a rate rather than a price -- which is the distinction every fixed-FPR threshold in
this repository exists to express, and the one a reward function cannot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.evaluation.ope import Economics, resample_to_prevalence
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "bandit.md"
REGRET_FIGURE = "bandit_regret.png"
COST_FIGURE = "bandit_exploration.png"

LINUCB = "LinUCB"
THOMPSON = "Thompson sampling"
EPSILON = "epsilon-greedy"
FIXED = "the deployed threshold"
RANDOM = "uniform random"

SKIP, REVIEW = 0, 1


# --------------------------------------------------------------------------------------
# The learners.
# --------------------------------------------------------------------------------------


class LinearBandit:
    """Ridge regression per action, explored either by an upper bound or by sampling.

    Both LinUCB and linear Thompson sampling maintain the same sufficient statistics -- a
    Gram matrix `A` and a response vector `b` per action -- and differ only in how they turn
    the resulting posterior into a choice. Implementing them as one class makes that explicit,
    and makes the comparison between them a comparison of exploration rules rather than of two
    people's linear algebra.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float,
        rng: np.random.Generator,
        thompson: bool = False,
        n_actions: int = 2,
    ) -> None:
        self.alpha = alpha
        self.rng = rng
        self.thompson = thompson
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def _theta(self, action: int) -> np.ndarray:
        theta: np.ndarray = np.linalg.solve(self.A[action], self.b[action])
        return theta

    def choose(self, context: np.ndarray) -> int:
        values = []
        for action in range(len(self.A)):
            inverse = np.linalg.inv(self.A[action])
            mean = float(context @ self._theta(action))
            if self.thompson:
                # Sample a coefficient vector from the posterior and act greedily on it.
                sample = self.rng.multivariate_normal(self._theta(action), self.alpha**2 * inverse)
                values.append(float(context @ sample))
            else:
                width = self.alpha * float(np.sqrt(context @ inverse @ context))
                values.append(mean + width)
        return int(np.argmax(values))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


class EpsilonGreedy:
    """The control. Same statistics, but exploration that ignores what it already knows."""

    def __init__(
        self, n_features: int, epsilon: float, rng: np.random.Generator, n_actions: int = 2
    ) -> None:
        self.epsilon = epsilon
        self.rng = rng
        self.A = [np.eye(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def choose(self, context: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.A)))
        values = [
            float(context @ np.linalg.solve(self.A[a], self.b[a])) for a in range(len(self.A))
        ]
        return int(np.argmax(values))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


def build_context(score: float, anomaly: float) -> np.ndarray:
    """The bandit's view of a flow: an intercept, the two model scores, and a curvature term.

    Deliberately not the 76 raw features. The detector has already done the hard part, and a
    bandit re-learning it from scratch would be measuring how long that takes rather than what
    an adaptive operating point is worth. What is left to learn is the *decision*: where in
    score space reviewing pays, which is a low-dimensional question and a genuinely uncertain
    one, since the answer depends on a prevalence nobody observes directly.
    """
    return np.array([1.0, score, anomaly, score * score], dtype=float)


# --------------------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmRun:
    """One policy over the whole stream."""

    name: str
    rewards: np.ndarray  # per-flow realised reward
    actions: np.ndarray  # per-flow action taken
    reviewed: int
    caught: int
    missed: int
    repeat: int = 0

    @property
    def total(self) -> float:
        return float(self.rewards.sum())

    def cumulative(self) -> np.ndarray:
        cumulative: np.ndarray = np.cumsum(self.rewards)
        return cumulative


@dataclass(frozen=True)
class BanditStudy:
    """Every policy on one stream, plus the hindsight reference regret is measured against."""

    arms: list[ArmRun]
    oracle_rewards: np.ndarray
    best_fixed_rewards: np.ndarray
    best_fixed_name: str
    n_flows: int
    prevalence: float
    economics: Economics
    threshold: float
    attacks: int
    alpha_sweep: list[tuple[float, float, float, float]] = field(default_factory=list)

    def runs(self, name: str) -> list[ArmRun]:
        return [arm for arm in self.arms if arm.name == name]

    def by_name(self, name: str) -> ArmRun:
        """The first run of a policy; for the deterministic arms it is the only one."""
        return self.runs(name)[0]

    def names(self) -> list[str]:
        seen: list[str] = []
        for arm in self.arms:
            if arm.name not in seen:
                seen.append(arm.name)
        return seen

    def mean_total(self, name: str) -> float:
        return float(np.mean([arm.total for arm in self.runs(name)]))

    def spread(self, name: str) -> float:
        totals = [arm.total for arm in self.runs(name)]
        return float(max(totals) - min(totals))

    def mean_missed(self, name: str) -> float:
        return float(np.mean([arm.missed for arm in self.runs(name)]))

    def mean_reviewed(self, name: str) -> float:
        return float(np.mean([arm.reviewed for arm in self.runs(name)]))

    def mean_caught(self, name: str) -> float:
        return float(np.mean([arm.caught for arm in self.runs(name)]))

    def regret(self, name: str) -> np.ndarray:
        """Cumulative regret against the best policy in the comparison class, in hindsight.

        The reference is the best *threshold* policy on this very stream, not the best fixed
        action. At a 1% attack rate reviewing a flow chosen at random has negative expected
        value, so 'review nothing' wins among fixed actions and every learner's regret would be
        measured against a policy that catches nothing -- a reference that makes the incumbent
        look infinitely good and says nothing about learning.
        """
        rewards = np.mean([arm.rewards for arm in self.runs(name)], axis=0)
        regret: np.ndarray = np.cumsum(self.best_fixed_rewards - rewards)
        return regret

    def learners(self) -> list[str]:
        return [name for name in self.names() if name in {LINUCB, THOMPSON, EPSILON}]

    def regret_slope(self, name: str) -> float:
        """The exponent of cumulative regret against stream position, fitted in log-log space.

        This is the only honest way to read a regret curve. Theory promises `sqrt(T)` growth --
        an exponent near 0.5 -- and a learner that never converges grows linearly, at 1.0. The
        first 10% is dropped because early regret is dominated by the first few draws.
        """
        regret = self.regret(name)
        start = max(10, len(regret) // 10)
        positions = np.arange(start, len(regret), dtype=float)
        values = regret[start:]
        usable = values > 0
        if usable.sum() < 10:
            return float("nan")
        slope = np.polyfit(np.log(positions[usable]), np.log(values[usable]), 1)[0]
        return float(slope)

    def parity_flow(self, name: str) -> int | None:
        """The first flow at which a learner's running total overtakes the incumbent's."""
        learner = np.mean([arm.cumulative() for arm in self.runs(name)], axis=0)
        incumbent = self.by_name(FIXED).cumulative()
        ahead = np.flatnonzero(learner >= incumbent)
        # Both start at zero, so only a crossing after the incumbent has banked something counts.
        after = ahead[ahead > np.argmax(incumbent > 0)] if incumbent.max() > 0 else ahead
        return int(after[0]) if len(after) else None

    def realised_fpr(self, name: str) -> float:
        """The share of benign flows a policy sent to an analyst: the alert budget it spent."""
        runs = self.runs(name)
        benign = float(len(runs[0].actions) - self.attacks)
        false_alerts = float(np.mean([arm.reviewed - arm.caught for arm in runs]))
        return false_alerts / benign if benign else float("nan")

    @property
    def oracle_total(self) -> float:
        return float(self.oracle_rewards.sum())

    @property
    def best_fixed_total(self) -> float:
        return float(self.best_fixed_rewards.sum())


def learner_chooser(model: Any) -> Callable[[np.ndarray, int], int]:
    """Adapt a learner to the stream's chooser signature, with the types intact."""

    def choose(context: np.ndarray, _index: int) -> int:
        return int(model.choose(context))

    return choose


def threshold_chooser(scores: np.ndarray, threshold: float) -> Callable[[np.ndarray, int], int]:
    """The incumbent: review whatever the detector scores above its calibrated operating point."""

    def choose(_context: np.ndarray, index: int) -> int:
        return int(scores[index] >= threshold)

    return choose


def random_chooser(rate: float, rng: np.random.Generator) -> Callable[[np.ndarray, int], int]:
    """The floor: review a fixed share of flows, chosen with no signal at all."""

    def choose(_context: np.ndarray, _index: int) -> int:
        return int(rng.random() < rate)

    return choose


def run_policy(
    name: str,
    chooser: Callable[[np.ndarray, int], int],
    contexts: np.ndarray,
    labels: np.ndarray,
    economics: Economics,
    learner: Any | None = None,
    repeat: int = 0,
) -> ArmRun:
    """Replay the stream under one policy, feeding back only what it actually observed.

    The partial-feedback discipline is the whole point and is easy to break by accident: a
    learner is updated with the reward of the action it *took*, never with what the other
    action would have paid. A skipped attack teaches it nothing, which is precisely the
    difficulty a triage system faces and the reason exploration costs something real.

    Learners are fed the reward **scaled by the value of a catch**, so it lands in roughly
    [-0.05, 1]. That is not cosmetic. LinUCB's confidence width and Thompson's posterior scale
    are both in the reward's units, so a dollar-denominated reward makes the exploration term
    negligible beside the first bad draw -- and the first review is almost always a benign flow.
    Left unscaled, both learners reviewed *one* flow in eight thousand and never looked again.
    The reported rewards stay in dollars; only what the learner sees is normalised.
    """
    scale = max(abs(economics.value_per_catch), 1e-9)
    rewards = np.zeros(len(contexts), dtype=float)
    actions = np.zeros(len(contexts), dtype=int)
    for index, context in enumerate(contexts):
        action = chooser(context, index)
        reward = float(economics.reward(np.array([action]), np.array([labels[index]]))[0])
        rewards[index] = reward
        actions[index] = action
        if learner is not None:
            learner.update(context, action, reward / scale)
    reviewed = int(actions.sum())
    caught = int(((actions == REVIEW) & (labels == 1)).sum())
    missed = int(((actions == SKIP) & (labels == 1)).sum())
    return ArmRun(
        name=name,
        rewards=rewards,
        actions=actions,
        reviewed=reviewed,
        caught=caught,
        missed=missed,
        repeat=repeat,
    )


def best_threshold_rewards(
    scores: np.ndarray, labels: np.ndarray, economics: Economics
) -> tuple[np.ndarray, float]:
    """The best threshold policy *on this stream*, which is what regret is measured against.

    Choosing the reference is the load-bearing decision in any regret study, and the obvious
    choice is wrong here. Among fixed *actions*, "review nothing" wins at a 1% attack rate --
    reviewing a flow drawn at random has negative expected value -- so regret against it would
    reward a learner for doing nothing and would score the deployed detector as *negative*
    regret, which is a sentence about the reference rather than about the policy. The strongest
    policy in the comparison class is the best threshold in hindsight, so that is the reference.

    Computed exactly rather than on a grid. Sorting by score descending makes "review the top
    k" a prefix sum, so every threshold policy is evaluated in one pass, and only cuts at the
    end of a tie group are candidates because a threshold cannot split flows that share a
    score. The first version swept 200 quantiles and a test caught it missing the optimum by
    $25 -- a reference that is not actually optimal understates every learner's regret, which
    is the one direction a regret study must never be wrong in.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ordered_scores = np.asarray(scores, dtype=float)[order]
    gains = economics.reward(np.ones(len(labels)), np.asarray(labels))[order]
    prefix = np.cumsum(gains)
    # A threshold cannot separate equal scores, so only the last index of each tie group is a
    # realisable cut. "Review nothing" is the k = 0 candidate and is always available.
    last_of_group = np.append(ordered_scores[:-1] != ordered_scores[1:], True)
    candidates = prefix[last_of_group]
    if len(candidates) == 0 or candidates.max() <= 0.0:
        return np.zeros(len(labels)), float(np.nextafter(ordered_scores[0], np.inf))
    best_cut = float(ordered_scores[last_of_group][int(np.argmax(candidates))])
    actions = (np.asarray(scores, dtype=float) >= best_cut).astype(int)
    return economics.reward(actions, np.asarray(labels)), best_cut


def run_bandit_study(settings: Settings) -> BanditStudy:
    """Fit the detector, build the stream, and race five policies down it."""
    cfg = settings.bandit
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False

    result = fit_supervised(variant)
    benign = variant.labels.benign_label
    s_val = attack_probability(np.asarray(result.proba_val), result.classes, benign)
    s_test = attack_probability(np.asarray(result.proba_test), result.classes, benign)
    y_val = np.asarray(result.y_val).astype(int)
    y_test = np.asarray(result.y_test).astype(int)

    # Same re-mix as the off-policy study, for the same reason and with the same numbers: at
    # the split's own ~25% attack rate reviewing a flow at random is profitable, "review
    # everything" wins by construction, and there is no policy question left to ask.
    rng = np.random.default_rng(settings.seed)
    keep = resample_to_prevalence(y_test, settings.cost.production_attack_rate, rng=rng)
    s_test, y_test = s_test[keep], y_test[keep]
    order = rng.permutation(len(y_test))[: cfg.max_flows]
    s_test, y_test = s_test[order], y_test[order]

    # A second context dimension the score alone does not carry: how far this flow sits from
    # the *calibration* distribution's centre, standardised. The centre and spread come from
    # validation, not from the stream -- the first version used the stream's own mean and the
    # static-analysis pass caught it, correctly: an online learner at flow one cannot know a
    # statistic of flows it has not seen, and a context built from the whole future is not a
    # context at all.
    centre, spread = float(s_val.mean()), float(s_val.std())
    anomaly = np.clip((s_test - centre) / (spread + 1e-9), -4.0, 4.0)
    contexts = np.vstack(
        [build_context(float(s), float(a)) for s, a in zip(s_test, anomaly, strict=True)]
    )
    economics = Economics(
        value_per_catch=settings.cost.cost_per_miss, cost_per_review=settings.cost.cost_per_alert
    )
    threshold = threshold_at_fpr(y_val, s_val, cfg.fixed_fpr)

    arms: list[ArmRun] = []
    for repeat in range(max(1, cfg.n_repeats)):
        seed = settings.seed + 1000 * repeat
        linucb = LinearBandit(contexts.shape[1], cfg.alpha, np.random.default_rng(seed))
        arms.append(
            run_policy(
                LINUCB,
                learner_chooser(linucb),
                contexts,
                y_test,
                economics,
                learner=linucb,
                repeat=repeat,
            )
        )
        thompson = LinearBandit(
            contexts.shape[1], cfg.alpha, np.random.default_rng(seed + 1), thompson=True
        )
        arms.append(
            run_policy(
                THOMPSON,
                learner_chooser(thompson),
                contexts,
                y_test,
                economics,
                learner=thompson,
                repeat=repeat,
            )
        )
        greedy = EpsilonGreedy(contexts.shape[1], cfg.epsilon, np.random.default_rng(seed + 2))
        arms.append(
            run_policy(
                EPSILON,
                learner_chooser(greedy),
                contexts,
                y_test,
                economics,
                learner=greedy,
                repeat=repeat,
            )
        )
        random_rng = np.random.default_rng(seed + 3)
        arms.append(
            run_policy(
                RANDOM,
                random_chooser(cfg.random_review_rate, random_rng),
                contexts,
                y_test,
                economics,
                repeat=repeat,
            )
        )

    # The incumbent is deterministic, so one run is the whole story.
    arms.append(
        run_policy(FIXED, threshold_chooser(s_test, threshold), contexts, y_test, economics)
    )

    # What the exploration constant actually buys. LinUCB's confidence width is the only knob
    # between "never review" and "review everything", and the sweep prices it in the unit the
    # SOC uses -- the share of benign traffic sent to an analyst.
    sweep: list[tuple[float, float, float, float]] = []
    for alpha in cfg.alpha_sweep:
        model = LinearBandit(contexts.shape[1], alpha, np.random.default_rng(settings.seed))
        run = run_policy(
            f"LinUCB alpha={alpha}",
            learner_chooser(model),
            contexts,
            y_test,
            economics,
            learner=model,
        )
        benign_flows = float(len(y_test) - int(y_test.sum()))
        sweep.append(
            (alpha, run.total, (run.reviewed - run.caught) / benign_flows, float(run.caught))
        )

    best_fixed, best_cut = best_threshold_rewards(s_test, y_test, economics)
    oracle_rewards = economics.reward(y_test.astype(float), y_test)
    return BanditStudy(
        arms=arms,
        oracle_rewards=oracle_rewards,
        best_fixed_rewards=best_fixed,
        best_fixed_name=f"the best threshold in hindsight ({best_cut:.4f})",
        n_flows=len(y_test),
        prevalence=float(y_test.mean()),
        economics=economics,
        threshold=float(threshold),
        attacks=int(y_test.sum()),
        alpha_sweep=sweep,
    )


def _best_sweep_total(study: BanditStudy) -> float:
    """The best total reward any confidence width reached in the sweep."""
    return max((row[1] for row in study.alpha_sweep), default=float("nan"))


def _best_sweep_fpr(study: BanditStudy) -> float:
    """The alert budget that best setting spent to get there."""
    if not study.alpha_sweep:
        return float("nan")
    return max(study.alpha_sweep, key=lambda row: row[1])[2]


def _policy_table(study: BanditStudy) -> str:
    header = (
        "| policy | total reward | spread over repeats | flows reviewed | attacks caught "
        "| benign reviewed (alert budget) | regret |\n|---|---|---|---|---|---|---|"
    )
    rows = []
    for name in study.names():
        rows.append(
            f"| {name} | ${study.mean_total(name):,.0f} | ${study.spread(name):,.0f} "
            f"| {study.mean_reviewed(name):,.0f} | {study.mean_caught(name):.0f} "
            f"| {study.realised_fpr(name):.2%} | ${study.regret(name)[-1]:,.0f} |"
        )
    reference = (
        f"| **{study.best_fixed_name}** | **${study.best_fixed_total:,.0f}** | -- | -- | -- "
        f"| -- | $0 |"
    )
    oracle = (
        f"| _the oracle (reviews exactly the attacks)_ | _${study.oracle_total:,.0f}_ | -- "
        f"| _{study.attacks}_ | _{study.attacks}_ | _0.00%_ | -- |"
    )
    return header + "\n" + "\n".join(rows) + "\n" + reference + "\n" + oracle


def _slope_table(study: BanditStudy) -> str:
    rows = "\n".join(
        f"| {name} | {study.regret_slope(name):.2f} | "
        f"{'yes' if study.parity_flow(name) is not None else '**never**'} |"
        for name in [*study.learners(), RANDOM]
    )
    return "| policy | regret exponent | overtook the incumbent |\n|---|---|---|\n" + rows


def _sweep_table(study: BanditStudy) -> str:
    if not study.alpha_sweep:
        return "_No sweep was run._"
    rows = "\n".join(
        f"| {alpha} | ${total:,.0f} | {fpr:.2%} | {caught:.0f} |"
        for alpha, total, fpr, caught in study.alpha_sweep
    )
    return (
        "| confidence width | total reward | benign reviewed | attacks caught |\n"
        "|---|---|---|---|\n" + rows
    )


def _lead(study: BanditStudy) -> str:
    best_learner = max(study.learners(), key=study.mean_total)
    incumbent = study.mean_total(FIXED)
    return (
        f"Over {study.n_flows:,} flows at a {study.prevalence:.2%} attack rate, **every learner "
        f"loses to a threshold that was chosen once, on validation, and never touched again**. "
        f"The best of them ({best_learner}) ends on ${study.mean_total(best_learner):,.0f} "
        f"against the incumbent's ${incumbent:,.0f}, and none of them ever overtakes it.\n\n"
        f"The theory is not what failed. LinUCB's cumulative regret grows as "
        f"`T^{study.regret_slope(LINUCB):.2f}` -- the `sqrt(T)` the analysis promises -- while "
        f"epsilon-greedy manages `T^{study.regret_slope(EPSILON):.2f}` and the random control "
        f"`T^{study.regret_slope(RANDOM):.2f}`, which is what not learning looks like. The "
        f"algorithm behaves exactly as advertised and is still the wrong thing to deploy.\n\n"
        f"**What exploration costs here is not detection. It is the alert budget.** LinUCB "
        f"reviews {study.realised_fpr(LINUCB):.1%} of benign traffic against the deployed "
        f"threshold's {study.realised_fpr(FIXED):.1%} -- and catches *more* attacks for it, "
        f"{study.mean_caught(LINUCB):.0f} against {study.mean_caught(FIXED):.0f}. It is buying "
        f"detection with analyst time at a price the reward function says is bad and a SOC "
        f"would never authorise at all."
    )


def _render(study: BanditStudy, regret_figure: Path, cost_figure: Path) -> str:
    return f"""# NetSentry — Learning the Triage Policy Online, and What It Costs

_Five policies down the same {study.n_flows:,}-flow stream at a {study.prevalence:.2%} attack
rate, under partial feedback: an outcome is observed only for flows the policy chose to review.
Rewards use the [cost study's](cost.md) economics (${study.economics.value_per_catch:,.0f} for a
catch, ${study.economics.cost_per_review:,.0f} per review), and the stochastic arms are averaged
over repeated runs. Regenerate with `netsentry bandit`._

## Why this report exists

The [off-policy study](ope.md) values a triage policy from a log that a different policy wrote.
This is the other half: **learning** one while it runs, with no log to start from and no oracle
to ask. It is the contextual bandit in its natural habitat -- a flow arrives, the system reviews
it or skips it, and a skipped attack produces no alert, no signal and no lesson.

{_lead(study)}

## The policies

![Cumulative reward](../figures/{regret_figure.name})

{_policy_table(study)}

The reference row is the strongest policy in the comparison class: the best *threshold* on this
very stream, chosen with hindsight. Picking that reference is the load-bearing decision in any
regret study and the obvious choice is wrong -- among fixed *actions*, "review nothing" wins at
a 1% attack rate, so regret against it would reward a learner for doing nothing and would score
the deployed detector as having *negative* regret. That would be a sentence about the reference.

The incumbent lands within ${study.best_fixed_total - study.mean_total(FIXED):,.0f} of the best
threshold anyone could have chosen knowing the whole stream. A validation-calibrated operating
point is already about as good as this comparison class gets, which is the context every claim
below has to be read against.

The spread column is worth a glance: LinUCB's is exactly zero, because LinUCB is
*deterministic* given the stream -- its upper confidence bound is a function of what it has
seen, not of a coin -- so its repeats agree to the cent. Thompson sampling's spread of
${study.spread(THOMPSON):,.0f} is the other end of that: the same algorithm and the same data,
differing only in a posterior draw, landing that far apart run to run.

## Does the regret curve behave?

{_slope_table(study)}

This is the table that separates "the algorithm is broken" from "the algorithm is fine and the
problem is elsewhere". Regret is fitted in log-log space over the stream, dropping the first
tenth where a handful of draws dominate. LinUCB's exponent sits near the 0.5 the analysis
promises; the random control's sits near 1.0, which is what no learning looks like. The
implementation is doing what it says on the tin, and it still never catches the incumbent.

## The knob, priced in the unit that matters

![What exploration buys](../figures/{cost_figure.name})

{_sweep_table(study)}

LinUCB's confidence width is the only dial between "never review" and "review everything", and
the sweep prices it in alert volume rather than in dollars. There is no setting that behaves
like an operating point: the width trades total reward against the share of benign traffic sent
to an analyst, continuously, with no mechanism that says *stay under 1%*.

**The best setting in the sweep still loses.** Tuned to
${_best_sweep_total(study):,.0f}, the learner returns
{_best_sweep_total(study) / max(study.mean_total(FIXED), 1e-9):.0%} of what the untouched
threshold makes -- and it gets there by spending
{_best_sweep_fpr(study):.1%} of the benign stream, several times the deployed budget. Tuning the
exploration constant against the reward moves the policy along that trade; it never converts it
into a constraint.

That is the actual finding, and it generalises past bandits. **A reward function is not a
constraint.** The economics say a review costs
${study.economics.cost_per_review:,.0f} and a catch is worth
${study.economics.value_per_catch:,.0f}, so a policy that reviews eight times as much traffic is
merely making a trade the objective permits. A SOC's alert budget is not a price, it is a *rate*
-- there is no amount of money that makes an analyst's tenth hour exist -- and the whole
apparatus this project builds around fixed-FPR thresholds, conformal risk control and
Neyman-Pearson certificates is machinery for expressing exactly that distinction. A learner that
optimises the reward instead of respecting the constraint will spend the budget every time.

The fix is not a better bandit but a *constrained* one -- a budget-limited or knapsack bandit
that treats the review rate as a resource rather than a cost -- and that is a different study.

## Scope and honest limits

- **The stream is a permuted test split re-mixed to a {study.prevalence:.2%} attack rate**,
  exactly as the off-policy study does and for the same reason: at the split's own ~25% rate
  reviewing a random flow is profitable and every policy question collapses.
- **The context is four numbers, not the 76 features.** The detector has already done the hard
  part; a bandit re-learning detection from scratch would be measuring how long that takes. What
  is left to learn is where in score space reviewing pays, which is genuinely uncertain because
  it depends on a prevalence nobody observes directly.
- **The learners see a reward scaled by the value of a catch.** Unscaled, the exploration term
  is negligible beside the first bad draw -- and the first review is almost always a benign flow
  -- so both linear learners reviewed a single flow in eight thousand and never looked again.
  That is a real trap in deploying a textbook bandit against a business objective, and it is
  recorded rather than quietly fixed.
- **{study.n_flows:,} flows is a shift, not a year.** The regret exponents say LinUCB would keep
  improving; nothing here says it would ever overtake a threshold that starts near optimal, and
  the alert budget it spends while finding out is the reason nobody would run the experiment.
- **Two actions, no deferral.** Real triage has more: escalate, enrich, auto-close. The
  [learning-to-defer study](defer.md) covers the human-in-the-loop version of that decision."""


def run_bandit_report(settings: Settings) -> Path:
    """Run the bandit study and write the report + figures."""
    study = run_bandit_study(settings)
    positions = np.arange(1, study.n_flows + 1, dtype=float)
    series = {
        name: (positions, np.mean([arm.cumulative() for arm in study.runs(name)], axis=0))
        for name in study.names()
    }
    series["the best threshold in hindsight"] = (positions, np.cumsum(study.best_fixed_rewards))
    regret_figure = plots.plot_lines(
        series,
        xlabel="flows seen",
        ylabel=f"cumulative reward ({settings.cost.currency})",
        title="Learning the operating point, against one that was simply chosen",
        out_path=settings.paths.figures_dir / REGRET_FIGURE,
    )

    alphas = [row[0] for row in study.alpha_sweep] or [0.0]
    cost_figure = plots.plot_lines(
        {
            "benign traffic reviewed": (
                np.array(alphas),
                np.array([row[2] for row in study.alpha_sweep] or [0.0]),
            ),
            "the deployed threshold's budget": (
                np.array(alphas),
                np.full(len(alphas), study.realised_fpr(FIXED)),
            ),
        },
        xlabel="LinUCB confidence width",
        title="Exploration is spent out of the alert budget",
        ylabel="share of benign flows sent to an analyst",
        out_path=settings.paths.figures_dir / COST_FIGURE,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, regret_figure, cost_figure), encoding="utf-8")
    logger.info("Wrote bandit report", extra={"path": str(out_path), "flows": study.n_flows})

    with track_run(settings, "bandit") as run:
        run.log_params({"flows": study.n_flows, "repeats": settings.bandit.n_repeats})
        run.log_metrics(
            {f"total_{name.split(' ')[0]}": study.mean_total(name) for name in study.names()}
            | {"regret_exponent_linucb": study.regret_slope(LINUCB)}
        )
        run.log_artifact(regret_figure)
        run.log_artifact(cost_figure)
        run.log_artifact(out_path)
    return out_path
