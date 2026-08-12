"""The arms race, as a game: what detection survives an attacker who adapts to the defence?

The evasion study measures a **one-shot** attack: train the detector, then let an attacker move
flows toward benign until they slip past. The hardening study answers it with one round of
adversarial training and re-measures. Both are honest and both stop one move too early, because
a real adversary sees the fix and moves again — and the number a defender actually needs is not
"how much does the attack cost me now" but "where does this end".

Framed as a game, three things become computable that the sequential studies cannot express.

**The attacker has a cost, and it is not the L2 norm.** Evading a flow-behaviour detector means
making the flow look benign, and a flow that looks benign *is* less of an attack: pad the
inter-arrival times of a DoS and it stops being a denial of service. The utility here is
therefore `(1 - detection) * (1 - fraction)` — the probability of getting through, times how
much attack is left when you do. That single term is what stops the attacker's best response
from being "mimic benign completely", and it is why the equilibrium sits somewhere interesting.

**The arms race can be simulated instead of assumed.** Starting from the clean model, each round
the attacker best-responds to the deployed detector and the defender retrains on the attack it
just saw. The trajectory either converges, cycles, or ratchets — and which of those happens is a
property of the model class and the attack surface, not something to be argued about.

**Commitment is worth something, and it can be priced.** A defender who moves first and knows
the attacker will best-respond does not want the model that is best against *today's* attack;
they want the model whose worst case, after the attacker re-optimises, is best. That is a
Stackelberg equilibrium (von Stackelberg 1934; Brückner & Scheffer 2011 for the classifier case),
and the gap between it and the myopic arms-race outcome is the value of thinking one move ahead
— reported here as a number rather than as advice.

The headline result is a **negative one, and it is kept**: at every operating point swept, the
attacker's utility-maximising move is to do *nothing*. A detector catching 8.9% of attacks is
already letting 91% through with the attack fully intact, and no disguise on offer buys more
evasion than it costs in attack value. That is arithmetic rather than a quirk of the utility
function — mimicry at fraction `f` only pays if it cuts detection by more than roughly `f` —
and it inverts the usual framing: evasion resistance is not a property to buy before the
detector works, it is a problem you *earn* by making the detector good enough to be worth
attacking. Because that conclusion rests entirely on what the disguise costs, the assumption is
swept rather than defended, and the exponent at which it flips is reported as the condition the
claim depends on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.feature_sets import numeric_features
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.evasion import controllable_indices, mimicry_perturb
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import StrategicConfig

logger = get_logger(__name__)

REPORT_NAME = "strategic.md"
FIGURE_NAME = "strategic_equilibrium.png"


# --------------------------------------------------------------------------------------
# Game-theoretic primitives. Pure functions over a payoff matrix, so they are testable
# without touching a model.
# --------------------------------------------------------------------------------------


def attacker_utility(detection: float, fraction: float, effectiveness_exponent: float) -> float:
    """Expected value to the attacker of mimicking benign traffic by `fraction`.

    `(1 - detection) * (1 - fraction)^k`: the chance of getting through, times how much attack
    survives the disguise. Without the second term the attacker's best response is always total
    mimicry, which is both trivial and wrong — a flow indistinguishable from benign traffic is
    not carrying out an attack. `k` controls how fast effectiveness decays; `k = 1` is linear.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    return float((1.0 - detection) * max(0.0, 1.0 - fraction) ** effectiveness_exponent)


def best_response(
    detections: np.ndarray, fractions: np.ndarray, effectiveness_exponent: float
) -> tuple[int, float]:
    """The attacker's utility-maximising mimicry level against one fixed defence.

    Returns `(index, utility)`. Ties go to the *smaller* fraction, since an attacker
    indifferent between two disguises prefers the cheaper one.
    """
    utilities = np.array(
        [
            attacker_utility(float(d), float(f), effectiveness_exponent)
            for d, f in zip(detections, fractions, strict=True)
        ]
    )
    return int(np.argmax(utilities)), float(np.max(utilities))


def stackelberg_solution(
    payoff: np.ndarray, fractions: np.ndarray, effectiveness_exponent: float
) -> tuple[int, int, float]:
    """The defender's best commitment, anticipating the attacker's best response.

    `payoff[i, j]` is detection when defence `i` meets attack `j`. The defender moves first and
    is not choosing the row that is best against today's attack — they are choosing the row whose
    detection is highest *after* the attacker re-optimises against it. Returns
    `(defence_index, attack_index, detection)`.
    """
    best_row, best_col, best_detection = 0, 0, -np.inf
    for i in range(payoff.shape[0]):
        col, _ = best_response(payoff[i], fractions, effectiveness_exponent)
        if payoff[i, col] > best_detection:
            best_row, best_col, best_detection = i, col, float(payoff[i, col])
    return best_row, best_col, best_detection


def pure_nash(
    payoff: np.ndarray, fractions: np.ndarray, effectiveness_exponent: float
) -> list[tuple[int, int]]:
    """Cells where neither side wants to move — the defender maximises detection, the attacker
    maximises utility, and each is already best-responding to the other."""
    equilibria = []
    for i in range(payoff.shape[0]):
        for j in range(payoff.shape[1]):
            attacker_ok = best_response(payoff[i], fractions, effectiveness_exponent)[0] == j
            defender_ok = int(np.argmax(payoff[:, j])) == i
            if attacker_ok and defender_ok:
                equilibria.append((i, j))
    return equilibria


def arms_race(
    payoff: np.ndarray,
    fractions: np.ndarray,
    effectiveness_exponent: float,
    rounds: int,
    start_defence: int = 0,
) -> list[tuple[int, int, float]]:
    """Simulate myopic alternating best responses: `(defence, attack, detection)` per round.

    Each round the attacker best-responds to the deployed defence and the defender then adopts
    whichever defence is strongest against *that* attack. Nobody looks ahead, which is exactly
    what makes the trajectory worth plotting: it either settles, or it cycles forever between
    two configurations that each look like a fix from inside the round that produced them.
    """
    trajectory = []
    defence = start_defence
    for _ in range(rounds):
        attack, _ = best_response(payoff[defence], fractions, effectiveness_exponent)
        trajectory.append((defence, attack, float(payoff[defence, attack])))
        defence = int(np.argmax(payoff[:, attack]))
    return trajectory


def cycle_length(trajectory: list[tuple[int, int, float]]) -> int:
    """Length of the repeating suffix of `(defence, attack)` states — 1 means it converged."""
    states = [(d, a) for d, a, _ in trajectory]
    if not states:
        return 0
    last = states[-1]
    for gap in range(1, len(states)):
        if states[-1 - gap] == last:
            return gap
    return len(states)


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class BudgetPoint:
    """The whole game, played at one false-alarm budget."""

    budget: float
    payoff: np.ndarray
    clean_detection: float
    best_reply: int
    detection_at_reply: float
    reply_utility: float

    @property
    def evasion_pays(self) -> bool:
        """Does the attacker's best reply involve disguising at all?"""
        return self.best_reply > 0


@dataclass
class StrategicStudy:
    """The payoff matrix at every operating point, and the game solved where it has content."""

    defence_names: list[str]
    fractions: list[float]
    effectiveness_exponent: float
    deployed: BudgetPoint
    points: list[BudgetPoint]
    frontier: BudgetPoint | None
    played_at: BudgetPoint
    cost_sensitivity: list[tuple[float, int, int]]
    trajectory: list[tuple[int, int, float]]
    cycle: int
    stackelberg: tuple[int, int, float]
    equilibria: list[tuple[int, int]]
    attacker_utility_at_equilibrium: float
    attacker_utility_undefended: float


def _fit_defence(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    centroid: np.ndarray,
    ctrl_idx: np.ndarray,
    train_fraction: float,
) -> SupervisedClassifier:
    """Fit a defence: the clean model, or one adversarially trained at a mimicry level.

    Adversarial training augments rather than replaces, so a defence tuned for one disguise does
    not simply forget undisguised attacks — that would make the arms race trivially cyclic for
    reasons of implementation rather than of strategy.
    """
    seed_everything(settings.seed)
    if train_fraction <= 0:
        return SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    attack_rows = x_train[y_train == 1]
    adv = mimicry_perturb(attack_rows, centroid, ctrl_idx, train_fraction)
    x_aug = np.vstack([x_train, adv])
    y_aug = np.concatenate([y_train, np.ones(len(adv), dtype=int)])
    return SupervisedClassifier(settings).fit(x_aug, y_aug, eval_set=(x_val, y_val))


def run_strategic(settings: Settings) -> StrategicStudy:
    """Build the defence-by-attack payoff matrix and solve the game three ways."""
    cfg: StrategicConfig = settings.strategic
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    feature_names = [f"numeric__{name}" for name in numeric_features()]
    ctrl_idx = controllable_indices(feature_names, variant.robustness.controllable_features)
    centroid = x_train[y_train == 0].mean(axis=0)
    x_attack = x_test[y_test == 1]
    if len(x_attack) > cfg.max_attack_flows:
        rng = np.random.default_rng(variant.seed)
        x_attack = x_attack[rng.choice(len(x_attack), cfg.max_attack_flows, replace=False)]

    fractions = np.array(cfg.attack_fractions, dtype=float)
    k = cfg.effectiveness_exponent
    budgets = sorted({*cfg.fpr_budgets, variant.thresholds.primary_fpr})

    # Fit each defence once, then score every (attack, budget) cell from the same scores: only
    # the threshold changes with the budget, so the operating-point sweep is nearly free.
    names: list[str] = []
    scores_by_defence: list[list[np.ndarray]] = []
    thresholds: list[dict[float, float]] = []
    for train_frac in cfg.defence_fractions:
        model = _fit_defence(
            variant, x_train, y_train, x_val, y_val, centroid, ctrl_idx, train_frac
        )
        s_val = attack_probability(np.asarray(model.predict_proba(x_val)), model.classes_, benign)
        thresholds.append({b: threshold_at_fpr(y_val, s_val, b) for b in budgets})
        per_attack = []
        for frac in fractions:
            adv = mimicry_perturb(x_attack, centroid, ctrl_idx, float(frac))
            per_attack.append(
                attack_probability(np.asarray(model.predict_proba(adv)), model.classes_, benign)
            )
        scores_by_defence.append(per_attack)
        names.append("clean (no hardening)" if train_frac <= 0 else f"hardened @ {train_frac:g}")

    points: list[BudgetPoint] = []
    for budget in budgets:
        payoff = np.array(
            [
                [float(np.mean(scores >= thresholds[i][budget])) for scores in per_attack]
                for i, per_attack in enumerate(scores_by_defence)
            ]
        )
        reply, utility = best_response(payoff[0], fractions, k)
        points.append(
            BudgetPoint(
                budget=budget,
                payoff=payoff,
                clean_detection=float(payoff[0, 0]),
                best_reply=reply,
                detection_at_reply=float(payoff[0, reply]),
                reply_utility=utility,
            )
        )

    deployed = next(p for p in points if p.budget == variant.thresholds.primary_fpr)
    # The game only has content where disguising is worth its cost. Below that the attacker's
    # best move is to do nothing, and every solution concept collapses onto the same cell.
    frontier = next((p for p in points if p.evasion_pays), None)
    played_at = frontier or deployed

    # Sensitivity: the negative result above depends entirely on how expensive the disguise is,
    # so sweep that assumption rather than defending it. A smaller exponent means the attack
    # retains more of its value while disguised -- a cheaper disguise.
    strongest = max(points, key=lambda p: p.clean_detection)
    cost_sensitivity = [
        (
            float(exponent),
            best_response(deployed.payoff[0], fractions, float(exponent))[0],
            best_response(strongest.payoff[0], fractions, float(exponent))[0],
        )
        for exponent in cfg.cost_sweep
    ]

    trajectory = arms_race(played_at.payoff, fractions, k, cfg.rounds)
    stack = stackelberg_solution(played_at.payoff, fractions, k)
    equilibria = pure_nash(played_at.payoff, fractions, k)

    return StrategicStudy(
        defence_names=names,
        fractions=[float(f) for f in fractions],
        effectiveness_exponent=k,
        deployed=deployed,
        points=points,
        frontier=frontier,
        played_at=played_at,
        cost_sensitivity=cost_sensitivity,
        trajectory=trajectory,
        cycle=cycle_length(trajectory),
        stackelberg=stack,
        equilibria=equilibria,
        attacker_utility_at_equilibrium=attacker_utility(
            float(played_at.payoff[stack[0], stack[1]]), fractions[stack[1]], k
        ),
        attacker_utility_undefended=attacker_utility(float(played_at.payoff[0, 0]), 0.0, k),
    )


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------


def run_strategic_report(settings: Settings) -> Path:
    """Run the strategic study and write the report + figure."""
    study = run_strategic(settings)
    fig = plots.plot_lines(
        {
            f"{p.budget:.1%} FPR budget": (
                np.array(study.fractions),
                np.array(
                    [
                        attacker_utility(float(d), f, study.effectiveness_exponent)
                        for d, f in zip(p.payoff[0], study.fractions, strict=True)
                    ]
                ),
            )
            for p in study.points
        },
        xlabel="attacker mimicry fraction (how far the flow is moved toward benign)",
        ylabel="attacker utility: P(get through) x attack value retained",
        title="When is disguising worth its cost? (clean model, by operating point)",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, fig), encoding="utf-8")
    logger.info("Wrote strategic report", extra={"path": str(out_path)})

    with track_run(settings, "strategic") as run:
        run.log_metrics(
            {
                "deployed_detection": study.deployed.clean_detection,
                "deployed_best_reply": study.fractions[study.deployed.best_reply],
                "frontier_budget": study.frontier.budget if study.frontier else float("nan"),
                "stackelberg_detection": study.stackelberg[2],
                "arms_race_final": study.trajectory[-1][2] if study.trajectory else 0.0,
                "cycle_length": float(study.cycle),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _payoff_table(study: StrategicStudy, point: BudgetPoint) -> str:
    header = "| defence \\ attack | " + " | ".join(f"{f:.0%} mimicry" for f in study.fractions)
    rows = [header + " | attacker's best reply |", "|---|" + "---|" * (len(study.fractions) + 1)]
    for i, name in enumerate(study.defence_names):
        br, _ = best_response(
            point.payoff[i], np.array(study.fractions), study.effectiveness_exponent
        )
        cells = " | ".join(
            (f"**{point.payoff[i, j]:.1%}**" if j == br else f"{point.payoff[i, j]:.1%}")
            for j in range(len(study.fractions))
        )
        rows.append(f"| {name} | {cells} | {study.fractions[br]:.0%} |")
    return "\n".join(rows)


def _frontier_table(study: StrategicStudy) -> str:
    rows = [
        "| FPR budget | clean-model detection | attacker's best reply | detection at that reply "
        "| is disguising worth it? |",
        "|---|---|---|---|---|",
    ]
    for p in study.points:
        mark = "**yes**" if p.evasion_pays else "no"
        deployed = " (deployed)" if p is study.deployed else ""
        rows.append(
            f"| {p.budget:.1%}{deployed} | {p.clean_detection:.1%} "
            f"| {study.fractions[p.best_reply]:.0%} mimicry | {p.detection_at_reply:.1%} "
            f"| {mark} |"
        )
    return "\n".join(rows)


def _trajectory_table(study: StrategicStudy) -> str:
    rows = ["| round | deployed defence | attacker plays | detection |", "|---|---|---|---|"]
    for r, (d, a, det) in enumerate(study.trajectory, start=1):
        rows.append(
            f"| {r} | {study.defence_names[d]} | {study.fractions[a]:.0%} mimicry | {det:.1%} |"
        )
    return "\n".join(rows)


def _deployed_read(study: StrategicStudy) -> str:
    p = study.deployed
    if p.evasion_pays:
        return (
            f"At the deployed {p.budget:.1%} budget the attacker's best reply is "
            f"**{study.fractions[p.best_reply]:.0%} mimicry**, and detection falls from "
            f"{p.clean_detection:.1%} to {p.detection_at_reply:.1%}. That is the number the "
            "one-shot evasion study reports; everything below asks what happens next."
        )
    return (
        f"**At the deployed operating point, evading this detector is irrational.** The clean "
        f"model catches {p.clean_detection:.1%} of undisguised attacks at the "
        f"{p.budget:.1%} false-alarm budget, so an attacker who does *nothing* already gets "
        f"{1 - p.clean_detection:.0%} of their traffic through with the attack fully intact. "
        "Every disguise on offer costs more attack value than it buys in evasion, so the "
        "utility-maximising move is the empty one — and the bold cells above sit in the leftmost "
        "column for every defence. This is not a quirk of the utility function; it is "
        "arithmetic. Mimicry at fraction `f` only pays when it cuts detection by more than "
        f"roughly `f`, and a detector at {p.clean_detection:.1%} does not have that much to give "
        "away in total. The adversarial-ML literature on evasion implicitly assumes a detector "
        "worth evading, and the honest reading of this table is that **the deployed operating "
        "point is not one**: the adversary's rational strategy against it is to ignore it."
    )


def _frontier_read(study: StrategicStudy) -> str:
    if study.frontier is None:
        return (
            "**No operating point in the sweep makes evasion worthwhile.** Even where the "
            f"detector catches {max(p.clean_detection for p in study.points):.1%} of attacks, "
            "doing nothing still beats every disguise on offer. That bounds the whole adversarial "
            "programme here: on this feature set and this model, the attacker's dominant strategy "
            "is to leave the traffic alone, and defensive effort is better spent raising "
            "detection than anticipating evasion."
        )
    f = study.frontier
    below = [p for p in study.points if p.budget < f.budget]
    below_text = ", ".join(f"{p.budget:.1%}" for p in below) or "none"
    return (
        f"The frontier sits at the **{f.budget:.1%} false-alarm budget**, where the clean model "
        f"reaches {f.clean_detection:.1%} detection. Below it — including at every budget this "
        f"project would actually deploy ({below_text}) — the attacker's best move is to do "
        f"nothing. Above it, disguising starts to pay: the attacker switches to "
        f"{study.fractions[f.best_reply]:.0%} mimicry and takes detection down to "
        f"{f.detection_at_reply:.1%}. That threshold is the useful output of this report. It is "
        "the detection rate a detector has to clear before an adversary has any reason to adapt "
        "to it, and it reframes the usual question: evasion resistance is not a property to buy "
        "before the detector works — it is a problem you *earn* by making the detector good "
        "enough to be worth attacking."
    )


def _sensitivity_table(study: StrategicStudy) -> str:
    strongest = max(study.points, key=lambda p: p.clean_detection)
    rows = [
        f"| disguise cost exponent `k` | best reply at {study.deployed.budget:.1%} FPR "
        f"| best reply at {strongest.budget:.1%} FPR | does evasion ever pay? |",
        "|---|---|---|---|",
    ]
    for exponent, at_deployed, at_strongest in study.cost_sensitivity:
        pays = "**yes**" if (at_deployed or at_strongest) else "no"
        note = " (as modelled)" if exponent == study.effectiveness_exponent else ""
        rows.append(
            f"| {exponent:g}{note} | {study.fractions[at_deployed]:.0%} mimicry "
            f"| {study.fractions[at_strongest]:.0%} mimicry | {pays} |"
        )
    return "\n".join(rows)


def _sensitivity_read(study: StrategicStudy) -> str:
    paying = [row for row in study.cost_sensitivity if row[1] or row[2]]
    if not paying:
        return (
            "Evasion fails to pay across the entire sweep, down to a disguise that costs the "
            "attacker almost nothing. At that point the result stops being a statement about the "
            "cost model and becomes one about the attack surface: interpolating toward the benign "
            "centroid simply does not move this model's score enough to matter, whatever it "
            "costs. The [monotone-constraint study](monotonic.md) reaches the same place from the "
            "other direction, by making the inflation family impossible rather than unprofitable."
        )
    critical = max(row[0] for row in paying)
    strongest = max(study.points, key=lambda p: p.clean_detection)
    return (
        f"Evasion becomes rational once the disguise is cheap enough — at `k = {critical:g}` and "
        f"below, where a {study.fractions[1]:.0%} disguise costs the attacker "
        f"{1 - (1 - study.fractions[1]) ** critical:.0%} of the attack's value rather than the "
        f"{1 - (1 - study.fractions[1]) ** study.effectiveness_exponent:.0%} the linear model "
        "charges. That is the honest form of this report's headline: the claim is not *evasion "
        "never pays*, it is **evasion does not pay unless disguising is nearly free**, and the "
        "threshold at which it flips is a number rather than an opinion. Which side of it a real "
        "adversary sits on is a question about attack semantics — how much of a DoS survives "
        "having its inter-arrival times padded — that this dataset cannot answer, so it is stated "
        "as a condition instead." + _where_it_flips(study, critical, strongest)
    )


def _where_it_flips(study: StrategicStudy, critical: float, strongest: BudgetPoint) -> str:
    """Which operating point concedes to a cheap disguise first, read off the sweep.

    The intuitive answer is the strongest detector, since that is where an attacker has the most
    to hide from. The table does not always agree, so it is read rather than assumed.
    """
    _, at_deployed, at_strongest = next(r for r in study.cost_sensitivity if r[0] == critical)
    if bool(at_deployed) == bool(at_strongest):
        return ""
    if at_deployed and not at_strongest:
        return (
            f" It also flips at the **deployed** {study.deployed.budget:.1%} budget before the "
            f"strongest {strongest.budget:.1%} one, which is the reverse of the intuitive answer "
            "and worth stating: a disguise removes a larger *share* of a weak detector's "
            "already-small detection, so the marginal value of hiding is highest exactly where "
            "detection is lowest. The consolation is that the stakes are lowest there too — at "
            f"{study.deployed.clean_detection:.1%} detection the attacker was already getting "
            f"{1 - study.deployed.clean_detection:.0%} of their traffic through without "
            "bothering to hide."
        )
    return (
        f" It flips first at the strongest {strongest.budget:.1%} operating point, where the "
        "attacker has the most detection to hide from."
    )


def _race_read(study: StrategicStudy) -> str:
    if not study.trajectory:
        return ""
    final = study.trajectory[-1]
    detections = [d for _, _, d in study.trajectory]
    if study.cycle == 1:
        shape = (
            f"The race **converges**: after round {max(1, len(detections) - 1)} neither side "
            f"changes, settling at {final[2]:.1%} detection with the defender running "
            f"*{study.defence_names[final[0]]}* against {study.fractions[final[1]]:.0%} mimicry."
        )
    else:
        shape = (
            f"The race **cycles with period {study.cycle}**: the defender keeps adopting the "
            "counter to the attack it just saw, the attacker keeps moving to whatever that "
            "counter is weakest against, and the pair never settle. Every individual round looks "
            "like a fix from inside the round that produced it, which is precisely the trap "
            "myopic hardening sets."
        )
    return (
        f"{shape} Detection over the rounds runs "
        + " -> ".join(f"{d:.1%}" for d in detections)
        + f". The swing between the best and worst round is "
        f"{max(detections) - min(detections):.1%} points, which is how much a defender who "
        "quotes the number from a *good* round is overstating what they have."
    )


def _stackelberg_read(study: StrategicStudy) -> str:
    d, a, det = study.stackelberg
    final = study.trajectory[-1][2] if study.trajectory else det
    gain = det - final
    cost_now = 1.0 - study.attacker_utility_at_equilibrium / max(
        study.attacker_utility_undefended, 1e-9
    )
    verdict = (
        f"Commitment is worth **{gain:+.1%} points** of detection here"
        if abs(gain) > 0.002
        else "Commitment is worth essentially nothing here"
    )
    return (
        f"The defender who moves first and knows the attacker will re-optimise picks "
        f"*{study.defence_names[d]}*, against which the best reply is "
        f"{study.fractions[a]:.0%} mimicry and detection holds at **{det:.1%}**. "
        f"{verdict}: the myopic race ends at {final:.1%}. Where the two differ, it is because the "
        "myopic defender optimises against the attack in front of them, which is a different "
        "objective from optimising against the attack that will *follow* their choice. Measured "
        f"on the attacker's own terms, the equilibrium strips **{cost_now:.0%}** of the value "
        "they extract from an undefended detector — the honest way to state a defence's worth, "
        "since the attacker's payoff is the thing a defence exists to reduce."
    )


def _equilibrium_read(study: StrategicStudy) -> str:
    if not study.equilibria:
        return (
            "There is **no pure-strategy equilibrium** in this matrix: for every "
            "defence-and-attack pair, one side strictly prefers to move. That is not a defect of "
            "the analysis, it is the formal statement of why the arms race does not end — and it "
            "is the argument for the commitment solution above, which does not require one, over "
            "any hope that repeated patching converges on its own."
        )
    cells = ", ".join(
        f"*{study.defence_names[i]}* vs {study.fractions[j]:.0%} mimicry"
        for i, j in study.equilibria
    )
    return (
        f"The matrix has {len(study.equilibria)} pure-strategy equilibrium "
        f"({'cell' if len(study.equilibria) == 1 else 'cells'}): {cells}. Neither side gains by "
        "moving unilaterally, so this is where a patient adversary and a patient defender end up "
        "regardless of who moves first — and it is the only detection figure in this repo that "
        "carries that property."
    )


def _render(study: StrategicStudy, fig: Path) -> str:
    played = study.played_at
    where = (
        f"the {played.budget:.1%} budget, the first operating point at which the attacker has "
        "any reason to adapt"
        if study.frontier is not None
        else f"the deployed {played.budget:.1%} budget — no operating point in the sweep makes "
        "evasion pay, so the game is degenerate everywhere and these are reported for "
        "completeness"
    )
    return f"""# NetSentry — The Arms Race as a Game: Strategic Equilibrium

_Synthetic stand-in. Honest temporal split. Attacker utility
`(1 - detection) * (1 - fraction)^{study.effectiveness_exponent:g}` — the chance of getting
through, times how much attack survives the disguise. Defences are the clean model plus
adversarial training at each mimicry level._

## Why this report exists

The [evasion study](robustness.md) measures a one-shot attack; the [hardening
study](hardening.md) answers it with one round of adversarial training and re-measures. Both stop
one move too early, because a real adversary sees the fix and moves again. Treating it as a game
makes three things computable: the attacker's **cost** (explicit, and not an L2 norm — a flow
that looks benign *is* less of an attack), the **arms race** (simulated rather than assumed), and
the value of **commitment** (a defender who moves first should not pick the model that is best
against today's attack).

It also forces a question the sequential studies never ask, which turns out to be the one worth
answering: **is this detector worth evading at all?**

## The payoff matrix at the deployed operating point

Detection for every defence against every attack, at the deployed {study.deployed.budget:.1%}
false-alarm budget. Bold marks the attacker's utility-maximising reply to each defence.

{_payoff_table(study, study.deployed)}

{_deployed_read(study)}

## How good must a detector be before it is worth evading?

The same defences and the same attacks, re-thresholded across operating points. Only the
threshold changes, so this isolates the effect of detection strength on the adversary's
incentive to adapt.

{_frontier_table(study)}

{_frontier_read(study)}

![Attacker utility by mimicry level and operating point](../figures/{fig.name})

## How sensitive is that to what the disguise costs?

The result above rests entirely on the attacker's cost model, so the assumption is swept rather
than defended. A smaller exponent `k` means the attack retains more of its value while disguised
— a cheaper disguise, and a more capable adversary.

{_sensitivity_table(study)}

{_sensitivity_read(study)}

## The myopic arms race

Played at {where}. Each round the attacker best-responds to what is deployed, and the defender
then adopts whatever is strongest against *that* attack. Neither looks ahead.

{_trajectory_table(study)}

{_race_read(study)}

## Commitment: the Stackelberg solution

{_stackelberg_read(study)}

## Is there an equilibrium at all?

{_equilibrium_read(study)}

## Scope

The defender's strategy set is finite and small — the clean model plus adversarial training at a
handful of mimicry levels — so this is a *game over deployable configurations*, not over all
possible detectors, and the equilibrium is an equilibrium of that restricted game. The attacker's
effectiveness decay is a modelling choice: linear in the mimicry fraction, defensible for
volumetric attacks and probably too generous for a slow-and-low one, so the exponent is a config
knob. It is also the assumption the headline is most sensitive to — a cheaper disguise moves the
frontier down — which is exactly why the frontier is reported as a threshold rather than as a
verdict. Attacks move in the same controllable-feature set the [evasion](robustness.md) and
[verification](verify_trees.md) studies use, so the three cannot drift apart. The structural
alternative is the [monotone-constraint](monotonic.md) model, which does not play this game at
all: it makes the inflation family impossible by construction rather than expensive, which is why
a constraint that costs nothing beats a defence that has to be re-derived every round."""
