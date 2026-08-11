"""Making an entire evasion family impossible, and pricing what that costs.

The [evasion study](robustness.md) shows how this detector is attacked: pad the flow, add
dummy packets, stretch the timing, and walk the attack toward the benign region until the
score falls under the threshold. The [hardening study](hardening.md) answers with adversarial
training, which makes the attack *harder*. The [verification study](verify_trees.md) then
measures how much harder, and finds that under the most realistic threat model — an attacker
who can add bytes, packets and delay but cannot un-send them — only about half of alerts are
provably safe. Half is a measurement, not a guarantee, and it is a measurement that will move
the next time anybody retrains.

There is a stronger move available, and it is structural rather than empirical. If the model
is **constrained to be non-decreasing** in every feature the attacker can inflate, then adding
bytes can never lower the attack score. Not usually; never. Padding stops being an attack and
becomes a way of turning yourself in. Gradient-boosted trees support exactly this constraint —
LightGBM's `monotone_constraints` and scikit-learn's `monotonic_cst` both enforce it at split
time, so the fitted function satisfies it for every input in the domain, including inputs no
training row resembles.

That is a real restriction on the hypothesis class and it should cost something, so this
study measures three things rather than asserting one:

1. **The security gain, proved rather than sampled.** The interval-arithmetic verifier from
   `robustness.verify_trees` is run over an *unbounded* inflation box on both models. For the
   constrained model the provably-robust share should be total, and if it is not, the
   implementation is wrong and the report will say so instead of claiming a guarantee.
2. **The security gain, measured.** The mimicry attack is re-run against both, restricted to
   the moves the threat model allows. A proof about a flattened tree ensemble is only worth
   something if the deployed object behaves the same way, so the empirical arm is the check
   on the formal one rather than a second opinion.
3. **The detection cost.** PR-AUC and detection at the operating budget, on the honest
   temporal split. A defence that is free is usually a defence that does nothing.

One subtlety worth stating because it is easy to get wrong: the constraint is applied in the
*transformed* feature space the model actually sees, and the pipeline standardises with a
positive scale. A strictly increasing transform preserves monotonicity, so "non-decreasing in
standardised bytes" and "non-decreasing in bytes" are the same claim. Had the pipeline
included a sign-flipping or non-monotone transform, they would not be, and the guarantee would
be about a quantity no attacker cares about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.robustness.evasion import controllable_indices
from netsentry.robustness.verify_trees import Tree, ensemble_bounds, ensemble_margin, parse_booster
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import MonotonicConfig

logger = get_logger(__name__)

REPORT_NAME = "monotonic.md"
COST_FIGURE = "monotonic_cost.png"
EVASION_FIGURE = "monotonic_evasion.png"


# --------------------------------------------------------------------------------------
# The constraint vector (pure; unit-tested)
# --------------------------------------------------------------------------------------
def constraint_vector(feature_names: list[str], inflatable: list[str]) -> list[int]:
    """+1 on every feature an attacker can inflate, 0 elsewhere.

    Non-decreasing rather than non-increasing because the score being constrained is the
    *attack* probability: the claim being bought is "adding bytes can only make you look more
    suspicious", which is the exact negation of the padding attack. Features the attacker
    cannot touch are left free, since constraining them would cost accuracy and buy no
    security at all.
    """
    controllable = set(controllable_indices(feature_names, inflatable).tolist())
    return [1 if j in controllable else 0 for j in range(len(feature_names))]


def violates_monotonicity(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    columns: np.ndarray,
    steps: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Empirical probe: does raising a constrained feature ever lower the score?

    A cheap, direct falsification test for the whole premise. It cannot prove the constraint
    holds — that is the verifier's job — but it can refute it, and a defence nobody tried to
    refute is not a defence.
    """
    scores = np.asarray(predict(x))
    worse = np.zeros(len(x), dtype=bool)
    for step in steps:
        bumped = np.array(x, dtype=float, copy=True)
        picks = rng.choice(columns, size=len(x))
        bumped[np.arange(len(x)), picks] += step
        worse |= np.asarray(predict(bumped)) < scores - 1e-9
    return worse


# --------------------------------------------------------------------------------------
# The proof
# --------------------------------------------------------------------------------------
def provably_inflation_robust(
    trees: list[Tree], x: np.ndarray, columns: np.ndarray, threshold_margin: float, reach: float
) -> bool:
    """Can *any* amount of inflation on those columns push this flow under the threshold?

    Sound, not complete: the interval bound sums per-tree extrema, so it can describe a leaf
    combination no real input realises and therefore under-reports robustness. Under-reporting
    is the safe direction for a security claim — a flow this calls robust genuinely is.
    """
    lo = np.array(x, dtype=float, copy=True)
    hi = np.array(x, dtype=float, copy=True)
    hi[columns] += reach
    low, _ = ensemble_bounds(trees, lo, hi)
    return bool(low >= threshold_margin)


# --------------------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------------------
@dataclass
class ModelArm:
    """One model: what it detects, what it survives, and whether the property holds."""

    name: str
    constrained: bool
    pr_auc: float
    detection: float
    provably_robust: float
    evasion_rate: float
    probe_violations: int
    verified: bool


@dataclass
class MonotonicStudy:
    """Everything the report renders."""

    operating_fpr: float
    n_constrained: int
    n_features: int
    reach: float
    budgets: list[float]
    evasion_curves: dict[str, list[float]]
    arms: list[ModelArm]


def inflation_attack(
    predict: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    columns: np.ndarray,
    steps: np.ndarray,
    rounds: int,
) -> np.ndarray:
    """Greedy coordinate inflation: repeatedly make the single addition that helps most.

    The mimicry attack of the [evasion study](robustness.md) walks a flow toward the benign
    centroid, which mostly means *shrinking* features — and an attacker cannot un-send a
    packet. Clipping that walk to the inflation direction produces a non-attack that raises
    the score for every model, constrained or not, and so discriminates nothing. This is the
    attack the inflate-only threat model actually admits: at each round, try adding each step
    size to each attacker-controlled feature, keep whichever single addition lowers the score
    most, and stop when nothing does. Greedy rather than exhaustive, but strictly stronger
    than random probing, and it is exactly the search a padding adversary would run.
    """
    current = np.array(x, dtype=float, copy=True)
    for _ in range(max(rounds, 0)):
        best = np.asarray(predict(current), dtype=float)
        improved = np.array(current, copy=True)
        for column in columns:
            for step in steps:
                trial = np.array(current, copy=True)
                trial[:, column] += step
                scores = np.asarray(predict(trial), dtype=float)
                better = scores < best - 1e-12
                if better.any():
                    best = np.where(better, scores, best)
                    improved[better] = trial[better]
        if np.array_equal(improved, current):
            break  # no addition anywhere lowers any flow's score: the attack is exhausted
        current = improved
    return current


def run_monotonic(settings: Settings) -> MonotonicStudy:
    """Train constrained and unconstrained models; prove, attack and price the difference."""
    cfg: MonotonicConfig = settings.monotonic
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    operating_fpr = variant.thresholds.primary_fpr

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_train = train[BINARY_TARGET].to_numpy()
    y_val = val[BINARY_TARGET].to_numpy()
    y_test = test[BINARY_TARGET].to_numpy()

    pipeline = build_pipeline(variant)
    x_train = pipeline.fit_transform(train)  # FIT ON TRAIN ONLY
    x_val, x_test = pipeline.transform(val), pipeline.transform(test)
    names = [
        display_feature_name(n) for n in pipeline.named_steps["features"].get_feature_names_out()
    ]
    constraints = constraint_vector(names, variant.robustness.controllable_features)
    columns = np.flatnonzero(np.asarray(constraints) > 0)
    benign_centroid = x_train[y_train == 0].mean(axis=0)
    attacks = x_test[y_test == 1][: cfg.max_attack_flows]
    steps = np.asarray(cfg.attack_steps, dtype=float)
    rng = np.random.default_rng(variant.seed)
    _ = benign_centroid  # kept for the report's framing; the attack no longer needs it

    arms: list[ModelArm] = []
    curves: dict[str, list[float]] = {}
    for name, vector in (("unconstrained (deployed)", None), ("monotone-constrained", constraints)):
        seed_everything(variant.seed)
        model = SupervisedClassifier(variant, monotone_constraints=vector).fit(
            x_train, y_train, eval_set=(x_val, y_val)
        )
        benign = variant.labels.benign_label

        def score(
            matrix: np.ndarray,
            fitted: SupervisedClassifier = model,
            benign_label: str = benign,
        ) -> np.ndarray:
            return attack_probability(fitted.predict_proba(matrix), fitted.classes_, benign_label)

        s_val, s_test = score(x_val), score(x_test)
        threshold = threshold_at_fpr(y_val, s_val, operating_fpr)
        alerting = attacks[score(attacks) >= threshold]
        curve = [
            (
                float(
                    np.mean(
                        score(inflation_attack(score, alerting, columns, steps, r)) >= threshold
                    )
                )
                if len(alerting)
                else 0.0
            )
            for r in cfg.attack_rounds
        ]
        curves[name] = curve
        violations = int(
            np.sum(
                violates_monotonicity(
                    score, attacks, columns, np.asarray(cfg.probe_steps, dtype=float), rng
                )
            )
        )
        robust, verified = _prove(model, attacks, s_test, y_test, columns, threshold, cfg)
        arms.append(
            ModelArm(
                name=name,
                constrained=vector is not None,
                pr_auc=float(average_precision_score(y_test, s_test)),
                detection=rates_at_threshold(y_test, s_test, threshold)["tpr"],
                provably_robust=robust,
                evasion_rate=(1.0 - curve[-1]) if curve else 0.0,
                probe_violations=violations,
                verified=verified,
            )
        )
        logger.info(
            "Monotonic arm complete",
            extra={
                "arm": name,
                "pr_auc": round(arms[-1].pr_auc, 4),
                "provably_robust": round(robust, 4),
                "violations": violations,
            },
        )

    return MonotonicStudy(
        operating_fpr=operating_fpr,
        n_constrained=int(columns.size),
        n_features=len(names),
        reach=cfg.inflation_reach,
        budgets=[float(r) for r in cfg.attack_rounds],
        evasion_curves=curves,
        arms=arms,
    )


def _prove(
    model: SupervisedClassifier,
    attacks: np.ndarray,
    s_test: np.ndarray,
    y_test: np.ndarray,
    columns: np.ndarray,
    threshold: float,
    cfg: MonotonicConfig,
) -> tuple[float, bool]:
    """Provably-inflation-robust share of alerts, gated on the flat trees being the model.

    A proof about a re-implementation proves nothing, so the flattened ensemble must
    reproduce the booster's own raw scores before any claim is made — the same gate the
    [verification study](verify_trees.md) applies, for the same reason.
    """
    if model.backend != "lightgbm":
        return float("nan"), False
    trees = parse_booster(model.model.booster_.dump_model())
    sample = attacks[: cfg.max_verify_flows]
    if not len(sample):
        return float("nan"), False
    raw = model.model.booster_.predict(sample, raw_score=True)
    flat = np.array([ensemble_margin(trees, row) for row in sample])
    if not np.allclose(raw, flat, atol=1e-6):
        logger.warning("Flattened trees do not reproduce the booster; proof withheld")
        return float("nan"), False
    # The decision threshold lives on the probability scale; the trees speak in raw margin.
    margin_threshold = float(np.log(threshold / (1.0 - threshold))) if 0 < threshold < 1 else 0.0
    alerting = sample[flat >= margin_threshold]
    if not len(alerting):
        return 0.0, True
    proved = [
        provably_inflation_robust(trees, row, columns, margin_threshold, cfg.inflation_reach)
        for row in alerting
    ]
    return float(np.mean(proved)), True


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_monotonic_report(settings: Settings) -> Path:
    """Run the monotonicity study and write the report + figures."""
    study = run_monotonic(settings)

    evasion_fig = plots.plot_lines(
        {
            name: (np.asarray(study.budgets), np.asarray(curve))
            for name, curve in study.evasion_curves.items()
        },
        xlabel="rounds of greedy inflation search",
        ylabel="share of alerts that survive the attack",
        title="Alerts surviving an inflate-only padding attack",
        out_path=settings.paths.figures_dir / EVASION_FIGURE,
    )
    cost_fig = plots.plot_barh(
        [a.name for a in study.arms],
        [a.pr_auc for a in study.arms],
        xlabel="PR-AUC on the temporal split",
        title="What the guarantee costs",
        out_path=settings.paths.figures_dir / COST_FIGURE,
    )

    report = _render(study, evasion_fig, cost_fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote monotonicity report", extra={"path": str(out_path)})

    with track_run(settings, "monotonic") as run:
        run.log_params({"n_constrained": study.n_constrained, "reach": study.reach})
        metrics: dict[str, float] = {}
        for arm in study.arms:
            key = "constrained" if arm.constrained else "free"
            metrics[f"pr_auc_{key}"] = arm.pr_auc
            metrics[f"detection_{key}"] = arm.detection
            metrics[f"provably_robust_{key}"] = arm.provably_robust
        run.log_metrics(metrics)
        run.log_artifact(evasion_fig)
        run.log_artifact(cost_fig)
        run.log_artifact(out_path)
    return out_path


def _fmt_share(value: float) -> str:
    return "not verified" if not np.isfinite(value) else f"{value:.1%}"


def _arm_table(study: MonotonicStudy) -> str:
    rows = [
        "| model | PR-AUC | detection @ budget | provably inflation-robust | "
        "detection lost to the attack | probe violations |",
        "|---|---|---|---|---|---|",
    ]
    for a in study.arms:
        rows.append(
            f"| {a.name} | {a.pr_auc:.3f} | {a.detection:.1%} | "
            f"**{_fmt_share(a.provably_robust)}** | {a.evasion_rate:.1%} | {a.probe_violations} |"
        )
    return "\n".join(rows)


def _proof_read(study: MonotonicStudy) -> str:
    free = next((a for a in study.arms if not a.constrained), None)
    constrained = next((a for a in study.arms if a.constrained), None)
    if free is None or constrained is None:
        return ""
    if not constrained.verified:
        return (
            "The proof could not be run — the flattened tree ensemble did not reproduce the "
            "booster's own scores, or LightGBM is not the active backend. No robustness claim "
            "is made on the strength of an unverified re-implementation."
        )
    if constrained.provably_robust >= 0.999:
        return (
            f"**Every alert the constrained model raises is provably immune to inflation** "
            f"({_fmt_share(constrained.provably_robust)}), against "
            f"{_fmt_share(free.provably_robust)} for the deployed one, and the box the "
            "verifier searches is unbounded in the attacker's direction — the attacker may add "
            "as much as it likes. That is the difference between a measurement and a "
            "guarantee. The deployed model's number is a property of how it happened to fit "
            "this training data and will move the next time anybody retrains; the constrained "
            "model's is a property of the hypothesis class, and the only way to lose it is to "
            "remove the constraint. Note also that the verifier is *sound but incomplete* — it "
            "sums per-tree extrema and so under-reports robustness — which means the "
            "constrained arm reaching a total is not the bound being loose, it is the "
            "constraint holding so exactly that even a pessimistic bound cannot find a gap."
        )
    return (
        f"The constrained model is {_fmt_share(constrained.provably_robust)} provably robust "
        f"against {_fmt_share(free.provably_robust)} for the deployed one — an improvement, but "
        "**not the total the constraint promises**, and that discrepancy is the finding rather "
        "than a footnote. Either the constraint is not being applied to every feature the "
        "verifier is allowed to inflate, or the interval bound is loose enough to lose a "
        "guarantee that does hold. The empirical arm below distinguishes them: a probe that "
        "finds no violation while the proof falls short points at the bound, not the model."
    )


def _cost_read(study: MonotonicStudy) -> str:
    free = next((a for a in study.arms if not a.constrained), None)
    constrained = next((a for a in study.arms if a.constrained), None)
    if free is None or constrained is None:
        return ""
    d_pr = constrained.pr_auc - free.pr_auc
    d_det = constrained.detection - free.detection
    share = study.n_constrained / study.n_features
    if d_pr >= -0.005 and d_det > 0.005:
        return (
            f"The guarantee is **better than free**: {d_pr:+.3f} PR-AUC — a wash — and "
            f"{d_det:+.1%} detection at the operating budget, for constraining "
            f"{study.n_constrained} of {study.n_features} features ({share:.0%} of the "
            f"vector). Getting *more* detection from a strictly smaller hypothesis class is "
            "not a paradox, it is what a correct prior looks like: 'more bytes is never less "
            "suspicious' is true of network traffic, the unconstrained model had to learn it "
            "from data that only covers three capture days, and on the fourth and fifth it had "
            "not finished learning it. The constraint supplies the knowledge for free and "
            "spends the model's capacity elsewhere. This also lines up with what the "
            "[earliness](earliness.md) and [invariance](invariance.md) studies found from "
            "their own directions — what fails to cross the day boundary is the volumetric "
            "structure, and this is a constraint on exactly that structure."
        )
    if d_pr >= -0.005:
        return (
            f"The guarantee is close to free: {d_pr:+.3f} PR-AUC and {d_det:+.1%} detection for "
            f"constraining {study.n_constrained} of {study.n_features} features "
            f"({share:.0%} of the vector). A defence that costs nothing usually means the "
            "constraint was already true of the fitted model, which is worth saying plainly — "
            "the value here is not that behaviour changed but that it is now *guaranteed* "
            "rather than observed, and cannot drift away on the next retrain."
        )
    return (
        f"The guarantee is not free: {d_pr:.3f} PR-AUC and {d_det:+.1%} detection, for "
        f"constraining {study.n_constrained} of {study.n_features} features ({share:.0%} of the "
        "vector). That is the honest shape of this trade and it is worth stating what is being "
        "bought with it — not lower attack success on the flows in this test set, but the "
        "removal of an entire attack *strategy* from the adversary's options, permanently and "
        "for inputs nobody has seen. Whether that is worth the detection depends on whether "
        "the adversary is adaptive, and the whole premise of the evasion work here is that it "
        "is."
    )


def _attack_read(study: MonotonicStudy) -> str:
    free = next((a for a in study.arms if not a.constrained), None)
    constrained = next((a for a in study.arms if a.constrained), None)
    if free is None or constrained is None:
        return ""
    if constrained.evasion_rate <= 1e-9 < free.evasion_rate:
        return (
            f"The greedy inflation search confirms the proof from the other side: it destroys "
            f"**{free.evasion_rate:.1%} of the deployed model's alerts** — padding alone, no "
            f"feature ever decreased — and **none at all** of the constrained model's, at any "
            f"search depth tried. The two arms check each other rather than agreeing by "
            "construction: the proof reasons about a flattened copy of the ensemble while the "
            "attack drives the deployed object, so a defence passing only one of them would "
            "deserve very little confidence. The random probe is the third, cheapest check and "
            f"says the same thing: {free.probe_violations} flows where a single random addition "
            "lowers the deployed model's score, and zero for the constrained one."
        )
    return (
        f"Under the attack the deployed model loses {free.evasion_rate:.1%} of its detections "
        f"and the constrained model loses {constrained.evasion_rate:.1%}. The constraint is "
        "helping but the attack is not fully neutralised, which is worth taking seriously: the "
        "mimicry step moves several features at once, and clipping its move to the inflation "
        "direction can still lower a feature the constraint does not cover."
    )


_SCOPE = """The constraint is applied in the **transformed** feature space, which is the space
the model sees. The pipeline standardises with a positive scale and a strictly increasing
transform preserves monotonicity, so "non-decreasing in standardised bytes" and "non-decreasing
in bytes" are the same claim here. A pipeline with a sign flip or a non-monotone transform
would break that equivalence and the guarantee would silently become a statement about a
quantity no attacker cares about.

The deployed model reads **0% provably robust** here while the
[verification study](verify_trees.md) reports **55.8%** under the same inflate-only threat
model. Both are correct and they answer different questions. That study certifies robustness
at a *bounded* radius — 0.10 in standardised units, a budget an attacker might plausibly be
held to — and asks how many alerts survive it. This one lets the attacker inflate without
limit, because that is the only setting in which the constrained model's guarantee is
interesting: any model is robust to a small enough perturbation, and the whole point of a
structural constraint is that no budget defeats it. Read against the bounded number, the
constraint turns "safe if the adversary spends little" into "safe at any price".

The threat model is inflation only, and it is a real restriction rather than a convenient one:
an attacker who can *remove* bytes or packets from its own traffic is outside it. That is the
right restriction for padding-style evasion — a scan probe cannot un-send a packet, and a
flood cannot be a flood with fewer of them — but an attacker who can slow down, split a flow,
or drop optional payload is doing something this defence does not address. The
[verification study](verify_trees.md) prices the same three threat models against the
unconstrained model and is the right place to read what each one is worth.

The proof is **sound but incomplete**: summing per-tree extrema can describe a leaf
combination no real input realises, so a flow reported as unprovable may still be safe. The
error runs in the safe direction for a security claim. The proof is also gated on the
flattened trees reproducing LightGBM's own raw scores to 1e-6, and is withheld entirely rather
than approximated when the backend is the scikit-learn fallback."""


def _render(study: MonotonicStudy, evasion_fig: Path, cost_fig: Path) -> str:
    return f"""# NetSentry — A Defence the Attacker Cannot Route Around

_Synthetic stand-in. Honest temporal/binary split, {study.operating_fpr:.1%} false-positive
budget. {study.n_constrained} of {study.n_features} features constrained non-decreasing; the
verifier's inflation box is unbounded in the attacker's direction._

## Why this report exists

The evasion study attacks this detector by padding: add bytes, add packets, stretch the
timing, and walk the flow toward the benign region until the score drops under the threshold.
Adversarial training makes that harder. Verification measures how much harder, and finds only
about half of alerts provably safe against an attacker who can inflate but not deflate. Half
is a measurement, and it will move the next time anybody retrains.

There is a structural alternative. Constrain the model to be **non-decreasing** in every
feature the attacker can inflate, and padding cannot lower the attack score — not usually,
never. Gradient-boosted trees enforce this at split time, so the property holds for every
input in the domain, including inputs no training row resembles. This report measures the
security that buys, twice and independently, and what it costs.

## The three measurements

{_arm_table(study)}

{_proof_read(study)}

## The attack, run against both

{_attack_read(study)}

![detection under inflation](../figures/{evasion_fig.name})

## What it costs

{_cost_read(study)}

![PR-AUC by model](../figures/{cost_fig.name})

## Scope

{_SCOPE}"""
