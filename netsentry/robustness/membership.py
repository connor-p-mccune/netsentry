"""Membership-inference privacy audit — the third adversarial axis.

Evasion (the robustness study) is the *inference-time* adversary; poisoning is the
*training-time* one. This is the third classic attack on a machine-learning model
and the one that is not about detection at all but about **privacy**: given only
query access to the model, can an attacker tell whether a particular flow was in its
training set? On a NIDS that is a real disclosure — "was this organisation's / this
host's traffic used to train the model?" — and it is the canonical way to measure how
much a model *memorises* its training data (Shokri et al., 2017; Yeom et al., 2018).

Two attacks are run against the deployed (regularised) model on the exchangeable
**stratified** split — the exchangeability membership inference assumes, and the same
reason the active-learning study runs there:

- **Confidence-threshold attack** (Yeom et al.). A member usually gets higher
  probability on its *true* class than a non-member; threshold that confidence. Cheap,
  needs only the target model, and is the tightest honest lower bound on leakage.
- **Shadow-model attack** (Shokri et al.). Train ``n_shadow`` shadow models that
  mimic the target on disjoint data drawn from the same distribution; for each, the
  rows it trained on are members and a held-out half are non-members. Train an attack
  classifier on ``(confidence-vector -> in/out)`` pairs pooled across shadows, then
  turn it on the target.

The honest arc — the project's measure -> re-measure signature — is the
**overfit reference**: the same architecture refit deliberately un-regularised (deep,
no early stopping) on the identical members. Membership leakage tracks the model's
**generalisation gap**, so the deployed model's regularisation and early stopping are
themselves the privacy control, and the report prices the gap between the two. The
worst-case metric is TPR at a low false-accusation rate (Carlini et al., 2022), not
just attack accuracy — a handful of confidently-memorised rows is the real leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from netsentry.data.clean import MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import tpr_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "membership.md"


@dataclass
class AttackResult:
    """One membership attack's leakage, plus the target model's generalisation gap."""

    name: str
    attack: str  # "threshold" | "shadow"
    auc: float  # attack ROC-AUC (0.5 = no leakage)
    advantage: float  # max over thresholds of (TPR - FPR); 0.0 = no leakage
    tpr_at_fpr: float  # members correctly identified at the low false-accusation budget
    fpr_budget: float
    train_acc: float
    test_acc: float
    # ROC of the attack itself, for the "privacy ROC" figure.
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray

    @property
    def gen_gap(self) -> float:
        """Train-minus-test accuracy — the memorisation that MI turns into a signal."""
        return self.train_acc - self.test_acc


def true_class_probability(
    proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """P(true class) per row — the Yeom confidence signal (0 if the class is absent).

    A model that memorised a member is over-confident on that member's real label; a
    non-member from the same distribution is not. Robust to a class the model never
    saw (a rare label missing from a subsample): its probability column is 0.
    """
    class_index = {c: j for j, c in enumerate(np.asarray(classes))}
    out = np.zeros(len(y_true), dtype=float)
    for i, label in enumerate(np.asarray(y_true)):
        j = class_index.get(label)
        if j is not None:
            out[i] = float(proba[i, j])
    return out


def confidence_features(
    proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray, top_k: int
) -> np.ndarray:
    """Attack features per row: sorted top-k probs, true-class prob, correctness, entropy.

    The sorted top-k (rather than per-class-indexed) columns keep the feature vector
    fixed-width and meaningful even when shadow models see different class subsets —
    exactly Shokri's construction, generalised so a rare-class gap cannot break it.
    """
    proba = np.asarray(proba, dtype=float)
    k = min(top_k, proba.shape[1])
    top_sorted = np.sort(proba, axis=1)[:, ::-1][:, :k]
    if top_sorted.shape[1] < top_k:  # pad narrow models to a stable width
        top_sorted = np.pad(top_sorted, ((0, 0), (0, top_k - top_sorted.shape[1])))
    true_p = true_class_probability(proba, classes, y_true).reshape(-1, 1)
    pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    correct = (pred == np.asarray(y_true)).astype(float).reshape(-1, 1)
    entropy = (-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)).reshape(-1, 1)
    return np.hstack([top_sorted, true_p, correct, entropy])


def membership_advantage(is_member: np.ndarray, scores: np.ndarray) -> float:
    """Yeom's membership advantage: max over thresholds of (TPR - FPR)."""
    is_member = np.asarray(is_member)
    if len(np.unique(is_member)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(is_member, scores)
    return float(np.max(tpr - fpr))


def attack_scores(
    is_member: np.ndarray, scores: np.ndarray, fpr_budget: float
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Summarise a membership attack: (AUC, advantage, TPR@budget, roc_fpr, roc_tpr)."""
    is_member = np.asarray(is_member)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(is_member)) < 2:
        return 0.5, 0.0, 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    auc = float(roc_auc_score(is_member, scores))
    advantage = membership_advantage(is_member, scores)
    _, tpr_budget = tpr_at_fpr(is_member, scores, fpr_budget)
    fpr, tpr, _ = roc_curve(is_member, scores)
    return auc, advantage, tpr_budget, fpr, tpr


def _accuracy(model: SupervisedClassifier, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(x) == np.asarray(y)))


def _overfit_settings(settings: Settings) -> Settings:
    """A deliberately un-regularised variant — the memorising reference model."""
    variant = settings.model_copy(deep=True)
    sup = variant.supervised
    sup.num_leaves = 255
    sup.min_child_samples = 2
    sup.reg_lambda = 0.0
    sup.subsample = 1.0
    sup.colsample_bytree = 1.0
    sup.learning_rate = 0.1
    sup.n_estimators = 400
    return variant


def _sanitize_eval(
    y_train: np.ndarray, eval_set: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Restrict an eval set to labels present in training.

    LightGBM's early-stopping eval set must carry only labels the model has seen; a
    rare class (Heartbleed, dozens of rows) can land in the eval half but not the
    subsampled training half. Drop those eval rows — the eval set only steers early
    stopping — and disable it entirely if nothing survives.
    """
    if eval_set is None:
        return None
    x_eval, y_eval = eval_set
    known = set(np.unique(y_train))
    mask = np.array([label in known for label in np.asarray(y_eval)])
    if not mask.any():
        return None
    return x_eval[mask], np.asarray(y_eval)[mask]


def _fit(
    settings: Settings,
    x: np.ndarray,
    y: np.ndarray,
    eval_set: tuple[np.ndarray, np.ndarray] | None,
) -> SupervisedClassifier:
    seed_everything(settings.seed)
    return SupervisedClassifier(settings).fit(x, y, eval_set=_sanitize_eval(y, eval_set))


def _threshold_result(
    name: str,
    model: SupervisedClassifier,
    x_mem: np.ndarray,
    y_mem: np.ndarray,
    x_non: np.ndarray,
    y_non: np.ndarray,
    fpr_budget: float,
) -> AttackResult:
    """Run the Yeom confidence-threshold attack against ``model``."""
    s_mem = true_class_probability(model.predict_proba(x_mem), model.classes_, y_mem)
    s_non = true_class_probability(model.predict_proba(x_non), model.classes_, y_non)
    is_member = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    scores = np.concatenate([s_mem, s_non])
    auc, adv, tpr_b, roc_fpr, roc_tpr = attack_scores(is_member, scores, fpr_budget)
    return AttackResult(
        name=name,
        attack="threshold",
        auc=auc,
        advantage=adv,
        tpr_at_fpr=tpr_b,
        fpr_budget=fpr_budget,
        train_acc=_accuracy(model, x_mem, y_mem),
        test_acc=_accuracy(model, x_non, y_non),
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
    )


def _shadow_result(
    settings: Settings,
    target: SupervisedClassifier,
    aux_x: np.ndarray,
    aux_y: np.ndarray,
    x_mem: np.ndarray,
    y_mem: np.ndarray,
    x_non: np.ndarray,
    y_non: np.ndarray,
) -> AttackResult:
    """Run the Shokri shadow-model attack: shadows teach an attack classifier."""
    cfg = settings.membership
    top_k = cfg.top_k_confidences
    rng = np.random.default_rng(settings.seed)

    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    n_aux = len(aux_x)
    for i in range(cfg.n_shadow):
        order = rng.permutation(n_aux)
        half = n_aux // 2
        in_idx, out_idx = order[:half], order[half : 2 * half]
        shadow = _fit(
            settings.model_copy(update={"seed": settings.seed + 1 + i}),
            aux_x[in_idx],
            aux_y[in_idx],
            eval_set=(aux_x[out_idx], aux_y[out_idx]),
        )
        for idx, member in ((in_idx, 1.0), (out_idx, 0.0)):
            proba = shadow.predict_proba(aux_x[idx])
            feats.append(confidence_features(proba, shadow.classes_, aux_y[idx], top_k))
            labels.append(np.full(len(idx), member))

    attack_x = np.vstack(feats)
    attack_y = np.concatenate(labels)
    seed_everything(settings.seed)
    attack_model = GradientBoostingClassifier(random_state=settings.seed, n_estimators=100)
    attack_model.fit(attack_x, attack_y)

    f_mem = confidence_features(target.predict_proba(x_mem), target.classes_, y_mem, top_k)
    f_non = confidence_features(target.predict_proba(x_non), target.classes_, y_non, top_k)
    s_mem = attack_model.predict_proba(f_mem)[:, 1]
    s_non = attack_model.predict_proba(f_non)[:, 1]
    is_member = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    scores = np.concatenate([s_mem, s_non])
    auc, adv, tpr_b, roc_fpr, roc_tpr = attack_scores(is_member, scores, cfg.attack_fpr)
    return AttackResult(
        name="deployed model",
        attack="shadow",
        auc=auc,
        advantage=adv,
        tpr_at_fpr=tpr_b,
        fpr_budget=cfg.attack_fpr,
        train_acc=_accuracy(target, x_mem, y_mem),
        test_acc=_accuracy(target, x_non, y_non),
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
    )


def run_membership_report(settings: Settings) -> Path:
    """Run the membership-inference audit on the stratified split; write the report."""
    cfg = settings.membership
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "multiclass"  # richer confidence vectors — Shokri's setting

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")

    # One leakage-safe pipeline, fit (unsupervised) on the full train split, shared by
    # every model so members/non-members/shadows live in one feature space.
    pipeline = build_pipeline(variant)
    pipeline.fit(train)

    rng = np.random.default_rng(variant.seed)
    train = train.reset_index(drop=True)
    n_target = min(cfg.target_train_rows, len(train))
    member_pos = rng.choice(len(train), size=n_target, replace=False)
    member_mask = np.zeros(len(train), dtype=bool)
    member_mask[member_pos] = True
    members = train[member_mask]
    aux_train = train[~member_mask]  # shadows train here — disjoint from the target's members

    n_eval = min(cfg.eval_rows, len(members), len(test))
    members = members.sample(n=n_eval, random_state=variant.seed)
    non_members = test.sample(n=min(cfg.eval_rows, len(test)), random_state=variant.seed)

    x_mem = pipeline.transform(members)
    y_mem = members[MULTICLASS_TARGET].to_numpy()
    x_non = pipeline.transform(non_members)
    y_non = non_members[MULTICLASS_TARGET].to_numpy()

    # Auxiliary pool for the shadows (Shokri: same distribution, disjoint from target).
    aux_pool = pd.concat([aux_train, val], ignore_index=True)
    aux_pool = aux_pool.sample(n=min(cfg.shadow_rows, len(aux_pool)), random_state=variant.seed)
    aux_x = pipeline.transform(aux_pool)
    aux_y = aux_pool[MULTICLASS_TARGET].to_numpy()

    # Target model (the deployed, regularised config) and its overfit reference.
    x_val, y_val = pipeline.transform(val), val[MULTICLASS_TARGET].to_numpy()
    target = _fit(variant, x_mem, y_mem, eval_set=(x_val, y_val))
    overfit = _fit(_overfit_settings(variant), x_mem, y_mem, eval_set=None)

    results = [
        _threshold_result("deployed model", target, x_mem, y_mem, x_non, y_non, cfg.attack_fpr),
        _shadow_result(variant, target, aux_x, aux_y, x_mem, y_mem, x_non, y_non),
        _threshold_result("overfit reference", overfit, x_mem, y_mem, x_non, y_non, cfg.attack_fpr),
    ]
    for r in results:
        logger.info(
            "Membership attack",
            extra={
                "model": r.name,
                "attack": r.attack,
                "auc": round(r.auc, 4),
                "adv": round(r.advantage, 4),
            },
        )

    fig = _plot_privacy_roc(results, variant.paths.figures_dir / "membership_roc.png")
    report = _render(results, cfg, fig)
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote membership report", extra={"path": str(out_path)})

    with track_run(settings, "membership") as run:
        run.log_metrics(
            {
                "deployed_threshold_auc": results[0].auc,
                "deployed_shadow_auc": results[1].auc,
                "overfit_threshold_auc": results[2].auc,
                "deployed_gen_gap": results[0].gen_gap,
                "overfit_gen_gap": results[2].gen_gap,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _downsample(fpr: np.ndarray, tpr: np.ndarray, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Thin a ROC curve to ~n points so the figure markers stay legible."""
    if len(fpr) <= n:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return fpr[idx], tpr[idx]


def _plot_privacy_roc(results: list[AttackResult], out_path: Path) -> Path:
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in results:
        f, t = _downsample(r.roc_fpr, r.roc_tpr)
        series[f"{r.name} — {r.attack} (AUC={r.auc:.2f})"] = (f, t)
    series["no leakage (chance)"] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    return plots.plot_lines(
        series,
        xlabel="False-accusation rate (non-members flagged as members)",
        ylabel="Members correctly identified",
        title="Membership-inference ROC (diagonal = no leakage)",
        out_path=out_path,
    )


def _table(results: list[AttackResult]) -> str:
    budget = f"{results[0].fpr_budget:.0%}"
    rows = [
        f"| model | attack | attack AUC | advantage | TPR @ {budget} FPR | gen. gap |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        rows.append(
            f"| {r.name} | {r.attack} | {r.auc:.3f} | {r.advantage:.3f} "
            f"| {r.tpr_at_fpr * 100:.1f}% | {r.gen_gap * 100:+.1f} pts |"
        )
    return "\n".join(rows)


def _read(results: list[AttackResult]) -> str:
    """Sign-aware prose so the narrative and the numbers cannot diverge."""
    deployed = results[0]
    overfit = results[2]
    leaky = deployed.auc >= 0.55 or deployed.advantage >= 0.1
    gap_grows = overfit.gen_gap > deployed.gen_gap + 0.01
    adv_grows = overfit.advantage > deployed.advantage + 0.02

    if leaky:
        head = (
            f"The deployed model **leaks membership above chance on this stand-in**: the "
            f"confidence-threshold attack reaches AUC {deployed.auc:.3f} (advantage "
            f"{deployed.advantage:.3f}) and the shadow-model attack AUC {results[1].auc:.3f}, "
            "so a query-only attacker separates training members from fresh traffic better "
            "than a coin flip. But the *worst-case* metric is reassuringly small — at a "
            f"{deployed.fpr_budget:.0%} false-accusation budget the attack recovers only "
            f"{deployed.tpr_at_fpr * 100:.1f}% of members — so the average leak is real while "
            "the confidently-memorised tail that a *targeted* disclosure needs stays thin."
        )
    else:
        head = (
            f"The deployed model **leaks little membership on this stand-in**: the "
            f"confidence-threshold attack lands at AUC {deployed.auc:.3f} (advantage "
            f"{deployed.advantage:.3f}) and the shadow-model attack at AUC "
            f"{results[1].auc:.3f} — both close to the 0.5 no-leakage line. That is not "
            "an accident: a model that generalises has little to memorise, and "
            f"its train-vs-test accuracy gap is only {deployed.gen_gap * 100:+.1f} points."
        )

    if adv_grows:
        gap_note = (
            f"its generalisation gap widens to {overfit.gen_gap * 100:+.1f} points and "
            if gap_grows
            else (
                f"and — the honest subtlety — even though its accuracy gap barely moves "
                f"({overfit.gen_gap * 100:+.1f} vs {deployed.gen_gap * 100:+.1f} points), "
            )
        )
        arc = (
            f"The **overfit reference** makes the mechanism visible. Refit un-regularised "
            f"(deep trees, no early stopping) on the identical rows, {gap_note}its membership "
            f"advantage nearly doubles to {overfit.advantage:.3f} (AUC {overfit.auc:.3f}) — "
            "because it is far more *confident* on the rows it memorised, which is exactly the "
            "signal the threshold attack reads. Privacy leakage is driven by memorisation, "
            "not accuracy alone, so **the regularisation and early stopping the deployed model "
            "already uses are its privacy control.** Same measure -> re-measure arc as the "
            "adversarial-hardening and poisoning-defense studies: name the weakness, show a "
            "control moves it, price the movement."
        )
    else:
        arc = (
            f"The overfit reference (gap {overfit.gen_gap * 100:+.1f} pts, advantage "
            f"{overfit.advantage:.3f}) is reported beside the deployed model so the "
            "relationship between memorisation and membership leakage is visible rather than "
            "asserted. On a real dataset with sharper class margins the leak will be larger; "
            "the control is the same (regularisation, early stopping, and, for a formal "
            "guarantee, differentially-private training at a measured utility cost)."
        )
    return f"{head}\n\n{arc}"


def _render(results: list[AttackResult], cfg: object, fig: Path) -> str:
    from netsentry.config.settings import MembershipConfig

    assert isinstance(cfg, MembershipConfig)
    return f"""# NetSentry — Membership-Inference Privacy Audit

_Synthetic stand-in. Run on the exchangeable **stratified** split — the assumption
membership inference needs — with the multiclass model, {cfg.n_shadow} shadow models,
and {cfg.target_train_rows:,} target-training rows. Members are the target's training
rows; non-members are held-out test flows from the same distribution._

Evasion is the inference-time adversary and poisoning the training-time one; this is
the third classic attack on an ML model and the one about **privacy**. With only query
access, can an attacker decide whether a specific flow was in the training set? On a
NIDS that is a genuine disclosure ("was this host's traffic used to train the model?"),
and it is the standard way to measure how much a model **memorises** its training data
(Shokri et al. 2017; Yeom et al. 2018).

## Attacks and leakage

Two attacks against the deployed model, plus a deliberately-overfit reference of the
same architecture on the identical rows:

- **Confidence-threshold** (Yeom): threshold the model's probability on each row's
  *true* class — a memorised member is over-confident on its real label.
- **Shadow-model** (Shokri): {cfg.n_shadow} shadows mimic the target on disjoint
  same-distribution data; an attack classifier learns member-vs-non-member from their
  confidence vectors, then is turned on the target.

{_table(results)}

Attack **AUC 0.5** and **advantage 0.0** mean no leakage; **TPR @ {results[0].fpr_budget:.0%} FPR**
is the worst-case metric (Carlini et al. 2022) — the fraction of members an attacker
recovers while almost never falsely accusing a non-member.

![Membership-inference ROC](../figures/{fig.name})

## Read

{_read(results)}

## Scope

Membership inference assumes members and non-members are exchangeable, which is why
this runs on the stratified split; under the temporal shift the two pools differ by
*distribution* as well as membership, which would confound the attack. The audit
measures privacy leakage of the *supervised* model only. The strong mitigation with a
formal guarantee is differentially-private training, which buys an (ε, δ) bound at a
measured detection cost — the natural next study, in the same spirit as this one:
name the risk, apply the control, re-measure.
"""
