"""Weak supervision: train the detector from the signature rules alone — zero labels.

Every supervised number in this project assumes someone labeled the training days. In a
real deployment nobody did: what a SOC actually has on day one is the incumbent signature
ruleset and a firehose of unlabeled flows. Data programming (Ratner, De Sa, Wu, Selsam &
Re, NeurIPS 2016 — the Snorkel line of work) turns exactly those assets into a trained
model, in two steps:

1. **Each signature becomes a labeling function (LF).** A rule votes *attack* on the flows
   it fires on and *abstains* everywhere else — the natural reading of a signature, which
   alerts or stays silent and never certifies benignness. The votes of all rules over the
   unlabeled training flows form a sparse, conflicting, incomplete label matrix.
2. **A generative label model resolves the votes without ground truth.** A two-class
   Dawid-Skene model (Dawid & Skene, 1979) treats the true label as latent and each LF as
   an independent noisy annotator with its own per-class vote distribution. EM alternates
   between inferring each flow's posterior attack probability and re-estimating every LF's
   accuracy, so the model learns *which signatures to trust, and how much*, purely from
   their agreement structure. Only **cast votes** enter the likelihood: a fired rule is
   evidence, an un-fired rule is missing data, never a benign vote.

The catch every data-programming paper states and this study measures: **agreement is the
only label-free evidence there is.** When labeling functions co-fire, their (dis)agreement
identifies who is right; when they never overlap, no method — EM, method-of-moments,
FlyingSquid-style triplets — can estimate accuracies, and an unanchored EM just drifts
(two failure modes were observed directly on this data and are documented in NOTES.md:
with abstention modelled as evidence, the dominant rule's silence testifies against every
other rule's flows and EM collapses them to benign in a rich-get-richer spiral; with cast
votes only and zero overlap, the likelihood becomes self-referential and bleeds toward
the prior). The label model here is therefore **agreement-gated**: given enough co-fire
mass it fits accuracies by EM (validated on planted overlapping LFs in the unit tests);
on disjoint signatures it says so and combines votes as a Bayesian believer at a stated
``signature_trust`` — and the report audits that belief against the ground truth the
model never saw.

One quantity is *not* learnable here, and the design says so instead of pretending: with
LFs that only ever vote attack or abstain, the class balance is unidentifiable (silence
could mean "benign" or "rare attack the rules miss" — the votes cannot tell). Snorkel's
label model takes ``class_balance`` as an input for exactly this reason, and so does this
study: ``class_prior`` is the one operator-supplied belief ("attacks are a small
minority of traffic" — a prior, never a label), the EM learns only the per-LF vote
tables under it, and a sensitivity sweep shows how much the student rides on the number.

The posteriors then train the ordinary downstream classifier (noise-aware: hard labels
weighted by the label model's confidence, ambiguous rows dropped). The payoff being
measured is Snorkel's central claim: the student sees the **full feature space** its
teachers never used, so it can generalise *past* them — in particular, onto the attacks
where **no rule fired at all**, where the ruleset's recall is 0 by construction.

One honest asymmetry is part of the design: the teachers may key on ``Destination Port``
(real signatures are port-scoped) while the student's feature pipeline still drops it —
weak supervision transfers the signatures' knowledge, not their memorisation. And every
number the label model states is audited post hoc against the ground truth it never saw,
so its assumptions are priced, not assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.rules import RuleEngine
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    import pandas as pd

    from netsentry.config import Settings
    from netsentry.config.settings import RuleDefinition, WeakSupervisionConfig

logger = get_logger(__name__)

REPORT_NAME = "weak_supervision.md"
FIGURE_NAME = "weak_supervision.png"

#: Vote values in the label matrix. Abstain is the LF staying silent, not a benign vote.
VOTE_ABSTAIN, VOTE_BENIGN, VOTE_ATTACK = -1, 0, 1


def votes_from_rules(df: pd.DataFrame, definitions: list[RuleDefinition]) -> np.ndarray:
    """The signature ruleset as a label matrix: fire = attack vote, silence = abstain.

    Shape ``(n_rows, n_rules)`` with values in {-1 abstain, 1 attack}. Signatures never
    vote benign — a rule that did not fire has said nothing, and encoding silence as a
    benign vote would smuggle in the exact assumption (no alert = clean) that the fixed
    class prior is meant to state explicitly instead.
    """
    matches = RuleEngine(definitions).matches(df)
    return np.where(matches.to_numpy(dtype=bool), VOTE_ATTACK, VOTE_ABSTAIN).astype(np.int8)


@dataclass
class LabelModel:
    """A fitted two-class Dawid-Skene generative model over LF votes.

    ``vote_probs[j, c, v]`` is P(LF j emits vote v | true class c), with the vote axis
    indexed as ``vote + 1`` (0 = abstain, 1 = benign, 2 = attack). ``prior`` is the
    estimated attack prevalence P(y = 1) — itself learned without labels.
    """

    prior: float
    vote_probs: np.ndarray
    n_iter: int
    converged: bool

    def posterior(self, votes: np.ndarray) -> np.ndarray:
        """P(y = 1 | cast votes) per row, computed in log space for numerical safety.

        Only cast votes contribute likelihood: an abstaining LF is missing data, so an
        all-abstain row's posterior is exactly the prior. (Including abstention factors
        is what triggers the rich-get-richer collapse documented in the module docstring.)
        """
        v_idx = np.asarray(votes, dtype=int) + 1
        log_like = np.zeros((len(v_idx), 2))
        for c in (0, 1):
            for j in range(self.vote_probs.shape[0]):
                cast = v_idx[:, j] != VOTE_ABSTAIN + 1
                log_like[cast, c] += np.log(self.vote_probs[j, c, v_idx[cast, j]])
        log_like[:, 1] += np.log(self.prior)
        log_like[:, 0] += np.log(1.0 - self.prior)
        # Stable softmax over the two classes.
        shift = log_like.max(axis=1, keepdims=True)
        like = np.exp(log_like - shift)
        return np.asarray(like[:, 1] / like.sum(axis=1))

    def implied_precision(self, j: int) -> float:
        """P(y = 1 | LF j voted attack): the precision the model believes signature j has."""
        p_attack = self.prior * self.vote_probs[j, 1, VOTE_ATTACK + 1]
        p_benign = (1.0 - self.prior) * self.vote_probs[j, 0, VOTE_ATTACK + 1]
        total = p_attack + p_benign
        return float(p_attack / total) if total > 0 else float("nan")


def fit_label_model(
    votes: np.ndarray, cfg: WeakSupervisionConfig, class_prior: float | None = None
) -> LabelModel:
    """Fit the Dawid-Skene label model by EM on the vote matrix alone (no labels).

    The class prior is **fixed** at ``class_prior`` (default ``cfg.class_prior``), not
    re-estimated: attack-or-abstain LFs cannot identify the class balance (all-abstain
    rows drift toward posterior 0.5 and the EM wanders), which is why Snorkel's label
    model takes class balance as an input. EM therefore learns only the per-LF vote
    tables. Initialisation anchors the polarity — rows with net attack votes start
    attack-leaning at ``signature_trust`` — breaking the label-switching symmetry, and
    ``smoothing`` is a Laplace pseudo-count keeping every vote probability off exact 0/1.
    The fit is fully deterministic: same votes, same model. Callers should reach it via
    :func:`build_label_model`, which refuses to fit when there is no agreement to fit on.
    """
    v = np.asarray(votes, dtype=int)
    n_rows, n_lfs = v.shape
    v_idx = v + 1
    one_hot = np.eye(3)[v_idx]  # (rows, lfs, 3)
    prior = float(np.clip(class_prior if class_prior is not None else cfg.class_prior, 1e-4, 0.9))

    net = (v == VOTE_ATTACK).sum(axis=1) - (v == VOTE_BENIGN).sum(axis=1)
    w = np.full(n_rows, prior)
    w[net > 0] = cfg.signature_trust
    w[net < 0] = 1.0 - cfg.signature_trust

    vote_probs = np.full((n_lfs, 2, 3), 1.0 / 3.0)
    converged = False
    n_iter = 0
    for n_iter in range(1, cfg.em_max_iter + 1):
        # M-step: re-estimate each LF's per-class vote table from the current posteriors.
        for c, weight in ((1, w), (0, 1.0 - w)):
            counts = np.einsum("i,ijv->jv", weight, one_hot)
            vote_probs[:, c, :] = (counts + cfg.smoothing) / (weight.sum() + 3.0 * cfg.smoothing)
        # E-step: posterior attack probability per row under the new tables + fixed prior.
        model = LabelModel(prior=prior, vote_probs=vote_probs, n_iter=n_iter, converged=False)
        w_new = model.posterior(v)
        delta = float(np.abs(w_new - w).mean())
        w = w_new
        if delta < cfg.em_tol:
            converged = True
            break
    return LabelModel(prior=prior, vote_probs=vote_probs, n_iter=n_iter, converged=converged)


def cofire_rows(votes: np.ndarray) -> int:
    """Rows carrying two or more cast votes — the agreement mass EM needs to learn from."""
    v = np.asarray(votes, dtype=int)
    return int(((v != VOTE_ABSTAIN).sum(axis=1) >= 2).sum())


def prior_belief_model(
    n_lfs: int, cfg: WeakSupervisionConfig, class_prior: float | None = None
) -> LabelModel:
    """A label model that *believes* rather than fits: every signature at ``signature_trust``.

    When signatures never co-fire there is no agreement to learn accuracies from, so the
    honest combiner states its beliefs instead: the posterior of a flow fired on by one
    signature is exactly ``signature_trust``, votes compose by naive-Bayes odds, and
    silence returns the class prior. Built as vote tables whose fire-likelihood ratio is
    ``trust/(1-trust) * (1-prior)/prior``, so ``implied_precision`` reads back the stated
    trust — the report's audit column then prices the belief against ground truth.
    """
    prior = float(np.clip(class_prior if class_prior is not None else cfg.class_prior, 1e-4, 0.9))
    trust = float(np.clip(cfg.signature_trust, 1e-3, 1.0 - 1e-3))
    ratio = (trust / (1.0 - trust)) * ((1.0 - prior) / prior)
    # Arbitrary fire scale — only the ratio enters the cast-votes posterior — kept small
    # enough that every table cell stays a probability.
    base = min(0.01, 0.5 / ratio)
    vote_probs = np.empty((n_lfs, 2, 3))
    vote_probs[:, 1, VOTE_ATTACK + 1] = base * ratio
    vote_probs[:, 0, VOTE_ATTACK + 1] = base
    vote_probs[:, 1, VOTE_BENIGN + 1] = base
    vote_probs[:, 0, VOTE_BENIGN + 1] = base * ratio
    vote_probs[:, 1, VOTE_ABSTAIN + 1] = 1.0 - vote_probs[:, 1, 1:].sum(axis=1)
    vote_probs[:, 0, VOTE_ABSTAIN + 1] = 1.0 - vote_probs[:, 0, 1:].sum(axis=1)
    return LabelModel(prior=prior, vote_probs=vote_probs, n_iter=0, converged=True)


def build_label_model(
    votes: np.ndarray, cfg: WeakSupervisionConfig, class_prior: float | None = None
) -> tuple[LabelModel, str]:
    """The agreement gate: fit accuracies by EM only when there is agreement to fit on.

    Returns the model and the mode it was built in — ``"agreement (EM)"`` when at least
    ``cfg.min_cofire_rows`` rows carry two or more votes, else ``"prior belief"``. An EM
    run on disjoint one-sided votes has no anchor (the likelihood is self-referential)
    and drifts wherever smoothing pushes it, so refusing to fit *is* the correct fit.
    """
    n_cofire = cofire_rows(votes)
    if n_cofire >= cfg.min_cofire_rows:
        return fit_label_model(votes, cfg, class_prior=class_prior), "agreement (EM)"
    logger.info(
        "Label model in prior-belief mode", extra={"cofire_rows": n_cofire, "n_lfs": votes.shape[1]}
    )
    return prior_belief_model(votes.shape[1], cfg, class_prior=class_prior), "prior belief"


def noise_aware_labels(
    posterior: np.ndarray, min_weight: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posteriors to training material: hard labels, confidence weights, and a keep mask.

    The hard label is the label model's most likely class; the weight ``|2p - 1|`` is its
    confidence, so a row the model is agnostic about teaches the student nothing. Rows
    below ``min_weight`` are dropped outright — training on a coin flip is pure noise.
    """
    p = np.asarray(posterior, dtype=float)
    y_hard = (p >= 0.5).astype(int)
    weights = np.abs(2.0 * p - 1.0)
    keep = weights >= min_weight
    return y_hard, weights, keep


def top_k_stats(
    scores: np.ndarray, y_true: np.ndarray, k: int, silent_mask: np.ndarray
) -> tuple[float, float, float]:
    """Precision / recall / rule-silent recall of a scorer's top-``k`` alerts.

    Matching every detector at the same alert volume (the union ruleset's own) makes the
    comparison threshold-free: no detector gets to buy recall with alerts the incumbent
    would never have raised. ``silent_mask`` marks the attacks no rule fired on — the
    region where the ruleset's recall is 0 by construction.
    """
    y = np.asarray(y_true, dtype=int)
    if k <= 0:
        return float("nan"), 0.0, 0.0
    top = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]
    hit = int(y[top].sum())
    n_attacks = int(y.sum())
    silent_total = int(y[silent_mask].sum())
    in_top = np.zeros(len(y), dtype=bool)
    in_top[top] = True
    silent_hit = int(y[in_top & silent_mask].sum())
    return (
        hit / k,
        hit / n_attacks if n_attacks else 0.0,
        silent_hit / silent_total if silent_total else float("nan"),
    )


@dataclass
class LFReadout:
    """One signature's coverage and its estimated-vs-true precision."""

    name: str
    coverage: float  # share of training flows the rule fires on
    est_precision: float  # what the label model believes, learned without labels
    true_precision: float  # ground truth, used only to validate the estimate


@dataclass
class DetectorOutcome:
    """One detector's showing on the honest temporal test split."""

    name: str
    pr_auc: float  # raw-score average precision; NaN for the binary union decision
    precision_at_k: float
    recall_at_k: float
    silent_recall_at_k: float  # recall on attacks no signature fired on


@dataclass
class PriorPoint:
    """The student's showing under one assumed class balance (the sensitivity sweep)."""

    assumed_prior: float
    weak_label_accuracy: float
    student_pr_auc: float


@dataclass
class WeakSupervisionStudy:
    """The full weak-supervision study: label model quality + detector comparison."""

    n_train: int
    n_test: int
    true_prior: float
    assumed_prior: float  # the configured operator belief; never tuned to the truth
    label_mode: str  # "agreement (EM)" or "prior belief" — what the gate chose
    n_cofire: int  # rows with two or more votes: the agreement mass available
    signature_trust: float  # the assumed precision the prior-belief mode states
    weak_label_accuracy: float  # posterior >= 0.5 vs the truth the model never saw
    kept_share: float  # training rows surviving the confidence floor
    em_iters: int
    em_converged: bool
    lfs: list[LFReadout]
    alert_volume: int  # the union ruleset's own test alert count (the matched volume)
    n_test_attacks: int
    n_silent_attacks: int  # test attacks no rule fired on
    outcomes: list[DetectorOutcome]
    prior_points: list[PriorPoint]


def run_weak_supervision(settings: Settings) -> WeakSupervisionStudy:
    """Fit the label model on unlabeled train votes, train the student, compare honestly."""
    cfg = settings.weak_supervision
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test").reset_index(drop=True)
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    y_test = test[BINARY_TARGET].to_numpy().astype(int)

    definitions = settings.rules.definitions
    votes_train = votes_from_rules(train, definitions)
    votes_val = votes_from_rules(val, definitions)
    votes_test = votes_from_rules(test, definitions)

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))

    def fit_student_at(prior: float) -> tuple[LabelModel, str, np.ndarray, float, float]:
        """Label model -> weak labels -> student, at one assumed class balance.

        Returns the label model, the gate's mode, the student's test scores, the weak
        labels' hidden accuracy, and the share of rows the confidence floor kept. The
        student never sees a true label: weak labels, weak eval set, weak weights.
        """
        lm, mode = build_label_model(votes_train, cfg, class_prior=prior)
        post_tr = lm.posterior(votes_train)
        post_v = lm.posterior(votes_val)
        y_weak, weights, keep = noise_aware_labels(post_tr, cfg.min_weight)
        if len(np.unique(y_weak[keep])) < 2:
            raise ValueError(
                "Weak labels collapsed to one class; the configured rules never fire on this data"
            )
        seed_everything(variant.seed)
        student_model = SupervisedClassifier(variant).fit(
            x_train[keep],
            y_weak[keep],
            eval_set=(x_val, (post_v >= 0.5).astype(int)),
            sample_weight=weights[keep],
        )
        scores = positive_scores(student_model.predict_proba(x_test), student_model.classes_)
        hidden_acc = float(((post_tr >= 0.5).astype(int) == y_train).mean())
        return lm, mode, scores, hidden_acc, float(keep.mean())

    label_model, label_mode, student_scores, weak_acc, kept_share = fit_student_at(cfg.class_prior)

    prior_points: list[PriorPoint] = []
    for prior in cfg.prior_sensitivity:
        if np.isclose(prior, cfg.class_prior):
            scores_p, acc_p = student_scores, weak_acc
        else:
            _, _, scores_p, acc_p, _ = fit_student_at(prior)
        prior_points.append(
            PriorPoint(
                assumed_prior=prior,
                weak_label_accuracy=acc_p,
                student_pr_auc=float(average_precision_score(y_test, scores_p)),
            )
        )

    lfs = [
        LFReadout(
            name=definitions[j].name,
            coverage=float((votes_train[:, j] == VOTE_ATTACK).mean()),
            est_precision=label_model.implied_precision(j),
            true_precision=(
                float(y_train[votes_train[:, j] == VOTE_ATTACK].mean())
                if (votes_train[:, j] == VOTE_ATTACK).any()
                else float("nan")
            ),
        )
        for j in range(len(definitions))
    ]

    # The fully-supervised ceiling: identical architecture, true labels.
    seed_everything(variant.seed)
    ceiling = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))
    ceiling_scores = positive_scores(ceiling.predict_proba(x_test), ceiling.classes_)

    union_fired = RuleEngine(definitions).decisions(test)
    silent_mask = ~union_fired
    label_model_scores = label_model.posterior(votes_test)

    k = int(union_fired.sum())
    union_precision = float(y_test[union_fired].mean()) if k else float("nan")
    union_recall = float(y_test[union_fired].sum() / y_test.sum()) if y_test.sum() else 0.0
    outcomes = [
        DetectorOutcome(
            name="signature union (the teachers)",
            pr_auc=float("nan"),
            precision_at_k=union_precision,
            recall_at_k=union_recall,
            silent_recall_at_k=0.0,  # by construction: silence is exactly where no rule fired
        )
    ]
    for name, scores in (
        ("label model only (votes, no features)", label_model_scores),
        ("weak student (trained on zero labels)", student_scores),
        ("supervised ceiling (true labels)", ceiling_scores),
    ):
        precision, recall, silent = top_k_stats(scores, y_test, k, silent_mask)
        outcomes.append(
            DetectorOutcome(
                name=name,
                pr_auc=float(average_precision_score(y_test, scores)),
                precision_at_k=precision,
                recall_at_k=recall,
                silent_recall_at_k=silent,
            )
        )
        logger.info(
            "Weak-supervision detector",
            extra={"detector": name, "pr_auc": round(outcomes[-1].pr_auc, 4)},
        )

    return WeakSupervisionStudy(
        n_train=len(train),
        n_test=len(test),
        true_prior=float(y_train.mean()),
        assumed_prior=cfg.class_prior,
        label_mode=label_mode,
        n_cofire=cofire_rows(votes_train),
        signature_trust=cfg.signature_trust,
        weak_label_accuracy=weak_acc,
        kept_share=kept_share,
        em_iters=label_model.n_iter,
        em_converged=label_model.converged,
        lfs=lfs,
        alert_volume=k,
        n_test_attacks=int(y_test.sum()),
        n_silent_attacks=int(y_test[silent_mask].sum()),
        outcomes=outcomes,
        prior_points=prior_points,
    )


def run_weak_supervision_report(settings: Settings) -> Path:
    """Run the weak-supervision study and write the report + figure."""
    study = run_weak_supervision(settings)

    fig = plots.plot_barh(
        [o.name for o in study.outcomes],
        [0.0 if np.isnan(o.silent_recall_at_k) else o.silent_recall_at_k for o in study.outcomes],
        xlabel="recall on attacks no signature fired on (matched alert volume)",
        title="Generalising past the teachers: rule-silent attack recall",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote weak-supervision report", extra={"path": str(out_path)})

    student = study.outcomes[2]
    ceiling = study.outcomes[3]
    with track_run(settings, "weak_supervision") as run:
        run.log_metrics(
            {
                "student_pr_auc": student.pr_auc,
                "ceiling_pr_auc": ceiling.pr_auc,
                "recovered_share": student.pr_auc / ceiling.pr_auc if ceiling.pr_auc else 0.0,
                "assumed_prior": study.assumed_prior,
                "true_prior": study.true_prior,
                "weak_label_accuracy": study.weak_label_accuracy,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _fmt(x: float, pct: bool = False) -> str:
    if np.isnan(x):
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.3f}"


def _lf_table(study: WeakSupervisionStudy) -> str:
    rows = [
        "| signature (labeling function) | coverage | model precision (no labels) "
        "| true precision | error |",
        "|---|---|---|---|---|",
    ]
    for lf in study.lfs:
        err = (
            "n/a"
            if np.isnan(lf.true_precision) or np.isnan(lf.est_precision)
            else f"{lf.est_precision - lf.true_precision:+.3f}"
        )
        rows.append(
            f"| {lf.name} | {lf.coverage:.1%} | {_fmt(lf.est_precision)} "
            f"| {_fmt(lf.true_precision)} | {err} |"
        )
    return "\n".join(rows)


def _lf_intro(study: WeakSupervisionStudy) -> str:
    if study.label_mode == "agreement (EM)":
        return (
            "The model column is estimated purely from vote agreement (EM); the true column "
            "(ground truth the model never saw) is shown only to validate it."
        )
    return (
        f"The signatures co-fire on only **{study.n_cofire}** of {study.n_train:,} rows — "
        "agreement is the sole label-free evidence about a labeling function's accuracy, and "
        "with none of it, *no* method (EM, method-of-moments, triplets) can estimate these "
        "numbers; an unanchored fit just drifts. The gate therefore refused to fit and the "
        f"model column states the configured trust ({study.signature_trust:.2f}) instead. The "
        "true column is the post-hoc audit of that belief — the spread in it is exactly what "
        "labels (or overlapping signatures) would have bought."
    )


def _detector_table(study: WeakSupervisionStudy) -> str:
    rows = [
        "| detector | PR-AUC | precision @ matched volume | recall @ matched volume "
        "| rule-silent attack recall |",
        "|---|---|---|---|---|",
    ]
    for o in study.outcomes:
        rows.append(
            f"| {o.name} | {_fmt(o.pr_auc)} | {_fmt(o.precision_at_k, pct=True)} "
            f"| {_fmt(o.recall_at_k, pct=True)} | {_fmt(o.silent_recall_at_k, pct=True)} |"
        )
    return "\n".join(rows)


def _read(study: WeakSupervisionStudy) -> str:
    union, label_only, student, ceiling = study.outcomes
    errors = [
        abs(lf.est_precision - lf.true_precision)
        for lf in study.lfs
        if not (np.isnan(lf.est_precision) or np.isnan(lf.true_precision))
    ]
    mae = float(np.mean(errors)) if errors else float("nan")
    recover = student.pr_auc / ceiling.pr_auc if ceiling.pr_auc else 0.0

    audited = [lf for lf in study.lfs if not np.isnan(lf.true_precision)]
    if study.label_mode != "agreement (EM)" and audited:
        worst = min(audited, key=lambda lf: lf.true_precision)
        best = max(audited, key=lambda lf: lf.true_precision)
        est_clause = (
            f"The audit prices the flat trust: **{worst.name}** is actually only "
            f"{worst.true_precision:.0%} precise against the stated "
            f"{study.signature_trust:.0%}, while {best.name} runs at "
            f"{best.true_precision:.0%}. With zero co-fire the label model *cannot* discover "
            "this from data — the weak labels inherit each signature's real precision as "
            "label noise the student has to absorb."
        )
    elif not np.isnan(mae) and mae <= 0.15:
        est_clause = (
            f"The label model estimated each signature's precision within "
            f"**{mae:.3f}** (mean absolute error) of the truth it never saw — the agreement "
            "structure of the votes alone, read under a coarse class-balance belief, carries "
            "that much."
        )
    else:
        est_clause = (
            f"The label model's precision estimates miss the truth by {mae:.3f} on average — "
            "the signatures' overlap is thin enough that the conditional-independence "
            "assumption visibly bends, reported as it fell."
        )
    if not np.isnan(student.silent_recall_at_k) and student.silent_recall_at_k >= 0.05:
        silent_clause = (
            f" The student's reach past its teachers is the headline: at the ruleset's own "
            f"alert volume it recovers **{student.silent_recall_at_k:.0%}** of the "
            f"{study.n_silent_attacks:,} test attacks no signature fired on — flows on which "
            f"the union's recall is 0 by construction — while the votes-only label model "
            f"manages {_fmt(label_only.silent_recall_at_k, pct=True)} (it cannot see past its "
            "inputs either; only the student, which reads the full feature space, can)."
        )
    else:
        silent_clause = (
            f" The student's reach past its teachers is thin at the matched volume: "
            f"{_fmt(student.silent_recall_at_k, pct=True)} of the {study.n_silent_attacks:,} "
            f"rule-silent test attacks, against the supervised ceiling's "
            f"{_fmt(ceiling.silent_recall_at_k, pct=True)} — taught only by teachers who never "
            "alert on those flows, its top alerts stay a smoothed union, reported plainly."
        )
    upset = (
        " The weak student *beating* its supervised ceiling on the temporal split is not a "
        "paradox: coarse signature-shaped labels encode behaviours that stay stable across "
        "days, while true labels let the model fit day-specific patterns that do not survive "
        "the shift — the leaderboard's simple-models-win-temporally finding, restated in the "
        "label dimension."
        if recover > 1.02
        else ""
    )
    ceiling_clause = (
        f" End to end, a detector trained on **zero labels** lands a PR-AUC of "
        f"{student.pr_auc:.3f} against the fully-supervised ceiling's {ceiling.pr_auc:.3f} "
        f"— **{recover:.0%} of the ceiling** — with the incumbent union at "
        f"{_fmt(union.precision_at_k, pct=True)} precision / "
        f"{_fmt(union.recall_at_k, pct=True)} recall at its own volume." + upset
    )
    return est_clause + silent_clause + ceiling_clause


def _prior_table(study: WeakSupervisionStudy) -> str:
    rows = [
        "| assumed P(attack) | hidden weak-label accuracy | student PR-AUC |",
        "|---|---|---|",
    ]
    for p in study.prior_points:
        marker = " (configured)" if np.isclose(p.assumed_prior, study.assumed_prior) else ""
        rows.append(
            f"| {p.assumed_prior:.2f}{marker} | {p.weak_label_accuracy:.1%} "
            f"| {p.student_pr_auc:.3f} |"
        )
    return "\n".join(rows)


def _prior_read(study: WeakSupervisionStudy) -> str:
    if not study.prior_points:
        return ""
    aucs = [p.student_pr_auc for p in study.prior_points]
    spread = max(aucs) - min(aucs)
    lo = min(p.assumed_prior for p in study.prior_points)
    hi = max(p.assumed_prior for p in study.prior_points)
    if spread <= 0.05:
        return (
            f"Misstating the prior by this whole range ({lo:.2f} to {hi:.2f}, against a true "
            f"prevalence of {study.true_prior:.2f}) moves the student's PR-AUC by only "
            f"{spread:.3f} — the belief has to be *coarse*, not correct."
        )
    return (
        f"The student's PR-AUC moves by {spread:.3f} across the {lo:.2f} to {hi:.2f} prior "
        f"range (true prevalence {study.true_prior:.2f}) — on this stand-in the assumed class "
        "balance is a real knob, reported as it fell; a deployment should bracket it the same "
        "way before trusting the weak model."
    )


def _render(study: WeakSupervisionStudy, fig: Path) -> str:
    if study.label_mode == "agreement (EM)":
        mode_line = (
            "label model in agreement mode ("
            + (
                f"EM converged in {study.em_iters} iterations"
                if study.em_converged
                else f"EM hit the {study.em_iters}-iteration cap"
            )
            + ")"
        )
    else:
        row_word = "row" if study.n_cofire == 1 else "rows"
        mode_line = (
            f"label model in **prior-belief mode** — the signatures co-fire on only "
            f"{study.n_cofire} {row_word}, so the agreement gate refused to fit accuracies"
        )
    return f"""# NetSentry — Weak Supervision (the signatures as teachers)

_Synthetic stand-in. Honest temporal/binary split; {study.n_train:,} training flows whose
labels were never shown to the label model or the student. Assumed class balance
{study.assumed_prior:.2f} (an operator belief, not tuned to the true {study.true_prior:.2f});
{mode_line}; the confidence floor kept {study.kept_share:.0%} of training rows. Every
detector is compared at the signature union's own test alert volume
({study.alert_volume:,} alerts against {study.n_test_attacks:,} true attacks)._

## Why this report exists

Every supervised result in this project assumes labeled training days; a real deployment
starts with none. What it does have is the incumbent signature ruleset. Data programming
(Ratner et al., NeurIPS 2016) treats each signature as a **labeling function** — voting
attack where it fires, abstaining elsewhere — and resolves the votes with a generative
label model, under one operator-supplied belief: a coarse class balance, required because
attack-or-abstain votes cannot identify it (the same reason Snorkel takes
``class_balance`` as an input). The label model is **agreement-gated**: signature
accuracies are estimable from votes only where signatures *co-fire*, so it fits them by
EM when that agreement mass exists and otherwise states a fixed trust and says so. Its
posteriors then train the ordinary downstream model, noise-aware. The question the study
prices: how much of the fully-supervised ceiling does a model trained on **zero labels**
recover, and can it detect attacks **no signature fires on** — where the ruleset's recall
is 0 by definition?

## What the label model can (and cannot) learn without labels

{_lf_table(study)}

{_lf_intro(study)} The weak labels themselves agree with the hidden truth on
**{study.weak_label_accuracy:.1%}** of training flows.

## Detectors on the honest temporal test split

{_detector_table(study)}

![Rule-silent attack recall](../figures/{fig.name})

{_read(study)}

## Sensitivity to the assumed class balance

{_prior_table(study)}

{_prior_read(study)}

## Scope

The label model assumes conditionally independent labeling functions wherever it fits at
all, and the agreement gate is the admission that hand-written rulesets — these included —
are usually engineered to be *disjoint*, which puts their accuracies beyond any label-free
estimator; the model-vs-true audit table is therefore part of the contract, not an
appendix. The
teachers may key on `Destination Port` (real signatures are port-scoped) while the
student's pipeline still drops it: weak supervision transfers the signatures' knowledge,
never their memorisation, so the leakage firewall holds in the weak regime too. And the
comparison is deliberately volume-matched — a detector only gets credit it could claim at
the incumbent's own alert budget. What this study does not claim: that weak supervision
replaces labeling. It prices the *starting point* — what a SOC can field before the first
labeled day exists — and the ceiling column is the argument for buying labels next (the
active-learning study prices exactly which ones)."""
