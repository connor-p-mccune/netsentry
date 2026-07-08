"""Distill the detector into rules a human can audit — and price the fidelity.

The rules baseline (``netsentry rules``) benchmarks *hand-written* signatures
against the model; this asks the inverse question: how much of the **learned**
model survives translation into an auditable form? A depth-limited decision tree
is trained to imitate the teacher's attack ranking (the raw score; the monotone
calibrator applies identically on top, so serving semantics are unchanged), then
judged two ways, because they answer different questions:

- **Fidelity to the teacher** — does the surrogate reproduce the teacher's
  *behavior*? Spearman rank correlation of their scores on the test split, plus
  decision agreement at a matched alert volume (both models alerting on the same
  fraction of validation traffic, so agreement is not an artifact of mismatched
  operating points).
- **Detection in its own right** — what does interpretability cost? The
  surrogate's PR-AUC and detection at the operating budget, beside the teacher's.

Two honest framings the report carries: a surrogate explains the teacher's
*behavior*, not its mechanism (high fidelity means "these rules mimic the model
here", not "the model reasons this way"); and a K-leaf tree emits only K distinct
scores, so tight false-positive budgets are unreachable by construction — score
quantization is itself an interpretability cost, made visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.confidence import pr_auc
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.features.feature_sets import display_feature_name
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "distill.md"


@dataclass
class DepthOutcome:
    """One surrogate depth: its size, its fidelity, and its own detection."""

    depth: int
    n_leaves: int
    spearman: float
    decision_agreement: float
    pr_auc: float
    tpr_at_budget: float


def matched_volume_threshold(reference_scores: np.ndarray, alert_fraction: float) -> float:
    """The threshold at which ``reference_scores`` alerts on ``alert_fraction`` of rows.

    Used to put the surrogate at the *same* validation alert volume as the teacher,
    so decision agreement compares behavior, not operating-point choices.
    """
    scores = np.asarray(reference_scores, dtype=float)
    if len(scores) == 0:
        return float("inf")
    fraction = float(np.clip(alert_fraction, 0.0, 1.0))
    return float(np.quantile(scores, 1.0 - fraction)) if fraction > 0 else float("inf")


def fidelity_metrics(
    teacher_scores: np.ndarray,
    surrogate_scores: np.ndarray,
    *,
    teacher_threshold: float,
    surrogate_threshold: float,
) -> dict[str, float]:
    """How faithfully the surrogate reproduces the teacher's behavior.

    Spearman compares the full rankings; decision agreement compares thresholded
    verdicts with each side at its own (volume-matched) threshold. Degenerate
    (constant) score vectors yield a Spearman of 0.0 rather than NaN.
    """
    teacher = np.asarray(teacher_scores, dtype=float)
    surrogate = np.asarray(surrogate_scores, dtype=float)
    if len(np.unique(teacher)) < 2 or len(np.unique(surrogate)) < 2:
        rank_corr = 0.0
    else:
        rank_corr = float(spearmanr(teacher, surrogate).statistic)
    agreement = float(np.mean((teacher >= teacher_threshold) == (surrogate >= surrogate_threshold)))
    return {"spearman": rank_corr, "decision_agreement": agreement}


def run_distill_report(settings: Settings) -> Path:
    """Distill the temporal model across tree depths and write the frontier report."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    cfg = settings.distill
    budget_fpr = settings.thresholds.fpr_targets[-1]

    result = fit_supervised(variant)
    bundle = result.bundle
    train = load_split(variant, "temporal", "train")
    val = load_split(variant, "temporal", "val")
    test = load_split(variant, "temporal", "test")
    y_test = test[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)

    # The surrogate imitates the teacher's raw attack ranking. The calibrator is a
    # monotone map applied identically on top (calibrator(surrogate) serves the same
    # semantics), and raw scores avoid isotonic tie-plateaus that would make rank
    # fidelity ill-behaved. Raw is also the scale of the headline PR-AUC, so the
    # teacher row here matches the evaluation report exactly.
    x_train = bundle.pipeline.transform(train)
    x_val = bundle.pipeline.transform(val)
    x_test = bundle.pipeline.transform(test)
    t_train = positive_scores(np.asarray(bundle.model.predict_proba(x_train)), result.classes)
    t_val = positive_scores(result.proba_val, result.classes)
    t_test = positive_scores(result.proba_test, result.classes)

    teacher_threshold = threshold_at_fpr(y_val, t_val, budget_fpr)
    teacher_alert_fraction = float(np.mean(t_val >= teacher_threshold))
    teacher_pr = pr_auc(y_test, t_test)
    teacher_tpr = rates_at_threshold(y_test, t_test, teacher_threshold)["tpr"]

    outcomes: list[DepthOutcome] = []
    rendered_rules = ""
    for depth in cfg.depths:
        seed_everything(variant.seed)
        surrogate = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=variant.seed,
        ).fit(x_train, t_train)
        s_val = surrogate.predict(x_val)
        s_test = surrogate.predict(x_test)
        surrogate_threshold = matched_volume_threshold(s_val, teacher_alert_fraction)
        fidelity = fidelity_metrics(
            t_test,
            s_test,
            teacher_threshold=teacher_threshold,
            surrogate_threshold=surrogate_threshold,
        )
        outcomes.append(
            DepthOutcome(
                depth=depth,
                n_leaves=int(surrogate.get_n_leaves()),
                spearman=fidelity["spearman"],
                decision_agreement=fidelity["decision_agreement"],
                pr_auc=pr_auc(y_test, s_test),
                tpr_at_budget=rates_at_threshold(
                    y_test, s_test, threshold_at_fpr(y_val, s_val, budget_fpr)
                )["tpr"],
            )
        )
        logger.info(
            "Distilled surrogate",
            extra={"depth": depth, "spearman": round(fidelity["spearman"], 3)},
        )
        if depth == cfg.report_depth:
            # Strip the ColumnTransformer prefix (numeric__...) for the human reader.
            names = [display_feature_name(str(n)) for n in bundle.feature_names()]
            lines = export_text(surrogate, feature_names=names).splitlines()
            if len(lines) > cfg.max_rule_lines:
                lines = [*lines[: cfg.max_rule_lines], "... (truncated)"]
            rendered_rules = "\n".join(lines)

    depths = np.array([o.depth for o in outcomes], dtype=float)
    fig = plots.plot_lines(
        {
            "fidelity to teacher (Spearman)": (depths, np.array([o.spearman for o in outcomes])),
            "surrogate PR-AUC": (depths, np.array([o.pr_auc for o in outcomes])),
            "teacher PR-AUC": (depths, np.full(len(outcomes), teacher_pr)),
        },
        xlabel="Surrogate tree depth",
        ylabel="Fidelity / PR-AUC",
        title="How much model fits in an auditable tree?",
        out_path=settings.paths.figures_dir / "distill.png",
    )

    report = _render(settings, outcomes, teacher_pr, teacher_tpr, budget_fpr, rendered_rules, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote distillation report", extra={"path": str(out_path)})

    with track_run(settings, "distill") as run:
        run.log_params({"depths": str(cfg.depths), "report_depth": cfg.report_depth})
        best = max(outcomes, key=lambda o: o.spearman)
        run.log_metrics(
            {
                "teacher_pr_auc": teacher_pr,
                "best_spearman": best.spearman,
                "best_depth": float(best.depth),
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _render(
    settings: Settings,
    outcomes: list[DepthOutcome],
    teacher_pr: float,
    teacher_tpr: float,
    budget_fpr: float,
    rendered_rules: str,
    fig: Path,
) -> str:
    cfg = settings.distill
    rows = [
        "| depth | leaves (rules) | fidelity (Spearman) | decision agreement | "
        f"surrogate PR-AUC | surrogate TPR @ {budget_fpr * 100:g}% FPR |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        rows.append(
            f"| {o.depth} | {o.n_leaves} | {o.spearman:.3f} | {o.decision_agreement:.1%} | "
            f"{o.pr_auc:.3f} | {o.tpr_at_budget:.1%} |"
        )
    chosen = next((o for o in outcomes if o.depth == cfg.report_depth), outcomes[-1])
    deepest = outcomes[-1]
    return f"""# NetSentry - Surrogate Distillation (the auditable approximation)

_Synthetic stand-in; the method is the point. A depth-limited decision tree is
trained to imitate the teacher's **attack ranking** over the temporal training
split (raw score; the monotone calibrator applies identically on top, so serving
semantics are unchanged) and judged on the temporal test split. Teacher: PR-AUC
**{teacher_pr:.3f}**, detection **{teacher_tpr:.1%}** at the
{budget_fpr * 100:g}% FP budget - the headline evaluation's own numbers._

## The question

The [rules baseline](rules.md) benchmarks hand-written signatures against the
model; this asks the inverse: how much of the *learned* model survives translation
into a form an auditor can read end-to-end? Fidelity says how well the surrogate
mimics the teacher; its own PR-AUC says what the translation costs.

{chr(10).join(rows)}

![Distillation frontier](../figures/{fig.name})

## Read

- At depth {deepest.depth} the surrogate tracks the teacher's ranking with Spearman
  **{deepest.spearman:.3f}** and agrees with **{deepest.decision_agreement:.1%}** of its
  volume-matched decisions - the share of deployed behavior that fits in
  {deepest.n_leaves} rules. The gap between the surrogate's PR-AUC and the teacher's
  {teacher_pr:.3f} is the price of auditability, stated per depth.
- **Score quantization is an interpretability cost.** A K-leaf tree emits only K
  distinct scores, so tight FP budgets are unreachable by construction: the
  surrogate's TPR at the {budget_fpr * 100:g}% budget moves in leaf-sized jumps.
  Anyone shipping "the interpretable version" inherits that granularity.
- **A surrogate explains behavior, not mechanism.** High fidelity means these rules
  reproduce the model's decisions on this traffic - not that the model "reasons"
  this way. The claim is scoped on purpose; over-reading global surrogates is a
  classic explainability failure.

## The depth-{chosen.depth} surrogate, in full ({chosen.n_leaves} leaves)

```text
{rendered_rules}
```

Complements: [SHAP](evaluation.md) attributes per-prediction contributions, the
[ablation](ablation.md) measures each family's causal value, the
[recourse study](recourse.md) gives the per-flow what-if - this gives the whole
model's closest small imitation, with its cost printed beside it.
"""
