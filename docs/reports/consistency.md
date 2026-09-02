# NetSentry -- Do the Reports Agree With Each Other?

_Every incumbent PR-AUC stated across 138 generated reports, against a ladder of 6 one-knob recomputations from the canonical configuration. Regenerate with `netsentry consistency`._

## Why this report exists

A great many reports here open by stating what the deployed model scores -- a baseline to beat, an incumbent to compare against, a control arm. [`netsentry claims`](claims.md) checks that each of those numbers appears in the report a reader is sent to. Nothing checks whether the reports agree **with one another**.

**A naive harvest finds 7 different answers across 12 reports. 4 of them are not disagreements at all.**

Every one reads like the same quantity -- the incumbent, the baseline, the static model, the arm to beat. Reading them properly, 2 state **a different metric** (ROC-AUC, a few words from a PR-AUC in the same sentence, and not comparable to one), and 2 belong to **the far side of a comparison** -- a retrained model, a per-site model -- where the qualifier that owns the number follows it rather than preceding it. Each is a mistake a reader skimming the same sentence makes as readily as a regular expression does, and the first version of this study made all of them.

What survives is **3 values spanning 0.021 PR-AUC** against a canonical configuration that scores 0.528 here. That is a real spread, and the ladder below says what produces it.

**It is not how the model was trained that makes the reports differ; it is what population it was scored on.** Cutting the training set to 12,000 rows or the ensemble to a tenth of its trees moves the number by at most 0.011 -- around the [resolution study](power.md)'s minimum detectable effect, so barely a difference at all. Changing *what is scored* moves it by up to 0.120: averaging over time-ordered batches instead of pooling costs 0.094, and scoring a single capture day moves it further still. The knobs a study is most likely to mention are the ones that matter least.

Of the surviving values, 1 is reached by exactly one rung, 2 by more than one. Nothing here is a disagreement nobody chose, which is the outcome this study was built to be able to *fail* to find.

**Bracketed is not explained**, and the distinction is the point. A value several rungs reach is consistent with more than one methodology choice, and the ladder cannot say which. Reporting that as an attribution would be the same error as calling a wide confidence interval a result.

## What the reports say

| value | reports stating it |
|---|---|
| **0.516** | [reuse](reuse.md) |
| **0.529** | [ablation](ablation.md), [cascade](cascade.md), [earliness](earliness.md), [gam](gam.md), [private_inference](private_inference.md) |
| **0.537** | [deep_tabular](deep_tabular.md), [operating_point](operating_point.md) |

The harvester collects a three-decimal value only when a word like *incumbent*, *deployed*, *baseline* or *static* appears within sixty characters before it, and only inside the band a PR-AUC for this model plausibly occupies. Without the band it collects coverage levels and p-values that happen to sit near the word *baseline*.

## What a naive harvest also picks up, and should not

| value | report | why it does not count | the sentence |
|---|---|---|---|
| 0.583 | [federated](federated.md) | belongs to the other arm of a comparison sentence | `* — it is CIC-IDS2017's clean baseline day — and yet its local model posts 0.583 PR-AUC...` |
| 0.668 | [hmeasure](hmeasure.md) | states a different metric (ROC-AUC, not average precision) | `AUC (Gradient-boosted trees (deployed) scores AUC 0.668 but H 0.180). ROC-AUC and the` |
| 0.693 | [openset](openset.md) | states a different metric (ROC-AUC, not average precision) | `| 0.523 | 0.9% | 0.1% | The deployed rule holds its own field: `1 - P(BENIGN)` reaches ...` |
| 0.544 | [streaming](streaming.md) | belongs to the other arm of a comparison sentence | `PR-AUC rises from **0.433** (static) to **0.544** (retrained) across the stre` |

**These are the interesting rows.** Every one was reported as a disagreement by the first version of this study, and every one is a mistake a human reader makes on the same sentence. *AUC* sitting a few words from a PR-AUC is a different metric with a different sensitivity to prevalence. `rises from 0.433 (static) to 0.544 (retrained)` puts the qualifier that owns the second number *after* it, so a rule that looks backwards attributes a retrained model's score to the frozen one. A per-site model in a table row is not the deployed model at all.

The reports are not wrong; each sentence is correct where it stands. But a number is only comparable with the construction that produced it, and prose puts the two far enough apart that a regular expression -- and a reader -- can lose the connection.

## What reproduces the spread

| configuration | what it models | PR-AUC | vs canonical |
|---|---|---|---|
| the canonical configuration | the full training split, the shipped hyperparameters, all later-day flows | **0.528** | -- |
| trained on 12,000 rows | what a study does when an expensive arm cannot afford the full split | **0.530** | +0.003 |
| 60 trees instead of 600 | what a study does when it has to refit the model dozens of times | **0.538** | +0.011 |
| scored on 33% of the later days | what a study does when it holds part of the split back for its own use | **0.512 - 0.544** _(range over draws)_ | -0.001 |
| averaged over 6 time-ordered batches | what a prequential study reports: a mean of per-batch scores | **0.434** | -0.094 |
| one capture day (Friday, the median of 2) | what a per-site or per-day arm reports | **0.648** | +0.120 |

Each row turns exactly one knob from the canonical configuration and scores the result the same way. None of them is wrong -- they are choices real studies here made, for reasons those studies give. The random rung is shown as a range because a study evaluating on a random third does not land on one number, and holding it to a point would call a correct value unexplained.

**The training knobs barely matter and the evaluation knobs dominate**, which is the useful result. Cutting the training set or thinning the ensemble moves the score by about what the [resolution study](power.md) says a difference needs to be worth reporting at all. Changing *what gets scored* moves it by ten times that. A study documenting its hyperparameters and not its evaluation population has documented the part that does not matter.

**Averaging over batches deserves its own sentence**, because it is the one that looks like an approximation and is not. PR-AUC depends on prevalence, and each time-ordered batch has its own; a mean of per-batch scores is therefore a different estimand from the pooled score rather than a noisier version of it. Two studies can disagree on this row while both being exactly right.

## Attribution

| stated | reached by | narrowest rung that reaches it | rungs that do | verdict |
|---|---|---|---|---|
| 0.516 | 0.512 - 0.544 | scored on 33% of the later days | 1 | **pinned** |
| 0.529 | 0.530 | trained on 12,000 rows | 3 | bracketed |
| 0.537 | 0.538 | 60 trees instead of 600 | 2 | bracketed |

Three verdicts, and the middle one is the honest addition. **Pinned** means exactly one rung reaches the value, which is as close to an attribution as this method gets. **Bracketed** means several do: the value is consistent with more than one methodology choice and the ladder cannot say which, so calling it explained would be the same error as calling a wide confidence interval a result. **Unexplained** is the one worth acting on, because it means two reports differ for a reason nobody chose.

Attribution takes the *narrowest* covering rung rather than the nearest, which matters once any rung is an interval: a wide enough range contains everything, and picking by distance alone would let it explain every value in the table. A point rung landing within tolerance is a much stronger claim than an interval that happens to contain the number, and the ranking says so.

## Scope and honest limits

- **This proves compatibility, not identity.** That a knob reproduces a value means the difference *could* have that cause. Establishing that it *does* would require reading each study's configuration, which is a human check this cannot replace.
- **Only one quantity is audited.** The incumbent's PR-AUC is the most frequently restated number here, which makes it the right place to start and leaves detection rates, coverage levels and latencies unchecked.
- **Text harvesting is approximate at both ends.** A report that states the incumbent's score in a table cell with no nearby qualifier is missed; a report that mentions a baseline in passing near an unrelated number could be over-collected. The band and the sixty-character window are what keep the second failure rare, at the cost of the first.
- **This report excludes itself from the corpus it audits.** It quotes other reports' numbers in its own tables, so leaving it in would let a previous run's output be harvested as fresh evidence -- a study reading its own output is measuring itself.
- **The ladder is not exhaustive.** It contains the knobs this repository's studies actually turn. A value it cannot reproduce may be a knob nobody thought to add rather than a genuine disagreement, which is why the verdict is *unexplained* rather than *wrong*.
- **Recomputation costs a refit per rung.** The numbers here come from the shipped configuration on the full split; running this under a reduced CI config produces a different canonical value and therefore a different ladder, which is the study's own point applied to itself.
