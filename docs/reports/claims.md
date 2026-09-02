# NetSentry -- Does the README Still Say What the Reports Say?

_Every precise number in 103 of the README's 119 sections, checked against the 139 generated reports on disk. Regenerate with `netsentry claims`._

## Why this report exists

The README quotes several hundred computed numbers, each produced by a study that can be regenerated with one command. Every one of them is a promise that running the command reproduces the figure. Nothing had ever checked that promise, and it is the kind that decays silently: a study's config changes, its report is regenerated, and the prose that quoted it three waves ago keeps its old number. The report is right, the README is wrong, and no test fails.

So this is a checker, in the same shape as [`netsentry mlint`](mlint.md) and for the same reason -- an invariant nobody enforces is an invariant nobody has.

**639 numbers in the README claim to come from a study. 630 of them do, 9 come from a different study than the one the section links, and 0 come from nowhere at all.**

Every precise number in the README is reproducible from a report on disk, which is the state this checker exists to keep. It was not the state when it was written.

The 9 traceable claims are a milder thing, separated rather than lumped in: the number is real and regenerable, but not from the report the section sends a reader to. Most are cross-references between studies and the rest are arithmetic the README performs on figures a report does state, so none of them is wrong -- they are simply unconfirmable where a reader would look. Their count is budgeted rather than driven to zero, for reasons the section below gives.

The checker's own evidence is measured rather than asserted. **97%** of verified claims appear in their report character for character; the rest are quoted at lower precision or in the other unit and are matched by arithmetic, which is correct but weaker. **29%** correspond to exactly one number in their report -- a quote that could be a rounding of thirty different figures is barely confirmation at all. Both limits are measured below rather than left for a reader to find.

## The verdicts

| verdict | claims | share | what it means |
|---|---|---|---|
| **verified** | 630 | 98.6% | the token appears in a report the section links; the promise holds |
| **traceable** | 9 | 1.4% | it appears in some *other* report; real, but not checkable where the reader is sent |
| **unsourced** | 0 | 0.0% | it appears in no report at all; the only class that fails the gate |

A **claim** is a number precise enough to be worth checking rather than a round figure someone chose: at least two decimals, or a percentage carrying one. A version string or a `30%` is not a claim; a configured budget like `0.1%` is, and verifies, because the report that spends the budget states it too.

Matching is numeric rather than textual, and getting that right took three passes. A quote of `2.31` does not assert the report says exactly 2.31 -- it asserts the report says something that *rounds* to 2.31, so a quote is treated as the interval it actually claims. A report stating `0.027` where the README says `2.7%` is the same fact in different units and is matched as such. Both extensions exist because the first version flagged roundings and unit conversions as faults, and a checker that cries wolf is a checker somebody switches off. Neither goes further: inferring that one number is another's difference or reciprocal would make the checker unfalsifiable, which is worse than making it strict.

## The claims sourced somewhere else

| README line | number | section | linked report |
|---|---|---|---|
| 366 | `8.5%` | Headline results | evaluation |
| 366 | `4.3%` | Headline results | evaluation |
| 367 | `0.537` | Headline results | evaluation |
| 1084 | `0.59` | One perturbation, shipped once | transport, universal |
| 1224 | `0.83` | Explaining the anomaly flag (why is this flow abnormal?) | anomaly_explain |
| 1386 | `12.7%` | Which Shapley value does the API ship? | shap_estimand |
| 2185 | `0.19` | Budgeted hyperparameter search, and the premises underneath it | multifidelity |
| 2363 | `0.017` | Deep tabular models vs the trees (the claim, checked) | deep_tabular |
| 2388 | `0.054` | Training for the operating point (partial AUC) | operating_point |

Three things end up here and none of them is a wrong number. Most are legitimate **cross-references** -- a section comparing two studies quotes the other one's figure. Some are **derived**: the README states a difference the report leaves the reader to compute, like the tree's `+0.017` between two sample sizes the report gives as 0.520 and 0.537. And a few sit in sections linking several reports, only one of which states them.

The checker deliberately does not chase these. Matching a claim against arithmetic over pairs of reported numbers would, across a corpus this size, match nearly anything -- and a check that always passes is not a check. The count is pinned as a budget instead: 9 today, and the build fails if it grows.

## Watching the checker fire

A rule nobody has watched fire is a rule nobody should trust, and a clean codebase makes a checker look identical to a checker that does nothing. So faults are injected into a copy of the README and the checker is rerun against each one.

| injected fault | what it models in practice | injected | detected | fails the build |
|---|---|---|---|---|
| a number drifts one digit | a study regenerated, the prose quoting it not updated | 25 | **88%** | **88%** |
| a claim moves to a section that cannot source it | prose reorganised, the link left pointing at the old study | 25 | **92%** | **92%** |
| a report link points at nothing | a study renamed or removed, its README section left behind | 25 | **100%** | **100%** |

The digit perturbation is the important one, because it is what real drift looks like: same magnitude, same precision, one digit different. A fault that replaced a number with something obviously wrong would test nothing.

**Running the harness changed the gate, and then corrected the harness.** *Detected* and *fails the build* began as one column, which hid a real weakness: under a gate counting only *unsourced* claims, a drift that lands on a figure appearing in some other report becomes **traceable** instead, and the build stays green. So the gate now pins both counts -- unsourced at 0, traceable at its current 9 -- the way `mlint` pins its violation count. Any claim leaving the verified class moves one of the two.

The first version of the harness also reported 100% detection, which was an artefact: it asked whether the *original* token still verified after being replaced, and a token that no longer exists never verifies. Asking the right question -- does the number now in the README verify? -- gives **88%**. The missing 12% is the checker's real blind spot: a one-digit drift sometimes lands on another figure the same report states, and arithmetic cannot tell that apart from the truth. A harness that had not been checked against itself would have reported the flattering number.

## What the checker cannot see

A claim is only as strong as the uniqueness of what it matched. **29%** of verified claims correspond to exactly one number in their report and are genuinely pinned; the rest could be a rounding of several, where the confirmation is weaker. **97%** match character for character rather than by rounding or unit conversion.

The blind spots follow directly, and the first one is measured rather than asserted:

- **A number that drifts onto another figure the same report already states** still matches. That is exactly the 12% of injected drifts the harness above does not catch, and it is why the number in that column is not 100%.
- **Prose that misdescribes a correct number** is invisible. `0.529` verifies whether the sentence around it says the model beats the baseline or loses to it.
- **Round figures are not claims.** A README saying "about 30%" where the report says 12% is not checked, because admitting one-significant-figure tokens would flood the class with page numbers, version strings and configured budgets.
- **Arithmetic the README performs is invisible.** A stated difference between two reported numbers cannot be verified without matching against derived quantities, which at this corpus size would match almost anything.

The remedy for all of them is the same and is already in place elsewhere: the reports are generated, not written, so the fix for a drifted number is to regenerate rather than to edit. This checker exists to notice when that has not happened.

## The other direction: links and orphans

- **Broken report links:** 0 (every section points at a report that exists)
- **Reports no README section links:** 26 of 139. Not a fault -- the [report index](INDEX.md) exists precisely so every study is reachable -- but the count is worth knowing, because a study nobody links is a study nobody reads.

## Scope and honest limits

- **Only the README is checked.** The same drift can happen in `docs/ARCHITECTURE.md`, the model card and the data card; extending the checker is a matter of adding paths, and it has not been done, so those documents carry no guarantee from this report.
- **The gate pins counts, not claims.** A traceable claim is tolerated because a cross-reference between studies is normal; what is refused is the *count* growing. That catches drift into either class, and it means a legitimate new cross-reference requires raising the budget deliberately -- which is the point, and is how `mlint` works.
- **This report is its own input.** The README section describing this checker quotes numbers from this report, so regenerating it changes the document it audits. That settles in one pass -- the totals are integers and integers are not claims -- but it is the one place in this repository where a study observes something it is part of, and worth naming rather than leaving for a reader to notice.
- **Whole-token matching is the mechanism and the limit.** Tokens are matched as numbers, not substrings, so `0.53` no longer verifies against a report that only says `0.5336`. It is still deliberately crude: it has no false positives a human would dispute, and its false negatives are enumerated above rather than discovered later.
