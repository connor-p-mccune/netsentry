# NetSentry - Champion/Challenger Promotion

_Synthetic stand-in; the method is the point. Challenger and champion scored on the
**same** frozen temporal test rows; deltas are paired-bootstrap (one resample scores
both models, so shared sampling noise cancels). Written by `netsentry promote`,
which exits non-zero on HOLD so a deploy pipeline can branch on the decision._

## Decision

**HOLD** - TPR@FPR delta CI lower bound -0.0188 breaches the -0.015 non-inferiority margin - detection regression risk

| role | bundle | backend / task / split | train rows | sha256 |
|---|---|---|---|---|
| champion | `champion.joblib` | lightgbm / binary / temporal | 28,034 | `68ab8984880c...` |
| challenger | `supervised_binary_temporal.joblib` | lightgbm / binary / temporal | 28,034 | `4d8d1fd49937...` |

| metric | delta (challenger - champion) |
|---|---|
| PR-AUC | +0.0001 (95% CI [-0.0022, +0.0025], p(challenger <= champion) = 0.468) |
| TPR at fpr_0.1pct (each model's own threshold) | -0.0149 (95% CI [-0.0188, -0.0117], p(challenger <= champion) = 1.000) |

## The policy, and why the margins are not hand-picked

Active policy: **non_inferiority** (margin 0.005 PR-AUC,
0.015 TPR).

- **Margins come from measurement.** The seed-sensitivity audit
  ([seed_variance.md](seed_variance.md)) measures how much these metrics move when
  *nothing* changes but the training seed. The non-inferiority margins are set just
  above that noise floor, so the gate cannot promote or demote on training luck.
- **`non_inferiority`** (default) rolls a routine retrain forward unless it is
  credibly *worse*: on drifting traffic, freshness has measured value (the streaming
  study shows retrained models recovering later-day detection), so parity is a
  reason to move, not to hold.
- **`superiority`** demands the delta CI exclude zero - the right bar for risky
  swaps (new architecture, new feature set) where churn itself has a cost.

Every decision is appended to `models/promotion_history.jsonl`; on promotion the
challenger is snapshotted to `models/champion.joblib` with a SHA-256 pointer, so a
later retrain overwriting the working bundle cannot silently rewrite the champion
(and a tampered snapshot fails the pointer check loudly).

## Hygiene

Promotion decisions reuse the frozen temporal test split, so they belong at release
cadence; in production this comparison runs on a fresh labeled window (or in shadow
- see the serving shadow-challenger, which produces exactly the paired scores this
report needs, on live traffic). Companion gates: `netsentry gate` decides *fit to
ship at all* (absolute bars); this decides *which of two ships* (relative).
