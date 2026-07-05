# NetSentry - Release Quality Gate

_Synthetic stand-in; the method is the point. The honest temporal/binary release
candidate, evaluated once against the frozen temporal test split. Bars come from
config (`gate.*`); this report is written by `netsentry gate`, which exits non-zero
on failure so CI and deploy pipelines can enforce it._

## Verdict

**PASS** - every bar cleared; the candidate is releasable under this policy.

| check | verdict | detail |
|---|---|---|
| leakage firewall | PASS | no identifier/port column in the fitted feature space |
| calibrator attached | PASS | method=isotonic |
| threshold profiles | PASS | all configured profiles present (fpr_0.1pct, fpr_1pct) |
| artifact smoke | PASS | scored 5 rows; probabilities in [0, 1] |
| PR-AUC floor | PASS | PR-AUC 0.529 vs floor 0.375 (1.5x the 0.250 random-ranker baseline) |
| too-good-to-be-true ceiling | PASS | PR-AUC 0.529 <= 0.999 ceiling |
| detection floor | PASS | TPR 9.1% at the 0.10% FP budget vs floor 5.0% |
| calibration quality | PASS | ECE 0.1057 vs max 0.15 |

Measured: PR-AUC **0.529** (attack prevalence 0.250),
TPR at the 0.10% FP budget **9.1%**, ECE
**0.1057**.

## What the bars encode

- **Structural checks** re-assert the honesty invariants on the artifact that would
  actually ship: no identifier/port column in the fitted feature space (the leakage
  firewall, re-checked at release rather than trusted to unit tests), a calibrator
  attached when configuration promises calibrated probabilities, every configured
  FPR profile present, and an end-to-end scoring smoke.
- **The too-good ceiling is deliberate.** A PR-AUC above 0.999
  *fails* the gate: on CIC-IDS-style data a near-perfect score is overwhelmingly more
  likely to be leakage than brilliance, so the gate refuses to ship it until a human
  explains it. This is the project's "treat a too-good number as a bug" habit turned
  into machinery.
- **Floors are relative where possible.** The PR-AUC floor is a multiple of the
  attack prevalence (a random ranker's PR-AUC), so the bar transfers across datasets
  with different base rates instead of encoding one dataset's difficulty.

## Hygiene

A release gate touches the frozen test split, so it belongs at release cadence, not
per-commit (repeated evaluation against one test set slowly erodes it). In
production the same bars run against a fresh labeled window; the config is the
policy either way. Companion: `netsentry promote` decides *champion vs challenger*
(relative), this gate decides *fit to ship at all* (absolute).
