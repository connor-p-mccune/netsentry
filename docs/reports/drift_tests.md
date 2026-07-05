# NetSentry — Statistical & Online Drift Detection

_Generated 2026-07-05 05:47 UTC. Reference: `train.parquet` vs current: `test.parquet`._

The [PSI drift report](drift.md) answers *how much* each feature moved. PSI is an
effect size with a rule-of-thumb cutoff, not a test, and it is computed on static
batches. This report adds the two things PSI cannot: **significance** (are the
shifts real, corrected for testing many features at once?) and **timing** (at what
point in a stream did the model's behaviour break?).

## Per-feature Kolmogorov-Smirnov tests (with Benjamini-Hochberg FDR)

A two-sample KS test per feature (reference vs current), then a Benjamini-Hochberg
procedure at FDR **0.05** across all 76 tested features — so the
"5 drifted" count controls the expected share of false alarms, rather than
flagging ~5% of stable features by chance.

**5 / 76 features drift significantly** at FDR 0.05.

| feature | KS statistic | p-value | drifted (FDR) |
|---|---|---|---|
| Total Fwd Packets | 0.1142 | 1.89e-150 | **yes** |
| Flow Duration | 0.0894 | 2.38e-92 | **yes** |
| Total Backward Packets | 0.0650 | 5.75e-49 | **yes** |
| SYN Flag Count | 0.0548 | 6.38e-35 | **yes** |
| Init_Win_bytes_forward | 0.0172 | 7.80e-04 | **yes** |
| Flow IAT Mean | 0.0141 | 1.07e-02 | no |
| Flow IAT Max | 0.0133 | 1.85e-02 | no |
| Fwd Packet Length Min | 0.0130 | 2.28e-02 | no |
| Bwd Packet Length Mean | 0.0126 | 2.96e-02 | no |
| Active Min | 0.0119 | 4.76e-02 | no |
| Total Length of Fwd Packets | 0.0114 | 6.24e-02 | no |
| Down/Up Ratio | 0.0114 | 6.35e-02 | no |
| Fwd PSH Flags | 0.0113 | 6.62e-02 | no |
| Bwd Avg Bulk Rate | 0.0108 | 9.29e-02 | no |
| Fwd Packet Length Mean | 0.0106 | 1.02e-01 | no |
| Flow Bytes/s | 0.0099 | 1.56e-01 | no |
| Fwd Avg Packets/Bulk | 0.0099 | 1.52e-01 | no |
| Bwd IAT Std | 0.0099 | 1.52e-01 | no |
| Subflow Fwd Bytes | 0.0098 | 1.60e-01 | no |
| CWE Flag Count | 0.0092 | 2.08e-01 | no |
| Flow IAT Min | 0.0088 | 2.59e-01 | no |
| Fwd IAT Max | 0.0088 | 2.59e-01 | no |
| Active Std | 0.0087 | 2.72e-01 | no |
| RST Flag Count | 0.0087 | 2.73e-01 | no |
| Bwd IAT Total | 0.0085 | 2.92e-01 | no |

## Online change detection (when did the stream break?)

The offline test says *whether*; these say *when*. A stream is built by placing a
reference (training-era) sample ahead of a current (later-day) sample, planting a
known change-point at the boundary so each detector can be judged against ground
truth. The stream is scored by the **deployed model** — what a production monitor
would actually watch — with Page-Hinkley on its attack-score stream (no labels
needed) and DDM on its error stream (labels needed).

- **Page-Hinkley** (model-score stream): alarmed at position 6,000 of the stream — after the reference→current boundary at 5,000 (an alarm just after the boundary is the detector catching the later-day shift).
- **DDM** (model-error stream): warning at 6,013, drift alarm at 7,133 (reference→current boundary at 5,000); the error rate climbing a statistically meaningful margin above its running minimum is the alarm.

## How to read this

- **KS + FDR** is the honest multi-feature drift count: a small p-value means the two
  samples are unlikely to share a distribution, and BH keeps the multiplicity in
  check. It complements PSI — PSI ranks *magnitude*, KS certifies *significance*.
- **Page-Hinkley** and **DDM** are *online*: they consume one observation at a time
  and raise an alarm at a specific index, which is what a production monitor needs —
  not "the batch drifted" but "alert now, at flow N." Here they locate the same
  later-day boundary the temporal split embodies, from the score and error streams
  respectively.
- All three are unsupervised-to-cheap early-warning signals: KS and Page-Hinkley
  need no labels at all; DDM needs only the eventual error signal.
