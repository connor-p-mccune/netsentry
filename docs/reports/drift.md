# NetSentry — Drift Report

_Generated 2026-07-01 16:01 UTC. Reference: `train.parquet` vs current: `test.parquet`._

Population Stability Index (PSI) per feature. Reading: **< 0.1** no
meaningful shift, **0.1-0.25** moderate, **>= 0.25**
major drift worth investigating.

## Summary

- **Max feature PSI: 0.1133** (moderate)
- Mean feature PSI: 0.0038
- Features with at least moderate drift: 1 / 76
- **Score drift (model output PSI): 0.0168** (none)

## Per-feature PSI (top 20)

| feature | PSI | severity |
|---|---|---|
| Total Fwd Packets | 0.1133 | moderate |
| Flow Duration | 0.0615 | none |
| Total Backward Packets | 0.0328 | none |
| SYN Flag Count | 0.0244 | none |
| Flow IAT Max | 0.0023 | none |
| Flow IAT Mean | 0.0022 | none |
| Init_Win_bytes_forward | 0.0020 | none |
| Fwd Packet Length Mean | 0.0015 | none |
| Down/Up Ratio | 0.0014 | none |
| Idle Std | 0.0014 | none |
| Fwd IAT Max | 0.0013 | none |
| Fwd Packets/s | 0.0012 | none |
| Active Min | 0.0012 | none |
| Init_Win_bytes_backward | 0.0012 | none |
| Total Length of Bwd Packets | 0.0012 | none |
| Packet Length Std | 0.0012 | none |
| Subflow Fwd Packets | 0.0011 | none |
| Bwd Packet Length Min | 0.0010 | none |
| Fwd PSH Flags | 0.0010 | none |
| Bwd Packet Length Mean | 0.0010 | none |

## How to read this

Input drift (feature PSI) often rises long before labels arrive, so it is an
early decay signal: when a feature's live distribution diverges from training,
the model is extrapolating. Score drift (the model's own output distribution
moving) is a complementary signal. In serving, `/metrics` exposes
`netsentry_feature_drift_psi_max` / `_mean` computed over a rolling window of
requests, so the same check runs continuously in production.
