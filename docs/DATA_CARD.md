# Data Card — CIC-IDS2017

## Source & license
- **Dataset:** CIC-IDS2017, Canadian Institute for Cybersecurity (CIC),
  University of New Brunswick. Cite the CIC and Sharafaldin et al. (2018).
- **License / usage:** per CIC terms — record the exact terms and any required
  citation/registration here.
- **Version used:** `<original | corrected re-release (Engelen et al., WTMC 2021)>`

## Contents
- ~5 days of labeled network flows (Mon–Fri); BENIGN plus ~14 attack types
  (DoS/DDoS variants, PortScan, Brute Force, Web Attacks, Bot, Infiltration,
  Heartbleed, …). Features are CICFlowMeter-derived flow statistics.

## Known issues (handled in `netsentry/data/clean.py`)
- **Leakage columns** (dropped): `Flow ID`, `Source IP`, `Destination IP`,
  `Source Port`, `Timestamp`. `Destination Port` handled deliberately (see below).
- **Whitespace** in column headers (stripped).
- **Inf / NaN** in `Flow Bytes/s`, `Flow Packets/s` (Inf→NaN→median impute, train-fit).
- **Duplicate rows** (dropped; count logged): `<N>`
- **Negative sentinels** (e.g. `Init_Win_bytes_forward == -1`): `<decision>`
- **Severe class imbalance**; rare classes (Heartbleed, Infiltration) have few
  samples — accuracy is not used as a headline metric.

## Label handling
- Consolidation map (e.g. `Web Attack – *` → `Web Attack`; `DoS *` grouping):
  `<document the exact mapping>`
- Targets produced: **binary** (benign/attack) and **multiclass**.

## `Destination Port` decision
- `<dropped from headline model | kept as categorical>` — rationale: it can let
  the model memorize attack-specific ports rather than behavior. Both variants
  evaluated; difference reported in `docs/`.

## Splits
- Temporal/by-day (headline), stratified (reference), leave-one-attack-out
  (anomaly). Persisted with content hashes for reproducibility.
