# Data Card — CIC-IDS2017

## Source & license
- **Dataset:** CIC-IDS2017, Canadian Institute for Cybersecurity (CIC),
  University of New Brunswick. Please cite the CIC and Sharafaldin et al. (2018),
  *"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization"*.
- **License / usage:** distributed by the CIC under their terms; access requires
  accepting those terms (effectively registration). Review and comply with the
  CIC terms before use; this repository ships **no** dataset bytes.
- **Version used:** original release. A community-corrected re-release exists
  (Engelen et al., WTMC 2021) that fixes known labelling errors; enable it via
  `data.use_corrected_labels` if you obtain it.

## How NetSentry obtains the data
- `netsentry download` is idempotent: it verifies any CSVs already in `data/raw/`
  and otherwise (a) downloads + checksum-verifies an archive from
  `data.source_url`, or (b) generates a **clearly-labelled synthetic** dataset
  when `data.allow_synthetic` is set and no source is configured, or (c) prints
  precise manual-download instructions.
- **Synthetic mode (development/CI only).** `netsentry/data/synthetic.py` emits a
  schema-faithful stand-in: the same columns, the same defects (Inf, `-1`
  sentinels, duplicates), genuine class imbalance with rare classes, a per-day
  attack layout, and `Destination Port` correlated with attack class. It exists
  so the pipeline, tests, and CI smoke-train can run without the real download.
  Synthetic results are always labelled as such and are **never** reported as
  real-world performance.

## Contents
- ~5 days of labelled network flows (Mon–Fri); `BENIGN` plus 14 attack types
  (FTP/SSH-Patator, the DoS family, Heartbleed, Web Attacks, Infiltration, Bot,
  PortScan, DDoS). Features are CICFlowMeter-derived flow statistics.
- The canonical column list, identifier/leaky columns, label vocabulary, and the
  day→attack layout are defined once in `netsentry/data/schema.py` (the single
  source of truth used by cleaning, features, and tests).

## Known issues (handled in `netsentry/data/clean.py`)
- **Leakage columns** (dropped before modelling): `Flow ID`, `Source IP`,
  `Source Port`, `Destination IP`, `Timestamp`, `External IP`, and the duplicate
  `Fwd Header Length.1`. These identify the flow/capture session, not behaviour.
- **`Destination Port`** is handled deliberately — see below.
- **Whitespace** in column headers (stripped on load).
- **Inf / NaN** in `Flow Bytes/s`, `Flow Packets/s` (Inf→NaN→median impute,
  fit on **train only**).
- **Duplicate rows** present in volume (dropped; count logged at clean time).
- **Negative sentinels** (e.g. `Init_Win_bytes_forward == -1` meaning "not set"):
  kept as an informative sentinel by default (`data.negative_sentinel_strategy`),
  rather than silently treated as a real byte count.
- **Severe class imbalance**; rare classes (Heartbleed, Infiltration) have very
  few samples — accuracy is therefore never used as a headline metric.

## Label handling
- Labels are normalised (whitespace and the en-dash/hyphen in `Web Attack` names)
  before mapping. Consolidation (in config): `Web Attack – {Brute Force, XSS, Sql
  Injection}` → `Web Attack`. The DoS sub-tools are **kept distinct** in the
  multiclass target (they are genuinely different tools with enough samples).
- Two targets are produced: **binary** (`BENIGN` vs attack) and **multiclass**.

## `Destination Port` decision
- **Dropped from the headline model.** It is legitimately predictive but lets the
  model memorise "attack X always used port Y" rather than learning behaviour. A
  second variant keeps it (encoded as a categorical via
  `features.encode_destination_port`) so the difference can be reported.

## Splits
- **Temporal/by-day** (headline; train Mon–Wed, test Thu–Fri), **stratified
  random** (optimistic reference), and **leave-one-attack-out** (for the anomaly
  detector). Validation is carved from the training split only. Splits are
  persisted with a content hash for reproducibility.
