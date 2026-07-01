# NetSentry — Model Card (auto-generated)

_Generated 2026-07-01 15:49 UTC from the deployed bundle.
This is the factual spec sheet; see [`MODEL_CARD.md`](../MODEL_CARD.md) for intended
use, limitations, and ethics, and [`reports/evaluation.md`](evaluation.md) for the
honest metrics._

## Artifact

| field | value |
|---|---|
| version | 0.1.0 |
| task | multiclass |
| training split | stratified |
| backend | lightgbm |
| features | 76 |
| training rows | 38400 |
| created | 2026-07-01T15:49:44.608553+00:00 |

## Classes (13)

BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, Infiltration, PortScan, SSH-Patator, Web Attack

## Calibration & operating points

- Probability calibration: **isotonic**
- Decision-threshold profiles (calibrated attack probability):

| profile | threshold |
|---|---|
| cost_optimal | 0.6034 |
| fpr_0.1pct | 0.9538 |
| fpr_1pct | 0.8000 |

## Attached components

| component | present |
|---|---|
| benign-only anomaly detector | yes |
| conformal prediction set | yes |
| drift self-monitoring reference | yes |

## Threat coverage

Detected attack classes map to **6 MITRE ATT&CK tactics** and
**8 techniques** — see [`reports/mitre.md`](mitre.md).

## Reproduce

Regenerate this card from the current artifact with `netsentry modelcard`; regenerate
the metrics/robustness/calibration evidence with `netsentry analyze`.
