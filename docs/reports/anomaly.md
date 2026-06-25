# NetSentry — Anomaly Detection (novel attacks)

_Benign-only detectors evaluated **leave-one-attack-out**: trained on benign traffic, scored on an attack class held out entirely. Synthetic stand-in data unless run on the real dataset._

Detection rate is measured at a **1.0% benign false-positive budget** (threshold calibrated on a benign validation set).

## iforest

| held-out attack | detection @ FPR | anomaly PR-AUC |
|---|---|---|
| DoS Hulk | 11.8% | 0.561 |
| DoS GoldenEye | 6.8% | 0.181 |
| PortScan | 3.8% | 0.305 |
| DDoS | 17.0% | 0.559 |
| **average** | **9.9%** | **0.401** |

## autoencoder

| held-out attack | detection @ FPR | anomaly PR-AUC |
|---|---|---|
| DoS Hulk | 12.2% | 0.551 |
| DoS GoldenEye | 5.5% | 0.205 |
| PortScan | 2.4% | 0.299 |
| DDoS | 20.8% | 0.573 |
| **average** | **10.2%** | **0.407** |

## Ensemble — supervised + anomaly on the temporal test

PR-AUC on later-day (partly novel) attacks:

| scorer | PR-AUC |
|---|---|
| supervised only | 0.478 |
| anomaly only | 0.393 |
| **ensemble (rank-mean)** | **0.506** |

Combining a supervised classifier (known attacks) with a benign-only anomaly detector (novel attacks) is the production pattern: neither alone covers both regimes.
