# NetSentry — Anomaly Detection (novel attacks)

_Benign-only detectors evaluated **leave-one-attack-out**: trained on benign traffic, scored on an attack class held out entirely. Synthetic stand-in data unless run on the real dataset._

Detection rate is measured at a **1.0% benign false-positive budget** (threshold calibrated on a benign validation set).

## iforest

| held-out attack | detection @ FPR | anomaly PR-AUC |
|---|---|---|
| FTP-Patator | 1.9% | 0.086 |
| SSH-Patator | 1.5% | 0.080 |
| DoS slowloris | 6.3% | 0.128 |
| DoS Slowhttptest | 3.2% | 0.092 |
| DoS Hulk | 9.2% | 0.588 |
| DoS GoldenEye | 4.0% | 0.196 |
| Bot | 1.1% | 0.055 |
| PortScan | 1.0% | 0.268 |
| DDoS | 10.9% | 0.553 |
| **average** | **4.3%** | **0.227** |

## autoencoder

| held-out attack | detection @ FPR | anomaly PR-AUC |
|---|---|---|
| FTP-Patator | 1.7% | 0.089 |
| SSH-Patator | 2.3% | 0.088 |
| DoS slowloris | 10.1% | 0.182 |
| DoS Slowhttptest | 7.4% | 0.128 |
| DoS Hulk | 18.8% | 0.669 |
| DoS GoldenEye | 4.4% | 0.185 |
| Bot | 1.1% | 0.048 |
| PortScan | 3.7% | 0.346 |
| DDoS | 26.7% | 0.693 |
| **average** | **8.5%** | **0.270** |

## Ensemble — supervised + anomaly on the temporal test

PR-AUC on later-day (partly novel) attacks:

| scorer | PR-AUC |
|---|---|
| supervised only | 0.529 |
| anomaly only | 0.433 |
| **ensemble (rank-mean)** | **0.537** |

Combining a supervised classifier (known attacks) with a benign-only anomaly detector (novel attacks) is the production pattern: neither alone covers both regimes.
