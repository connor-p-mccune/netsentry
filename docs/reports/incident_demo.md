# NetSentry — Incident Report

_Input: `demo_flows.csv` — 66 flows scored through model
version 0.2.0 at the `fpr_1pct` threshold profile; 6 flows
alerted, grouped into **3 incident(s)** (same predicted class,
stream-contiguous, benign gaps ≤ 3 bridged)._

## Summary

| # | class | flows | rows | peak prob | ATT&CK | services |
|---|---|---|---|---|---|---|
| 1 | **PortScan** | 4 | 3-9 | 0.977 | T1046 | DNS |
| 2 | **PortScan** | 1 | 25-25 | 0.977 | T1046 | DNS |
| 3 | **DoS Hulk** | 1 | 66-66 | 0.954 | T1499 | HTTP |

## Incidents

### Incident 1: PortScan

- **Flows:** 4 (rows 3-9); peak probability 0.977, mean 0.977; 4 also flagged anomalous
- **ATT&CK:** Discovery / [T1046 Network Service Discovery](https://attack.mitre.org/techniques/T1046/)
- **Services:** DNS
- **Sources:** 10.0.0.14, 10.0.0.10, 10.0.0.22, 10.0.0.26
- **Targets:** 10.0.0.1
- **Recommended actions:** auto_alert: 4
- **Behavioural tell (most-cited SHAP feature):** Flow Bytes/s

### Incident 2: PortScan

- **Flows:** 1 (rows 25-25); peak probability 0.977, mean 0.977; 1 also flagged anomalous
- **ATT&CK:** Discovery / [T1046 Network Service Discovery](https://attack.mitre.org/techniques/T1046/)
- **Services:** DNS
- **Sources:** 10.0.0.18
- **Targets:** 10.0.0.1
- **Recommended actions:** auto_alert: 1
- **Behavioural tell (most-cited SHAP feature):** Flow Bytes/s

### Incident 3: DoS Hulk

- **Flows:** 1 (rows 66-66); peak probability 0.954, mean 0.954; 1 also flagged anomalous
- **ATT&CK:** Impact / [T1499 Endpoint Denial of Service](https://attack.mitre.org/techniques/T1499/)
- **Services:** HTTP
- **Sources:** 198.51.100.7
- **Targets:** 10.0.0.8
- **Recommended actions:** auto_alert: 1
- **Behavioural tell (most-cited SHAP feature):** Flow Bytes/s

## How to read this

An *incident* here is a contiguity heuristic over per-flow verdicts — nearby
same-class alerts are assumed to be one operation, the same correlation
assumption the campaigns study prices. The grouping adds no detection: every
verdict, probability, and recommended action comes from the same engine and
operating threshold the API serves, and a silent attack stays silent no matter
how its neighbours are grouped. Probabilities are calibrated scores; "services"
come from `Destination Port` as routing metadata (never a model feature); the
ATT&CK mapping is indicative of the CIC-IDS2017 scenarios.
