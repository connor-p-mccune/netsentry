# NetSentry — MITRE ATT&CK Coverage

Each detected attack class is mapped to a MITRE ATT&CK tactic and technique, so a
prediction carries something an analyst can pivot on — not just a class name. The
serving API returns this mapping in the `mitre` field of every attack prediction.

> These are **indicative** mappings for the CIC-IDS2017 capture scenarios (the
> dataset is not natively labelled with ATT&CK IDs). They encode the behaviour each
> class represents, and are the single source of truth shared by serving and this report.

**Coverage:** 12 attack classes across **6 tactics**
(Command and Control, Credential Access, Discovery, Execution, Impact, Initial Access) and **8 techniques**.

| attack class | tactic | technique |
|---|---|---|
| Bot | Command and Control | [T1071 Application Layer Protocol](https://attack.mitre.org/techniques/T1071/) |
| FTP-Patator | Credential Access | [T1110 Brute Force](https://attack.mitre.org/techniques/T1110/) |
| SSH-Patator | Credential Access | [T1110 Brute Force](https://attack.mitre.org/techniques/T1110/) |
| PortScan | Discovery | [T1046 Network Service Discovery](https://attack.mitre.org/techniques/T1046/) |
| Infiltration | Execution | [T1204 User Execution](https://attack.mitre.org/techniques/T1204/) |
| DDoS | Impact | [T1498 Network Denial of Service](https://attack.mitre.org/techniques/T1498/) |
| DoS GoldenEye | Impact | [T1499 Endpoint Denial of Service](https://attack.mitre.org/techniques/T1499/) |
| DoS Hulk | Impact | [T1499 Endpoint Denial of Service](https://attack.mitre.org/techniques/T1499/) |
| DoS Slowhttptest | Impact | [T1499.002 Service Exhaustion Flood](https://attack.mitre.org/techniques/T1499/002/) |
| DoS slowloris | Impact | [T1499.002 Service Exhaustion Flood](https://attack.mitre.org/techniques/T1499/002/) |
| Heartbleed | Initial Access | [T1190 Exploit Public-Facing Application](https://attack.mitre.org/techniques/T1190/) |
| Web Attack | Initial Access | [T1190 Exploit Public-Facing Application](https://attack.mitre.org/techniques/T1190/) |

## Why this matters

Detection is only the first step; response needs context. Tagging a flagged flow
with its ATT&CK technique lets a SOC correlate NetSentry alerts with EDR/SIEM
detections that speak the same language, prioritise by tactic (a Credential-Access
brute force vs an Impact DoS), and measure detection coverage against the framework
their threat model is written in.
