# NetSentry — vulnpipe Triage

_Findings re-ranked by fused risk = severity + NetSentry attack probability +
anomaly flag. A vulnerability on a host whose traffic looks like an active attack
is prioritised over the same severity on a quiet host._

| # | id | asset | severity | attack p | anomaly | risk |
|---|---|---|---|---|---|---|
| 1 | CVE-2024-0001 | web01 | high | 1.00 | no | 0.725 |
| 2 | CVE-2024-0005 | app01 | high | 0.92 | no | 0.696 |
| 3 | CVE-2024-0002 | db01 | critical | 0.44 | no | 0.655 |
| 4 | CVE-2024-0003 | web02 | medium | 1.00 | no | 0.600 |
| 5 | CVE-2024-0004 | cache01 | low | 0.22 | no | 0.202 |

## Wiring real vulnpipe output

Map each vulnpipe finding to a `VulnFinding` (id, `severity` or `cvss`, asset) and
attach the host's network-flow features as `flow`. The fusion weights live in
config (`triage.*`). Run `netsentry triage --findings <file.json>`.
