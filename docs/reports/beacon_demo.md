# NetSentry — Beaconing / C2 Periodicity (synthetic demo)

The classifier scores each flow in isolation and drops every identifier, so it is
blind to **beaconing** — a host calling a command-and-control server on a fixed
cadence (MITRE ATT&CK **Command and Control**, e.g. T1071 Application Layer
Protocol). No single callback looks anomalous; the regularity of the schedule is
the tell. This analytic scores each talker pair's connection-time regularity, the
cross-flow complement to the per-flow model.

**Candidates:** 7 talker pair(s) with enough events; **1**
scored at or above the 0.85 regularity flag line.

The synthetic capture plants a single periodic beacon; the detector ranks it first (`10.0.0.7 -> 45.77.12.9:8443`, regularity 0.975, period 1.0 min) above the jittery benign talkers — the mechanic, on data with a known answer.

| talker pair | connections | period | regularity | CV |
|---|---|---|---|---|
| `10.0.0.7 -> 45.77.12.9:8443` **[flag]** | 360 | 1.0 min | 0.975 | 0.04 |
| `10.0.0.12 -> 93.184.216.36:443` | 52 | 4.6 min | 0.435 | 0.98 |
| `10.0.0.15 -> 93.184.216.39:443` | 52 | 5.0 min | 0.433 | 0.91 |
| `10.0.0.13 -> 93.184.216.37:80` | 54 | 5.7 min | 0.394 | 0.81 |
| `10.0.0.11 -> 93.184.216.35:443` | 59 | 4.8 min | 0.310 | 0.82 |
| `10.0.0.10 -> 93.184.216.34:80` | 62 | 3.2 min | 0.209 | 0.92 |
| `10.0.0.14 -> 93.184.216.38:443` | 60 | 3.4 min | 0.200 | 1.06 |

## How to read this

Regularity is a robust dispersion of the inter-arrival times (median absolute
deviation over the median interval), in `[0, 1]`: **1.0 is a perfectly periodic
beacon, 0.0 is bursty human traffic.** The coefficient of variation (std/mean of
the intervals) is shown alongside — a beacon has CV near zero.

This is a **hunt lead generator, not a verdict.** A legitimate periodic service —
NTP, a monitoring poll, a cron job, a software-update check — is also regular and
will score high; the analytic surfaces *candidates* ranked by regularity for a
human to triage, and adds no detection to the model's per-flow verdicts. It reads
the timestamp and identity columns as metadata only — exactly the fields the model
is forbidden to see — which is why it can catch what the model, by construction,
cannot.
