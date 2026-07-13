# NetSentry — Host-Graph Analytics (synthetic demo)

The classifier scores each flow in isolation and drops every identifier, so it is
blind to the two attacks whose signal lives in the **graph** of who talks to whom:
**scanning** (one host fanning out across many destinations or ports — ATT&CK
Discovery, T1046) and **lateral movement** (a reached host pivoting deeper — ATT&CK
Lateral Movement, T1021). This analytic reconstructs the host communication graph
and surfaces both — the cross-flow, topology-aware complement to the per-flow model,
the way beaconing is its timing-aware complement.

The synthetic capture plants a horizontal sweep, a vertical sweep, and one multi-hop pivot among benign talkers that only reach the internet; the planted sweep (`203.0.113.9`, 40 distinct hosts) tops the scan table and the planted pivot (`203.0.113.5` -> `10.0.0.5` -> `10.0.0.20` -> `10.0.0.50` -> `10.0.0.51`) is recovered whole (5 hosts, 4 internal hops) — the mechanic, on data with a known answer.

## Scan fan-out (Discovery)

Sources whose distinct-destination (horizontal) or distinct-port (vertical) spread
reaches the 20-target fan-out line. A single scan probe is an unremarkable
short flow — which is exactly why the temporal model misses PortScan, a later-day
class it never trained on — but the spread across the graph is not subtle.

| source | kind | distinct hosts | distinct ports | flows |
|---|---|---|---|---|
| `203.0.113.9` | horizontal | 40 | 1 | 40 |
| `203.0.113.7` | vertical | 1 | 30 | 30 |

## Lateral-movement chains (Lateral Movement)

Directed paths that enter the network and then hop internal-to-internal. Ordinary
egress to the internet cannot form a chain (the destination is external), so a long
internal path is the pivot pattern a responder wants to see whole rather than as
disconnected alerts.

| movement chain | hosts | internal hops |
|---|---|---|
| `203.0.113.5` -> `10.0.0.5` -> `10.0.0.20` -> `10.0.0.50` -> `10.0.0.51` | 5 | 4 |

## How to read this

This is a **hunt-lead generator, not a verdict.** A vulnerability scanner, a
monitoring poller, a backup server, or an administrator's jump box all fan out or
pivot as a matter of course and will surface here; the analytic ranks *candidates*
by structural suspicion for a human to triage, and adds no detection to the model's
per-flow verdicts. It reads the ``Src IP`` / ``Dst IP`` / ``Dst Port`` columns as
metadata only — exactly the identifiers the model is forbidden to see — which is why
it can catch what the model, by construction, cannot. Internal/external is an RFC1918
heuristic; on a real deployment substitute the site's own address plan.
