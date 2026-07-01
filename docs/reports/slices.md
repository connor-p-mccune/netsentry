# NetSentry — Per-Class Detection Slices

_Synthetic stand-in. Detection rate per attack class on the honest **temporal** test
split, at the 1%-FPR operating point (raw attack score threshold
0.867, matching the evaluation report)._

The aggregate PR-AUC hides which attacks are caught. On the temporal split the
test-day attacks are largely **novel** to the model, so this is the concrete "known
vs unknown" breakdown.

| attack class | test support | detection |
|---|---|---|
| DDoS | 2,442 | 52.7% |
| Infiltration | 42 | 4.8% |
| Web Attack | 288 | 1.7% |
| Bot | 351 | 1.7% |
| PortScan | 3,114 | 0.3% |

Best caught: **DDoS** (53%); most evasive: **PortScan** (0%). Low-detection classes are exactly where the benign-only anomaly detector
earns its keep — the supervised model cannot recall an attack type it never trained
on, which is the whole argument for pairing the two.
