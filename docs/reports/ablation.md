# NetSentry — Feature-Group Ablation

_Synthetic stand-in. Leave-one-family-out on the honest **temporal** split (binary
attack vs benign). Baseline (all families): PR-AUC 0.529, detection
21.0% at the 1%-FPR operating point. Each row
refits the model with that behavioural family's columns removed; the delta is the
family's marginal value given all the others._

## Why ablation, not just SHAP

SHAP attributes a *given* prediction to features; it cannot say what the model would
lose if a whole family were unavailable, because a high-SHAP feature may be redundant
with one the model would fall back on. Ablation answers that directly by removing the
family and refitting — the causal-flavoured complement to SHAP's attribution.

| family removed | features | PR-AUC | Δ PR-AUC | detection | Δ detection |
|---|---|---|---|---|---|
| flow rates | 4 | 0.224 | **+0.305** | 0.8% | +20.2 pts |
| TCP flags | 12 | 0.464 | **+0.065** | 20.8% | +0.2 pts |
| packet size | 16 | 0.525 | **+0.004** | 20.9% | +0.1 pts |
| header/window/bulk | 11 | 0.526 | **+0.003** | 21.1% | -0.1 pts |
| timing/IAT | 23 | 0.568 | **-0.039** | 21.5% | -0.5 pts |
| volume/counts | 10 | 0.641 | **-0.112** | 28.2% | -7.2 pts |

## Read

The most load-bearing family is **flow rates** (-0.305 PR-AUC,
+20.2 pts detection when removed): the honest temporal
signal leans on it most, and it is a *ratio* family that transfers across days.
 Conversely, removing **timing/IAT** (+0.039), **volume/counts** (+0.112) *improved* the honest PR-AUC. That is not a licence to prune — it is the signature of **overfitting to the temporal shift**: those families encode absolute scales (packet/flow volumes, durations) that differ between the Mon–Wed training attacks and the Thu–Fri test attacks, so the model learns day-specific thresholds that mislead on later days. The rate family, being a ratio, transfers better. Acting on this would mean selecting features on *validation* (never this test split — that is the leakage the project exists to avoid); the ablation only tells you where to look.

This lines up with the adversarial-robustness study — the rate/timing features
ablation shows carry the transferable signal are exactly the attacker-controllable
ones the evasion attack exploits, which is why a classifier resting on them needs the
benign-only anomaly detector beside it. Ablation measures *marginal* value given the
rest, not standalone value: a family can look redundant here yet be the only signal in
another regime.
