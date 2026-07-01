# NetSentry — Counterfactual Recourse

_Synthetic stand-in. For each flagged flow, the smallest set of moves to
attacker-controllable features (toward the benign centroid) that drops it below the
operating threshold (fpr_1pct, threshold 0.800).
Deltas are in standardized model-space units._

SHAP explains *why* a flow fired; this explains *what would clear it* — the
analyst's what-if, and the flip side of the [robustness study](robustness.md): the
same controllable features an attacker exploits are the ones that define recourse.
**5/5** example hits can be cleared within
5 changes.


### Example 1 — score 1.000 → 0.603 (cleared after 2 change(s))

- **decrease** `Flow Duration` by 0.86 std
- **decrease** `Idle Mean` by 2.29 std

### Example 2 — score 1.000 → 0.048 (cleared after 2 change(s))

- **decrease** `Flow Packets/s` by 2.22 std
- **decrease** `Total Backward Packets` by 3.37 std

### Example 3 — score 1.000 → 0.309 (cleared after 2 change(s))

- **decrease** `Flow Packets/s` by 3.39 std
- **decrease** `Total Backward Packets` by 4.99 std

### Example 4 — score 1.000 → 0.522 (cleared after 1 change(s))

- **decrease** `Flow Packets/s` by 5.94 std

### Example 5 — score 1.000 → 0.674 (cleared after 1 change(s))

- **decrease** `Flow Bytes/s` by 2.95 std

## Why this matters

A flagged flow with a reason *and* a recourse is triage-ready: the reason points the
analyst at the behaviour, the recourse quantifies how far from the benign manifold it
sits. A hit with **no** recourse (no small controllable change clears it) is a
high-confidence detection; one cleared by a single tweak is worth a second look.
