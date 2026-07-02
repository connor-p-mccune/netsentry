# NetSentry — Analysis Index

_Regenerated 2026-07-02 11:02 UTC via `netsentry analyze`. Synthetic stand-in unless run on the real dataset._

| report | what it covers | status |
|---|---|---|
| Operational evaluation | PR-AUC, TPR@FPR, per-class, calibration | [open](evaluation.md) |
| Cost-sensitive thresholds | decision-theoretic operating point | [open](cost.md) |
| Conformal prediction | coverage guarantee + selective alerting | [open](conformal.md) |
| Adversarial robustness | evasion (mimicry + query search) | [open](robustness.md) |
| Training-set poisoning | label flips + benign-pool contamination | [open](poisoning.md) |
| Drift monitoring | feature/score PSI, train vs test | [open](drift.md) |
| Per-class detection | which temporal-split attacks are caught | [open](slices.md) |
| Rules-vs-model baseline | hand-written signatures at a matched FPR budget | [open](rules.md) |
| Feature-group ablation | which behavioural families carry detection | [open](ablation.md) |
| Counterfactual recourse | minimal change that clears a hit | [open](recourse.md) |
| Active learning | uncertainty vs random labeling efficiency | [open](active_learning.md) |
| MITRE ATT&CK coverage | attack class -> tactic/technique | [open](mitre.md) |
| Provenance & supply chain | CycloneDX SBOM + model-integrity manifest | [open](provenance.md) |
