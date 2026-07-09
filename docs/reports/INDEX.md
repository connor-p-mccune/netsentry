# NetSentry — Analysis Index

_Regenerated 2026-07-04 04:53 UTC via `netsentry analyze`. Synthetic stand-in unless run on the real dataset._

| report | what it covers | status |
|---|---|---|
| Operational evaluation | PR-AUC, TPR@FPR, per-class, calibration | [open](evaluation.md) |
| Cost-sensitive thresholds | decision-theoretic operating point | [open](cost.md) |
| Alert-queue capacity | detection vs analyst budget; lift over random triage | [open](alert_queue.md) |
| Conformal prediction | coverage guarantee + selective alerting | [open](conformal.md) |
| Adversarial robustness | evasion (mimicry + query search) | [open](robustness.md) |
| Training-set poisoning | label flips + benign-pool contamination | [open](poisoning.md) |
| Adversarial hardening | adversarial training vs mimicry, re-measured | [open](hardening.md) |
| Label-noise audit | confident-learning flags + planted-flip self-validation | [open](label_audit.md) |
| Drift monitoring | feature/score PSI, train vs test | [open](drift.md) |
| Statistical drift | per-feature KS+FDR, online Page-Hinkley/DDM | [open](drift_tests.md) |
| Prequential streaming | static vs retrained model on the later-day stream | [open](streaming.md) |
| Retrain-trigger policy | when to retrain: never / periodic / drift-triggered / every batch | [open](retrain_policy.md) |
| Self-training | pseudo-labels on the unlabeled stream vs the labeled ceiling | [open](selftrain.md) |
| Model-family leaderboard | every family through one honest protocol; the gap replicates | [open](leaderboard.md) |
| Per-class detection | which temporal-split attacks are caught | [open](slices.md) |
| Campaign detection | the (day, class) operation as the unit: first alerts and silent campaigns | [open](campaigns.md) |
| Per-service parity | detection/false-alarm equity across services | [open](subgroups.md) |
| Novelty distance | detection vs distance-to-training; the split gap decomposed | [open](novelty.md) |
| Leave-one-day-out | temporal sensitivity: every day takes a turn as the future | [open](lodo.md) |
| Rules-vs-model baseline | hand-written signatures at a matched FPR budget | [open](rules.md) |
| Feature-group ablation | which behavioural families carry detection | [open](ablation.md) |
| Counterfactual recourse | minimal change that clears a hit | [open](recourse.md) |
| Importance stability | are the shipped explanations stable across refits | [open](importance_stability.md) |
| Surrogate distillation | the model's closest auditable imitation, with fidelity priced | [open](distill.md) |
| Active learning | uncertainty vs random labeling efficiency | [open](active_learning.md) |
| Seed sensitivity | the training-noise floor under every reported metric | [open](seed_variance.md) |
| Release gate | honesty invariants + metric floors the candidate must clear | [open](gate.md) |
| Promotion decision | champion vs challenger, paired-bootstrap deltas (via `netsentry promote`) | [open](promotion.md) |
| MITRE ATT&CK coverage | attack class -> tactic/technique | [open](mitre.md) |
| ATT&CK Navigator layer | detection coverage as a loadable Navigator layer | [open](attack_navigator_layer.json) |
| Provenance & supply chain | CycloneDX SBOM + model-integrity manifest | [open](provenance.md) |
