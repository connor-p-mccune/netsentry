# NetSentry — Analysis Index

_Regenerated 2026-07-04 04:53 UTC via `netsentry analyze`. Synthetic stand-in unless run on the real dataset._

| report | what it covers | status |
|---|---|---|
| Operational evaluation | PR-AUC, TPR@FPR, per-class, calibration | [open](evaluation.md) |
| H-measure | a coherent, cost-explicit alternative to ROC-AUC (Hand 2009) | [open](hmeasure.md) |
| Cost-sensitive thresholds | decision-theoretic operating point | [open](cost.md) |
| Alert-queue capacity | detection vs analyst budget; lift over random triage | [open](alert_queue.md) |
| SOC queue simulation | FIFO vs score-priority attack-SLA under queueing load | [open](socsim.md) |
| Base-rate stress test | alert precision vs production prevalence (Axelsson 1999) | [open](base_rate.md) |
| Conformal prediction | coverage guarantee + selective alerting | [open](conformal.md) |
| Adaptive conformal | coverage restored online under drift (ACI) | [open](adaptive_conformal.md) |
| Adversarial robustness | evasion (mimicry + query search) | [open](robustness.md) |
| Training-set poisoning | label flips + benign-pool contamination | [open](poisoning.md) |
| Poisoning defense | audit-and-drop sanitization vs label flips, re-measured | [open](poisoning_defense.md) |
| Backdoor poisoning | trigger trojan (BadNets) + spectral-signatures defense (Tran et al. 2018) | [open](backdoor.md) |
| Membership inference | privacy leakage: does the model memorise its training data | [open](membership.md) |
| Differential privacy | the (ε, δ) guarantee priced: detection & leakage vs ε | [open](dp.md) |
| Model extraction | stealing the model by query: fidelity, stolen detection, transfer evasion | [open](extraction.md) |
| Adversarial hardening | adversarial training vs mimicry, re-measured | [open](hardening.md) |
| Certified robustness | randomized smoothing: a provable L2 radius per flow (Cohen et al. 2019) | [open](certify.md) |
| Label-noise audit | confident-learning flags + planted-flip self-validation | [open](label_audit.md) |
| Training-data valuation | KNN-Shapley value per flow: mislabel detection + value-guided pruning | [open](data_value.md) |
| Prediction-powered inference | attack prevalence from few labels + the model, with valid CIs (Angelopoulos 2023) | [open](ppi.md) |
| Drift monitoring | feature/score PSI, train vs test | [open](drift.md) |
| Statistical drift | per-feature KS+FDR, online Page-Hinkley/DDM | [open](drift_tests.md) |
| Anytime-valid drift | conformal test martingale: a Ville-bounded false-alarm rate at any stopping time | [open](exchangeability.md) |
| Prequential streaming | static vs retrained model on the later-day stream | [open](streaming.md) |
| Retrain-trigger policy | when to retrain: never / periodic / drift-triggered / every batch | [open](retrain_policy.md) |
| Threshold refresh | the label-cheap lever vs retraining; budget compliance under drift | [open](refresh.md) |
| Self-training | pseudo-labels on the unlabeled stream vs the labeled ceiling | [open](selftrain.md) |
| Weak supervision | the signatures as labeling functions: a detector trained on zero labels (Ratner 2016) | [open](weak_supervision.md) |
| Model-family leaderboard | every family through one honest protocol; the gap replicates | [open](leaderboard.md) |
| Leakage attribution | reproduce the field's ~99% and price each leakage source | [open](leakage.md) |
| Per-class detection | which temporal-split attacks are caught | [open](slices.md) |
| Campaign detection | the (day, class) operation as the unit: first alerts and silent campaigns | [open](campaigns.md) |
| Per-service parity | detection/false-alarm equity across services | [open](subgroups.md) |
| Novelty distance | detection vs distance-to-training; the split gap decomposed | [open](novelty.md) |
| Leave-one-day-out | temporal sensitivity: every day takes a turn as the future | [open](lodo.md) |
| Rules-vs-model baseline | hand-written signatures at a matched FPR budget | [open](rules.md) |
| Feature-group ablation | which behavioural families carry detection | [open](ablation.md) |
| Counterfactual recourse | minimal change that clears a hit | [open](recourse.md) |
| Importance stability | are the shipped explanations stable across refits | [open](importance_stability.md) |
| Feature interactions | Friedman's H-statistic: which features the model has entangled | [open](interactions.md) |
| Exemplar explanations | do the nearest known training flows vouch for the alerts | [open](exemplars.md) |
| Anchor explanations | high-precision IF-THEN rules with a coverage trade-off (Ribeiro et al. 2018) | [open](anchors.md) |
| Anomaly attribution | why a flow is abnormal: per-feature anomaly explanations + a faithfulness check | [open](anomaly_explain.md) |
| Surrogate distillation | the model's closest auditable imitation, with fidelity priced | [open](distill.md) |
| Active learning | uncertainty vs random labeling efficiency | [open](active_learning.md) |
| Seed sensitivity | the training-noise floor under every reported metric | [open](seed_variance.md) |
| Release gate | honesty invariants + metric floors the candidate must clear | [open](gate.md) |
| Promotion decision | champion vs challenger, paired-bootstrap deltas (via `netsentry promote`) | [open](promotion.md) |
| MITRE ATT&CK coverage | attack class -> tactic/technique | [open](mitre.md) |
| ATT&CK Navigator layer | detection coverage as a loadable Navigator layer | [open](attack_navigator_layer.json) |
| Provenance & supply chain | CycloneDX SBOM + model-integrity manifest | [open](provenance.md) |
