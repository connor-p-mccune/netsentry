# NetSentry — Analysis Index

_Refreshed 2026-08-21 14:48 UTC. `netsentry analyze` regenerates every report listed here and rewrites this index; a row without a link did not produce its report. Synthetic stand-in unless run on the real dataset._

| report | what it covers | status |
|---|---|---|
| Operational evaluation | PR-AUC, TPR@FPR, per-class, calibration | [open](evaluation.md) |
| H-measure | a coherent, cost-explicit alternative to ROC-AUC (Hand 2009) | [open](hmeasure.md) |
| Cost-sensitive thresholds | decision-theoretic operating point | [open](cost.md) |
| Alert-queue capacity | detection vs analyst budget; lift over random triage | [open](alert_queue.md) |
| SOC queue simulation | FIFO vs score-priority attack-SLA under queueing load | [open](socsim.md) |
| Base-rate stress test | alert precision vs production prevalence (Axelsson 1999) | [open](base_rate.md) |
| Neyman-Pearson thresholds | a finite-sample guarantee that the FP budget holds (Tong, Feng & Li 2018) | [open](neyman_pearson.md) |
| Time-to-detection survival | Kaplan-Meier with the never-detected campaigns still in the denominator | [open](survival.md) |
| Decision latency | when a flow verdict can first exist, and what deciding earlier costs | [open](earliness.md) |
| Learning to defer | which flows are worth an analyst's time; when 'abstain where unsure' is the wrong policy (Madras et al. 2018) | [open](defer.md) |
| Taxonomy-aware errors | hierarchical P/R/F1 over the ATT&CK tree; which mistakes actually cost (Kiritchenko 2006) | [open](hierarchy.md) |
| Causal invariance | ICP screening + IRM over capture days, with the premise checked first (Peters 2016, Arjovsky 2019) | [open](invariance.md) |
| Monotone constraints | an entire evasion family made impossible by construction, proved and priced | [open](monotonic.md) |
| Byzantine-robust aggregation | one lying site destroys FedAvg; median / trimmed mean / Krum priced (Blanchard 2017, Yin 2018) | [open](byzantine.md) |
| Group DRO | train for the worst service, not the average one, against the cheap serving-side fix (Sagawa et al. 2020) | [open](dro.md) |
| Deterministic verification | a sound, absolute robustness radius for the deployed ensemble by interval arithmetic (Chen et al. 2019) | [open](verify_trees.md) |
| Epistemic vs aleatoric | ambiguity or ignorance: uncertainty decomposed over an ensemble, tested on never-seen attack classes | [open](uncertainty.md) |
| Off-policy evaluation | value a triage policy you never deployed: IPS/SNIPS/doubly-robust (Dudik, Langford & Li 2011) | [open](ope.md) |
| Extreme-value thresholds | peaks-over-threshold GPD fit: operating points past the edge of the data (Siffer et al. 2017) | [open](evt.md) |
| Conformal alert FDR | a false-discovery-rate guarantee on the alert batch: conformal p-values + BH (Bates et al. 2023) | [open](alert_fdr.md) |
| Distribution-free risk control | bound the miss rate the contract names, not the false-positive rate the threshold targets (Angelopoulos et al. 2021, 2022) | [open](risk_control.md) |
| Conformal prediction | coverage guarantee + selective alerting | [open](conformal.md) |
| Adaptive conformal | coverage restored online under drift (ACI) | [open](adaptive_conformal.md) |
| Adversarial robustness | evasion (mimicry + query search) | [open](robustness.md) |
| Training-set poisoning | label flips + benign-pool contamination | [open](poisoning.md) |
| Adversarial hardening | adversarial training vs mimicry, re-measured | [open](hardening.md) |
| Certified robustness | randomized smoothing: a provable L2 radius per flow (Cohen et al. 2019) | [open](certify.md) |
| Sensor failure | the deployed model with a broken exporter: missing / stuck / mis-assembled fields | [open](degradation.md) |
| Automatic slice discovery | search for the underperforming regions nobody predefined, with a permuted null and a confirmation half (Chung et al. 2019) | [open](slice_discovery.md) |
| Cost-aware feature acquisition | buy the expensive features only for the flows whose verdict is in doubt, against a random-gating control on the same budget | [open](acquisition.md) |
| Budgeted sampling | score a fraction of the stream and estimate the rest: Horvitz-Thompson against four designs, including the one with no estimator at all | [open](sampling.md) |
| Multi-objective selection | a Pareto front over detection, cost and evasion-resistance, and the front members no weighted sum can reach (Deb et al. 2002) | [open](pareto.md) |
| Server-side batching | amortise the fixed cost of a scoring call across the requests already queued, and find the load below which waiting is a loss | [open](batching.md) |
| Budgeted cascade | two-stage inference: the compute handed back and the detection it costs | [open](cascade.md) |
| Sequential host decisions | how many flows before a host can be called compromised (Wald's SPRT, 1945) | [open](sequential.md) |
| Federated training | detection when traffic cannot be pooled: FedAvg vs pooled vs alone (McMahan 2017) | [open](federated.md) |
| Self-supervised pretraining | learn the representation from unlabelled flows, with PCA and an untrained encoder as the controls (VIME 2020, SCARF 2022) | [open](pretrain.md) |
| DP synthetic release | share the traffic instead of the model: train-synthetic/test-real under a budget (PrivBayes family, Zhang et al. 2017) | [open](dp_synth.md) |
| Secure aggregation | federate without the coordinator seeing any site's update -- and what hiding it costs in robustness (Bonawitz et al. 2017) | [open](secagg.md) |
| Poisoning defense | audit-and-drop sanitization vs label flips, re-measured | [open](poisoning_defense.md) |
| Detection SLOs | error budgets and multiwindow burn-rate alerting, with the rules generated | [open](slo.md) |
| Point-in-time feature store | host context joined correctly vs over the whole capture: the temporal leak, priced | [open](feature_store.md) |
| Closed-loop control | alert volume held at the analyst budget by feedback, and the attack on the loop | [open](control.md) |
| Operating-point training | a partial-AUC surrogate against cross-entropy, scored at every false-positive budget | [open](operating_point.md) |
| Deep tabular models | MLP and FT-Transformer against the boosted incumbent under one shared protocol | [open](deep_tabular.md) |
| Budgeted hyperparameter search | successive halving and Hyperband at an equal budget, after measuring the two premises they rest on (Li et al. 2018) | [open](multifidelity.md) |
| Online learning | one-pass Hoeffding tree + ADWIN, prequentially, against static and periodic retraining | [open](online.md) |
| Continual learning | class-incremental updates: forgetting, replay and the compute argument, measured | [open](continual.md) |
| Multivariate drift (MMD) | kernel two-sample testing: the joint change the per-feature monitors cannot see | [open](mmd.md) |
| Optimal transport | a drift distance in units and the coupling that explains it, then the distance an attacker has to travel (Cuturi 2013) | [open](transport.md) |
| Strategic equilibrium | the arms race as a game: myopic race vs commitment, with the attacker cost priced | [open](strategic.md) |
| Metamorphic testing | a label-free correctness oracle, validated by injected mutants (Chen 1998, Xie 2011) | [open](metamorphic.md) |
| Backdoor poisoning | trigger trojan (BadNets) + spectral-signatures defense (Tran et al. 2018) | [open](backdoor.md) |
| Membership inference | privacy leakage: does the model memorise its training data | [open](membership.md) |
| Differential privacy | the (epsilon, delta) guarantee priced: detection & leakage vs epsilon | [open](dp.md) |
| Machine unlearning | SISA exact deletion: sharding tax, per-request cost, verified forgetting (Bourtoule et al. 2021) | [open](unlearn.md) |
| Model extraction | stealing the model by query: fidelity, stolen detection, transfer evasion | [open](extraction.md) |
| Model watermarking | prove ownership by backdooring: exact binomial test, innocent control, extraction survival (Adi et al. 2018) | [open](watermark.md) |
| Label-noise audit | confident-learning flags + planted-flip self-validation | [open](label_audit.md) |
| Training-data valuation | KNN-Shapley value per flow: mislabel detection + value-guided pruning | [open](data_value.md) |
| Prediction-powered inference | attack prevalence from few labels + the model, with valid CIs (Angelopoulos 2023) | [open](ppi.md) |
| Label-shift correction | recover + correct for the deployment prior with zero labels (BBSE + MLLS/EM) | [open](label_shift.md) |
| Drift monitoring | feature/score PSI, train vs test | [open](drift.md) |
| Statistical drift | per-feature KS+FDR, online Page-Hinkley/DDM | [open](drift_tests.md) |
| Anytime-valid drift | conformal test martingale: a Ville-bounded false-alarm rate at any stopping time | [open](exchangeability.md) |
| Covariate shift | diagnose the temporal gap via a domain classifier + price importance-weighted retraining (Shimodaira 2000, Bickel 2009) | [open](covariate_shift.md) |
| Prequential streaming | static vs retrained model on the later-day stream | [open](streaming.md) |
| Retrain-trigger policy | when to retrain: never / periodic / drift-triggered / every batch | [open](retrain_policy.md) |
| Threshold refresh | the label-cheap lever vs retraining; budget compliance under drift | [open](refresh.md) |
| Self-training | pseudo-labels on the unlabeled stream vs the labeled ceiling | [open](selftrain.md) |
| Weak supervision | the signatures as labeling functions: a detector trained on zero labels (Ratner 2016) | [open](weak_supervision.md) |
| PU learning | confirmed attacks + unlabeled traffic: c recovery, weighted retrain, honest budgets (Elkan-Noto 2008) | [open](pu_learning.md) |
| Expert advice (online) | track the best model under drift with a regret bound: Hedge + fixed-share | [open](experts.md) |
| Model-family leaderboard | every family through one honest protocol; the gap replicates | [open](leaderboard.md) |
| Leakage attribution | reproduce the field's ~99% and price each leakage source | [open](leakage.md) |
| Per-class detection | which temporal-split attacks are caught | [open](slices.md) |
| Campaign detection | the (day, class) operation as the unit: first alerts and silent campaigns | [open](campaigns.md) |
| Per-service parity | detection/false-alarm equity across services | [open](subgroups.md) |
| Attack-family discovery | clustering the flagged pile into campaigns, with k chosen without labels | [open](discovery.md) |
| Novelty distance | detection vs distance-to-training; the split gap decomposed | [open](novelty.md) |
| Rare-class rate estimation | partial pooling so a twelve-flow class does not read like a thousand-flow one | [open](rare_rates.md) |
| Open-set recognition | the test days share no attack class with training: which novelty rule notices (Scheirer 2013, Dhamija 2018) | [open](openset.md) |
| Leave-one-day-out | temporal sensitivity: every day takes a turn as the future | [open](lodo.md) |
| Rules-vs-model baseline | hand-written signatures at a matched FPR budget | [open](rules.md) |
| Feature-group ablation | which behavioural families carry detection | [open](ablation.md) |
| Counterfactual recourse | minimal change that clears a hit | [open](recourse.md) |
| SHAP estimand audit | which Shapley value the API ships, graded against the coalition sum and against the two quantities it is usually confused with (Janzing et al. 2020) | [open](shap_estimand.md) |
| Importance stability | are the shipped explanations stable across refits | [open](importance_stability.md) |
| Predictive multiplicity | how arbitrary is the verdict across equally-good models (Marx et al. 2020) | [open](multiplicity.md) |
| Partial dependence & ICE | the response-curve shape of the top features | [open](partial_dependence.md) |
| Feature interactions | Friedman's H-statistic: which features the model has entangled | [open](interactions.md) |
| Exemplar explanations | do the nearest known training flows vouch for the alerts | [open](exemplars.md) |
| Anchor explanations | high-precision IF-THEN rules with a coverage trade-off (Ribeiro et al. 2018) | [open](anchors.md) |
| Anomaly attribution | why a flow is abnormal: per-feature anomaly explanations + a faithfulness check | [open](anomaly_explain.md) |
| Influence functions | which training flows caused a verdict, validated against real LOO (Koh & Liang 2017) | [open](influence.md) |
| Optimal sparse trees | how far greedy CART sits from the provably optimal tree, with a certificate (Hu, Rudin & Seltzer 2019) | [open](optimal_tree.md) |
| Surrogate distillation | the model's closest auditable imitation, with fidelity priced | [open](distill.md) |
| Glass-box additive model | a model that is its own explanation, and the capacity dial that shows what the honest split actually punishes (Lou, Caruana & Gehrke 2012) | [open](gam.md) |
| Active learning | uncertainty vs random labeling efficiency | [open](active_learning.md) |
| Seed sensitivity | the training-noise floor under every reported metric | [open](seed_variance.md) |
| Release gate | honesty invariants + metric floors the candidate must clear | [open](gate.md) |
| Anytime-valid A/B | when the shadow model can be promoted: peeking-safe confidence sequences | [open](sequential_ab.md) |
| MITRE ATT&CK coverage | attack class -> tactic/technique | [open](mitre.md) |
| Private inference | score a flow under two-party secret sharing so neither side sees the other's secret, then read the model out with queries the server cannot refuse | [open](private_inference.md) |
| Private indicator sharing | ask a peer whether they have seen an indicator without telling them which: DH private set intersection, and the dictionary attack on the hashing it replaces | [open](psi.md) |
| MITRE ATLAS coverage | the detector as a target: this repo's own ML attack surface, with the gaps named | [open](atlas.md) |
| ATT&CK Navigator layer | detection coverage as a loadable Navigator layer | [open](attack_navigator_layer.json) |
| Sigma detection rules | the signature baseline exported as portable Sigma rules | [open](sigma/README.md) |
| Streaming quantiles | estimate the threshold's quantile in fixed memory, graded in alert volume rather than in quantile error (Jain & Chlamtac 1985; Dunning) | [open](quantiles.md) |
| Streaming sketches | host analytics at line rate in fixed memory, with every bound checked against exact truth (Cormode 2005, Flajolet 2007) | [open](sketches.md) |
| Tamper-evident alert ledger | hash-chained alert history: every edit attempted, and what verification catches | [open](ledger.md) |
| Conformance mapping | NIST AI RMF and EU AI Act obligations mapped to artifacts, with every claim verified against the repository | [open](compliance.md) |
| Online triage learning | a contextual bandit learning the operating point under partial feedback, and the alert budget its exploration spends | [open](bandit.md) |
| Serving lifecycle conformance | the API contract as a state machine, driven through random operation sequences, with deliberately broken services proving the checker fails | [open](state_machine.md) |
| Anomaly-score semantics | is the anomaly score a density estimate or a complexity measure: six benign-only detectors, a control that learns nothing, and the size component regressed out | [open](density.md) |
| ML-invariant static analysis | the leakage rules enforced by a parser, with the rule set graded by injecting the violations it claims to catch | [open](mlint.md) |
| Proof-carrying verdicts | commit to the ensemble as a Merkle tree and prove each verdict against it, with seven forgeries executed and the leakage priced | [open](attestation.md) |
| Provenance & supply chain | CycloneDX SBOM + model-integrity manifest | [open](provenance.md) |
