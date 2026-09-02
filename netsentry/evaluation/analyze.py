"""Run the full analysis suite and write an index — one-command reproducibility.

Regenerates every model-analysis report (operational evaluation + calibration,
cost-sensitive thresholds, conformal prediction, adversarial robustness, drift) and
writes an ``INDEX.md`` linking them with one-line summaries and a pass/fail status.
Each report is run defensively, so one failure does not abort the rest — the index
records which succeeded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from netsentry.data.dp_synth import run_dp_synth_report
from netsentry.evaluation.ablation import run_ablation_report
from netsentry.evaluation.acquisition import run_acquisition_report
from netsentry.evaluation.active_learning import run_active_learning_report
from netsentry.evaluation.adaptive_conformal import run_adaptive_conformal_report
from netsentry.evaluation.alert_fdr import run_alert_fdr_report
from netsentry.evaluation.alert_queue import run_alert_queue_report
from netsentry.evaluation.bandit import run_bandit_report
from netsentry.evaluation.baserate import run_base_rate_report
from netsentry.evaluation.campaigns import run_campaigns_report
from netsentry.evaluation.conformal import run_conformal_report
from netsentry.evaluation.consistency import run_consistency_report
from netsentry.evaluation.cost import run_cost_report
from netsentry.evaluation.data_value import run_data_value_report
from netsentry.evaluation.defer import run_defer_report
from netsentry.evaluation.discovery import run_discovery_report
from netsentry.evaluation.earliness import run_earliness_report
from netsentry.evaluation.evt import run_evt_report
from netsentry.evaluation.gate import run_gate
from netsentry.evaluation.hierarchy import run_hierarchy_report
from netsentry.evaluation.hmeasure import run_hmeasure_report
from netsentry.evaluation.hull import run_hull_report
from netsentry.evaluation.label_audit import run_label_audit_report
from netsentry.evaluation.label_shift import run_label_shift_report
from netsentry.evaluation.leaderboard import run_leaderboard_report
from netsentry.evaluation.leakage import run_leakage_report
from netsentry.evaluation.lodo import run_lodo_report
from netsentry.evaluation.multiplicity import run_multiplicity_report
from netsentry.evaluation.neyman_pearson import run_neyman_pearson_report
from netsentry.evaluation.novelty import run_novelty_report
from netsentry.evaluation.ope import run_ope_report
from netsentry.evaluation.openset import run_openset_report
from netsentry.evaluation.pareto import run_pareto_report
from netsentry.evaluation.power import run_power_report
from netsentry.evaluation.ppi import run_ppi_report
from netsentry.evaluation.rare_rates import run_rare_rates_report
from netsentry.evaluation.report import run_evaluation
from netsentry.evaluation.reuse import run_reuse_report
from netsentry.evaluation.risk_control import run_risk_control_report
from netsentry.evaluation.rules import run_rules_report
from netsentry.evaluation.sampling import run_sampling_report
from netsentry.evaluation.seed_variance import run_seed_variance_report
from netsentry.evaluation.sequential_ab import run_sequential_ab_report
from netsentry.evaluation.slice_discovery import run_slice_discovery_report
from netsentry.evaluation.slices import run_slices_report
from netsentry.evaluation.socsim import run_socsim_report
from netsentry.evaluation.subgroups import run_subgroups_report
from netsentry.evaluation.survival import run_survival_report
from netsentry.evaluation.uncertainty import run_uncertainty_report
from netsentry.explain.anchors import run_anchors_report
from netsentry.explain.anomaly_explain import run_anomaly_explain_report
from netsentry.explain.counterfactual import run_recourse_report
from netsentry.explain.distill import run_distill_report
from netsentry.explain.exemplars import run_exemplars_report
from netsentry.explain.importance_stability import run_importance_stability_report
from netsentry.explain.influence import run_influence_report
from netsentry.explain.interactions import run_interactions_report
from netsentry.explain.optimal_tree import run_optimal_tree_report
from netsentry.explain.partial_dependence import run_partial_dependence_report
from netsentry.explain.shap_estimand import run_shap_estimand_report
from netsentry.features.store_report import run_store_report
from netsentry.governance.attestation import run_attestation_report
from netsentry.governance.claims import run_claims_report
from netsentry.governance.compliance import run_compliance_report
from netsentry.governance.ledger_report import run_ledger_report
from netsentry.governance.mlint import run_mlint_report
from netsentry.governance.provenance import run_provenance_report
from netsentry.intel.atlas import run_atlas_report
from netsentry.intel.navigator import run_navigator_export
from netsentry.intel.psi import run_psi_report
from netsentry.intel.report import run_mitre_report
from netsentry.intel.sequential import run_sequential_report
from netsentry.intel.sigma import run_sigma_export
from netsentry.intel.sketches import run_sketches_report
from netsentry.log import get_logger
from netsentry.models.density import run_density_report
from netsentry.models.gam import run_gam_report
from netsentry.models.monotonic import run_monotonic_report
from netsentry.monitoring.control import run_control_report
from netsentry.monitoring.covariate_shift import run_covariate_shift_report
from netsentry.monitoring.exchangeability import run_exchangeability_report
from netsentry.monitoring.experts import run_experts_report
from netsentry.monitoring.mmd import run_mmd_report
from netsentry.monitoring.quantiles import run_quantile_report
from netsentry.monitoring.refresh import run_refresh_report
from netsentry.monitoring.report import run_drift_report, run_drift_tests_report
from netsentry.monitoring.retrain_policy import run_retrain_policy_report
from netsentry.monitoring.slo import run_slo_report
from netsentry.monitoring.streaming import run_streaming_report
from netsentry.monitoring.transport import run_transport_report
from netsentry.robustness.backdoor import run_backdoor_report
from netsentry.robustness.calibration_attack import run_calibration_attack_report
from netsentry.robustness.certify import run_certify_report
from netsentry.robustness.composition import run_composition_report
from netsentry.robustness.degradation import run_degradation_report
from netsentry.robustness.dp import run_dp_report
from netsentry.robustness.extraction import run_extraction_report
from netsentry.robustness.hardening import run_hardening_report
from netsentry.robustness.membership import run_membership_report
from netsentry.robustness.metamorphic import run_metamorphic_report
from netsentry.robustness.poisoning import run_poisoning_report
from netsentry.robustness.report import run_robustness_report
from netsentry.robustness.sanitize import run_sanitize_report
from netsentry.robustness.strategic import run_strategic_report
from netsentry.robustness.universal import run_universal_report
from netsentry.robustness.verify_trees import run_verify_trees_report
from netsentry.robustness.watermark import run_watermark_report
from netsentry.serving.batching import run_batching_report
from netsentry.serving.cascade import run_cascade_report
from netsentry.serving.lifecycle import run_lifecycle_report
from netsentry.serving.private_inference import run_private_inference_report
from netsentry.serving.side_channel import run_side_channel_report
from netsentry.training.byzantine import run_byzantine_report
from netsentry.training.continual import run_continual_report
from netsentry.training.deep_tabular import run_deep_tabular_report
from netsentry.training.determinism import run_determinism_report
from netsentry.training.dro import run_dro_report
from netsentry.training.federated import run_federated_report
from netsentry.training.invariance import run_invariance_report
from netsentry.training.multifidelity import run_multifidelity_report
from netsentry.training.online import run_online_report
from netsentry.training.operating_point import run_operating_point_report
from netsentry.training.pretrain import run_pretrain_report
from netsentry.training.pu_learning import run_pu_learning_report
from netsentry.training.secagg import run_secagg_report
from netsentry.training.selftrain import run_selftrain_report
from netsentry.training.unlearn import run_unlearn_report
from netsentry.training.weak_supervision import run_weak_supervision_report

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

INDEX_NAME = "INDEX.md"


def _run_gate_report(settings: Settings) -> Path:
    """Adapter: the gate writes its report either way; enforcement is the CLI's job."""
    out, _ = run_gate(settings)
    return out


# (title, description, output filename, runner). Runners take only Settings.
_ANALYSES: list[tuple[str, str, str, Callable[[Settings], Path]]] = [
    (
        "Operational evaluation",
        "PR-AUC, TPR@FPR, per-class, calibration",
        "evaluation.md",
        run_evaluation,
    ),
    (
        "H-measure",
        "a coherent, cost-explicit alternative to ROC-AUC (Hand 2009)",
        "hmeasure.md",
        run_hmeasure_report,
    ),
    (
        "Calibration poisoning",
        "the deployed threshold is a quantile of benign validation scores, so its breakdown "
        "point is the false-positive budget itself -- the tighter the budget, the cheaper "
        "the attack; with a trimmed and a median-of-days defence priced both ways",
        "calibration_attack.md",
        run_calibration_attack_report,
    ),
    (
        "Cross-report consistency",
        "every report states what the incumbent scores; this checks whether they agree, "
        "and "
        "reproduces the spread by turning one methodology knob at a time",
        "consistency.md",
        run_consistency_report,
    ),
    (
        "Compositional failure",
        "every safeguard here was validated with one thing wrong; a 2^4 factorial over "
        "shift, sensor outage, evasion and prevalence collapse asks whether the guarantees "
        "and the monitors survive two at once",
        "composition.md",
        run_composition_report,
    ),
    (
        "Statistical resolution",
        "how big a difference has to be on this split before it means anything: bootstrap "
        "intervals per metric, paired vs unpaired comparison, an exact permutation null, "
        "and several of the differences this project has already published, against the "
        "resulting bar",
        "power.md",
        run_power_report,
    ),
    (
        "Documentation claims",
        "every precise number the README quotes, checked against the report that generates "
        "it: verified, traceable to another study, or unsourced -- with an injection harness "
        "measuring whether the checker fires",
        "claims.md",
        run_claims_report,
    ),
    (
        "Held-out reuse",
        "how many times this package reads the sealed split, what selecting on it costs "
        "measured against a never-queried half, and whether Thresholdout or a confidence "
        "gate closes the gap without losing the ability to find a real improvement",
        "reuse.md",
        run_reuse_report,
    ),
    (
        "Operating-point frontier",
        "is the deployed cut on the ROC convex hull, does the gain a coin promises survive the "
        "later days, and what net benefit says without a threshold (Provost & Fawcett 2001)",
        "hull.md",
        run_hull_report,
    ),
    ("Cost-sensitive thresholds", "decision-theoretic operating point", "cost.md", run_cost_report),
    (
        "Alert-queue capacity",
        "detection vs analyst budget; lift over random triage",
        "alert_queue.md",
        run_alert_queue_report,
    ),
    (
        "SOC queue simulation",
        "FIFO vs score-priority attack-SLA under queueing load",
        "socsim.md",
        run_socsim_report,
    ),
    (
        "Base-rate stress test",
        "alert precision vs production prevalence (Axelsson 1999)",
        "base_rate.md",
        run_base_rate_report,
    ),
    (
        "Neyman-Pearson thresholds",
        "a finite-sample guarantee that the FP budget holds (Tong, Feng & Li 2018)",
        "neyman_pearson.md",
        run_neyman_pearson_report,
    ),
    (
        "Time-to-detection survival",
        "Kaplan-Meier with the never-detected campaigns still in the denominator",
        "survival.md",
        run_survival_report,
    ),
    (
        "Decision latency",
        "when a flow verdict can first exist, and what deciding earlier costs",
        "earliness.md",
        run_earliness_report,
    ),
    (
        "Learning to defer",
        "which flows are worth an analyst's time; when 'abstain where unsure' is the wrong "
        "policy (Madras et al. 2018)",
        "defer.md",
        run_defer_report,
    ),
    (
        "Taxonomy-aware errors",
        "hierarchical P/R/F1 over the ATT&CK tree; which mistakes actually cost "
        "(Kiritchenko 2006)",
        "hierarchy.md",
        run_hierarchy_report,
    ),
    (
        "Causal invariance",
        "ICP screening + IRM over capture days, with the premise checked first "
        "(Peters 2016, Arjovsky 2019)",
        "invariance.md",
        run_invariance_report,
    ),
    (
        "Monotone constraints",
        "an entire evasion family made impossible by construction, proved and priced",
        "monotonic.md",
        run_monotonic_report,
    ),
    (
        "Byzantine-robust aggregation",
        "one lying site destroys FedAvg; median / trimmed mean / Krum priced "
        "(Blanchard 2017, Yin 2018)",
        "byzantine.md",
        run_byzantine_report,
    ),
    (
        "Group DRO",
        "train for the worst service, not the average one, against the cheap serving-side "
        "fix (Sagawa et al. 2020)",
        "dro.md",
        run_dro_report,
    ),
    (
        "Deterministic verification",
        "a sound, absolute robustness radius for the deployed ensemble by interval "
        "arithmetic (Chen et al. 2019)",
        "verify_trees.md",
        run_verify_trees_report,
    ),
    (
        "Epistemic vs aleatoric",
        "ambiguity or ignorance: uncertainty decomposed over an ensemble, tested on "
        "never-seen attack classes",
        "uncertainty.md",
        run_uncertainty_report,
    ),
    (
        "Off-policy evaluation",
        "value a triage policy you never deployed: IPS/SNIPS/doubly-robust "
        "(Dudik, Langford & Li 2011)",
        "ope.md",
        run_ope_report,
    ),
    (
        "Extreme-value thresholds",
        "peaks-over-threshold GPD fit: operating points past the edge of the data "
        "(Siffer et al. 2017)",
        "evt.md",
        run_evt_report,
    ),
    (
        "Conformal alert FDR",
        "a false-discovery-rate guarantee on the alert batch: conformal p-values + BH "
        "(Bates et al. 2023)",
        "alert_fdr.md",
        run_alert_fdr_report,
    ),
    (
        "Distribution-free risk control",
        "bound the miss rate the contract names, not the false-positive rate the threshold "
        "targets (Angelopoulos et al. 2021, 2022)",
        "risk_control.md",
        run_risk_control_report,
    ),
    (
        "Conformal prediction",
        "coverage guarantee + selective alerting",
        "conformal.md",
        run_conformal_report,
    ),
    (
        "Adaptive conformal",
        "coverage restored online under drift (ACI)",
        "adaptive_conformal.md",
        run_adaptive_conformal_report,
    ),
    (
        "Adversarial robustness",
        "evasion (mimicry + query search)",
        "robustness.md",
        run_robustness_report,
    ),
    (
        "Training-set poisoning",
        "label flips + benign-pool contamination",
        "poisoning.md",
        run_poisoning_report,
    ),
    (
        "Universal perturbation",
        "one vector fitted once and shipped as a constant: no queries at attack time, "
        "transferable across models, and structurally impossible against monotone constraints "
        "(Moosavi-Dezfooli et al. 2017)",
        "universal.md",
        run_universal_report,
    ),
    (
        "Adversarial hardening",
        "adversarial training vs mimicry, re-measured",
        "hardening.md",
        run_hardening_report,
    ),
    (
        "Certified robustness",
        "randomized smoothing: a provable L2 radius per flow (Cohen et al. 2019)",
        "certify.md",
        run_certify_report,
    ),
    (
        "Sensor failure",
        "the deployed model with a broken exporter: missing / stuck / mis-assembled fields",
        "degradation.md",
        run_degradation_report,
    ),
    (
        "Automatic slice discovery",
        "search for the underperforming regions nobody predefined, with a permuted null and a "
        "confirmation half (Chung et al. 2019)",
        "slice_discovery.md",
        run_slice_discovery_report,
    ),
    (
        "Cost-aware feature acquisition",
        "buy the expensive features only for the flows whose verdict is in doubt, against a "
        "random-gating control on the same budget",
        "acquisition.md",
        run_acquisition_report,
    ),
    (
        "Budgeted sampling",
        "score a fraction of the stream and estimate the rest: Horvitz-Thompson against four "
        "designs, including the one with no estimator at all",
        "sampling.md",
        run_sampling_report,
    ),
    (
        "Multi-objective selection",
        "a Pareto front over detection, cost and evasion-resistance, and the front members no "
        "weighted sum can reach (Deb et al. 2002)",
        "pareto.md",
        run_pareto_report,
    ),
    (
        "Server-side batching",
        "amortise the fixed cost of a scoring call across the requests already queued, and "
        "find the load below which waiting is a loss",
        "batching.md",
        run_batching_report,
    ),
    (
        "Budgeted cascade",
        "two-stage inference: the compute handed back and the detection it costs",
        "cascade.md",
        run_cascade_report,
    ),
    (
        "Sequential host decisions",
        "how many flows before a host can be called compromised (Wald's SPRT, 1945)",
        "sequential.md",
        run_sequential_report,
    ),
    (
        "Federated training",
        "detection when traffic cannot be pooled: FedAvg vs pooled vs alone (McMahan 2017)",
        "federated.md",
        run_federated_report,
    ),
    (
        "Self-supervised pretraining",
        "learn the representation from unlabelled flows, with PCA and an untrained encoder as "
        "the controls (VIME 2020, SCARF 2022)",
        "pretrain.md",
        run_pretrain_report,
    ),
    (
        "DP synthetic release",
        "share the traffic instead of the model: train-synthetic/test-real under a budget "
        "(PrivBayes family, Zhang et al. 2017)",
        "dp_synth.md",
        run_dp_synth_report,
    ),
    (
        "Secure aggregation",
        "federate without the coordinator seeing any site's update -- and what hiding it "
        "costs in robustness (Bonawitz et al. 2017)",
        "secagg.md",
        run_secagg_report,
    ),
    (
        "Poisoning defense",
        "audit-and-drop sanitization vs label flips, re-measured",
        "poisoning_defense.md",
        run_sanitize_report,
    ),
    (
        "Detection SLOs",
        "error budgets and multiwindow burn-rate alerting, with the rules generated",
        "slo.md",
        run_slo_report,
    ),
    (
        "Point-in-time feature store",
        "host context joined correctly vs over the whole capture: the temporal leak, priced",
        "feature_store.md",
        run_store_report,
    ),
    (
        "Closed-loop control",
        "alert volume held at the analyst budget by feedback, and the attack on the loop",
        "control.md",
        run_control_report,
    ),
    (
        "Operating-point training",
        "a partial-AUC surrogate against cross-entropy, scored at every false-positive budget",
        "operating_point.md",
        run_operating_point_report,
    ),
    (
        "Deep tabular models",
        "MLP and FT-Transformer against the boosted incumbent under one shared protocol",
        "deep_tabular.md",
        run_deep_tabular_report,
    ),
    (
        "Budgeted hyperparameter search",
        "successive halving and Hyperband at an equal budget, after measuring the two "
        "premises they rest on (Li et al. 2018)",
        "multifidelity.md",
        run_multifidelity_report,
    ),
    (
        "Online learning",
        "one-pass Hoeffding tree + ADWIN, prequentially, against static and periodic retraining",
        "online.md",
        run_online_report,
    ),
    (
        "Continual learning",
        "class-incremental updates: forgetting, replay and the compute argument, measured",
        "continual.md",
        run_continual_report,
    ),
    (
        "Multivariate drift (MMD)",
        "kernel two-sample testing: the joint change the per-feature monitors cannot see",
        "mmd.md",
        run_mmd_report,
    ),
    (
        "Optimal transport",
        "a drift distance in units and the coupling that explains it, then the distance an "
        "attacker has to travel (Cuturi 2013)",
        "transport.md",
        run_transport_report,
    ),
    (
        "Strategic equilibrium",
        "the arms race as a game: myopic race vs commitment, with the attacker cost priced",
        "strategic.md",
        run_strategic_report,
    ),
    (
        "Metamorphic testing",
        "a label-free correctness oracle, validated by injected mutants (Chen 1998, Xie 2011)",
        "metamorphic.md",
        run_metamorphic_report,
    ),
    (
        "Backdoor poisoning",
        "trigger trojan (BadNets) + spectral-signatures defense (Tran et al. 2018)",
        "backdoor.md",
        run_backdoor_report,
    ),
    (
        "Membership inference",
        "privacy leakage: does the model memorise its training data",
        "membership.md",
        run_membership_report,
    ),
    (
        "Differential privacy",
        "the (epsilon, delta) guarantee priced: detection & leakage vs epsilon",
        "dp.md",
        run_dp_report,
    ),
    (
        "Machine unlearning",
        "SISA exact deletion: sharding tax, per-request cost, verified forgetting "
        "(Bourtoule et al. 2021)",
        "unlearn.md",
        run_unlearn_report,
    ),
    (
        "Model extraction",
        "stealing the model by query: fidelity, stolen detection, transfer evasion",
        "extraction.md",
        run_extraction_report,
    ),
    (
        "Model watermarking",
        "prove ownership by backdooring: exact binomial test, innocent control, extraction "
        "survival (Adi et al. 2018)",
        "watermark.md",
        run_watermark_report,
    ),
    (
        "Label-noise audit",
        "confident-learning flags + planted-flip self-validation",
        "label_audit.md",
        run_label_audit_report,
    ),
    (
        "Training-data valuation",
        "KNN-Shapley value per flow: mislabel detection + value-guided pruning",
        "data_value.md",
        run_data_value_report,
    ),
    (
        "Prediction-powered inference",
        "attack prevalence from few labels + the model, with valid CIs (Angelopoulos 2023)",
        "ppi.md",
        run_ppi_report,
    ),
    (
        "Label-shift correction",
        "recover + correct for the deployment prior with zero labels (BBSE + MLLS/EM)",
        "label_shift.md",
        run_label_shift_report,
    ),
    ("Drift monitoring", "feature/score PSI, train vs test", "drift.md", run_drift_report),
    (
        "Statistical drift",
        "per-feature KS+FDR, online Page-Hinkley/DDM",
        "drift_tests.md",
        run_drift_tests_report,
    ),
    (
        "Anytime-valid drift",
        "conformal test martingale: a Ville-bounded false-alarm rate at any stopping time",
        "exchangeability.md",
        run_exchangeability_report,
    ),
    (
        "Covariate shift",
        "diagnose the temporal gap via a domain classifier + price importance-weighted "
        "retraining (Shimodaira 2000, Bickel 2009)",
        "covariate_shift.md",
        run_covariate_shift_report,
    ),
    (
        "Prequential streaming",
        "static vs retrained model on the later-day stream",
        "streaming.md",
        run_streaming_report,
    ),
    (
        "Retrain-trigger policy",
        "when to retrain: never / periodic / drift-triggered / every batch",
        "retrain_policy.md",
        run_retrain_policy_report,
    ),
    (
        "Threshold refresh",
        "the label-cheap lever vs retraining; budget compliance under drift",
        "refresh.md",
        run_refresh_report,
    ),
    (
        "Self-training",
        "pseudo-labels on the unlabeled stream vs the labeled ceiling",
        "selftrain.md",
        run_selftrain_report,
    ),
    (
        "Weak supervision",
        "the signatures as labeling functions: a detector trained on zero labels (Ratner 2016)",
        "weak_supervision.md",
        run_weak_supervision_report,
    ),
    (
        "PU learning",
        "confirmed attacks + unlabeled traffic: c recovery, weighted retrain, honest budgets "
        "(Elkan-Noto 2008)",
        "pu_learning.md",
        run_pu_learning_report,
    ),
    (
        "Expert advice (online)",
        "track the best model under drift with a regret bound: Hedge + fixed-share",
        "experts.md",
        run_experts_report,
    ),
    (
        "Model-family leaderboard",
        "every family through one honest protocol; the gap replicates",
        "leaderboard.md",
        run_leaderboard_report,
    ),
    (
        "Leakage attribution",
        "reproduce the field's ~99% and price each leakage source",
        "leakage.md",
        run_leakage_report,
    ),
    (
        "Per-class detection",
        "which temporal-split attacks are caught",
        "slices.md",
        run_slices_report,
    ),
    (
        "Campaign detection",
        "the (day, class) operation as the unit: first alerts and silent campaigns",
        "campaigns.md",
        run_campaigns_report,
    ),
    (
        "Per-service parity",
        "detection/false-alarm equity across services",
        "subgroups.md",
        run_subgroups_report,
    ),
    (
        "Attack-family discovery",
        "clustering the flagged pile into campaigns, with k chosen without labels",
        "discovery.md",
        run_discovery_report,
    ),
    (
        "Novelty distance",
        "detection vs distance-to-training; the split gap decomposed",
        "novelty.md",
        run_novelty_report,
    ),
    (
        "Rare-class rate estimation",
        "partial pooling so a twelve-flow class does not read like a thousand-flow one",
        "rare_rates.md",
        run_rare_rates_report,
    ),
    (
        "Open-set recognition",
        "the test days share no attack class with training: which novelty rule notices "
        "(Scheirer 2013, Dhamija 2018)",
        "openset.md",
        run_openset_report,
    ),
    (
        "Leave-one-day-out",
        "temporal sensitivity: every day takes a turn as the future",
        "lodo.md",
        run_lodo_report,
    ),
    (
        "Rules-vs-model baseline",
        "hand-written signatures at a matched FPR budget",
        "rules.md",
        run_rules_report,
    ),
    (
        "Feature-group ablation",
        "which behavioural families carry detection",
        "ablation.md",
        run_ablation_report,
    ),
    (
        "Counterfactual recourse",
        "minimal change that clears a hit",
        "recourse.md",
        run_recourse_report,
    ),
    (
        "SHAP estimand audit",
        "which Shapley value the API ships, graded against the coalition sum and against the "
        "two quantities it is usually confused with (Janzing et al. 2020)",
        "shap_estimand.md",
        run_shap_estimand_report,
    ),
    (
        "Importance stability",
        "are the shipped explanations stable across refits",
        "importance_stability.md",
        run_importance_stability_report,
    ),
    (
        "Predictive multiplicity",
        "how arbitrary is the verdict across equally-good models (Marx et al. 2020)",
        "multiplicity.md",
        run_multiplicity_report,
    ),
    (
        "Partial dependence & ICE",
        "the response-curve shape of the top features",
        "partial_dependence.md",
        run_partial_dependence_report,
    ),
    (
        "Feature interactions",
        "Friedman's H-statistic: which features the model has entangled",
        "interactions.md",
        run_interactions_report,
    ),
    (
        "Exemplar explanations",
        "do the nearest known training flows vouch for the alerts",
        "exemplars.md",
        run_exemplars_report,
    ),
    (
        "Anchor explanations",
        "high-precision IF-THEN rules with a coverage trade-off (Ribeiro et al. 2018)",
        "anchors.md",
        run_anchors_report,
    ),
    (
        "Anomaly attribution",
        "why a flow is abnormal: per-feature anomaly explanations + a faithfulness check",
        "anomaly_explain.md",
        run_anomaly_explain_report,
    ),
    (
        "Influence functions",
        "which training flows caused a verdict, validated against real LOO (Koh & Liang 2017)",
        "influence.md",
        run_influence_report,
    ),
    (
        "Optimal sparse trees",
        "how far greedy CART sits from the provably optimal tree, with a certificate "
        "(Hu, Rudin & Seltzer 2019)",
        "optimal_tree.md",
        run_optimal_tree_report,
    ),
    (
        "Surrogate distillation",
        "the model's closest auditable imitation, with fidelity priced",
        "distill.md",
        run_distill_report,
    ),
    (
        "Glass-box additive model",
        "a model that is its own explanation, and the capacity dial that shows what the "
        "honest split actually punishes (Lou, Caruana & Gehrke 2012)",
        "gam.md",
        run_gam_report,
    ),
    (
        "Active learning",
        "uncertainty vs random labeling efficiency",
        "active_learning.md",
        run_active_learning_report,
    ),
    (
        "Seed sensitivity",
        "the training-noise floor under every reported metric",
        "seed_variance.md",
        run_seed_variance_report,
    ),
    (
        "Release gate",
        "honesty invariants + metric floors the candidate must clear",
        "gate.md",
        _run_gate_report,
    ),
    (
        "Anytime-valid A/B",
        "when the shadow model can be promoted: peeking-safe confidence sequences",
        "sequential_ab.md",
        run_sequential_ab_report,
    ),
    ("MITRE ATT&CK coverage", "attack class -> tactic/technique", "mitre.md", run_mitre_report),
    (
        "Private inference",
        "score a flow under two-party secret sharing so neither side sees the other's "
        "secret, then read the model out with queries the server cannot refuse",
        "private_inference.md",
        run_private_inference_report,
    ),
    (
        "Private indicator sharing",
        "ask a peer whether they have seen an indicator without telling them which: DH private "
        "set intersection, and the dictionary attack on the hashing it replaces",
        "psi.md",
        run_psi_report,
    ),
    (
        "MITRE ATLAS coverage",
        "the detector as a target: this repo's own ML attack surface, with the gaps named",
        "atlas.md",
        run_atlas_report,
    ),
    (
        "ATT&CK Navigator layer",
        "detection coverage as a loadable Navigator layer",
        "attack_navigator_layer.json",
        run_navigator_export,
    ),
    (
        "Sigma detection rules",
        "the signature baseline exported as portable Sigma rules",
        "sigma/README.md",
        run_sigma_export,
    ),
    (
        "Streaming quantiles",
        "estimate the threshold's quantile in fixed memory, graded in alert volume rather than "
        "in quantile error (Jain & Chlamtac 1985; Dunning)",
        "quantiles.md",
        run_quantile_report,
    ),
    (
        "Streaming sketches",
        "host analytics at line rate in fixed memory, with every bound checked against exact "
        "truth (Cormode 2005, Flajolet 2007)",
        "sketches.md",
        run_sketches_report,
    ),
    (
        "Tamper-evident alert ledger",
        "hash-chained alert history: every edit attempted, and what verification catches",
        "ledger.md",
        run_ledger_report,
    ),
    (
        "Conformance mapping",
        "NIST AI RMF and EU AI Act obligations mapped to artifacts, with every claim verified "
        "against the repository",
        "compliance.md",
        run_compliance_report,
    ),
    (
        "Online triage learning",
        "a contextual bandit learning the operating point under partial feedback, and the "
        "alert budget its exploration spends",
        "bandit.md",
        run_bandit_report,
    ),
    (
        "Response side channel",
        "the verdict read off the length and timing of the reply, and which change to the "
        "contract actually closes it",
        "side_channel.md",
        run_side_channel_report,
    ),
    (
        "Serving lifecycle conformance",
        "the API contract as a state machine, driven through random operation sequences, with "
        "deliberately broken services proving the checker fails",
        "state_machine.md",
        run_lifecycle_report,
    ),
    (
        "Anomaly-score semantics",
        "is the anomaly score a density estimate or a complexity measure: six benign-only "
        "detectors, a control that learns nothing, and the size component regressed out",
        "density.md",
        run_density_report,
    ),
    (
        "ML-invariant static analysis",
        "the leakage rules enforced by a parser, with the rule set graded by injecting the "
        "violations it claims to catch",
        "mlint.md",
        run_mlint_report,
    ),
    (
        "Proof-carrying verdicts",
        "commit to the ensemble as a Merkle tree and prove each verdict against it, with "
        "seven forgeries executed and the leakage priced",
        "attestation.md",
        run_attestation_report,
    ),
    (
        "Reproducibility audit",
        "change one thing at a time and hash what comes out: which of byte, function and "
        "verdict reproducibility actually holds, and which guarantees depend on which",
        "determinism.md",
        run_determinism_report,
    ),
    (
        "Provenance & supply chain",
        "CycloneDX SBOM + model-integrity manifest",
        "provenance.md",
        run_provenance_report,
    ),
]


@dataclass
class AnalysisEntry:
    """The outcome of running one analysis in the suite."""

    title: str
    description: str
    filename: str
    ok: bool
    error: str | None = None


def write_index(reports_dir: Path, entries: list[AnalysisEntry]) -> Path:
    """Write the analysis index linking each report with its status."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NetSentry — Analysis Index",
        "",
        f"_Refreshed {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. `netsentry analyze` "
        "regenerates every report listed here and rewrites this index; a row without a link "
        "did not produce its report. Synthetic stand-in unless run on the real dataset._",
        "",
        "| report | what it covers | status |",
        "|---|---|---|",
    ]
    for e in entries:
        status = f"[open]({e.filename})" if e.ok else f"failed — {e.error}"
        lines.append(f"| {e.title} | {e.description} | {status} |")
    lines.append("")
    out = reports_dir / INDEX_NAME
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_full_analysis(settings: Settings) -> Path:
    """Run every analysis report and write the index; return the index path."""
    entries: list[AnalysisEntry] = []
    for title, description, filename, runner in _ANALYSES:
        try:
            runner(settings)
            entries.append(AnalysisEntry(title, description, filename, ok=True))
            logger.info("Analysis done", extra={"report": title})
        except Exception as exc:  # one report failing must not abort the suite
            logger.warning("Analysis failed (%s): %s", title, exc)
            entries.append(AnalysisEntry(title, description, filename, ok=False, error=str(exc)))
    index = write_index(settings.paths.reports_dir, entries)
    n_ok = sum(e.ok for e in entries)
    logger.info(
        "Wrote analysis index", extra={"path": str(index), "ok": n_ok, "total": len(entries)}
    )
    return index
