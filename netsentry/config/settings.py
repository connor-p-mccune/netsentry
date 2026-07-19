"""Typed configuration for NetSentry.

Every tunable knob — seed, paths, split strategy, model hyperparameters, decision
thresholds — lives here and is populated from YAML (see ``configs/``) with
environment-variable overrides (prefix ``NETSENTRY_``, nested delimiter ``__``).
No magic numbers in code: if a number affects behaviour, it belongs in config.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Per-call YAML payload, injected by the loader as a *low-priority* settings
# source so environment variables override YAML (which overrides model defaults).
_yaml_overrides: ContextVar[dict[str, Any] | None] = ContextVar("_yaml_overrides", default=None)


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """Feed merged YAML into Settings below env vars, with leaf-level deep-merge."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Values are supplied wholesale by __call__; per-field lookup is unused.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_yaml_overrides.get() or {})


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths resolve against the working directory."""

    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("docs/reports")
    figures_dir: Path = Path("docs/figures")
    mlruns_dir: Path = Path("mlruns")


class DataConfig(BaseModel):
    """Dataset acquisition and raw-handling knobs."""

    source_url: str | None = None
    archive_name: str = "cic-ids2017.zip"
    archive_sha256: str | None = None  # verify the downloaded archive if provided
    expected_csv_count: int = 8
    use_corrected_labels: bool = False
    # When the real dataset is unavailable, a clearly-labelled synthetic dataset
    # with the same schema and quirks can be generated for tests/CI/demos.
    allow_synthetic: bool = True
    synthetic_rows: int = 60000
    synthetic_attack_fraction: float = 0.22
    drop_duplicates: bool = True
    negative_sentinel_columns: list[str] = Field(
        default_factory=lambda: ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
    )
    negative_sentinel_strategy: Literal["keep", "nan"] = "keep"


class LabelConfig(BaseModel):
    """Label consolidation and target construction."""

    benign_label: str = "BENIGN"
    # Raw -> consolidated multiclass label. Web-attack variants are near-identical
    # and tiny, so they are merged; DoS sub-tools are kept distinct (documented in
    # DATA_CARD.md). Cleaning normalises whitespace/dashes before applying this.
    consolidation: dict[str, str] = Field(
        default_factory=lambda: {
            "Web Attack - Brute Force": "Web Attack",
            "Web Attack - XSS": "Web Attack",
            "Web Attack - Sql Injection": "Web Attack",
        }
    )


class SplitConfig(BaseModel):
    """How train/val/test are formed. Temporal is the honest headline split."""

    strategy: Literal["temporal", "stratified"] = "temporal"
    day_column: str = "Day"
    train_days: list[str] = Field(default_factory=lambda: ["Monday", "Tuesday", "Wednesday"])
    test_days: list[str] = Field(default_factory=lambda: ["Thursday", "Friday"])
    stratified_test_size: float = 0.2
    val_size: float = 0.2  # carved from TRAIN only, for thresholds/early stopping
    persist: bool = True


class FeatureConfig(BaseModel):
    """Feature pipeline configuration (the leakage firewall)."""

    feature_set: str = "full_no_port"
    scaler: Literal["standard", "robust", "none"] = "standard"
    impute_strategy: Literal["median", "mean"] = "median"
    encode_destination_port: bool = False
    destination_port_top_k: int = 32


class SupervisedConfig(BaseModel):
    """Supervised classifier. ``auto`` prefers LightGBM, falls back to sklearn."""

    backend: Literal["auto", "lightgbm", "hist_gbdt"] = "auto"
    task: Literal["binary", "multiclass"] = "multiclass"
    class_weight: Literal["balanced", "none"] = "balanced"
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 50
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50
    n_jobs: int = -1
    tune: bool = False
    tune_trials: int = 25


class AutoencoderConfig(BaseModel):
    """Benign-only PyTorch autoencoder (optional ``ae`` extra)."""

    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32, 16])
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5


class AnomalyConfig(BaseModel):
    """Unsupervised novel-attack detection."""

    iforest_n_estimators: int = 200
    iforest_max_samples: str = "auto"
    iforest_contamination: float = 0.02
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    target_fpr: float = 0.01
    loao_min_samples: int = 50  # skip leave-one-attack-out for classes rarer than this
    detectors: list[str] = Field(default_factory=lambda: ["iforest", "autoencoder"])


class ThresholdConfig(BaseModel):
    """Operating points. Thresholds are chosen on validation at a target FPR."""

    fpr_targets: list[float] = Field(default_factory=lambda: [0.001, 0.01])
    primary_fpr: float = 0.001
    assumed_flows_per_day: int = 1_000_000  # for the alerts/day estimate
    calibrate: bool = True
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"


class CostConfig(BaseModel):
    """Cost model for decision-theoretic threshold selection (the SOC economics).

    Every raised alert costs analyst time; every missed attack costs an expected
    loss. The cost-optimal threshold minimises total expected cost — a more
    defensible operating point than a round-number FPR. Values are illustrative
    and meant to be overridden per deployment."""

    cost_per_alert: float = 25.0  # triage cost of any raised alert (analyst time)
    cost_per_miss: float = 500.0  # expected loss from a missed attack flow
    # Production attack base rate for the daily extrapolation. The synthetic test
    # split is ~22% attack, which is wildly higher than real traffic; using a
    # realistic prior keeps alerts/day and $/day from being degenerate.
    production_attack_rate: float = 0.01
    currency: str = "$"
    grid_points: int = 300  # threshold grid resolution for the cost sweep


class AlertQueueConfig(BaseModel):
    """Capacity-constrained triage: the detection a fixed analyst budget actually buys.

    The cost report picks an expected-cost-minimising threshold; this asks the
    complementary operational question a SOC lead faces on Monday morning — "my team
    can work K alerts a day; ranking flows by risk, how many attacks do we catch, and
    how much better is that than triaging K flows at random?" Detection and precision
    are evaluated at a realistic production base rate (not the synthetic test mix), so
    the alert-per-day and analyst-headcount figures are not degenerate."""

    alert_budgets_per_day: list[int] = Field(
        default_factory=lambda: [50, 100, 250, 500, 1000, 2500]
    )
    minutes_per_alert: float = 10.0  # analyst triage time budgeted per alert
    analyst_minutes_per_day: float = 420.0  # ~7 productive hours per analyst per day


class SocSimConfig(BaseModel):
    """Discrete-event simulation of the analyst queue at the deployed operating point.

    The alert-queue study is static capacity planning — at budget K, what fraction
    of attacks does the ranking put in the queue. This adds the dimension a
    fraction cannot show: **time**. Real alerts arrive over a shift (benign false
    positives roughly uniform, attack alerts clustered into campaigns), analysts
    are finite servers with a per-alert service time, and a saturated queue makes
    the triage discipline decide *which* attacks are reviewed before the shift
    ends. FIFO works the oldest ticket; score-priority lets a high-risk attack
    jump a benign false-positive pileup. The study sweeps analyst headcount so the
    saturation knee is visible, and every point is a median over ``n_runs`` seeded
    arrival draws. The timeline is a documented model (CIC-IDS2017 carries no
    per-flow wall-clock), driven by the model's *real* score distribution and
    labels."""

    horizon_minutes: float = 480.0  # one analyst shift
    arrivals_per_shift: int = 300  # alerts entering the queue over the shift (sampled)
    minutes_per_alert_mean: float = 8.0  # mean exponential service time per alert
    sla_minutes: float = 30.0  # an attack alert must reach an analyst within this
    n_campaigns: int = 4  # attack alerts cluster into this many bursts
    campaign_spread_minutes: float = 15.0  # burst width (std dev of arrival jitter)
    analyst_counts: list[int] = Field(default_factory=lambda: [2, 3, 4, 6, 8])
    n_runs: int = 20  # seeded arrival draws per (headcount, discipline); medians reported


class BaseRateConfig(BaseModel):
    """Base-rate stress test: the operating points re-read at deployment prevalences.

    Axelsson's base-rate fallacy (1999): alert precision is governed by the attack
    prevalence at least as much as by the detector's conditional rates, so an FPR
    budget that looks strict on a ~22% test mix can still bury analysts at a
    1-in-10,000 production base rate. The priors sweep should span the orders of
    magnitude a deployment could plausibly sit at."""

    priors: list[float] = Field(default_factory=lambda: [0.00001, 0.0001, 0.001, 0.01, 0.1])
    precision_target: float = 0.9  # queue precision used for the required-FPR inversion


class CaptureConfig(BaseModel):
    """Raw packet-capture ingestion (PCAP -> CIC flow features).

    Timeouts mirror CICFlowMeter's flow semantics so features computed from a
    capture line up with the training data: a flow ends after ``flow_timeout_us``
    of silence (or a TCP close), and the active/idle features split the packet
    timeline at gaps longer than ``activity_timeout_us``."""

    flow_timeout_us: int = 120_000_000  # idle time (us) after which a 5-tuple starts a new flow
    activity_timeout_us: int = 5_000_000  # gap (us) separating active periods (Active/Idle stats)


class ValidationConfig(BaseModel):
    """Thresholds for the input data-quality gates (fail loudly on bad input)."""

    max_nan_fraction: float = 0.5  # warn if a feature column exceeds this missing rate
    max_duplicate_fraction: float = 0.2  # warn above this exact-duplicate share


class EvaluationConfig(BaseModel):
    """Uncertainty quantification for the reported metrics."""

    bootstrap_samples: int = 1000  # resamples for metric confidence intervals
    bootstrap_alpha: float = 0.05  # 1 - alpha is the CI level (0.05 -> 95%)
    learning_curve_fractions: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0]
    )


class SubgroupsConfig(BaseModel):
    """Per-service detection-parity audit at a single global threshold.

    Groups the honest-split test flows by the service implied by ``Destination Port``
    — a field the model never sees, since it is dropped to prevent port-memorisation —
    and measures detection rate and false-positive rate per service at one global
    operating threshold. Large unintended gaps are the operational analogue of an
    equalized-odds fairness audit: they show where a per-service threshold would beat
    one global cut, and which services a global cut floods with false positives."""

    min_support: int = 100  # flows a service needs before its rates are reported


class CampaignsConfig(BaseModel):
    """Campaign-level detection: the (day, attack-class) operation as the unit.

    A campaign counts as alerted when >= 1 flow crosses the operating threshold;
    ``k_confirm`` is the conservative reading (a single hit may not start an
    investigation if nothing correlates the alerts). The framing changes the
    numerator only — benign flows have no campaign structure, so alert volume is
    still priced per flow by the FPR budget."""

    k_confirm: int = 5  # alerts a campaign needs to count as confidently detected


class NoveltyConfig(BaseModel):
    """Novelty-distance study: detection as a function of distance to the training set.

    For every test attack, the Euclidean distance (in the pipeline's standardized
    feature space) to its nearest training attack measures how *novel* the flow is to
    the model. Binning detection rate by that distance, for both split strategies,
    exposes the mechanism behind the temporal-vs-stratified gap: whether the shuffled
    split flatters because its test attacks sit near training twins (a composition
    effect over one decay curve), or because performance at matched novelty also
    shifts. Reference/query caps keep the k-NN index fast on the full dataset."""

    max_reference: int = 30000  # cap on training attacks indexed for the NN lookup
    max_queries: int = 10000  # cap on test attacks scored per split
    n_bins: int = 5  # distance bins (quantile edges over the pooled distances)
    # A test attack closer than this (standardized units, summed over ~77 dims) to a
    # training attack is a near-twin — on the real CIC data these are the shuffled
    # split's leakage; exact duplicates were already dropped in cleaning.
    twin_epsilon: float = 0.5


class ConformalConfig(BaseModel):
    """Split-conformal prediction: distribution-free coverage + selective alerting.

    The model emits a *set* per flow with a finite-sample guarantee that the true
    label is inside with probability >= 1 - alpha. Ambiguous (both-label) and empty
    (neither-label, i.e. novel) sets are routed to a human, so the analyst only sees
    the flows the model is genuinely unsure about."""

    alpha: float = 0.1  # target error rate; coverage target is 1 - alpha
    alphas_grid: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])


class AdaptiveConformalConfig(BaseModel):
    """Adaptive conformal inference (Gibbs & Candes 2021) on the labeled stream.

    Static split-conformal loses its guarantee when drift breaks exchangeability
    (the conformal report's temporal finding); ACI steers alpha online from the
    realized coverage errors — alpha_(t+1) = alpha_t + gamma (alpha - err_t) —
    which restores a long-run coverage guarantee under *arbitrary* shift, at the
    price of label feedback and wider (more often human-reviewed) sets. ``gamma``
    trades reaction speed against set-size stability; ``label_delay`` models the
    triage lag before ground truth arrives."""

    gamma: float = 0.005  # ACI step size
    window: int = 2000  # trailing-window size for the rolling-coverage figure
    label_delay: int = 0  # flows between a decision and its label feeding back


class MonitoringConfig(BaseModel):
    """Data-drift monitoring (PSI) — the production-decay early-warning system."""

    psi_bins: int = 10
    psi_moderate: float = 0.1  # PSI >= this is a moderate distribution shift
    psi_major: float = 0.25  # PSI >= this is a major shift worth investigating
    serving_window: int = 500  # flows buffered before serving recomputes drift gauges
    reference_rows: int = 5000  # reference sample summarised into the serving bundle


class DistillConfig(BaseModel):
    """Surrogate distillation: the model's closest small, auditable imitation.

    A depth-limited decision tree is trained to imitate the teacher's calibrated
    attack score (classic model distillation) and judged on fidelity (Spearman +
    decision agreement at matched alert volume) and on its own detection — so the
    price of auditability is a measured number per depth, not a vibe. The chosen
    ``report_depth`` tree is rendered into the report in full."""

    depths: list[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6])
    report_depth: int = 4  # the depth whose rules are rendered in the report
    min_samples_leaf: int = 50  # leaf support floor: rules must describe real traffic
    max_rule_lines: int = 80  # cap the rendered rule text in the report


class ImportanceStabilityConfig(BaseModel):
    """Explanation-trust audit: are the model's feature importances stable across refits?

    The API ships SHAP top-features as a product contract, and the report shows a global
    importance ranking — but a ranking from a *single* fit could be an artifact of one
    lucky sample. This refits the model on bootstrap resamples of the training data,
    recomputes global importance each time, and measures how much the ranking moves: a
    high rank correlation and top-k overlap means the explanations are trustworthy, not
    noise. It is the honesty check behind treating explainability as a contract."""

    n_bootstrap: int = 15  # bootstrap refits of the training data
    top_k: int = 10  # size of the top-feature set whose stability is tracked
    permutation_repeats: int = 3  # only for the model-agnostic permutation fallback
    max_val_rows: int = 4000  # cap validation rows for the permutation fallback (speed)


class PartialDependenceConfig(BaseModel):
    """Partial dependence + ICE: the response-curve shape of the top model features.

    Complements the SHAP importance ranking (which features), the ablation (a
    family's causal value), and the importance-stability audit (is the ranking
    trustworthy) with the one thing none of them show — how the predicted attack
    probability *moves* as a feature sweeps its range. Computed in raw feature space
    through the fitted pipeline, so the axis is interpretable and there is no
    train/serve skew. ``grid_trim_quantile`` clips the sweep to the feature's central
    mass so a single outlier does not stretch the grid into empty space."""

    top_k: int = 6  # most-important features to profile
    grid_points: int = 20  # sweep resolution per feature
    ice_samples: int = 40  # individual ICE curves drawn under each PDP
    sample_rows: int = 500  # validation rows the PDP is averaged over
    grid_trim_quantile: float = 0.05  # trim each tail before building the grid


class InteractionsConfig(BaseModel):
    """Feature-interaction strength via Friedman's H-statistic (Friedman & Popescu 2008).

    The partial-dependence study shows each top feature's marginal response but assumes
    independence; this measures the interaction that assumption hides. The pairwise H is
    the share of a feature pair's joint-partial-dependence variance that is *not*
    explained by summing the two marginals — 0 (additive) to 1 (fully entangled). It is
    estimated on the honest temporal model over a background sample, through the fitted
    pipeline, so it reads against the PDP. ``top_k`` features give ``top_k*(top_k-1)/2``
    pairs; ``sample_rows`` is the Monte-Carlo background (cost is quadratic in it per
    pair, so keep it modest); ``max_pairs_reported`` caps the ranked table."""

    top_k: int = 5  # top features (by model importance) whose pairwise H is measured
    sample_rows: int = 150  # background sample the H-statistic is estimated over
    max_pairs_reported: int = 12  # ranked interacting pairs shown in the report


class AnomalyExplainConfig(BaseModel):
    """Per-feature attribution for anomaly flags — the unsupervised mirror of SHAP.

    The anomaly detector emits only a score; this names the behaviours behind a flag
    by model-agnostic benign occlusion (reset each feature to its benign reference,
    re-score, and read the drop). ``max_explained`` caps the flagged flows attributed
    (occlusion re-scores once per feature); ``top_k`` features are listed per attack
    class (a class needs ``min_class_flags`` flags to be profiled); ``report_features``
    sets the global table/figure length; ``faithfulness_k`` is the deletion-test width
    (the top-k vs random-k score-drop comparison that validates the attributions)."""

    max_explained: int = 400  # flagged flows attributed (occlusion cost scales with this)
    top_k: int = 6  # features listed per attack class
    report_features: int = 12  # features in the global ranking table/figure
    min_class_flags: int = 10  # flagged flows a class needs before it is profiled
    faithfulness_k: int = 5  # features occluded in the top-k-vs-random deletion check


class AnchorsConfig(BaseModel):
    """High-precision IF-THEN anchor rules for a verdict (Ribeiro, Singh & Guestrin 2018).

    SHAP attributes a verdict, the counterfactual finds the smallest clearing change, and
    exemplars point at similar cases — but none states a **sufficient condition**. An anchor
    is a short conjunction of feature predicates such that, whenever they hold, the model
    returns this verdict with high **precision** (>= ``precision_threshold``); of the many
    such rules the useful one has high **coverage**. Each candidate feature is discretised
    into ``n_bins`` quantile bins and a greedy search pins the flagged flow to its own bins,
    adding at each step the predicate that most raises precision (estimated on a background
    of ``background_rows`` real flows satisfying the rule, requiring ``min_match`` supporting
    rows), until a lower confidence bound at width ``confidence_z`` clears the threshold or
    the rule reaches ``max_predicates``. ``top_k_features`` (by model importance) are eligible
    predicates; ``n_explained`` flagged test flows are anchored, and each anchor's precision
    is re-validated on a held-out background. Runs on the exchangeable stratified/binary split,
    where the model's decision boundary is well-populated and the held-out background is
    exchangeable with the reference the rules are grown on."""

    top_k_features: int = 8  # features (by importance) eligible as anchor predicates
    n_bins: int = 5  # quantile bins each feature is discretised into
    precision_threshold: float = 0.95  # target precision (tau) the anchor must clear
    max_predicates: int = 4  # maximum clauses in one anchor
    background_rows: int = 4000  # reference flows the precision/coverage are estimated on
    min_match: int = 30  # minimum background rows satisfying a rule to trust its precision
    n_explained: int = 25  # flagged flows anchored and reported
    confidence_z: float = 1.64  # z for the one-sided precision lower confidence bound


class ExemplarConfig(BaseModel):
    """Exemplar (case-based) explanations: nearest known training flows per query.

    A class-balanced sample of the training split (so rare attack classes are
    represented, not drowned by benign volume) is held in the fitted pipeline's
    standardized space; retrieval is exact k-NN. The report audits agreement
    (are exemplar-supported alerts more precise?) and distance-as-novelty before
    the API ships ``similar_flows``. Sized to stay embeddable in a bundle."""

    per_class: int = 200  # exemplars kept per class label
    k: int = 5  # neighbours retrieved per query flow
    examples: int = 5  # example alerts rendered in the report


class GateConfig(BaseModel):
    """Release quality gate: the bars a candidate must clear before it ships.

    Structural honesty checks (leakage firewall on the fitted artifact, calibrator
    present, threshold profiles complete, scoring smoke) always run; these knobs set
    the performance floors — and one deliberate *ceiling*: a PR-AUC above
    ``max_pr_auc`` fails the gate because on this data a near-perfect score is
    overwhelmingly more likely to be leakage than skill. Floors are relative to the
    attack prevalence where possible so the policy transfers across base rates.
    Defaults are set to pass the synthetic stand-in with headroom; tune per
    deployment."""

    min_pr_auc_lift: float = 1.5  # PR-AUC >= lift x prevalence (random-ranker baseline)
    max_pr_auc: float = 0.999  # above this, assume leakage until a human explains it
    min_tpr_at_primary_fpr: float = 0.05  # detection floor at the primary FP budget
    # ECE of the *calibrated* score on the honest test split. Under temporal shift a
    # validation-fit calibrator honestly degrades (~0.11 on the stand-in, vs ~0.12
    # raw); the bar allows that documented headroom while still catching a grossly
    # mis-calibrated probability.
    max_ece: float = 0.15


class PromotionConfig(BaseModel):
    """Champion/challenger promotion policy (the decision layer before serving).

    Margins are non-inferiority bands on the paired-bootstrap deltas, calibrated
    from the seed-sensitivity audit: PR-AUC moves ~0.002 sd and TPR@0.1%FPR ~0.006 sd
    across seeds on the stand-in, so the defaults sit just above that training-noise
    floor — a promotion decided inside the band would be a decision about luck.
    ``non_inferiority`` rolls routine retrains forward unless credibly worse (right
    under drift, where freshness has measured value); ``superiority`` additionally
    demands the delta CI exclude zero (right for risky architecture swaps)."""

    policy: Literal["non_inferiority", "superiority"] = "non_inferiority"
    metric_margin: float = 0.005  # PR-AUC non-inferiority margin (~3x seed sd)
    tpr_margin: float = 0.015  # TPR@primary-FPR margin (~2.5x seed sd)
    require_tpr_non_inferior: bool = True
    n_boot: int = 1000  # paired-bootstrap resamples for the delta CIs


class SeedVarianceConfig(BaseModel):
    """Training-noise audit: refit the honest model across seeds, report the spread.

    Bootstrap CIs quantify *data* noise (resampling the evaluation rows); this
    measures *training* noise (row/feature subsampling, tie-breaking) by refitting
    the same config at consecutive seeds. The metric standard deviation across those
    refits is the noise floor any model-to-model comparison must clear, and the
    evidence behind the promotion gate's non-inferiority margin (PromotionConfig)."""

    n_seeds: int = 5  # refits at consecutive seeds, base seed first


class DriftDetectorConfig(BaseModel):
    """Statistical / online concept-drift detectors — significance, not just PSI magnitude.

    PSI reports how *far* a distribution moved but carries no notion of significance,
    and it is computed on static batches. These add the two things PSI cannot: a
    per-feature two-sample **Kolmogorov-Smirnov** test (with Benjamini-Hochberg FDR
    control, so 'how many features genuinely drifted' is an honest count, not a
    threshold on an effect size), and two classic **online** detectors that answer
    *when* the stream broke — **Page-Hinkley** on the model-score stream and **DDM**
    (Gama et al. 2004) on the model-error stream."""

    ks_fdr_alpha: float = 0.05  # Benjamini-Hochberg false-discovery rate for the KS tests
    ph_delta: float = 0.005  # Page-Hinkley magnitude tolerance (drift allowed before alarming)
    ph_lambda: float = 50.0  # Page-Hinkley alarm threshold on the cumulative deviation
    ddm_warn_level: float = 2.0  # DDM warning zone: error rate >= min + warn * sigma_min
    ddm_drift_level: float = 3.0  # DDM drift alarm: error rate >= min + drift * sigma_min
    # DDM's cumulative error-rate estimate is volatile at small n and its 3-sigma band
    # tightens as the stream grows, so a real-data error stream needs a substantial
    # warmup to establish a stable baseline before the detector is armed.
    ddm_min_samples: int = 2000
    max_features_reported: int = 25  # cap the per-feature KS table in the report


class ExchangeabilityConfig(BaseModel):
    """Anytime-valid drift detection via a conformal test martingale (Vovk et al. 2003).

    The windowed drift detectors (PSI, KS+FDR, Page-Hinkley, DDM) either need a reference
    window or spend their false-alarm budget at a declared moment. A conformal test
    martingale spends none: it bets against the null that the stream is **exchangeable**,
    accumulating a non-negative martingale that stays a fair game under the null and grows
    without bound under drift, so by **Ville's inequality** alarming at ``M_t >= 1/alpha``
    has false-alarm probability at most ``alpha`` at *any* stopping time. ``alpha`` sets
    that budget (and the ``1/alpha`` alarm line); ``stream_len`` is the number of flows per
    stream; the drift stream turns attack-heavy at ``change_point`` with attack fraction
    ``post_change_attack_rate``; ``n_bets`` is the size of the power-martingale mixture grid;
    ``n_null_streams`` independent exchangeable streams estimate the empirical false-alarm
    rate against the Ville bound. Uses the deployed temporal/binary attack score as the
    nonconformity measure, so the test watches the same signal the detector acts on."""

    alpha: float = 0.01  # false-alarm budget; alarm when M_t >= 1/alpha
    stream_len: int = 2000  # flows per stream
    change_point: int = 1000  # the drift stream turns attack-heavy here
    post_change_attack_rate: float = 0.8  # attack fraction after the change point
    n_bets: int = 19  # power-martingale mixture grid size (epsilons in the open unit interval)
    n_null_streams: int = 50  # independent exchangeable streams for the false-alarm estimate


class RobustnessConfig(BaseModel):
    """Adversarial-evasion evaluation: how detection degrades under an attacker.

    The threat model is an attacker who shapes the *controllable* parts of a flow
    (volume, timing, sizes — by padding, dummy packets, delays) to look benign,
    while the protocol-structural fields stay fixed. Budgets are in standardized
    feature-space units (the model's own scale)."""

    # CIC features an attacker can plausibly manipulate without breaking the attack.
    controllable_features: list[str] = Field(
        default_factory=lambda: [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Idle Mean",
            "Active Mean",
        ]
    )
    mimicry_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    search_budgets: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 3.0])
    search_iterations: int = 150
    max_attack_samples: int = 3000  # cap evaluated attack flows so the study stays fast
    profile: str = "fpr_1pct"  # operating point the attacker tries to slip under
    recourse_max_steps: int = 5  # max features a counterfactual explanation may change


class MembershipConfig(BaseModel):
    """Membership-inference privacy audit: how much does the model memorise its data?

    The third classic adversarial axis after evasion (inference-time) and poisoning
    (training-time). With only query access, can an attacker tell whether a flow was in
    the training set (Shokri et al. 2017; Yeom et al. 2018)? Runs on the exchangeable
    stratified split — the assumption MI needs. ``target_train_rows`` sizes the member
    pool; ``n_shadow`` shadow models feed the Shokri attack classifier; ``top_k_confidences``
    is the width of the sorted-probability feature vector; ``attack_fpr`` is the low
    false-accusation budget for the worst-case TPR (Carlini et al. 2022). Deliberately a
    few thousand rows and a handful of shadows so the audit stays fast."""

    target_train_rows: int = 6000  # rows the target model trains on (the members)
    eval_rows: int = 3000  # members and non-members each capped to this for the attack
    n_shadow: int = 8  # shadow models mimicking the target (Shokri)
    shadow_rows: int = 6000  # auxiliary pool per study, split in/out across shadows
    top_k_confidences: int = 3  # sorted top-k probabilities used as attack features
    attack_fpr: float = 0.01  # low false-accusation budget for the worst-case TPR


class DPConfig(BaseModel):
    """Differentially-private training: the formal privacy control, priced.

    The membership audit measures leakage; this prices the mitigation with a formal
    guarantee. DP-SGD logistic models are trained at a sweep of Gaussian
    ``noise_multipliers`` (0.0 is the non-private reference), each priced on the
    same axis: the epsilon it spends (at a fixed ``delta``), the detection it keeps,
    and the membership leak it closes (the same Yeom attack the membership audit
    runs). ``l2_clip`` bounds each flow's per-example gradient; ``epochs`` /
    ``batch_size`` / ``lr`` are the optimiser knobs the accountant reads as
    (steps, sampling rate). Deliberately a few thousand rows and a linear model so
    the study stays fast and the accountant stays exact."""

    noise_multipliers: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    l2_clip: float = 1.0  # per-example gradient L2 clip norm (the influence bound)
    epochs: int = 60
    lr: float = 0.5
    batch_size: int = 256
    l2_reg: float = 1e-4  # weight decay (a private prior; the bias is never penalised)
    delta: float = 1e-5  # the (epsilon, delta) budget's delta, fixed across the sweep
    target_train_rows: int = 6000  # rows the models train on (the members)
    eval_rows: int = 3000  # members/non-members each capped to this for the attack
    primary_fpr: float = 0.001  # operating point for the TPR utility column
    attack_fpr: float = 0.01  # low false-accusation budget for the worst-case leak


class ExtractionConfig(BaseModel):
    """Model-extraction (model-stealing) attack: is the deployed model stealable by query?

    The fourth classic adversarial axis after evasion, poisoning, and membership
    inference — the one about the confidentiality of the *model*. A surrogate is
    trained purely on the victim's returned scores over the attacker's own
    same-distribution traffic (no ground-truth labels), and its fidelity (agreement
    with the victim) and stolen detection (PR-AUC) are swept over ``query_budgets``.
    ``round_decimals`` sets the precision of the 'rounded' query-response defense
    (the label-only defense returns the top-1 class); ``transfer_*`` parametrise the
    black-box transfer-evasion attack the stolen surrogate enables — an L2 search of
    radius ``transfer_budget`` (standardised units, matching the robustness study)
    for ``transfer_iterations`` random restarts over up to ``max_attack_samples``
    attack flows, scored at the ``transfer_fpr`` operating point. Runs on the
    exchangeable stratified/binary split; deliberately a few thousand rows and a
    generic surrogate so the study stays fast."""

    query_budgets: list[int] = Field(default_factory=lambda: [250, 500, 1000, 2000, 4000])
    round_decimals: int = 1  # precision of the 'rounded' query-response defense
    max_eval_rows: int = 4000  # held-out rows for fidelity/PR-AUC measurement
    transfer_budget: float = 2.0  # L2 evasion budget (standardised units) for the transfer attack
    transfer_iterations: int = 100  # random-restart search iterations for the transfer attack
    transfer_fpr: float = 0.01  # victim operating point the transfer attack tries to slip under
    max_attack_samples: int = 1500  # attack flows perturbed in the transfer experiment


class CertifyConfig(BaseModel):
    """Certified robustness via randomized smoothing (Cohen, Rosenfeld & Kolter 2019).

    The formal-guarantee counterpart to the empirical evasion study: the smoothed
    classifier (majority vote under Gaussian noise) comes with a provable L2 radius
    ``R = sigma * Phi^-1(p_A)``, where ``p_A`` is a Clopper-Pearson lower bound (at
    confidence 1 - ``alpha``) on the majority-vote probability over ``n_samples`` noise
    draws. ``sigmas`` sweep the accuracy/robustness frontier (more noise certifies farther
    but detects less); ``radii_grid`` sets the certified-accuracy sweep; ``max_flows``
    class-balanced test flows are certified (cost is ``n_samples`` model scorings per
    flow, so both are kept modest); ``target_fpr`` sets the base detector's operating
    point. Radii are in standardised-feature units — the same scale as the evasion search
    budgets, so the reports read against each other. Runs on the stratified/binary split."""

    sigmas: list[float] = Field(default_factory=lambda: [0.25, 0.5, 1.0])
    n_samples: int = 1000  # Monte-Carlo noise draws per flow for the certificate
    alpha: float = 0.001  # 1 - alpha is the certificate's confidence level
    max_flows: int = 300  # class-balanced test flows certified (n_samples scorings each)
    target_fpr: float = 0.01  # operating point of the base detector being smoothed
    radii_grid: list[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 1.5, 2.0])


class HardeningConfig(BaseModel):
    """Adversarial training against the feature-space mimicry the evasion study runs.

    The robustness study *measures* how mimicry collapses detection; this closes the
    loop and *acts* on it. Training is augmented with mimicry-perturbed copies of the
    attack flows — the same move the attacker makes, so the classifier learns that a
    flow shaped toward the benign centroid on its attacker-controllable features is
    still an attack. Adversarial training is expected to trade a little clean
    detection for robustness; the report measures both sides of that trade rather than
    asserting the win, in keeping with the project's honesty thesis."""

    # Mimicry fractions synthesized into the training set. Including 1.0 means the
    # model trains on exactly the fully-mimicked attack the evasion study produces.
    mimicry_train_fractions: list[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0])
    max_augmented: int = 6000  # cap on synthesized adversarial rows (keeps refits fast)


class RuleClause(BaseModel):
    """One comparison in a signature rule: ``feature OP value`` (NaN never matches)."""

    feature: str
    op: Literal["ge", "le", "eq"]
    value: float


class RuleDefinition(BaseModel):
    """A named, human-auditable signature that fires when every clause holds."""

    name: str
    description: str
    clauses: list[RuleClause]


def _clause(feature: str, op: Literal["ge", "le", "eq"], value: float) -> RuleClause:
    return RuleClause(feature=feature, op=op, value=value)


def _default_rules() -> list[RuleDefinition]:
    """Signatures a SOC would plausibly hand-write for the CIC-IDS2017 attack mix.

    Thresholds are in raw feature units (packets/s, microseconds, bytes) and are
    deliberately conservative — a signature's job is precision on the pattern it
    encodes, not coverage. Note rules are *allowed* to key on ``Destination Port``:
    port-scoping is exactly what real signatures do, whereas the ML model drops the
    port to avoid memorising it — a contrast the rules report calls out.
    """
    return [
        RuleDefinition(
            name="volumetric-flood",
            description="High packet- and byte-rate flood (DoS Hulk / DDoS style)",
            clauses=[
                _clause("Flow Packets/s", "ge", 800.0),
                _clause("Flow Bytes/s", "ge", 8000.0),
            ],
        ),
        RuleDefinition(
            name="port-scan-sweep",
            description="Short, SYN-heavy, low-volume probe (PortScan style)",
            clauses=[
                _clause("SYN Flag Count", "ge", 4.0),
                _clause("Flow Duration", "le", 20000.0),
                _clause("Total Fwd Packets", "le", 5.0),
            ],
        ),
        RuleDefinition(
            name="slow-drip-dos",
            description="Connection held open with sparse traffic (slowloris style)",
            clauses=[
                _clause("Flow Duration", "ge", 600000.0),
                _clause("Flow IAT Mean", "ge", 50000.0),
                _clause("Total Fwd Packets", "le", 8.0),
            ],
        ),
        RuleDefinition(
            name="ftp-bruteforce",
            description="Rapid repeated connections to FTP (Patator style)",
            clauses=[
                _clause("Destination Port", "eq", 21.0),
                _clause("SYN Flag Count", "ge", 4.0),
                _clause("Total Fwd Packets", "ge", 20.0),
            ],
        ),
        RuleDefinition(
            name="ssh-bruteforce",
            description="Rapid repeated connections to SSH (Patator style)",
            clauses=[
                _clause("Destination Port", "eq", 22.0),
                _clause("SYN Flag Count", "ge", 4.0),
                _clause("Total Fwd Packets", "ge", 20.0),
            ],
        ),
        RuleDefinition(
            name="tls-heartbeat-exfil",
            description="Oversized TLS responses to tiny requests (Heartbleed style)",
            clauses=[
                _clause("Destination Port", "eq", 443.0),
                _clause("Bwd Packet Length Max", "ge", 300.0),
                _clause("Total Length of Bwd Packets", "ge", 4000.0),
            ],
        ),
    ]


class RulesConfig(BaseModel):
    """Hand-written signature baseline the ML model is benchmarked against.

    Rules are config, not code, so an operator can audit, tune, or extend them the
    way they would a Suricata ruleset — and the comparison report re-runs unchanged.
    """

    definitions: list[RuleDefinition] = Field(default_factory=_default_rules)


class RetrainPolicyConfig(BaseModel):
    """Retrain-trigger policy study: when should the drift signal pull the lever?

    The streaming study shows retraining recovers what drift costs; this prices
    *when*. Four policies ride the same prequential stream — never (floor), every
    batch (ceiling), periodic (the calendar default), and drift-triggered (retrain
    when the deployed model's own score-PSI breaches the major-drift line, with a
    cooldown) — and the report is the efficiency frontier: mean batch PR-AUC vs
    number of retrains. The trigger threshold defaults to ``monitoring.psi_major``,
    the same line the Prometheus alert fires on, so measurement, alert, and action
    share one number."""

    n_batches: int = 8  # finer than the streaming study so triggers have room to differ
    periodic_every: int = 3  # the calendar baseline: retrain every k-th batch
    psi_trigger: float | None = None  # score-PSI retrain trigger; None -> psi_major
    cooldown_batches: int = 2  # min batches between drift-triggered retrains


class RefreshConfig(BaseModel):
    """Threshold-refresh study: the label-cheap adaptation lever, priced.

    Between a frozen deployment and a full retrain sits re-choosing only the
    decision threshold on a trailing window of recently labeled flows, at the same
    FPR budget. The study decomposes drift's cost into operating-point drift (the
    score distribution moved — a refresh fixes it) and ranking drift (the model is
    blind to new attack types — only retraining fixes it). Refreshed cuts are
    chosen on the prequentially *emitted* scores, so no model picks its threshold
    on flows it trained on."""

    n_batches: int = 8  # matches the retrain-policy stream so results compare
    window_batches: int = 2  # trailing labeled batches the refreshed cut is chosen on


class StreamingConfig(BaseModel):
    """Prequential streaming simulation: does retraining recover from drift?

    The drift monitor *measures* decay; this closes the loop to the *action*.
    Later-day test flows arrive as a time-ordered stream of batches, and two
    policies are compared prequentially (score each batch, then learn from it): a
    **static** model frozen at deploy versus one **retrained** on each labeled batch.
    The gap is the value of continuous learning against later-day, partly-novel
    attacks — and the reason labels (see the active-learning study) are the cost."""

    n_batches: int = 6  # time-ordered windows the later-day stream is split into
    retrain: bool = True  # compare a retrained model against the static one


class ActiveLearningConfig(BaseModel):
    """Analyst-labeling-budget study: does querying uncertain flows beat random?

    Labels are the scarce resource in a SOC (an analyst's time), so the question is
    label *efficiency*: starting from a small labeled seed, which flows should the
    analyst label next to most improve detection. Uncertainty sampling (query flows
    nearest the decision boundary) is compared against a random baseline. Runs on the
    stratified split, where the pool and test are exchangeable — the assumption
    active learning needs, and the one the temporal shift deliberately breaks."""

    seed_size: int = 500  # initial randomly-labeled flows
    query_batch: int = 500  # flows labeled per round
    rounds: int = 8  # labeling rounds after the seed
    max_pool: int = 20000  # cap the unlabeled pool so the study stays fast
    strategies: list[str] = Field(default_factory=lambda: ["uncertainty", "random"])


class LeakageConfig(BaseModel):
    """Leakage-attribution ladder: reproduce the field's ~99% and price each source.

    The executable form of the project's thesis. Starting from the honest temporal
    model, three leakage sources are added back one at a time — a shuffled split, the
    memorisable ``Destination Port``, and a synthetic per-campaign session identifier
    standing in for Flow ID / Source IP — and each rung's PR-AUC gain is that source's
    contribution to the inflation. ``max_rows`` caps each split so the four refits stay
    fast; the identifier injection is a controlled demonstration of the anti-pattern the
    ``remainder="drop"`` firewall exists to stop, never something the pipeline adopts."""

    max_rows: int = 30000  # per-split row cap for the ladder refits (keeps it fast)


class DataValueConfig(BaseModel):
    """Training-data valuation via exact KNN-Shapley (Jia et al., VLDB 2019).

    Values each training flow by its game-theoretic contribution to a K-nearest-
    neighbour classifier's accuracy on held-out traffic, in the fitted pipeline's
    standardised space — signed, so a negative value flags a flow that pulls the
    classifier the wrong way (a mislabel signature). ``k`` is the neighbour count of
    the valuation utility; ``reference_rows`` training flows are valued against
    ``query_rows`` held-out flows (the closed form is O(N log N) per query, so both
    can be sizeable). ``planted_flip_rate`` seeds the self-validating mislabel-recovery
    check; ``prune_fractions`` drive the value-guided pruning experiment (each fraction
    costs three deployed-model refits, so keep the list short); ``report_classes`` caps
    the per-class value table. Runs on the exchangeable stratified/binary split."""

    k: int = 10  # neighbours in the KNN utility the Shapley value is defined against
    reference_rows: int = 5000  # training flows valued
    query_rows: int = 2000  # held-out flows the value is measured against
    planted_flip_rate: float = 0.05  # label flips planted for the mislabel-recovery check
    prune_fractions: list[float] = Field(default_factory=lambda: [0.05, 0.1])
    report_classes: int = 10  # classes shown in the per-class mean-value table


class PPIConfig(BaseModel):
    """Prediction-powered inference: estimate attack prevalence from few labels + the model.

    A SOC never labels a full day of traffic; it labels a small audit sample and lets
    the model score the rest. The classical estimate (label the sample, ignore the
    model) is valid but wide; imputing every flow with the model is tight but biased
    by the model's own error, so its interval does not cover the truth. Prediction-
    powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Science* 2023)
    keeps the model's tightness *and* classical validity by correcting the imputed
    estimate with the model's measured bias on the labelled sample (the "rectifier").

    ``label_budgets`` are the audit-sample sizes swept; at each, the three intervals'
    half-widths and their empirical coverage of the true test prevalence are measured
    over ``n_trials`` random label draws; ``alpha`` sets the confidence level
    (1 - ``alpha``). Runs on the exchangeable stratified/binary split, because PPI's
    validity assumes the labelled audit is a random sample of the scored population —
    exactly what the temporal split deliberately violates."""

    label_budgets: list[int] = Field(default_factory=lambda: [100, 250, 500, 1000])
    n_trials: int = 300  # random label-draw trials per budget for coverage/width
    alpha: float = 0.1  # 1 - alpha confidence level for every interval


class InfluenceConfig(BaseModel):
    """Influence functions: which training flows are responsible for a verdict (Koh & Liang 2017).

    Data valuation (KNN-Shapley) scores a flow's *global* contribution; this answers the
    per-prediction question — for *this* verdict, which training flows pushed it, and would
    removing them flip it? Influence functions estimate the effect of up-weighting a training
    point on a test loss via the inverse-Hessian-vector product, exactly and in closed form
    for the convex logistic model (the deployed gradient-boosted model is not twice-
    differentiable, so this runs on the logistic baseline — the same surrogate-scope honesty
    as the distillation study). ``l2`` is the logistic regularisation (its inverse is the
    ``C`` passed to the fit and sets the Hessian damping); ``n_explained`` test flows get a
    most-influential-training-flow table; ``top_k`` training flows are listed each way;
    ``loo_sample`` training points are actually retrained-without to validate the influence
    estimate against ground-truth leave-one-out; ``mislabel_flip_rate`` plants label flips to
    check that self-influence surfaces them (a second, independent mislabel detector next to
    the confident-learning audit and KNN-Shapley)."""

    l2: float = 1.0  # logistic L2 strength; Hessian damping = l2, fit C = 1 / l2
    n_explained: int = 4  # test flows to explain with their most-influential training flows
    top_k: int = 6  # training flows listed per direction (helpful / harmful)
    loo_sample: int = 60  # training points actually retrained-without for the LOO validation
    mislabel_flip_rate: float = 0.05  # planted flips for the self-influence mislabel check
    max_train: int = 6000  # cap training rows (keeps the Hessian solve + LOO retrains fast)


class LabelShiftConfig(BaseModel):
    """Label-shift estimation and correction from unlabelled deployment traffic.

    Base-rate stress and PPI both turn on the deployment attack prevalence; PPI estimates it
    from a handful of labels. Label shift asks the harder question — recover the shifted
    prior with **zero** deployment labels — and then *correct* the classifier for it.
    Under the label-shift assumption (the class-conditional feature law p(x|y) is fixed;
    only the prior p(y) moves, exactly what resampling to a target prevalence produces),
    two cited estimators apply. **BBSE** (Lipton, Wang & Smola, ICML 2018) solves the linear
    system ``C w = mu`` where ``C`` is the source confusion matrix and ``mu`` the target's
    predicted-label distribution, giving the importance weights ``w = q(y)/p(y)`` from the
    black-box predictor's *hard* labels (robust to miscalibration). **MLLS/EM** (Saerens,
    Latinne & Decaestecker, 2002) maximises the target likelihood over the prior by EM on
    the *soft* posteriors (efficient when calibrated). Corrected posteriors reweight each
    class by ``w`` and renormalise. ``target_priors`` are the true deployment prevalences
    swept; at each, estimation error and post-correction calibration are measured over
    ``n_trials`` resamples of the exchangeable stratified/binary test set to that prior."""

    target_priors: list[float] = Field(default_factory=lambda: [0.02, 0.05, 0.15, 0.35, 0.6])
    n_trials: int = 40  # resamples of the test set to each target prior
    target_size: int = 4000  # rows per simulated deployment sample
    em_max_iter: int = 200  # MLLS/EM iteration cap
    em_tol: float = 1e-7  # MLLS/EM convergence tolerance on the prior change


class HMeasureConfig(BaseModel):
    """The H-measure: a coherent alternative to ROC-AUC (Hand 2009).

    Averaging over thresholds, ROC-AUC implicitly weights false-positive against
    false-negative cost by a distribution that depends on each classifier's own score
    distribution, so cross-model comparisons are made under different, incomparable cost
    assumptions. The H-measure fixes an **explicit, shared** Beta prior on the cost
    parameter for every classifier and reports the normalised expected minimum loss.
    ``prior_alpha``/``prior_beta`` set the default symmetric severity prior (Hand's
    Beta(2, 2)); ``cost_skew_alpha``/``cost_skew_beta`` set a second, SOC-flavoured prior
    that puts mass where a missed attack costs more than a false alarm — a cost stance
    ROC-AUC structurally cannot express. ``grid_points`` is the cost-grid resolution for
    the loss integral. Runs on the honest temporal/binary split across the deployed model
    and two references."""

    prior_alpha: float = 2.0  # default symmetric severity prior Beta(a, b) (Hand 2009)
    prior_beta: float = 2.0
    cost_skew_alpha: float = 2.0  # cost-skewed prior: mass toward cheap false positives...
    cost_skew_beta: float = 4.0  # ...i.e. expensive missed attacks (the SOC's real stance)
    grid_points: int = 2000  # cost-grid resolution for the loss-curve quadrature


class LeaderboardConfig(BaseModel):
    """Model-family leaderboard: every family through one shared honest protocol.

    The claim under test is not "which model wins" but whether the
    stratified-minus-temporal gap replicates across families — if it does, the
    gap is a property of the evaluation, not of any single model. Baselines run
    at sensible defaults on purpose (only the deployed model is tuned), and the
    report says so."""

    families: list[str] = Field(
        default_factory=lambda: ["majority", "naive_bayes", "logistic", "random_forest", "gbdt"]
    )
    rf_n_estimators: int = 200


class SelfTrainConfig(BaseModel):
    """Self-training (pseudo-labeling) study on the unlabeled later-day stream.

    The streaming study prices labeled retraining; this prices the label-free
    shortcut — retrain on the model's own confident scores over the unlabeled
    adaptation window. Taus are on the raw score scale; flows between them are
    abstentions. The known risk under drift, which the report audits directly, is
    novel attacks scoring confidently benign and being learned as benign."""

    adaptation_fraction: float = 0.5  # leading share of the test stream seen unlabeled
    tau_attack: float = 0.98  # raw score at/above which a flow is pseudo-labeled attack
    tau_benign: float = 0.02  # raw score at/below which a flow is pseudo-labeled benign
    max_pseudo_per_class: int = 20000  # cap per side, most confident first


class ExpertsConfig(BaseModel):
    """Online prediction with expert advice: track the best model as drift shifts it.

    The leaderboard finds that different model families win on different splits, and the
    streaming/retrain studies show *which* model is best drifts over the week. Rather than
    pick one in advance, combine them online: each model is an "expert", and a
    prediction-with-expert-advice algorithm (Cesa-Bianchi & Lugosi 2006) weights them by
    their running loss with a **provable regret bound** — no distributional assumptions,
    no retraining, labels revealed prequentially. **Hedge** (exponential weights) competes
    with the best *fixed* expert in hindsight; **fixed-share** (Herbster & Warmuth 1998)
    mixes a little mass back to every expert each step so it can *track* a best expert that
    changes across the stream, competing with the best *sequence* of experts. ``experts``
    are the pooled families (from the leaderboard builder); ``fixed_share_alpha`` is the
    per-step switching mass; ``eta`` is the learning rate (``auto`` uses the optimal
    ``sqrt(8 ln N / T)``); ``loss_clip`` bounds the per-step log-loss so the regret bound's
    range assumption holds. Runs on the honest temporal/binary stream — the drift the
    tracking guarantee is for."""

    experts: list[str] = Field(default_factory=lambda: ["logistic", "random_forest", "gbdt"])
    fixed_share_alpha: float = 0.02  # per-step mass shared to all experts (enables tracking)
    eta: float | str = "auto"  # Hedge learning rate; "auto" = sqrt(8 ln N / T)
    loss_clip: float = 5.0  # cap per-step log-loss (prob clipped to keep it bounded)


class WeakSupervisionConfig(BaseModel):
    """Weak supervision: train the detector from the signature rules alone, zero labels.

    Data programming (Ratner et al., NeurIPS 2016) reads each hand-written signature as
    a **labeling function** — it votes attack when it fires and abstains otherwise — and
    fits a Dawid-Skene-style generative label model by EM to estimate every signature's
    accuracy *without any ground truth*, from the votes' agreement structure only.
    The label model's posteriors become probabilistic training labels for the ordinary
    downstream model, which sees the full feature space its teachers never used and so
    can generalise past them. Two quantities are not identifiable from attack-or-abstain
    votes and are therefore **stated, not fitted**: ``class_prior`` (silence could mean
    benign or a missed rare attack — the same reason Snorkel takes ``class_balance`` as an
    input; ``prior_sensitivity`` sweeps it) and, when the signatures never co-fire,
    the per-signature accuracies themselves — agreement is the only label-free evidence,
    so the label model is **agreement-gated**: with at least ``min_cofire_rows`` rows
    carrying two or more votes it fits per-LF accuracies by EM, otherwise it combines
    votes as a Bayesian believer at ``signature_trust`` (the operator's "a deployed
    signature is usually right"), and the report audits that belief against ground truth.
    ``em_max_iter``/``em_tol`` bound the EM fit; ``smoothing`` is the Laplace pseudo-count
    keeping vote tables off 0/1; ``min_weight`` drops training rows whose posterior is too
    ambiguous to teach with (noise-aware confidence weighting)."""

    class_prior: float = 0.15  # assumed P(attack): a coarse operator belief, never a label
    prior_sensitivity: list[float] = Field(default_factory=lambda: [0.05, 0.15, 0.30])
    # Assumed precision of a fired signature. Doubles as the EM's polarity anchor (a fired
    # rule initially reads attack-leaning) and as the fixed trust of the prior-belief
    # combiner when agreement is too thin to estimate accuracies from.
    signature_trust: float = 0.8
    min_cofire_rows: int = 50  # rows with >= 2 votes needed before EM may fit accuracies
    em_max_iter: int = 200  # EM iteration cap for the generative label model
    em_tol: float = 1e-6  # EM convergence tolerance on the mean posterior change
    # Laplace pseudo-count for the per-LF vote tables. Deliberately strong: with weak
    # smoothing EM self-confirms a fired-alone signature to precision exactly 1.0 (the
    # naive-Bayes saturation); ~5 pseudo-counts damp that while real agreement still moves
    # the tables.
    smoothing: float = 5.0
    min_weight: float = 0.05  # drop rows whose |2 * posterior - 1| confidence is below this


class PULearnConfig(BaseModel):
    """Positive-unlabeled learning: train from confirmed attacks + unlabeled traffic.

    A real SOC labels only the attacks incident response confirms; everything else is
    unlabeled, not verified benign, and contains the attacks nobody caught. Under SCAR
    (labels Selected Completely At Random from the positives), Elkan & Noto (KDD 2008)
    relate the labeled-vs-unlabeled classifier ``g`` to the true posterior through one
    estimable constant ``c = p(labeled | attack)``, which buys corrected scores, a
    hidden-attack prevalence estimate, a weighted retrain (each unlabeled flow enters as
    part-positive, part-negative), and a de-contaminated FPR denominator for threshold
    selection. ``label_fracs`` sweeps the confirmed fraction; ``headline_frac`` picks the
    setting the operating-point analysis runs at (must be in the sweep); ``budget_fpr``
    is that analysis's false-positive budget; ``score_clip`` keeps ``g`` off 0/1 before
    the posterior odds ratio; ``max_weighted_rows`` caps the duplicated Elkan-Noto
    design matrix (seeded subsample beyond it)."""

    label_fracs: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75])
    headline_frac: float = 0.25  # the sweep point the budget analysis reads at
    budget_fpr: float = 0.01  # FPR budget for the three-cuts comparison
    score_clip: float = 1e-3  # clip g away from 0/1 before w = ((1-c)/c) g/(1-g)
    max_weighted_rows: int = 120_000  # cap on the duplicated weighted design matrix


class AlertFDRConfig(BaseModel):
    """Conformal alert selection with a false-discovery-rate guarantee on the batch.

    The base-rate study shows a fixed FPR does not control the precision of the alert queue.
    This does, with a guarantee: calibrate on held-out benign flows, form each test flow's
    conformal p-value (the smoothed rank of its attack score among the benign nulls), and
    select alerts by Benjamini-Hochberg at a target FDR ``q``. Bates et al. (Annals of
    Statistics 2023) prove the conformal p-values are PRDS, so BH controls FDR on them — the
    benign share of the alerts is at most ``q``, distribution-free, at any prevalence.
    ``q_levels`` are validated (realized FDP <= q averaged over draws); ``q_headline`` is the
    level the prevalence sweep runs at; ``prevalences`` are the production priors the batch is
    resampled to (the base-rate axis); ``fixed_fpr`` is the baseline cut chosen on benign
    calibration scores; ``batch_size`` is the alert batch judged per trial; ``n_trials``
    averages over calibration/test resamples so the marginal guarantee is what is measured;
    ``tolerance`` is the finite-sample slack allowed before a level is flagged uncontrolled.
    Runs on the exchangeable stratified/binary split (conformal validity needs it)."""

    q_levels: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30])
    q_headline: float = 0.10  # target FDR the prevalence sweep holds
    prevalences: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.05, 0.2])
    fixed_fpr: float = 0.01  # the uncontrolled baseline threshold's benign budget
    batch_size: int = 5000  # alerts judged per trial
    n_trials: int = 200  # calibration/test resamples the rates average over
    tolerance: float = 0.02  # finite-sample slack on the FDR bound before flagging


class PoisoningConfig(BaseModel):
    """Training-set poisoning study: how detection degrades as labels are corrupted.

    The evasion study covers the inference-time adversary; this covers the
    training-time one. Label flips model an attacker who corrupts the labeling
    source so their attack flows are recorded as benign; benign-pool contamination
    models attack traffic present during the 'clean' capture the anomaly detector
    normalises on. Rates are fractions (of attack training rows, and of the benign
    training pool, respectively)."""

    label_flip_rates: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.25, 0.5])
    contamination_rates: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.05, 0.1, 0.2])


class BackdoorConfig(BaseModel):
    """Targeted backdoor (trojan) poisoning + the spectral-signatures defense.

    The poisoning study covers the *availability* attack (random flips degrade everything);
    this is the *integrity* one (Gu et al. 2017, BadNets): the attacker plants attack flows
    wearing a rare **trigger** — exact values in attacker-controllable fields — labeled
    BENIGN, so the model learns "trigger means benign" while clean metrics barely move,
    then wears the trigger at attack time. ``trigger`` maps raw feature names to the
    planted values (defaults are fields an attacker sets directly: the TCP window via
    socket options, packet pacing via delays). ``poison_rates`` are injected fractions of
    the labeled pool; the defense (Tran et al., NeurIPS 2018) runs at ``defense_rate``:
    score every benign-labeled row by its squared projection on the top singular direction
    of the centered class representation, drop the top ``removal_multiplier`` x injected
    count (the paper's over-removal), refit, re-measure."""

    trigger: dict[str, float] = Field(
        default_factory=lambda: {"Init_Win_bytes_forward": 4242.0, "Fwd IAT Min": 4242.0}
    )
    poison_rates: list[float] = Field(default_factory=lambda: [0.002, 0.005, 0.01, 0.02])
    defense_rate: float = 0.01  # the budget the defense arc (audit -> remove -> refit) runs at
    removal_multiplier: float = 1.5  # remove this many times the injected count, by score


class SanitizeConfig(BaseModel):
    """Audit-and-drop defense against poisoned training labels, re-measured.

    The poisoning study prices the training-time attack; this prices the cheapest
    defense an operator can actually run: the confident-learning audit
    (``label_audit.folds`` out-of-fold models) over *all* labeled data — train and
    validation together, because threshold selection is poisoned too — dropping
    every flagged row in both directions, then refitting. ``flip_rates`` should
    share its range with ``poisoning.label_flip_rates`` so the two curves read
    against each other; ``max_rows`` caps the combined labeled pool because every
    rate costs ``folds + 2`` full model fits."""

    flip_rates: list[float] = Field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5])
    max_rows: int = 30000  # combined train+val cap (each rate is folds+2 fits)


class LabelAuditConfig(BaseModel):
    """Confident-learning-style label-noise audit of the training split.

    CIC-IDS2017 has community-documented label errors (the Engelen et al. WTMC 2021
    corrections exist for a reason); this audit *finds* candidate errors rather than
    assuming them away. Out-of-fold predictions flag rows whose model score is as
    extreme as the typical score of the *opposite* class (class-conditional mean
    thresholds). The audit validates itself by planting a known fraction of label
    flips and measuring how many it recovers, and at what precision."""

    folds: int = 3  # out-of-fold prediction folds (train split only; test untouched)
    planted_flip_rate: float = 0.05  # attack rows flipped benign for the recovery check
    max_rows: int = 30000  # subsample cap so the k-fold study stays fast


class TransferConfig(BaseModel):
    """Threshold transfer onto a foreign dataset: what re-buys the FPR budget.

    The cross-dataset study's verdict is that the ranking transfers but the
    operating point does not ("re-choose thresholds on labeled local traffic");
    this prices that advice. Four policies are compared at the primary FPR
    budget on the foreign set: the transplanted source threshold, an
    unsupervised quantile matched on the *unlabeled* target scores (valid only
    while the stream is benign-dominated — the report measures the violation at
    the test mix and at a production-like mix), a threshold chosen on ``k``
    labeled target flows for each ``label_budgets`` entry (redrawn
    ``n_resamples`` times so small-sample noise is reported, not hidden), and
    the all-label oracle."""

    label_budgets: list[int] = Field(default_factory=lambda: [50, 100, 250, 500, 1000, 2500])
    n_resamples: int = 30  # seeded redraws per label budget
    compliance_factor: float = 2.0  # realized FPR within this factor of budget counts as held


class CrossDatasetConfig(BaseModel):
    """Synthetic 'foreign' (NetFlow-schema) dataset for cross-dataset generalization."""

    rows: int = 20000
    attack_fraction: float = 0.30
    name: str = "synthetic-netflow"


class IncidentConfig(BaseModel):
    """Incident-report generation from scored flows (`netsentry incident`).

    Consecutive same-class alerts are one incident; up to ``gap_tolerance``
    non-alert rows in between are bridged, because real attack traffic
    interleaves with background. A contiguity heuristic, stated as such in the
    report — it re-reads per-flow verdicts, it does not create detection."""

    gap_tolerance: int = 3  # non-alert rows an incident may bridge before closing
    top_talkers: int = 5  # sources/targets/services listed per incident


class BeaconConfig(BaseModel):
    """Beaconing / C2 periodicity detection over connection timelines.

    The per-flow classifier drops every identifier and scores flows in isolation,
    so it cannot see a host calling home on a fixed cadence (ATT&CK Command and
    Control). This unsupervised, identity-aware analytic groups connections by
    talker pair and scores the regularity of their inter-arrival times. A pair needs
    ``min_events`` connections before periodicity is judgeable; ``score_threshold``
    is the regularity flag line (1.0 = perfectly periodic, 0.0 = bursty). Reads the
    timestamp/identity columns as metadata only — the fields the model never sees."""

    timestamp_column: str = "Timestamp"
    min_events: int = 8  # connections a pair needs before its regularity is scored
    score_threshold: float = 0.85  # regularity at/above which a pair is flagged
    by_port: bool = True  # group by (src, dst, dst_port) rather than (src, dst)
    top_n: int = 20  # ranked candidates rendered in the report


class GraphConfig(BaseModel):
    """Host-communication-graph analytics: scan fan-out + lateral-movement chains.

    The per-flow classifier drops every identifier and scores flows in isolation, so
    it is structurally blind to attacks whose signal lives in the *topology* — a
    source fanning out across the network (scanning) or a reached host pivoting deeper
    (lateral movement). This identity-aware analytic reconstructs the graph from the
    ``Src IP`` / ``Dst IP`` / ``Dst Port`` columns (metadata the model never sees). A
    source needs ``min_fanout`` distinct destinations *or* ports to count as a scan;
    a movement chain needs ``min_chain_hosts`` hosts. Runtime on large graphs is
    bounded by ``max_depth`` (chain search depth) and ``max_starts`` (entry nodes
    the depth-first search launches from)."""

    min_fanout: int = 20  # distinct hosts/ports a source must reach to be a scan candidate
    by_port: bool = True  # also score vertical (per-port) fan-out, not just horizontal
    min_chain_hosts: int = 3  # hosts in a movement chain (a->b->c) before it is reported
    max_depth: int = 8  # cap on chain search depth (bounds the DFS on large graphs)
    max_starts: int = 500  # entry nodes the chain search launches from (runtime bound)
    top_n: int = 20  # ranked candidates rendered per table


class StixConfig(BaseModel):
    """STIX 2.1 threat-intel bundle export from scored detections.

    Emits a standards-conformant bundle (identity, attack-pattern, indicator,
    observed-data + SCOs, sighting, relationship) a TAXII server or intel platform
    (MISP, OpenCTI) can ingest directly. ``tlp`` selects the Traffic Light Protocol
    marking-definition applied to every object; the default AMBER matches the
    limited-distribution posture of an internal detection feed."""

    identity_name: str = "NetSentry ML-NIDS"
    tlp: Literal["white", "green", "amber", "red"] = "amber"


class TriageConfig(BaseModel):
    """Weights for fusing CVE severity with NetSentry's live-traffic risk signals."""

    severity_weight: float = 0.5
    model_weight: float = 0.35
    anomaly_weight: float = 0.15


class MLflowConfig(BaseModel):
    """Experiment tracking. Falls back to local file logging if MLflow is absent."""

    enabled: bool = True
    experiment_name: str = "netsentry"
    tracking_uri: str | None = None  # defaults to paths.mlruns_dir when unset


class ServingConfig(BaseModel):
    """FastAPI inference service."""

    model_config = ConfigDict(protected_namespaces=())

    host: str = "0.0.0.0"
    port: int = 8000
    artifact_path: Path | None = None  # defaults to the latest bundle in models_dir
    # Optional shadow challenger: a second bundle scored silently on every request.
    # It never affects responses; it emits disagreement metrics (score delta +
    # decision disagreement) to Prometheus — live paired evidence for `netsentry
    # promote`, gathered on production traffic instead of the frozen test split.
    shadow_artifact_path: Path | None = None
    default_threshold_profile: str = "fpr_0.1pct"
    max_batch_size: int = 1000
    top_k_features: int = 5
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_payloads: bool = False
    # Optional API-key auth on the prediction endpoints (via the X-API-Key header).
    # Unset -> open (dev default); set via NETSENTRY_SERVING__API_KEY in production.
    api_key: str | None = None
    rate_limit_per_minute: int = 0  # 0 disables the per-client fixed-window rate limit
    # Behavioral canaries: validation flows embedded in the bundle with their
    # build-time scores, replayed at load (and via `netsentry canary`) to prove this
    # runtime reproduces the model that was validated. `verify` checks the bytes;
    # the canary checks the behavior — env skew moves scores without moving a byte.
    canary_rows: int = 8  # validation flows embedded at bundle build (class-mixed)
    canary_tolerance: float = 1e-6  # max |score now - score at build| before failing
    canary_strict: bool = False  # refuse to start serving on canary failure (prod: true)
    # Canary-gated hot reload: POST /admin/reload swaps the live bundle in place, but
    # only after the candidate reproduces its own embedded canaries in this runtime
    # (a mismatch is rejected 409 and the current model keeps serving). Off by
    # default — an operational surface is opt-in — and guarded by the same API key
    # as the prediction endpoints. Candidates must live under models_dir.
    reload_enabled: bool = False


class Settings(BaseSettings):
    """Root configuration object, assembled from YAML + environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="NETSENTRY_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    project_name: str = "netsentry"
    seed: int = 42

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    supervised: SupervisedConfig = Field(default_factory=SupervisedConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    alert_queue: AlertQueueConfig = Field(default_factory=AlertQueueConfig)
    base_rate: BaseRateConfig = Field(default_factory=BaseRateConfig)
    socsim: SocSimConfig = Field(default_factory=SocSimConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    seed_variance: SeedVarianceConfig = Field(default_factory=SeedVarianceConfig)
    subgroups: SubgroupsConfig = Field(default_factory=SubgroupsConfig)
    campaigns: CampaignsConfig = Field(default_factory=CampaignsConfig)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    adaptive_conformal: AdaptiveConformalConfig = Field(default_factory=AdaptiveConformalConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    drift_detectors: DriftDetectorConfig = Field(default_factory=DriftDetectorConfig)
    exchangeability: ExchangeabilityConfig = Field(default_factory=ExchangeabilityConfig)
    importance_stability: ImportanceStabilityConfig = Field(
        default_factory=ImportanceStabilityConfig
    )
    exemplars: ExemplarConfig = Field(default_factory=ExemplarConfig)
    anomaly_explain: AnomalyExplainConfig = Field(default_factory=AnomalyExplainConfig)
    anchors: AnchorsConfig = Field(default_factory=AnchorsConfig)
    partial_dependence: PartialDependenceConfig = Field(default_factory=PartialDependenceConfig)
    interactions: InteractionsConfig = Field(default_factory=InteractionsConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    membership: MembershipConfig = Field(default_factory=MembershipConfig)
    dp: DPConfig = Field(default_factory=DPConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    certify: CertifyConfig = Field(default_factory=CertifyConfig)
    hardening: HardeningConfig = Field(default_factory=HardeningConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    refresh: RefreshConfig = Field(default_factory=RefreshConfig)
    retrain_policy: RetrainPolicyConfig = Field(default_factory=RetrainPolicyConfig)
    leaderboard: LeaderboardConfig = Field(default_factory=LeaderboardConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    data_value: DataValueConfig = Field(default_factory=DataValueConfig)
    ppi: PPIConfig = Field(default_factory=PPIConfig)
    influence: InfluenceConfig = Field(default_factory=InfluenceConfig)
    label_shift: LabelShiftConfig = Field(default_factory=LabelShiftConfig)
    hmeasure: HMeasureConfig = Field(default_factory=HMeasureConfig)
    selftrain: SelfTrainConfig = Field(default_factory=SelfTrainConfig)
    weak_supervision: WeakSupervisionConfig = Field(default_factory=WeakSupervisionConfig)
    experts: ExpertsConfig = Field(default_factory=ExpertsConfig)
    pu_learning: PULearnConfig = Field(default_factory=PULearnConfig)
    alert_fdr: AlertFDRConfig = Field(default_factory=AlertFDRConfig)
    poisoning: PoisoningConfig = Field(default_factory=PoisoningConfig)
    backdoor: BackdoorConfig = Field(default_factory=BackdoorConfig)
    sanitize: SanitizeConfig = Field(default_factory=SanitizeConfig)
    label_audit: LabelAuditConfig = Field(default_factory=LabelAuditConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    crossdata: CrossDatasetConfig = Field(default_factory=CrossDatasetConfig)
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    incident: IncidentConfig = Field(default_factory=IncidentConfig)
    beacon: BeaconConfig = Field(default_factory=BeaconConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    stix: StixConfig = Field(default_factory=StixConfig)
    triage: TriageConfig = Field(default_factory=TriageConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Precedence (high to low): init kwargs > env > .env > YAML > defaults."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    def mlflow_tracking_uri(self) -> str:
        """Resolve the MLflow tracking URI, defaulting to a local file store."""
        if self.mlflow.tracking_uri:
            return self.mlflow.tracking_uri
        return self.paths.mlruns_dir.resolve().as_uri()
