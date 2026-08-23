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


class RareRatesConfig(BaseModel):
    """Hierarchical (partial-pooling) estimation of per-class detection rates.

    ``split`` picks which protocol supplies the counts: the stratified split is the default here
    because it is the one where every attack class appears at test, including the genuinely tiny
    ones the report is about, and the question being asked is about *estimation uncertainty*
    rather than about the detection level (which the temporal split still owns). ``level`` is the
    interval level, ``grid_points`` the resolution of the empirical-Bayes hyperparameter search,
    ``coverage_replicates`` the number of simulated redraws behind the coverage check, and
    ``target_half_width`` the precision the sample-size table solves for."""

    split: Literal["stratified", "temporal"] = "stratified"
    level: float = 0.95
    grid_points: int = 60  # per axis of the (mean, concentration) marginal-likelihood grid
    coverage_replicates: int = 400
    target_half_width: float = 0.05  # the +/- precision the sample-size table asks for


class OpenSetConfig(BaseModel):
    """Open-set recognition: rank the novelty rules on classes the model was never taught.

    The temporal split contains no attack class the training days showed, so the deployment is
    an open-set problem and the deployed rule (``attack_prob`` = ``1 - P(BENIGN)``) is only one
    candidate novelty score. ``rules`` names the field, all computed from artefacts the
    deployment already has; ``fusion_members`` are the rules the rank-average ``fused`` rule
    combines. ``budgets`` are the false-alarm budgets the unknown-detection rate is read at
    (the first is used for the per-class breakdown). ``holdout_counts`` drive the openness
    sweep on the stratified split — how many attack classes (rarest first) to withhold from
    training. ``mahalanobis_shrinkage`` shrinks the pooled covariance toward a scaled identity
    so the precision matrix survives a class with a handful of rows, and ``max_rows`` caps each
    split because the feature-space scorers are dense."""

    rules: list[str] = Field(
        default_factory=lambda: [
            "attack_prob",
            "msp",
            "entropy",
            "margin",
            "mahalanobis",
            "iforest",
            "fused",
        ]
    )
    fusion_members: list[str] = Field(default_factory=lambda: ["attack_prob", "mahalanobis"])
    budgets: list[float] = Field(default_factory=lambda: [0.01, 0.001])
    primary_budget: float = 0.01  # budget the per-unknown-class breakdown is read at
    holdout_counts: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    mahalanobis_shrinkage: float = 0.1
    max_rows: int = 30000  # per-split cap (the dense scorers are the bottleneck)


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


class ControlConfig(BaseModel):
    """Closed-loop threshold control: hold alert volume at the analyst budget under drift.

    ``target_alert_rate`` is the fraction of flows the analyst team can actually review, so the
    setpoint is that rate times ``batch_rows``. The actuator is ``log10`` of the alert-rate
    parameter -- the parameterisation in which the plant is nearly unit gain, so ``kp`` near 1 is
    roughly deadbeat and a gain means the same thing across the operating range. ``max_step``
    rate-limits it in decades per batch. ``gain_sweep`` locates the stability boundary
    empirically and ``delay_sweep`` prices late feedback. The attack block defines the
    control-loop attack: ``decoys_per_batch`` loud flows (above ``decoy_quantile`` of the score
    distribution) for ``attack_batches`` batches, whose only purpose is to make the controller
    raise its own threshold; ``freeze_above`` and ``guarded_max_step`` are the mitigation."""

    batch_rows: int = 500
    target_alert_rate: float = 0.02
    kp: float = 0.6
    ki: float = 0.1
    max_step: float = 0.2  # decades of alert rate per batch
    tracker_step: float = 0.05
    gain_sweep: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 1.5, 2.5])
    delay_sweep: list[int] = Field(default_factory=lambda: [0, 1, 2, 5])
    # The settling band has to clear the setpoint's own counting noise: at 10 alerts a
    # batch, Poisson variation alone is about 30%, so a tighter band would measure the
    # arrival process rather than the controller.
    settling_tolerance: float = 0.5
    attack_start_batch: int = 20
    attack_batches: int = 10
    decoys_per_batch: int = 250
    decoy_quantile: float = 0.98
    freeze_above: float = 0.5  # decades of error beyond which the integrator stops learning
    guarded_max_step: float = 0.05
    recovery_tolerance: float = 0.1  # relative, on the realised alert rate


class OperatingPointConfig(BaseModel):
    """Training *for* the false-positive budget instead of for the loss.

    ``budgets`` are the operating points every arm is scored at; ``train_budgets`` are the ones a
    partial-AUC network is trained for (one model each), which is what turns the result into a
    matrix rather than a number. ``batch_rows`` is part of the objective's specification rather
    than a performance knob: the surrogate ranks positives against the top
    ``ceil(alpha * n_negatives)`` negatives *in the batch*, so at a 0.1% budget a batch of 1,000
    supplies exactly one, and the estimate is only as good as the batch is large."""

    budgets: list[float] = Field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05])
    train_budgets: list[float] = Field(default_factory=lambda: [0.001, 0.01])
    batch_rows: int = 4096
    epochs: int = 30
    max_train_rows: int = 12000


class DeepTabularConfig(BaseModel):
    """Deep tabular models against the boosted incumbent, under one shared protocol.

    ``hidden_sizes``/``dropout`` size the MLP; ``token_dim``, ``n_heads`` and ``n_blocks`` size
    the FT-Transformer's per-feature tokens and attention stack. Early stopping watches
    validation **PR-AUC** with ``patience`` epochs of grace, because under 20% prevalence the
    loss and the deployment metric do not agree and the metric is what ships. ``data_fractions``
    is the sample-efficiency sweep -- the "neural models need more data" claim, tested rather
    than repeated."""

    hidden_sizes: list[int] = Field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    token_dim: int = 32
    n_heads: int = 4
    n_blocks: int = 2
    batch_size: int = 1024
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 4
    # Every arm sees the same capped training set. The cap exists because attention over 76
    # feature tokens is the expensive thing here, and a study nobody can afford to re-run is a
    # study nobody will check -- so the transformer's cost sets the budget and the *whole*
    # comparison is held to it, rather than quietly giving the trees more data.
    pauc_temperature: float = 0.5  # score margin at which a pair stops contributing gradient
    max_train_rows: int = 12000
    data_fractions: list[float] = Field(default_factory=lambda: [0.15, 0.5, 1.0])


class OnlineConfig(BaseModel):
    """Prequential streaming: a one-pass learner against the batch pipeline that ships.

    ``batch_rows`` is the prequential unit (score the batch, then learn from it), ``warmup_rows``
    the training days every arm starts from, and ``max_stream_rows`` the later-day stream they
    are all judged on. The tree parameters are the VFDT's: ``grace_period`` amortises the split
    test, ``split_delta`` is the Hoeffding bound's confidence and ``tie_threshold`` the escape
    hatch for genuinely equal candidates. ``label_delays`` is the honest part -- online learning
    assumes the label arrives with the flow, and a SOC's arrives hours later, so the assumption
    is swept rather than stated."""

    batch_rows: int = 1000
    warmup_rows: int = 20000
    max_stream_rows: int = 25000
    retrain_every: int = 8000  # flows between full refits for the periodic-retrain arm
    grace_period: int = 200
    split_delta: float = 1e-6
    tie_threshold: float = 0.05
    n_thresholds: int = 10
    max_depth: int = 12
    min_leaf_samples: float = 20.0
    adwin_delta: float = 0.002
    label_delays: list[int] = Field(default_factory=lambda: [0, 1, 5, 20])


class ContinualConfig(BaseModel):
    """Class-incremental updates: what folding in a new attack family costs the old ones.

    Each capture day is one task. ``train_fraction`` splits a day *by position* rather than at
    random -- an attack burst is a run of near-duplicate flows, so a shuffled within-day split
    would score memory as retention. ``buffer_rows`` is the replay reservoir the headline policy
    carries, and ``buffer_sweep`` traces the stability-plasticity frontier between its two
    degenerate ends: an empty buffer *is* naive fine-tuning, and a buffer larger than the history
    is a warm-started full retrain. ``max_rows_per_task`` bounds the per-day work."""

    train_fraction: float = 0.6
    max_rows_per_task: int = 20000
    buffer_rows: int = 4000
    buffer_sweep: list[int] = Field(default_factory=lambda: [0, 500, 2000, 8000, 32000])
    bench_rows: int = 5000  # flows scored to price each final model's inference cost


class MMDConfig(BaseModel):
    """Multivariate drift: the kernel two-sample test, and the marginal monitors it backstops.

    ``window_rows`` is the window each side of the test gets; ``permutations`` the exact-null
    budget for the headline tests and ``power_permutations`` the cheaper budget the repeated
    sweeps use (power is dominated by the window, not by the permutation count, so spending the
    budget on repeats buys more). ``repeats``/``power_repeats`` size the false-alarm and power
    estimates. ``shift_sigmas`` and ``n_faulted_features`` define the two controlled faults --
    a mean shift both monitor families can see, and the same features permuted across rows,
    which preserves every marginal *exactly* and is therefore invisible to all of them by
    construction. ``psi_threshold`` is the operator's PSI alarm level, and ``psi_bins`` mirrors
    the deployed monitor so the comparison is against what actually ships."""

    window_rows: int = 1000
    permutations: int = 200
    power_permutations: int = 100
    alpha: float = 0.05
    repeats: int = 30  # null draws behind the false-alarm rate
    power_repeats: int = 20  # draws per (fault, window) cell
    window_sweep: list[int] = Field(default_factory=lambda: [125, 250, 500, 1000])
    cost_sweep: list[int] = Field(default_factory=lambda: [250, 500, 1000, 2000])
    shift_sigmas: float = 0.25
    n_faulted_features: int = 6
    # The dependence sweep. The modelled features of the synthetic stand-in are very nearly
    # independent, under which a row-permutation fault is a no-op rather than an invisible
    # change -- so the joint test's reach is measured on controlled windows whose pairwise
    # dependence is a dial and whose marginals are identical at every setting.
    dependence_rhos: list[float] = Field(default_factory=lambda: [0.0, 0.15, 0.3, 0.6, 0.9])
    stream_features: int = 20
    psi_threshold: float = 0.2
    psi_bins: int = 10
    bandwidth_points: int = 500  # subsample behind the median heuristic
    attribution_points: int = 400  # subsample behind the per-feature (marginal) MMD


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


class WatermarkConfig(BaseModel):
    """Model watermarking: prove ownership by backdooring the detector on purpose.

    Watermarking (Adi et al., USENIX Security 2018) embeds a secret set of trigger flows with
    owner-chosen **random** labels during training so the model memorises them; ownership is
    later proven by querying a suspect on the keys. Because the labels are fair coins, an
    innocent model agrees with them only at chance, so the null is exactly `Binomial(K, 0.5)`
    and the ownership test returns an exact p-value (computed in log-space, no scipy). The
    study also measures the fidelity tax (watermarked vs clean detection) and survival under
    model extraction (reusing the extraction attack). ``n_keys`` is the watermark size (more
    keys = a smaller ownership p-value); ``trigger_scale`` places the keys in the standardised
    feature-space tails (off the data manifold, memorable, collision-free); ``extraction_queries``
    is the surrogate-stealing budget for the survival test; ``decision_threshold_log10p`` is the
    log10 p-value below which ownership is declared proven. Runs on the temporal/binary split."""

    n_keys: int = 256  # secret watermark keys embedded in training
    trigger_scale: float = 4.0  # std of the off-manifold trigger draws in standardised space
    extraction_queries: int = 4000  # surrogate-stealing budget for the survival test
    decision_threshold_log10p: float = -6.0  # ownership proven when log10 p <= this


class UnlearnConfig(BaseModel):
    """Machine unlearning via SISA: delete a flow without retraining from scratch.

    SISA (Bourtoule et al., IEEE S&P 2021) shards the training flows, trains one isolated
    submodel per shard, and averages their attack-probabilities. A deletion request retrains
    only the shard(s) that held the deleted flows, so unlearning is cheap and — because the
    other submodels are untouched and per-shard training is deterministic — provably identical
    to a fresh ensemble on the surviving data (exact unlearning, not an approximate scrub).
    ``shard_counts`` sweeps S for the sharding-tax curve; ``headline_shards`` is the S the
    deletion-cost and exactness demos run at; ``delete_counts`` are the batch sizes priced
    against a full retrain (with the coupon-collector expectation); ``cost_trials`` averages the
    random-deletion cost; ``verify_deletions`` is the batch size the exactness + membership
    payoff is demonstrated on. Runs on the temporal/binary split."""

    shard_counts: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    headline_shards: int = 8  # S for the deletion-cost + exactness demos (must be in shard_counts)
    delete_counts: list[int] = Field(default_factory=lambda: [1, 5, 25, 100])
    cost_trials: int = 200  # random-deletion trials the shard-touch cost averages over
    verify_deletions: int = 5  # batch size the exactness + forgetting demo deletes


class EarlinessConfig(BaseModel):
    """Decision latency: when the deployed verdict can first exist, and what earlier costs.

    Flow exporters emit one record per *finished* flow, so most CICFlowMeter statistics are
    undefined until the flow is over and the deployed detector is structurally a post-mortem
    one. The study prices that in two halves: the wait (computed per flow from whether the
    exporter saw a FIN/RST — a flow that merely stops is held until the idle timer
    ``capture.flow_timeout_us`` expires) and the detection lost by deciding earlier, by
    refitting on the nested feature tiers in ``features.feature_sets.availability_sets``.
    ``in_flight_horizon_us`` is how long an in-flight detector is allowed to accumulate
    packets before its intensive statistics mean anything (a flow shorter than the horizon
    ends first, so its verdict is bounded by its own duration). ``horizons_s`` are the
    x-positions of the detected-in-time frontier, log-spaced because the interesting range
    spans milliseconds to minutes; ``min_class_flows`` is the support a class needs before it
    gets its own row."""

    in_flight_horizon_us: int = 1_000_000  # packets accumulated before an in-flight verdict
    horizons_s: list[float] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 5.0, 15.0, 60.0, 120.0, 300.0]
    )
    min_class_flows: int = 50  # test flows a class needs before it gets its own row
    unclosed_shares: list[float] = Field(  # sensitivity sweep the stand-in cannot supply
        default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )


class HierarchyConfig(BaseModel):
    """Taxonomy-aware multiclass evaluation: not every misclassification costs the same.

    The flat multiclass metric charges the same for confusing ``DoS Hulk`` with ``DoS
    GoldenEye`` (same playbook, same containment) as for confusing it with ``BENIGN`` (no
    response at all). The study scores against the four-level ATT&CK taxonomy already in
    ``intel.attack_mapping`` — verdict / tactic / technique / class — using hierarchical
    precision/recall/F1 (Kiritchenko et al. 2006), and compares the deployed flat classifier
    against a local-classifier-per-parent-node one. The ``cost_*`` fields are a stated
    playbook schedule in arbitrary units, not a measurement: they encode an ordering nobody
    would dispute (a missed attack costs more than the wrong playbook, which costs more than
    a sibling name) so that "88% accurate" can be restated as an expected response cost.
    ``min_class_rows`` is the support a class needs before it gets its own row."""

    cost_within_technique: float = 0.1  # a sibling name; the same playbook runs
    cost_within_tactic: float = 0.3  # right intent, wrong technique
    cost_cross_tactic: float = 1.0  # the wrong playbook runs
    cost_false_alarm: float = 1.0  # an investigation with nothing at the end of it
    cost_missed_attack: float = 5.0  # no investigation at all
    min_class_rows: int = 30  # test rows a class needs before it gets its own row

    def error_costs(self) -> dict[str, float]:
        """Per-error-kind playbook cost, keyed as ``evaluation.hierarchy.ERROR_KINDS``."""
        return {
            "exact": 0.0,
            "within_technique": self.cost_within_technique,
            "within_tactic": self.cost_within_tactic,
            "cross_tactic": self.cost_cross_tactic,
            "false_alarm": self.cost_false_alarm,
            "missed_attack": self.cost_missed_attack,
        }


class DeferConfig(BaseModel):
    """Learning to defer: which flows are worth an analyst's time, under a budget.

    Conformal abstention declines to decide where the *model* is unsure, which assumes the
    human is better there. Madras et al. (2018) state the decision properly as a comparison
    of two expected losses under a review budget, and that reframing makes the analyst the
    experimental variable. Three are simulated — skill that is constant, skill that tracks
    the model's confidence, and skill that tracks the flow's distance from the training data
    — via ``analyst_base_skill`` (accuracy at the middle of the range) and ``analyst_spread``
    (how sharply it varies across it), so an analyst can be made strong or weak without
    changing *which* flows they are strong on. ``budget_fractions`` sweeps the share of test
    flows that may be reviewed and ``operating_budget_fraction`` picks the row the tables
    report; ``reference_rows`` sizes the training sample the novelty distance is measured
    against. Reviews are charged whether or not the human was right, without which
    "defer everything" wins by construction."""

    analyst_base_skill: float = 0.8  # accuracy at the middle of the covariate range
    analyst_spread: float = 0.35  # how sharply skill varies across it
    budget_fractions: list[float] = Field(
        default_factory=lambda: [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    )
    operating_budget_fraction: float = 0.01  # the budget the summary tables report
    reference_rows: int = 4000  # training rows the novelty distance is measured against
    min_rows_per_bin: int = 500  # validation flows per cell of the skill estimator
    cost_false_positive: float = 25.0  # matches the cost study's analyst-time figure
    cost_false_negative: float = 500.0  # expected loss from a missed attack flow
    cost_review: float = 25.0  # an analyst's time, charged right or wrong


class InvarianceConfig(BaseModel):
    """Causal-invariance methods over capture days, and a premise check before them.

    Invariant Causal Prediction (Peters et al. 2016) keeps features whose relationship to the
    label is stable across environments; Invariant Risk Minimization (Arjovsky et al. 2019)
    penalises representations on which different environments would prefer different
    classifiers. Both assume environments that differ in nuisance structure while sharing the
    label mechanism, which CIC-IDS2017's days do not — so the report measures the per-day
    class composition first and reads everything else in that light. ``min_strength`` and
    ``max_dispersion`` are the screen: a feature must point the same way every day, carry at
    least that much mean ``|AUC - 0.5|``, and vary in magnitude by no more than that
    coefficient of variation. Both were fixed before the transfer numbers were seen, because
    a screen tuned until its subset wins is selecting on the outcome. ``penalty_weights``
    sweeps IRMv1's penalty on a linear head (weight 0 is the ERM control, identical in every
    other respect); ``steps``, ``learning_rate`` and ``l2`` are that head's optimiser."""

    min_strength: float = 0.02  # mean |AUC - 0.5| a feature needs to be worth screening
    max_dispersion: float = 0.75  # allowed coefficient of variation of that strength
    penalty_weights: list[float] = Field(default_factory=lambda: [0.0, 1.0, 10.0, 100.0, 1000.0])
    steps: int = 300  # full-batch gradient steps for the linear head
    learning_rate: float = 0.5
    l2: float = 1e-4
    plot_features: int = 20  # strongest features shown in the stability figure


class MonotonicConfig(BaseModel):
    """Monotone constraints as a structural evasion defence, priced against detection.

    The evasion attack works by inflation: pad the flow until the score falls under the
    threshold. A model constrained non-decreasing in every attacker-inflatable feature cannot
    be attacked that way at all -- adding bytes can only raise suspicion -- and both backends
    enforce the constraint at split time, so it holds for every input rather than for the ones
    that resemble training rows. The inflatable set is ``robustness.controllable_features``,
    shared with the evasion and verification studies so the three cannot drift apart.
    ``inflation_reach`` is how far the verifier lets the attacker inflate in standardised
    units (large enough to be effectively unbounded for a standardised feature);
    ``attack_steps`` and ``attack_rounds`` drive the greedy inflation
    search (at each round, the single addition that lowers the score most is kept);
    ``probe_steps`` are the random bumps the falsification probe tries. ``max_attack_flows``
    and ``max_verify_flows`` bound the work, since both the search and the interval
    propagation are per-flow."""

    inflation_reach: float = 1e6  # standardised units the verifier lets the attacker add
    attack_steps: list[float] = Field(default_factory=lambda: [0.25, 1.0, 4.0])
    attack_rounds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    probe_steps: list[float] = Field(default_factory=lambda: [0.1, 1.0, 10.0])
    max_attack_flows: int = 500  # alerting flows driven through the greedy attack
    max_verify_flows: int = 400  # flows the interval verifier proves (it is per-flow)


class OptimalTreeConfig(BaseModel):
    """Provably optimal sparse decision trees by branch and bound, against greedy CART.

    The distilled surrogate is grown by CART, which is greedy, so nobody knows how much
    accuracy greediness costs. At interpretable sizes that is now computable exactly (Hu,
    Rudin & Seltzer 2019; Lin et al. 2020): the search minimises weighted error plus
    ``penalties`` per leaf and reports whether the space was **exhausted**, which is the
    difference between a proof and a best effort. ``n_features`` and ``n_thresholds`` set the
    binarisation (features ranked by single-feature separation, cut at quantiles rather than
    at purity-optimal points, since a purity-optimal threshold is a greedy split smuggled into
    an exhaustive search). ``max_depth`` and ``node_budget`` bound the search; an uncertified
    row is reported as an upper bound rather than as the optimum. ``max_train_rows`` subsamples
    for tractability, and greedy is scored on the same rows so the gap is like-for-like."""

    n_features: int = 8  # strongest features entering the binarisation
    n_thresholds: int = 3  # quantile cuts per feature
    max_depth: int = 3  # depth limit for both the search and the greedy baseline
    penalties: list[float] = Field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02, 0.05])
    node_budget: int = 400_000  # search nodes before the certificate is withdrawn
    max_train_rows: int = 8000  # training rows the search runs on


class SketchConfig(BaseModel):
    """Streaming sketches for host analytics at line rate, with every bound checked.

    The host-graph scan detector keeps a set of destinations per source, which grows with the
    traffic and fails during exactly the incident it was bought for. Count-Min (Cormode &
    Muthukrishnan 2005), HyperLogLog (Flajolet et al. 2007), Misra-Gries (1982) and reservoir
    sampling (Vitter 1985) answer the same questions in fixed memory, and the study grades
    each guarantee against exact ground truth rather than citing it. The stream is synthetic
    because cleaning drops the identity columns before any model sees them; it is
    ``zipf_exponent``-skewed rather than uniform because hash collisions hurt most under skew,
    which is the regime real traffic is in. ``countmin_epsilons`` and ``countmin_delta`` size
    the Count-Min tables from the guarantee wanted; ``hll_precisions`` sweeps register counts;
    ``top_k`` is the shortlist whose ordering must survive approximation."""

    n_flows: int = 50_000  # synthetic stream length (the sketches are pure Python)
    n_hosts: int = 1_500
    zipf_exponent: float = 1.1  # heavy-tailed talker volumes, as real traffic is
    scanners: int = 3  # planted high-fan-out sources with a known answer
    scanner_targets: int = 600
    countmin_epsilons: list[float] = Field(default_factory=lambda: [0.01, 0.001, 0.0001])
    countmin_delta: float = 0.01  # failure probability the width/depth are sized for
    hll_precisions: list[int] = Field(default_factory=lambda: [6, 8, 10, 12])
    top_k: int = 10  # shortlist whose ranking must survive the approximation
    heavy_hitter_k: int = 32  # Misra-Gries counters (guarantees any 1/k-share host)
    reservoir_size: int = 5_000


class DiscoveryConfig(BaseModel):
    """Unsupervised attack-family discovery over the flows the detector flags.

    Novel-attack detection ends with an unstructured pile of anomalies; clustering turns it
    into candidate campaigns an analyst can triage as groups. The methodological crux is that
    ``k`` is chosen by **silhouette score on the unlabelled features** (``k_candidates``,
    subsampled to ``silhouette_sample`` rows for speed) — choosing it by whichever value best
    reproduces the labels would be leakage wearing an unsupervised costume, so labels are
    opened only afterwards to grade a decision already made. A family counts as *discovered*
    when some cluster is at least ``min_purity`` made of it and holds ``min_cluster_size``
    flows. ``novel_quantile`` sets the relative distance beyond which a cluster is called a
    candidate *new* family rather than named after the nearest known centroid (an absolute
    distance would be unit-dependent). ``baseline_trials`` sizes the random-assignment control
    that gives the ARI a reference point. Runs on the stratified split, where every family
    appears in the test set so "would it have been discovered" is well posed."""

    flag_fpr: float = 0.01  # budget defining the flagged pile that gets clustered
    k_candidates: list[int] = Field(default_factory=lambda: [2, 3, 4, 5, 6, 8, 10, 12, 16])
    silhouette_sample: int = 3000  # rows the silhouette is evaluated on (it is O(n^2))
    max_flows: int = 6000  # cap on the clustered pile
    min_purity: float = 0.6  # dominant-class share that makes a cluster a discovery
    min_cluster_size: int = 20  # a coherent campaign, not a statistical coincidence
    novel_quantile: float = 0.8  # distance quantile above which a cluster is "new"
    min_class_rows: int = 30  # training rows a class needs to enter the naming catalogue
    baseline_trials: int = 20  # random assignments averaged for the ARI control


class SequentialABConfig(BaseModel):
    """Anytime-valid shadow-model comparison (Robbins 1970; Howard et al. 2021).

    A fixed-sample test earns its error rate by being evaluated **once**, at a sample size
    fixed in advance; checking it every morning as shadow data arrives inflates the
    false-positive rate toward certainty. A confidence sequence is valid simultaneously at
    every sample size, so an operator may peek and stop freely. ``rho`` is the mixture
    parameter — small tightens the boundary early (catch a big effect fast), large tightens
    it later (resolve a small effect eventually); ``alpha`` is the coverage level and
    ``power`` feeds the fixed-n comparison. ``n_null_trials`` / ``null_obs`` / ``checkpoints``
    size the simulation that *measures* the peeking inflation rather than asserting it. The
    challenger is the deployed configuration with a different capacity and learning rate, and
    the comparison is paired per flow — the entire statistical advantage of a shadow
    deployment over splitting traffic."""

    alpha: float = 0.05  # coverage level, held uniformly over all sample sizes
    power: float = 0.80  # target power for the fixed-n comparison row
    rho: float = 1.0  # mixture parameter: where the anytime boundary is tightest
    max_stream: int = 20000  # cap on the shadow stream length
    challenger_num_leaves: int = 31
    challenger_learning_rate: float = 0.08
    n_null_trials: int = 400  # null streams used to measure the peeking error rate
    null_obs: int = 2000  # observations per null stream
    checkpoints: int = 20  # how often the peeking team looks at the dashboard
    warmup: int = 500  # prefix used to fix the sub-Gaussian scale proxy (never re-estimated)


class FederatedConfig(BaseModel):
    """Federated averaging across sites that cannot pool raw traffic (McMahan et al. 2017).

    Flow records carry who talked to whom; for a hospital group or an MSSP's client estates
    "send us your flow logs" ends the project, and training alone is expensive because each
    site only sees the attacks that hit it. FedAvg trains locally, shares only **weights**,
    averages them by sample count, and repeats. Sites here are the capture days of the
    training split — already non-IID (different attacks per day), the regime where FedAvg is
    known to degrade via client drift. ``rounds`` and ``local_epochs`` set the
    communication/computation trade (more local work per round saves bandwidth and increases
    drift); the local-only and centralized arms get the same total budget so the comparison
    is about data access, not compute. ``noise_multipliers`` adds DP-FedAvg arms: each site's
    update is clipped to ``clip_norm`` (bounding one *site's* influence — the privacy unit
    that matches the threat) and Gaussian noise is added to the aggregate, accounted at
    ``delta`` with the same Renyi accountant the DP study uses. The model is linear because
    FedAvg averages parameters and a boosted forest has none to average."""

    rounds: int = 10  # federated aggregation rounds
    local_epochs: int = 2  # local passes per site per round (more = more client drift)
    batch_size: int = 256
    learning_rate: float = 0.1
    l2: float = 0.0001
    clip_norm: float = 5.0  # L2 bound on one site's update, for the DP arms
    noise_multipliers: list[float] = Field(default_factory=lambda: [0.5, 2.0])
    delta: float = 1e-5  # DP delta the per-site epsilon is reported at


class SecAggConfig(BaseModel):
    """Secure aggregation over the federation (Bonawitz et al., CCS 2017).

    FedAvg sends weights instead of flows, which moves the data-protection problem rather
    than solving it: an update is a function of the data, and the study shows a coordinator
    naming the attack family a site holds from the update alone. Secure aggregation removes
    the channel — pairwise Diffie-Hellman masks that cancel in the sum, a self-mask that
    stops a coordinator unmasking a live site by declaring it dropped, and Shamir shares so
    the round survives real dropouts. ``shards_per_day`` splits each capture day so the
    federation is large enough for a recovery threshold and an anonymity set to mean
    something; ``threshold_fraction`` sets the `t`-of-`n` share threshold. ``scale_bits`` is
    the fixed-point scale the field encoding uses (``scale_bits_sweep`` measures both of its
    failure modes: a quantization floor and a wraparound ceiling). ``group_sizes`` sweeps the
    privacy/robustness frontier — larger groups hide more and leave less for a robust
    aggregation rule to see — and ``range_bound`` is the coordinate bound an *ideal* range
    proof would enforce, used to price the strongest attack such a proof still permits."""

    shards_per_day: int = 4  # sites per capture day (3 days -> 12 participants)
    rounds: int = 6  # federated aggregation rounds
    local_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 0.1
    l2: float = 0.0001
    threshold_fraction: float = 0.5  # Shamir recovery threshold as a fraction of the sites
    scale_bits: int = 20  # fixed-point scale used for the headline run
    scale_bits_sweep: list[int] = Field(default_factory=lambda: [0, 8, 20, 32, 40, 44, 46, 48])
    dropout_counts: list[int] = Field(default_factory=lambda: [0, 1, 3, 6, 7])
    group_sizes: list[int] = Field(default_factory=lambda: [1, 2, 3, 6, 12])
    privacy_rounds: int = 3  # rounds the identification attack is averaged over
    reference_benign_rows: int = 2000  # benign flows behind each per-family reference update
    attack_scale: float = 10.0  # sign-flip amplification of the single malicious site
    range_bounds: list[float] = Field(default_factory=lambda: [0.02, 0.05, 0.1, 0.25, 1.0])
    cost_sites: list[int] = Field(default_factory=lambda: [4, 8, 16, 32])


class DPSynthConfig(BaseModel):
    """Differentially-private synthetic flow release (PrivBayes family, Zhang et al. 2017).

    The federated and secure-aggregation studies keep the data still and move the
    computation; this asks whether the *data* can move instead, as a synthetic release with a
    formal guarantee. Features are binned on a **public** signed-log grid spanning
    ``domain_min``..``domain_max`` — derived from the schema, never from the data, because
    taking min/max from the capture is already a query about one record. ``n_bins`` sets the
    grid resolution (finer grids model the data better and split the budget further per
    cell). ``epsilons`` is the swept budget; ``prior_budget_fraction`` is the slice spent on
    the class prior, with the rest divided across one conditional marginal per feature
    (sequential composition within a class, parallel across classes under add/remove
    neighbouring). ``structures`` chooses which parent sets to release; the non-private
    ``oracle Chow-Liu`` arm (``include_oracle_structure``) bounds what a private structure
    search could ever buy before anybody spends budget building one. The audit arms run a
    nearest-neighbour membership attack against the release itself."""

    n_bins: int = 24  # bins per feature on the public signed-log grid
    domain_min: float = -1.0  # the CICFlowMeter "not set" sentinel is the low end
    domain_max: float = 1e9  # declared public upper bound, not measured from the data
    epsilons: list[float] = Field(default_factory=lambda: [0.5, 1.0, 4.0, 16.0])
    prior_budget_fraction: float = 0.05  # share of epsilon spent on the class prior
    structures: list[str] = Field(default_factory=lambda: ["independent", "public families"])
    include_oracle_structure: bool = True
    oracle_epsilons: list[float] = Field(default_factory=lambda: [4.0])
    max_released_rows: int = 30000
    repeats: int = (
        3  # synthesis draws per arm (the mechanism is randomised; one draw is not a result)
    )
    n_estimators: int = 250  # downstream model size (kept small: this grid trains many)
    audited_structure: str = "public families"
    audited_epsilons: list[float] = Field(default_factory=lambda: [0.5, 4.0])
    audit_rows: int = 800  # members / non-members put to the membership attack
    audit_release_rows: int = 3000  # released rows the attacker searches over


class PretrainConfig(BaseModel):
    """Self-supervised pretraining on unlabelled flows (VIME 2020, SCARF 2022).

    The fifth answer to the label shortage, and the only one that changes the *inputs*: learn
    a representation from unlabelled traffic, then fit a small head on whatever labels exist.
    Both pretext tasks share one encoder and one corruption operator (replace
    ``corruption_rate`` of a row's features with values drawn from the same column elsewhere
    in the pool) so the comparison is between objectives. ``label_budgets`` sweeps the labels
    the practitioner has (0 means "all of them"), each drawn ``repeats`` times because a
    hundred-label draw is high variance. Two unlabelled pools are compared: the training days
    and ``deployment_pool_day`` — inputs only, labels never touched — with the *later*
    ``evaluation_day`` held out, because splitting the test days at random would put the same
    attack burst on both sides. Controls (PCA, an untrained encoder, and boosted trees on raw
    features) are not optional extras: they are what separates "the pretext task worked" from
    "a lower-dimensional projection is easier for a linear model"."""

    embedding_dim: int = 64
    hidden_sizes: list[int] = Field(default_factory=lambda: [128])
    epochs: int = 30
    batch_size: int = 512  # also the contrastive difficulty: negatives come from the batch
    learning_rate: float = 1e-3
    corruption_rate: float = 0.3
    reconstruction_weight: float = 2.0  # VIME's alpha on the reconstruction head
    temperature: float = 0.5  # SCARF's InfoNCE temperature
    max_pool_rows: int = 20000
    label_budgets: list[int] = Field(default_factory=lambda: [100, 250, 1000, 4000, 0])
    repeats: int = 3  # label draws per budget
    eval_fpr: float = 0.01
    certification_confidence: float = 0.95  # the confidence the FPR floor is quoted at
    boosted_estimators: int = 200
    deployment_pool_day: str = "Thursday"  # unlabelled adaptation pool (earlier test day)
    evaluation_day: str = "Friday"  # held-out evaluation (strictly later)


class RiskControlConfig(BaseModel):
    """Distribution-free control of a named risk (Angelopoulos et al. 2021, 2022).

    Every operating point here is chosen by fixing a false-positive budget, which implies a
    miss rate nobody wrote down. This controls the miss rate directly, two ways.
    **Conformal risk control** picks the extreme threshold whose inflated empirical risk
    ``(n R + B) / (n + 1)`` clears ``alpha`` and guarantees ``E[R] <= alpha`` -- an
    *expectation* bound, which ``n_trials`` simulated calibrate-and-deploy cycles show being
    exceeded on individual deployments about half the time. **Learn then Test** treats each
    grid threshold as a hypothesis with a Hoeffding-Bentkus p-value and returns the certified
    set, buying ``P(R > alpha) <= delta``. ``multi_alphas`` x ``volume_budgets`` runs both
    constraints at once (intersection-union p-values, Bonferroni across the grid, because the
    two risks move in opposite directions); an empty result is a certificate of infeasibility,
    not a failure. ``class_alpha`` re-runs the promise per attack family, where the affordable
    ones live."""

    alphas: list[float] = Field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5])
    delta: float = 0.1  # the high-probability level Learn-then-Test certifies at
    grid_size: int = 200  # candidate thresholds, taken as quantiles of the score distribution
    n_trials: int = 200  # simulated calibrate-and-deploy cycles behind the exceedance column
    calibration_fraction: float = 0.5
    multi_alphas: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5])
    volume_budgets: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.05])
    class_alpha: float = 0.25  # per-class miss-rate target
    class_min_support: int = 30  # attacks a class needs before its promise is testable


class SamplingConfig(BaseModel):
    """Scoring a fraction of the stream, and estimating what was skipped (Horvitz-Thompson 1952).

    At line rate the model cannot see every flow. The cascade makes scoring cheaper at full
    coverage and the sketches count without scoring; this asks what to do when the budget is
    genuinely hard. ``budgets`` is the fraction of flows the model may score. Four designs
    compete: uniform, stratified by service (proportional and Neyman allocation), priority
    sampling with inclusion probability proportional to a cheap pre-filter's score, and greedy
    top-k. ``floor`` is the minimum inclusion probability under the priority design -- the
    exploration budget that keeps every flow reachable and therefore keeps the Horvitz-Thompson
    estimator defined; greedy has no floor by construction, which is why no unbiased estimator
    of the stream's attack total exists under it. ``n_simulations`` draws each design repeatedly
    so the reported confidence intervals can have their *coverage* measured rather than
    asserted."""

    budgets: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25])
    floor: float = 0.002  # minimum inclusion probability under the priority design
    n_simulations: int = 200  # draws per design per budget (coverage is measured, not assumed)


class SliceDiscoveryConfig(BaseModel):
    """Automatic discovery of underperforming feature regions (SliceFinder, Chung et al. 2019).

    The per-class and per-service studies slice on a partition somebody chose in advance;
    this searches for the regions nobody had a hypothesis about. Every feature is quantile-
    binned into ``n_bins`` literals and a beam of width ``beam`` searches conjunctions to
    ``depth``, keeping slices with at least ``min_support`` flows. Because a search over
    hundreds of thousands of candidate regions finds terrible-looking slices in a model with
    no weaknesses at all, three defences run with it: Benjamini-Hochberg control of the
    false-discovery rate at ``q``, a *permuted-loss* null calibration reported before any real
    finding, and a discovery/confirmation split so the winner's curse is measured rather than
    inherited. ``top_n`` bounds how many surviving slices get re-measured and rendered."""

    n_bins: int = 10  # quantile bins per feature (each bin becomes one literal)
    depth: int = 2  # conjunction depth of the search
    beam: int = 25  # slices carried from one depth to the next
    min_support: int = 100  # flows a slice needs before it is scored
    q: float = 0.05  # Benjamini-Hochberg false-discovery rate across candidates
    alpha: float = 0.05  # the uncorrected level, reported alongside for contrast
    top_n: int = 12  # significant slices carried into confirmation and the report


class BatchingConfig(BaseModel):
    """Server-side micro-batching of single-flow requests.

    The API scores one flow per request and `/predict/batch` asks the caller to batch, which
    a collector shipping records as flows close cannot do. Almost all the cost of scoring one
    flow is *fixed* (frame construction, transformer dispatch, ensemble setup), so a server
    that holds arriving requests for a few milliseconds amortises a constant rather than
    trading accuracy for speed. ``batch_sizes`` and ``timing_repeats`` measure the real
    service curve through the deployed scoring path (median of repeats, because a GC pause is
    not a property of the batch size); the fit splits it into a fixed and a marginal term.
    ``arrival_rates`` then drives a discrete-event simulation of three policies -- no
    batching, batch-on-arrival, and adaptive waiting up to ``max_wait_ms`` for ``max_batch``
    to fill -- reported at p50/p95/p99 because a mean latency on a queue describes nobody.
    ``wait_sweep_ms`` sweeps the one knob at ``headline_rate``."""

    batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    timing_repeats: int = 7  # median over repeats at each batch size
    n_estimators: int = 300  # the served model's size (timing depends on it, so it is pinned)
    max_batch: int = 64
    max_wait_ms: float = 5.0
    arrival_rates: list[float] = Field(
        default_factory=lambda: [5.0, 20.0, 50.0, 200.0, 800.0, 2000.0, 5000.0]
    )
    wait_sweep_ms: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 20.0])
    headline_rate: float = 2000.0  # the load the max-wait sweep is run at
    n_requests: int = 20000  # simulated arrivals per policy per rate


class ParetoConfig(BaseModel):
    """Multi-objective model selection by NSGA-II (Deb et al. 2002).

    Every model choice in this project collapses several things into one number and sorts on
    it, which either hides the other axes or hard-codes an exchange rate somebody invented.
    This evolves a Pareto front over the boosted model's hyperparameters against three
    objectives that genuinely conflict -- detection at the false-positive budget, inference
    cost, and detection surviving a padding attack of ``evasion_factor``. ``population_size``
    x ``generations`` sets the evaluation budget, matched exactly by a random-search control
    so the algorithm has to earn its complexity on hypervolume rather than on reputation.
    ``n_weights`` weight vectors are then drawn from the simplex to find which front members
    *no* weighted-sum objective can select -- a fact about the front's convexity, and the
    argument for reporting a front instead of a score. ``max_train_rows`` caps training so a
    few hundred fits stay affordable, and the cap applies to every candidate."""

    population_size: int = 10
    generations: int = 4
    crossover_eta: float = 15.0  # simulated-binary crossover distribution index
    mutation_eta: float = 20.0  # polynomial mutation distribution index
    mutation_rate: float = 0.25  # per-coordinate mutation probability
    evasion_factor: float = 1.5  # multiplier applied to volume-like features by the attacker
    n_weights: int = 20000  # weight vectors sampled when testing reachability
    max_train_rows: int = 8000


class PSIConfig(BaseModel):
    """Private set intersection for indicator sharing (Meadows 1986; Huberman et al. 1999).

    Sigma and STIX export what a detector found; both assume the sharing decision is already
    made. The step before it leaks: asking a peer whether they have seen an indicator tells
    them you are interested in it. This runs the Diffie-Hellman PSI protocol between two
    organisations' indicator lists, built from attack destinations on ``org_a_day`` and
    ``org_b_day`` with a constructed ``overlap`` -- constructed because this stand-in draws
    addresses independently per row, so a measured intersection would measure the generator
    rather than the protocol. ``hash_samples`` sizes the SHA-256 rate measurement behind the
    dictionary attack on hash-based sharing (the practice PSI replaces), ``port_indicators``
    the fully-executed recovery against a 16-bit space, ``inflation_sizes`` the attack on PSI
    itself (a party that submits a padded candidate set learns the peer's membership for all of
    it), and ``cost_sizes`` the cost sweep."""

    org_a_day: str = "Friday"
    org_b_day: str = "Thursday"
    list_size: int = 400  # indicators per organisation (each is one 2048-bit exponentiation)
    overlap: int = 40  # shared infrastructure planted in both lists
    hash_samples: int = 200000  # bare hashes timed, for the raw-primitive rate
    address_sample: int = 400000  # addresses actually enumerated, formatting included
    port_indicators: int = 50  # hashed ports recovered by exhausting the 16-bit space
    honest_size: int = 40  # indicators the dishonest party actually holds
    universe_hit_rate: float = 0.25  # share of a submitted universe that reaches the peer
    inflation_sizes: list[int] = Field(default_factory=lambda: [100, 400, 1600])
    cost_sizes: list[int] = Field(default_factory=lambda: [50, 100, 200, 400])


class AcquisitionConfig(BaseModel):
    """Cost-aware feature acquisition: buy expensive features only where they change the answer.

    Every other study hands the model all 76 statistics; an exporter cannot, because a TCP flag
    count falls out of a header it already parsed while an inter-arrival distribution needs
    per-packet state for the whole conversation. ``prices`` assigns a per-flow computation cost
    to each behavioural family (an assumption, stated here rather than buried, with
    ``alternate_prices`` re-running the whole frontier flat as the sensitivity check).
    Four policies compete: cheapest-first fixed tiers, a greedy static subset, adaptive
    acquisition that escalates only flows whose score sits within ``bands`` of the decision
    threshold in rank space, and a random-gating control that spends the same budget without
    the uncertainty signal -- without which "adaptive wins" is unfalsifiable."""

    prices: dict[str, float] = Field(
        default_factory=lambda: {
            "TCP flags": 1.0,
            "header/window/bulk": 1.5,
            "volume/counts": 2.0,
            "packet size": 4.0,
            "flow rates": 6.0,
            "timing/IAT": 10.0,
        }
    )
    alternate_prices: dict[str, float] = Field(
        default_factory=lambda: {
            "TCP flags": 1.0,
            "header/window/bulk": 1.0,
            "volume/counts": 1.0,
            "packet size": 1.0,
            "flow rates": 1.0,
            "timing/IAT": 1.0,
        }
    )
    bands: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.2])
    keep_fractions: list[float] = Field(default_factory=lambda: [0.02, 0.1, 0.3])
    n_estimators: int = 120  # the greedy search fits many subsets, so this stays modest
    max_train_rows: int = 12000  # applied to every policy, so the comparison stays fair


class QuantileConfig(BaseModel):
    """Streaming quantile estimation for the deployed threshold.

    Every operating point here is a quantile of the score distribution, and every study that
    re-derives one assumes the scores can be sorted -- which requires storing them, which a
    stream does not allow. The sketches study counts in fixed memory but estimates no
    quantiles. This builds four estimators from scratch (reservoir sampling, P-squared,
    a merging t-digest, and a fixed-bin histogram over the score's natural [0, 1] range) and
    grades them against exact truth *and* against the alert volume each threshold actually
    delivers, because a threshold error in the fourth decimal is a large multiple of the alerts
    at a 0.1% budget. None of these estimators forgets, which is the property an operator has
    to design around, so the stream's second half is test-day traffic: real drift from the
    project's own data rather than an injected shift."""

    stream_rows: int = 200000  # replayed benign scores: long enough for the estimators to settle
    reservoir_sizes: list[int] = Field(default_factory=lambda: [1000, 10000, 50000])
    compressions: list[float] = Field(default_factory=lambda: [50.0, 200.0, 1000.0])
    histogram_bins: list[int] = Field(default_factory=lambda: [1000, 10000, 100000])
    # Drift is taken from the project's own data rather than injected: the stream's second
    # half is test-day benign traffic, so the regime change is the one the model already has.


class DensityConfig(BaseModel):
    """Is the anomaly score a density estimate or a complexity measure?

    The autoencoder has shipped since phase 5 on the premise that reconstruction error ranks
    novelty, which is false in general: an autoencoder reconstructs *simple* inputs well
    whether or not they are anomalous. Six benign-only detectors go through the deployed
    leave-one-attack-out protocol -- the two incumbents, two genuine densities (Mahalanobis
    and a diagonal mixture), the autoencoder's linear shadow (PCA reconstruction error), and a
    control that never sees the training data (the squared norm of the standardised vector).
    Every score is then correlated against that norm and rank-residualised against it, which
    is what separates *unlikely under benign traffic* from *large*."""

    methods: list[str] = Field(
        default_factory=lambda: [
            "isolation forest (deployed)",
            "autoencoder (deployed)",
            "PCA reconstruction (linear autoencoder)",
            "Mahalanobis distance (Gaussian density)",
            "Gaussian mixture (diagonal)",
            "kernel density estimate",
            "vector norm (learns nothing)",
        ]
    )
    gmm_components: int = 8
    pca_components: int = 16
    kde_samples: int = 2000  # KDE scoring is O(train x test); the subsample bounds the study
    ridge: float = 1e-3  # flow features are rank-deficient, so the covariance needs it
    max_train_rows: int = 20000
    max_attacks: int = 9


class AttestationConfig(BaseModel):
    """Proof-carrying verdicts: verifying the computation, not the artefact it ran from.

    ``verify`` hashes the bundle at rest and the ledger hash-chains the alert history; neither
    covers the moment a verdict is issued. Hashing a decision tree bottom-up turns it into a
    Merkle tree, so a root-to-leaf path plus its sibling hashes is an authentication path an
    auditor can check against a published root -- without the model and without re-running
    inference. The forgeries are executed rather than argued about, and the leakage the
    certificates create is measured in the same units the attacker would spend."""

    max_flows: int = 600
    timing_repeats: int = 20
    replay_trials: int = 100  # how often a certificate still verifies for a moved flow
    # A certificate covers the leaf region a flow fell into, so the question is how big that
    # region is. Perturbations are in standardised feature units, like every other budget here.
    replay_epsilons: list[float] = Field(
        default_factory=lambda: [1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    )
    leak_counts: list[int] = Field(default_factory=lambda: [1, 10, 50, 100, 200, 400])
    response_bytes: int = 512  # a typical /predict body, for the size comparison
    alerts_per_day: float = 5000.0  # certifying only the alerts, not the whole stream


class GamConfig(BaseModel):
    """The glass box: a generalized additive model fitted beside the deployed ensemble.

    Everything in ``netsentry/explain`` is post hoc -- an approximation of the deployed model
    whose own error has to be measured. An additive model is legible by construction: it *is*
    a sum of one-dimensional curves, so the explanation is exact rather than attributed, and
    the curves are lookup tables an operator can edit without retraining. The study prices what
    that costs against the boosted incumbent, decomposes the gap with a bounded number of
    pairwise terms, and measures surgical edits chosen on validation and scored on the later
    days (Lou, Caruana & Gehrke 2012; Caruana et al. 2015)."""

    n_bins: int = 32  # bins per shape function, for the recovery harness
    rounds: int = 60  # cycles over every feature
    learning_rate: float = 0.2
    l2: float = 1.0  # ridge on the Newton step, so a sparse bin cannot swing the curve
    # The capacity dials. Bins per shape function is resolution, parameter count and capacity
    # in one integer; the boosting schedule is the second, independent dial, and it is here
    # because one hyperparameter behaving a certain way is an anecdote.
    bin_ladder: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    round_ladder: list[int] = Field(default_factory=lambda: [1, 3, 10, 30, 120])
    pair_candidates: int = 16  # top features by swing that pairwise terms may be drawn from
    pair_ladder: list[int] = Field(default_factory=lambda: [1, 4, 16])
    pair_rounds: int = 60
    budget: float = 0.01  # the shared false-positive budget every arm is scored at
    top_features: int = 8
    plot_features: int = 4
    n_edits: int = 6
    min_removed: int = 3  # an edit clearing one alarm on validation is noise, not a finding
    # Two budgets on purpose: at the deployed one there are only a few dozen false alarms on
    # validation to choose an edit from, and whether that is why editing fails is a question.
    edit_budgets: list[float] = Field(default_factory=lambda: [0.01, 0.10])
    # The recovery harness: a known additive truth the fitter has to return, including a pure
    # noise component it has to leave flat.
    recovery_rows: int = 4000
    recovery_rounds: int = 200


class TransportConfig(BaseModel):
    """Optimal transport: a drift distance with units, and the coupling that explains it.

    PSI, KS and MMD all return a scalar with no operational unit and no statement about where
    the mass went. Transport returns both: the cost is in the ground metric's units (a
    training standard deviation, on this feature space) and the plan says which flows
    correspond to which. The regularisation is what makes it tractable and is also a bias, so
    the solver is graded against the exact linear-assignment optimum before anything depends
    on it, and every distance is quoted against a same-population floor because the empirical
    Wasserstein distance converges as ``n^(-1/d)`` and ``d`` is 76 here."""

    max_rows: int = 1000  # the coupling is O(n*m) in memory and cubic to solve exactly
    projections: int = 200  # sliced-Wasserstein directions
    permutations: int = 199
    # Entropic strengths as multiples of the median pairwise cost, so the sweep means the same
    # thing whatever the feature space's scale. The sweep doubles as the accuracy grading and
    # as the centroid-to-partner dial.
    reg_scales: list[float] = Field(default_factory=lambda: [0.5, 0.2, 0.1, 0.05, 0.02])
    max_iter: int = 300
    tol: float = 1e-6
    psi_bins: int = 10
    top_features: int = 8
    # Perturbation budgets in standard deviations, matched across targeting strategies so the
    # comparison is about which target is worth aiming at.
    budgets: list[float] = Field(default_factory=lambda: [1.0, 2.0, 4.0, 6.0, 8.0])
    profile: str = "fpr_1pct"
    budget: float = 0.01
    adapt_rows: int = 1200


class BanditConfig(BaseModel):
    """Learning the triage policy online, under partial feedback.

    The off-policy study values a policy from someone else's log; this learns one while it
    runs, observing an outcome only for the flows it chose to review. LinUCB and linear
    Thompson sampling share the same sufficient statistics and differ only in how they
    explore, epsilon-greedy is the control that says whether the sophistication pays, and the
    deployed fixed threshold is the incumbent that learns nothing and risks nothing. Regret is
    measured against the best fixed policy in hindsight -- but the number the study exists for
    is the exploration cost denominated in *missed attacks*, because that is the currency a
    SOC actually pays while a learner is finding out."""

    max_flows: int = 20000
    alpha: float = 1.0  # the confidence width (LinUCB) / posterior scale (Thompson)
    epsilon: float = 0.1
    fixed_fpr: float = 0.01  # the incumbent's operating point, chosen on validation
    random_review_rate: float = 0.05
    n_repeats: int = 5  # exploration is stochastic; one run of a bandit is an anecdote
    # The confidence width is the only knob between "never review" and "review everything";
    # the sweep prices it in the unit a SOC uses, which is alert volume rather than dollars.
    alpha_sweep: list[float] = Field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0])


class LifecycleConfig(BaseModel):
    """Model-based (stateful) testing of the serving lifecycle.

    Every part of the lifecycle has a single-request test; none of them covers the sequences,
    which is where the two-step bugs live -- a reload that half-succeeds, a guard that stops
    applying after a swap, a health endpoint reporting a version it no longer serves. The
    contract is written as a state machine holding only what an observer can check, the real
    application is driven through random operation sequences, and model and service are
    compared after every step. ``mutant_steps`` re-runs the identical walk against deliberately
    broken services, because a conformance machine that has never failed is indistinguishable
    from one that cannot."""

    steps: int = 200
    # Two operations build an entire inference engine (seconds, not milliseconds); the rest are
    # free. The schedule allocates the expensive ones explicitly and fills the rest with cheap
    # ones, because a weighted draw once produced a headline run in which a *successful* reload
    # was never exercised at all.
    min_heavy: int = 4
    min_light: int = 8


class MlintConfig(BaseModel):
    """Static analysis of this project's own ML invariants.

    The rules in `.claude/rules/ml.md` are enforced today by discipline, review and tests --
    all three of which act on code that has already been written. A linter acts on the diff.
    Each rule is a syntactic translation of a prose invariant (fit on train only, no global
    statistics, no identifier columns, seed everything, no hardcoded operating point, never
    lead with accuracy), so each is deliberately incomplete and states its own blind spot.
    ``probe_host`` names the real module the mutation harness injects violations into, because
    a rule that has only ever fired on a two-line fixture has not been shown to survive a
    module that does real work. ``max_violations`` is the CI budget: above it, the command
    exits non-zero."""

    roots: list[str] = Field(default_factory=lambda: ["netsentry"])
    # Tests deliberately construct leaky fixtures to assert the pipeline refuses them, so
    # linting them would report the test suite's own negative controls as violations.
    exclude: list[str] = Field(default_factory=lambda: ["__pycache__", "/tests/"])
    # NS003 only means something where a column could become a feature. Addresses are routing
    # metadata in `intel/`, `capture/` and `serving/watch.py` by design.
    identifier_scope: list[str] = Field(
        default_factory=lambda: [
            "netsentry/features",
            "netsentry/models",
            "netsentry/training",
            "netsentry/explain",
        ]
    )
    rules: list[str] = Field(default_factory=list)  # empty enables every rule
    probe_host: str = "netsentry/features/pipeline.py"
    # A ratchet, not a target: the three standing violations are the feature store's
    # as-of join keys, audited in docs/reports/mlint.md. A fourth fails the build.
    max_violations: int = 3


class SequentialConfig(BaseModel):
    """Sequential host-compromise decisions by Wald's SPRT (1945).

    A per-flow FPR is not a host-level guarantee: at rate `f` over `n` flows a clean host
    trips at least one alert with probability ``1 - (1-f)^n``, which approaches certainty for
    chatty hosts. The SPRT accumulates log-likelihood evidence flow by flow and stops at the
    first boundary crossing, controlling **both** error rates and — among tests with those
    rates — minimising the expected number of flows (Wald & Wolfowitz 1948). Both likelihoods
    come from the deployed operating point measured on *validation* (an alerting flow is
    TPR/FPR times more likely on a compromised host), so nothing new is fitted.
    ``alpha`` is the tolerated false escalation of a clean host, ``beta`` the tolerated miss;
    ``max_flows`` bounds the observation window (a stream that never crosses is reported
    *undecided*, which is a real outcome, not a failure). ``compromise_mix`` is the headline
    attack share of a compromised host's traffic and ``compromise_mixes`` the sweep. Host
    streams are composed from real test-set scores because the identifier columns that would
    carry host identity are dropped before modelling — the project's leakage rule."""

    alpha: float = 0.01  # tolerated probability of escalating a clean host
    beta: float = 0.10  # tolerated probability of missing a compromised one
    max_flows: int = 1000  # observation window per host before reporting undecided
    n_hosts: int = 400  # simulated streams per arm
    compromise_mix: float = 0.10  # headline attack share of a compromised host's flows
    compromise_mixes: list[float] = Field(
        default_factory=lambda: [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
    )


class CascadeConfig(BaseModel):
    """Budgeted two-stage inference: spend the expensive model only where it changes the answer.

    A cheap logistic stage-1 runs on every flow; only flows above its cut reach the deployed
    boosted model. The cut is chosen **on validation** as the quantile that forwards
    ``keep_fractions`` of the *deployed model's own alerts* — an explicit escape budget
    rather than a round number, and one that needs no labels, so it can be re-derived on
    live traffic. ``max_escape`` is the budget the headline operating point must respect;
    ``latency_calls`` is how many single-row predictions each stage is timed over (median,
    because a GC pause is not a property of the model). Both stages share the one fitted
    feature pipeline, so no second preprocessing path can skew the comparison. A cascade can
    only ever *remove* alerts, so the FPR budget survives it and the trade is recall-side."""

    keep_fractions: list[float] = Field(default_factory=lambda: [1.0, 0.99, 0.95, 0.9, 0.75, 0.5])
    max_escape: float = 0.05  # escape budget the headline operating point must respect
    stage1_max_iter: int = 2000  # logistic solver iterations for the cheap stage
    latency_calls: int = 300  # single-row predictions each stage is timed over


class DegradationConfig(BaseModel):
    """Serve-time sensor-failure audit: the deployed model with a quietly broken input.

    Not an adversary — a Tuesday. Exporters drop counters (``missing``, filled by the
    train-fitted median imputer), wedge on a constant (``stuck``, modelled as zero, the
    pessimistic end), and mis-assemble records (``shuffled``: real values on the wrong
    flows, so every marginal is intact and only the joint is destroyed). Faults are applied
    per behavioural feature family — the realistic granularity, since one exporter module
    owns the timing statistics and another owns the byte counters — and scored through the
    **unchanged** pipeline, model, and frozen primary-FPR threshold, because that is what a
    real incident looks like. Each fault's worst-feature PSI is checked against the deployed
    drift monitor's thresholds, so the report can separate visible outages from silent ones.
    ``silent_tpr_drop`` is the relative detection loss above which a monitor-quiet fault is
    called a silent failure. Runs on the honest temporal/binary split."""

    modes: list[str] = Field(default_factory=lambda: ["missing", "stuck", "shuffled"])
    silent_tpr_drop: float = 0.25  # relative detection loss that makes a quiet fault serious


class MultiplicityConfig(BaseModel):
    """Predictive multiplicity over the Rashomon set (Marx, Calmon & Ustun, ICML 2020).

    Every metric in this project describes one model, but the training protocol has free
    choices no metric adjudicates — seed, row/column subsample, leaf count, learning rate.
    Vary them and you get a family of statistically indistinguishable models; a flow whose
    verdict flips across that family was decided by an arbitrary choice, not by evidence.
    ``n_models`` candidates are drawn from a *plausible* neighbourhood of the deployed
    configuration (a wild grid would be a strawman), each decided at its own
    validation-calibrated primary-FPR threshold so nobody wins by alerting more.
    ``epsilon`` is the headline Rashomon tolerance — relative PR-AUC slack, because a fixed
    absolute slack means different things at different base rates — and ``epsilon_sweep``
    shows how much freedom a little more slack buys. ``review_bands`` are the
    vote-fraction abstention bands the study prices as a three-way routing policy.
    Runs on the honest temporal/binary split at the deployed operating point."""

    n_models: int = 12  # candidates drawn from the plausible modelling neighbourhood
    epsilon: float = 0.05  # headline Rashomon tolerance (relative PR-AUC slack)
    epsilon_sweep: list[float] = Field(default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    subsample_choices: list[float] = Field(default_factory=lambda: [0.7, 0.8, 0.9, 1.0])
    colsample_choices: list[float] = Field(default_factory=lambda: [0.6, 0.8, 1.0])
    num_leaves_choices: list[int] = Field(default_factory=lambda: [31, 63, 127])
    learning_rate_choices: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.08])
    review_bands: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.4, 0.6), (0.2, 0.8), (0.01, 0.99)]
    )


class CovariateShiftConfig(BaseModel):
    """Covariate-shift diagnosis + importance-weighted correction on the temporal gap.

    How much of the temporal-vs-stratified gap is covariate shift (`p(x)` moves, `p(y|x)`
    holds), and does importance weighting close it? A domain classifier trained to tell a
    train flow from a test flow (Bickel et al. 2009; the classifier two-sample test of
    Lopez-Paz & Oquab 2017) gives both a shift detector (its held-out AUC) and the density
    ratio `w(x) = p_test/p_train` (its calibrated odds), cross-fit so no flow scores a model
    that saw it. The detector is refit with those weights (importance-weighted ERM,
    Shimodaira 2000) and scored against the unweighted baseline and the stratified ceiling —
    the honest test of whether the gap is covariate shift IW can fix or concept shift it
    cannot. ``domain_classifier`` is the family that estimates the ratio (a well-calibrated
    ``logistic`` gives smoother weights than trees); ``n_folds`` is the cross-fit depth;
    ``weight_clip`` bounds the density-ratio tail so a few extreme weights cannot dominate
    the retrain. Runs on the temporal/binary split — the shift the study is about."""

    domain_classifier: str = "logistic"  # family for the train-vs-test density ratio
    n_folds: int = 4  # cross-fit folds for out-of-sample ratios + the C2ST AUC
    weight_clip: float = 20.0  # cap on w(x) to bound importance-weight variance


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


class NeymanPearsonConfig(BaseModel):
    """Neyman-Pearson thresholds: a finite-sample guarantee on the false-positive budget.

    The headline operating point is an empirical quantile of a finite benign validation
    sample, so the rate it achieves on unseen traffic is a random variable — and a biased
    one, exceeding the budget about half the time. The NP umbrella algorithm (Tong, Feng &
    Li, JMLR 2018) replaces that with a threshold chosen as an order statistic, giving
    ``P(true FPR > alpha) <= delta`` for a finite sample with no distributional assumption
    beyond a continuous score. ``delta`` is the headline confidence level and
    ``delta_sweep`` prices tightening it; ``calibration_sizes`` sweeps the closed-form cost
    of the guarantee against how much benign traffic is available for calibration (there is
    a hard floor below which no threshold certifies the budget at all);
    ``split_calibration_size`` and ``n_splits`` drive the Monte-Carlo arm that checks the
    closed form against a measurement. Runs on the honest temporal/binary split with raw
    (uncalibrated) scores — the calibrator is monotone but creates ties that would corrupt
    an order statistic."""

    delta: float = 0.05  # tolerated probability that the realized FPR exceeds the budget
    delta_sweep: list[float] = Field(default_factory=lambda: [0.20, 0.10, 0.05, 0.01])
    calibration_sizes: list[int] = Field(
        default_factory=lambda: [1_000, 3_000, 10_000, 30_000, 100_000, 1_000_000]
    )
    split_calibration_size: int = 3_000  # benign flows per Monte-Carlo calibration draw
    n_splits: int = 400  # calibration/holdout re-draws behind the measured violation rate
    n_sims: int = 20_000  # rank-space replicates behind the exact violation-rate simulation


class SurvivalConfig(BaseModel):
    """Time-to-detection with right-censoring: the campaigns nobody ever caught.

    The campaign study averages first-alert latency over the campaigns that *raised* an
    alert, which conditions on success and deletes the worst outcomes. Kaplan-Meier (1958)
    keeps an undetected burst in the at-risk denominator for every flow it was observed
    without inventing an event time for it, so the median and the restricted mean describe
    the deployed detector rather than its lucky half; a log-rank test then compares two
    operating points using every burst including the censored ones. Attack flows are chopped
    into ``episode_flows``-length bursts within each (day, class) stream — a whole campaign
    gives a handful of subjects, and fixed-length windows make the censoring administrative
    (follow-up ends because the window ended, never because of anything about the attack),
    which is exactly the independence the estimator assumes. ``min_episodes`` is the support
    a class needs before its own curve is reported."""

    episode_flows: int = 50  # hostile flows per burst; also the follow-up horizon
    min_episodes: int = 5  # bursts an attack class needs before it gets its own row


class ByzantineConfig(BaseModel):
    """Byzantine-robust aggregation: the federated study's missing threat model.

    FedAvg averages, averaging is linear, and a linear aggregate has no bounded influence —
    one site sending a large enough vector moves the global model anywhere. Federation is
    exactly where such a site is plausible, since the reason to federate is that the other
    members' data cannot be inspected. Three attacks (sign flip, Gaussian noise, and a
    label flip whose update looks entirely ordinary) are run against four aggregation rules:
    the mean, coordinate-wise median and trimmed mean (Yin et al., ICML 2018), and Krum,
    which elects rather than averages (Blanchard et al., NeurIPS 2017). ``shards_per_day``
    splits each capture day into that many sites so a Byzantine minority is meaningful;
    ``malicious_counts`` is the sweep; ``trim`` is the trimmed mean's tolerance parameter;
    ``sign_flip_scale`` and ``gaussian_sigma`` size the two loud attacks. The clean-case row
    prices what each defence costs when nobody is lying."""

    shards_per_day: int = 4  # sites per capture day (total sites = days x shards)
    rounds: int = 8  # federated aggregation rounds
    malicious_counts: list[int] = Field(default_factory=lambda: [1, 2, 4, 6])
    trim: int = 2  # values dropped from each end per coordinate by the trimmed mean
    sign_flip_scale: float = 10.0  # amplification on the negated honest update
    gaussian_sigma: float = 5.0  # scale of the pure-noise update


class DROConfig(BaseModel):
    """Group DRO: minimise the worst service's loss instead of the average one.

    The per-service parity audit shows one global threshold treats services unequally,
    because empirical risk minimisation optimises a mean the bulk service dominates. Sagawa
    et al. (ICLR 2020) replace that objective with the worst group's and solve the saddle
    point by online exponentiated gradient: upweight whichever group is doing worst, refit,
    repeat. Groups are services resolved from the destination port (routing metadata, never
    a model feature), matching the audit. ``n_rounds`` and ``step_size`` drive the
    adversary; ``min_group_size`` drops services too rare to carry a stable per-group loss.
    The study scores DRO against plain ERM *and* against the serving-side per-service
    threshold already shipped, because a training-time fix has to beat the cheap incumbent
    rather than an absent one."""

    group_by: Literal["day", "service"] = "day"  # the partition the adversary reweights
    n_rounds: int = 8  # DRO rounds; each is a full weighted refit
    step_size: float = 2.0  # exponentiated-gradient step on the group weights
    min_group_size: int = 200  # training flows a group needs to be included


class VerifyTreesConfig(BaseModel):
    """Deterministic (sound) robustness verification of the deployed tree ensemble.

    The evasion study gives an upper bound on the attack radius and randomized smoothing
    gives a probabilistic lower bound for a *smoothed* surrogate. A tree ensemble is
    piecewise-constant over axis-aligned boxes, so interval arithmetic gives an **absolute**
    lower bound for the deployed model itself: propagate a box down each tree, sum the
    per-tree extremes, and if the bound still clears the threshold no perturbation inside the
    box can flip the verdict. Bounding trees independently makes it sound but incomplete —
    the exact answer is a max-clique search over consistent leaf tuples (Chen et al., NeurIPS
    2019) — so the study sandwiches every flow between the certificate and a real attack and
    reports the gap. ``max_radius`` bounds the bisection, ``bisection_steps`` its precision,
    ``budget`` is the radius the robust-share headline is quoted at, ``n_flows`` caps how many
    caught attacks are verified, and ``attack_samples`` sizes the random search behind the
    upper bound. ``exactness_checks``/``exactness_tolerance`` gate the whole report: the
    flattened trees must reproduce LightGBM's own raw score or the run aborts, since a proof
    about a re-implementation proves nothing about what is deployed."""

    n_flows: int = 120  # caught attack flows verified per threat model
    max_radius: float = 1.0  # bisection ceiling, in standardised feature units
    bisection_steps: int = 12  # radius precision: max_radius / 2^steps
    budget: float = 0.10  # radius the "provably robust" headline share is quoted at
    attack_samples: int = 60  # random probes inside the box, per attack radius trial
    exactness_checks: int = 200  # flows the flattened ensemble is checked against LightGBM on
    exactness_tolerance: float = 1e-6  # largest reconstruction error tolerated before aborting


class UncertaintyConfig(BaseModel):
    """Epistemic vs aleatoric decomposition over a bagged, re-seeded ensemble.

    One attack score is asked to mean both "this looks benign" and "I have never seen
    anything like this", and a SOC should treat those flows differently. An ensemble
    separates them: aleatoric uncertainty is the members' average entropy (irreducible
    noise), epistemic is the entropy of their average minus that — the mutual information
    between the label and the choice of member (Houlsby et al. 2011; Depeweg et al. 2018).
    Members share hyperparameters and differ only by bootstrap draw and seed, the tabular
    analogue of a deep ensemble (Lakshminarayanan et al. 2017); varying hyperparameters is a
    different question that the multiplicity study asks. ``n_models`` sets ensemble size and
    ``bag_fraction`` how much of the training split each member draws; ``coverages`` are the
    abstention levels of the risk-coverage curves. The temporal split supplies a falsifiable
    test — attack classes present only on the later days — so the claim that epistemic
    uncertainty tracks unfamiliarity is checked rather than asserted."""

    n_models: int = 10  # ensemble members behind the decomposition
    bag_fraction: float = 0.8  # bootstrap draw size per member, as a share of the train split
    n_holdout_classes: int = 3  # attack classes deleted from training, one controlled arm each
    min_holdout_flows: int = 100  # test flows a class needs before it can carry an arm
    coverages: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])


class OPEConfig(BaseModel):
    """Off-policy evaluation of triage policies from a logged, partially-labelled stream.

    A SOC only labels what it reviewed, so scoring a candidate threshold on its own logs
    measures the *deployed* policy's selection rather than the candidate's value. Treating
    triage as a contextual bandit gives four estimators — direct method, IPS
    (Horvitz-Thompson 1952), SNIPS (Swaminathan & Joachims 2015) and doubly robust (Dudik,
    Langford & Li 2011) — and this dataset's full labels make the true policy value
    computable, so each can be scored against it. ``logging_fpr`` is the deployed threshold
    the logs came from; ``exploration`` is the share of decisions the logging policy
    randomises (without it the propensities are 0/1 and no candidate is identified);
    ``candidate_fprs`` are the policies valued offline; ``exploration_sweep`` prices the
    randomisation budget against the regret of choosing a policy with a bad estimate.
    Rewards come from ``cost`` so this report and the cost study share one currency."""

    logging_fpr: float = 0.001  # the deployed operating point that generated the logs
    exploration: float = 0.05  # headline share of triage decisions randomised
    exploration_sweep: list[float] = Field(
        default_factory=lambda: [0.0, 0.005, 0.02, 0.05, 0.10, 0.20]
    )
    candidate_fprs: list[float] = Field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    )
    n_replicates: int = 120  # replicate logs behind the headline estimator comparison
    sweep_replicates: int = 60  # replicate logs per exploration-budget row


class EVTConfig(BaseModel):
    """Extreme-value (peaks-over-threshold) estimation of the operating point.

    At a 0.1% budget the deployed threshold is pinned by a handful of benign order
    statistics, and one order of magnitude tighter it stops existing (``n * alpha < 1``
    degenerates to the sample maximum). Pickands-Balkema-de Haan says exceedances over a
    high threshold converge to a Generalized Pareto, so the tail can be *fitted* from
    hundreds of flows and extrapolated (Siffer et al., KDD 2017). ``tail_quantile`` is
    where the tail is declared to begin and ``tail_quantile_sweep`` exposes that choice as
    the bias-variance dial it is; ``budgets`` are the operating points compared on real
    scores; ``sim_budgets``/``sim_n``/``sim_trials`` drive the controlled arm against
    populations with closed-form tails (exponential, heavy, bounded) — the only place the
    realized false-positive rate can be computed exactly, and therefore the only place the
    comparison can be decided. ``grid_points`` is the resolution of the log-spaced search
    over Grimshaw's profile likelihood before golden-section refinement."""

    tail_quantile: float = 0.95  # where the fitted tail is declared to start
    tail_quantile_sweep: list[float] = Field(default_factory=lambda: [0.90, 0.95, 0.98, 0.99])
    budgets: list[float] = Field(default_factory=lambda: [0.01, 0.001, 0.0001, 0.00001])
    sim_budgets: list[float] = Field(default_factory=lambda: [0.001, 0.0001, 0.00001])
    sim_n: int = 5_000  # calibration flows per simulated replicate (matches the real split)
    sim_trials: int = 400  # replicates behind each simulated cell
    grid_points: int = 400  # log-spaced candidates for the GPD profile likelihood


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


class SLOConfig(BaseModel):
    """Detection SLOs and the multiwindow burn-rate policy derived from them.

    The objectives are given as *budgets* (the tolerable bad-event share) because that is the
    quantity every downstream number is a function of: ``alert_ratio_objective_budget`` is the
    live, label-free SLI the generated Prometheus rules evaluate, and
    ``false_alarm_objective_budget`` is the retrospective one that needs confirmed-benign labels
    and therefore cannot page. ``period_days`` is the compliance window the budget is measured
    over, ``regression_multiplier`` is the alert-ratio lift the replay injects to measure
    detection time against the closed form, ``regression_sweep`` is the range of lifts the
    policy table is priced across, and ``assumed_error_ratio`` stands in for the serving error
    rate until the service has run long enough to supply one. ``headroom`` is the multiple of
    the *measured* healthy alert ratio the calibrated budget allows: an objective the system
    already violates when nothing is wrong makes every burn-rate alert meaningless, so the
    report checks the specified objective against reality and calibrates when it fails.
    ``rules_dir`` is where the generated rule file lands — next to the hand-written alerts the
    compose stack already loads."""

    alert_ratio_objective_budget: float = 0.02  # tolerable share of scored flows that alert
    false_alarm_objective_budget: float = 0.01  # tolerable share of benign flows that alert
    availability_objective: float = 0.999  # request success ratio
    assumed_error_ratio: float = 0.0005  # stand-in serving error rate
    period_days: int = 30
    headroom: float = 2.0  # multiple of the measured healthy rate the calibrated budget allows
    regression_multiplier: float = 30.0  # alert-ratio lift the replay steps to (an abrupt
    # break: only a large, fast lift exercises the page rows, which are the rows whose
    # windows fit inside the replayed capture; the sweep covers the gentle end)
    regression_sweep: list[float] = Field(default_factory=lambda: [1.5, 3.0, 10.0, 50.0])
    replay_hours: float = 16.0  # wall-clock the replayed capture days stand for (2 x 8h)
    rules_dir: Path = Path("docker/prometheus")


class LedgerConfig(BaseModel):
    """Tamper-evident alert ledger: a hash chain over the alerts the service emits.

    ``path`` is the append-only chain and ``anchor_path`` the published ``(count, head_hash)``
    pair -- the only thing that makes tail-truncation detectable, so it belongs somewhere the
    ledger's writer does not control. ``enabled`` gates the spool watcher's sealing step (off by
    default: sealing is cheap but it is still a write the operator should opt into), and
    ``demo_alerts`` bounds the ledger the tamper report builds."""

    enabled: bool = False  # seal alerts emitted by the spool watcher
    path: Path = Path("data/ledger/alerts.jsonl")
    anchor_path: Path = Path("data/ledger/anchor.json")
    demo_alerts: int = 500  # alerts the tamper-evidence report seals


class FeatureStoreConfig(BaseModel):
    """Point-in-time-correct host context, and the temporal leak the naive join creates.

    lookback_seconds bounds the window each flow's context is aggregated over -- strictly
    earlier events only, which is what a serving path could reproduce at request time.
    max_rows caps the raw capture the study reads (the as-of sweep is linear, but the raw
    files carry every identifier column and are the largest thing here). The entity is the source
    host; identifiers are used to *compute* the aggregates and never reach the model, which sees
    four behaviour counts."""

    lookback_seconds: float = 60.0
    max_rows: int = 60000  # raw rows read for the host-structure diagnostic
    # The controlled stream the mechanism is demonstrated on, because the stand-in draws a fresh
    # address for every flow and therefore has no host to have context about.
    n_hosts: int = 400
    n_scanners: int = 20
    benign_flows: int = 12  # upper bound on an ordinary host's connections
    scanner_flows: int = 60  # connections in a scanner's burst
    scan_gap_seconds: float = 1.5
    stream_seconds: int = 8 * 3600


class StrategicConfig(BaseModel):
    """The evasion arms race as a game: payoff matrix, myopic race, and the commitment solution.

    defence_fractions are the deployable defences (0 is the clean model; the rest are
    adversarial training at that mimicry level), and attack_fractions the attacker's
    strategies. effectiveness_exponent controls how fast an attack loses its point as it is
    disguised -- (1 - fraction)^k -- and it is the term that stops the attacker's best reply
    from being total mimicry; the qualitative conclusions were checked to survive changing it.
    fpr_budgets re-thresholds the same fitted defences across operating points, which is
    what locates the *evasion frontier*: below some detection level the attacker's best move is
    to do nothing, because the disguise costs more attack value than it buys. rounds is the
    length of the simulated myopic race, and max_attack_flows bounds the per-cell scoring
    work (the matrix is |defences| x |attacks| evaluations, reused across every budget)."""

    defence_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.15, 0.3, 0.5])
    attack_fractions: list[float] = Field(default_factory=lambda: [0.0, 0.15, 0.3, 0.5, 0.75])
    effectiveness_exponent: float = 1.0  # linear decay of attack value with disguise
    fpr_budgets: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.25, 0.5])
    cost_sweep: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.25, 0.1, 0.05])
    rounds: int = 6
    max_attack_flows: int = 4000


class MetamorphicConfig(BaseModel):
    """Metamorphic relations as a label-free correctness oracle, validated by mutation.

    ``clock_factors`` are the exporter re-timing multipliers (kept near unity: a large dilation
    changes the traffic's character rather than just how it was recorded, so it would no longer
    be a semantics-preserving transformation). ``significant_digits`` is the precision a
    serialised payload is rounded to. ``max_single_rows`` caps the single-vs-batch relation,
    which is the only one that costs one model call per flow. The kill matrix puts three oracles
    against the same mutants: a *structural* relation violation (exact, so no tolerance), a
    labelled PR-AUC drop beyond ``accuracy_tolerance``, and a canary deviation beyond
    ``canary_tolerance`` on ``canary_rows`` pinned flows. ``stale_fraction`` is the training
    share behind the deliberately under-trained control mutant."""

    clock_factors: list[float] = Field(default_factory=lambda: [1.1, 0.9])
    significant_digits: int = 6
    max_rows: int = 8000  # unlabelled probe flows the suite runs on
    max_single_rows: int = 300  # per-flow calls for the single-vs-batch relation
    accuracy_tolerance: float = 0.01  # PR-AUC drop the labelled oracle would call a regression
    canary_rows: int = 8  # pinned flows the reference-comparison oracle checks
    canary_tolerance: float = 1e-6  # score deviation the canary oracle calls a failure
    stale_fraction: float = 0.1  # training share behind the under-trained control mutant


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
    openset: OpenSetConfig = Field(default_factory=OpenSetConfig)
    rare_rates: RareRatesConfig = Field(default_factory=RareRatesConfig)
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
    neyman_pearson: NeymanPearsonConfig = Field(default_factory=NeymanPearsonConfig)
    evt: EVTConfig = Field(default_factory=EVTConfig)
    ope: OPEConfig = Field(default_factory=OPEConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    verify_trees: VerifyTreesConfig = Field(default_factory=VerifyTreesConfig)
    dro: DROConfig = Field(default_factory=DROConfig)
    byzantine: ByzantineConfig = Field(default_factory=ByzantineConfig)
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)
    earliness: EarlinessConfig = Field(default_factory=EarlinessConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    defer: DeferConfig = Field(default_factory=DeferConfig)
    invariance: InvarianceConfig = Field(default_factory=InvarianceConfig)
    monotonic: MonotonicConfig = Field(default_factory=MonotonicConfig)
    optimal_tree: OptimalTreeConfig = Field(default_factory=OptimalTreeConfig)
    sketches: SketchConfig = Field(default_factory=SketchConfig)
    multiplicity: MultiplicityConfig = Field(default_factory=MultiplicityConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    sequential: SequentialConfig = Field(default_factory=SequentialConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    secagg: SecAggConfig = Field(default_factory=SecAggConfig)
    dp_synth: DPSynthConfig = Field(default_factory=DPSynthConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    risk_control: RiskControlConfig = Field(default_factory=RiskControlConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    slice_discovery: SliceDiscoveryConfig = Field(default_factory=SliceDiscoveryConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    pareto: ParetoConfig = Field(default_factory=ParetoConfig)
    psi: PSIConfig = Field(default_factory=PSIConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    quantiles: QuantileConfig = Field(default_factory=QuantileConfig)
    mlint: MlintConfig = Field(default_factory=MlintConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    gam: GamConfig = Field(default_factory=GamConfig)
    attestation: AttestationConfig = Field(default_factory=AttestationConfig)
    density: DensityConfig = Field(default_factory=DensityConfig)
    sequential_ab: SequentialABConfig = Field(default_factory=SequentialABConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    covariate_shift: CovariateShiftConfig = Field(default_factory=CovariateShiftConfig)
    unlearn: UnlearnConfig = Field(default_factory=UnlearnConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    poisoning: PoisoningConfig = Field(default_factory=PoisoningConfig)
    backdoor: BackdoorConfig = Field(default_factory=BackdoorConfig)
    sanitize: SanitizeConfig = Field(default_factory=SanitizeConfig)
    metamorphic: MetamorphicConfig = Field(default_factory=MetamorphicConfig)
    strategic: StrategicConfig = Field(default_factory=StrategicConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    mmd: MMDConfig = Field(default_factory=MMDConfig)
    continual: ContinualConfig = Field(default_factory=ContinualConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    deep_tabular: DeepTabularConfig = Field(default_factory=DeepTabularConfig)
    operating_point: OperatingPointConfig = Field(default_factory=OperatingPointConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
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
