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


class MonitoringConfig(BaseModel):
    """Data-drift monitoring (PSI) — the production-decay early-warning system."""

    psi_bins: int = 10
    psi_moderate: float = 0.1  # PSI >= this is a moderate distribution shift
    psi_major: float = 0.25  # PSI >= this is a major shift worth investigating
    serving_window: int = 500  # flows buffered before serving recomputes drift gauges
    reference_rows: int = 5000  # reference sample summarised into the serving bundle


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


class CrossDatasetConfig(BaseModel):
    """Synthetic 'foreign' (NetFlow-schema) dataset for cross-dataset generalization."""

    rows: int = 20000
    attack_fraction: float = 0.30
    name: str = "synthetic-netflow"


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
    default_threshold_profile: str = "fpr_0.1pct"
    max_batch_size: int = 1000
    top_k_features: int = 5
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_payloads: bool = False


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
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    crossdata: CrossDatasetConfig = Field(default_factory=CrossDatasetConfig)
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
