"""Pydantic request/response models — the serving feature contract.

A request carries a ``flow`` mapping of CICFlowMeter feature name -> value.
Unknown feature keys and non-numeric values are rejected with 422. Missing
features are allowed (the fitted pipeline imputes them), so partial flows are
accepted. CIC column names contain spaces and slashes (not valid Python
identifiers), hence the mapping shape rather than one field per feature.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from netsentry.data import schema

_ALLOWED_FEATURES = set(schema.FEATURE_COLUMNS)


class FlowRequest(BaseModel):
    """A single network flow as a feature-name -> value mapping."""

    model_config = ConfigDict(extra="forbid")
    flow: dict[str, float]

    @field_validator("flow")
    @classmethod
    def _reject_unknown_features(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - _ALLOWED_FEATURES
        if unknown:
            raise ValueError(f"unknown feature columns: {sorted(unknown)[:5]}")
        return value


class BatchRequest(BaseModel):
    """A bounded batch of flows."""

    model_config = ConfigDict(extra="forbid")
    flows: list[dict[str, float]]

    @field_validator("flows")
    @classmethod
    def _validate_flows(cls, value: list[dict[str, float]]) -> list[dict[str, float]]:
        for flow in value:
            unknown = set(flow) - _ALLOWED_FEATURES
            if unknown:
                raise ValueError(f"unknown feature columns: {sorted(unknown)[:5]}")
        return value


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class SimilarFlow(BaseModel):
    """A nearest known training flow — case-based evidence behind a verdict."""

    label: str
    day: str | None = None
    distance: float  # Euclidean, in the pipeline's standardized units


class PredictionResponse(BaseModel):
    predicted_class: str
    is_attack: bool
    attack_probability: float
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    top_features: list[FeatureContribution]
    model_version: str
    threshold_profile: str
    # Conformal selective-prediction outputs (present when the bundle carries a
    # calibrated conformal set): the prediction set and the recommended SOC action.
    prediction_set: list[str] | None = None
    recommended_action: str | None = None
    # MITRE ATT&CK technique for the predicted attack class (None when benign).
    mitre: dict[str, str] | None = None
    # Nearest known training flows (case-based explanation). Opt-in via
    # ?exemplars=true; None when not requested or the bundle carries no index.
    similar_flows: list[SimilarFlow] | None = None
    # Why the anomaly detector flagged this flow: the top features by benign-occlusion
    # contribution (the unsupervised mirror of top_features). Opt-in via
    # ?anomaly_explain=true; populated only for flagged flows on a bundle that carries
    # an anomaly detector + benign reference, else None.
    anomaly_features: list[FeatureContribution] | None = None


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class CanaryStatus(BaseModel):
    """Behavioral self-test result: does this runtime reproduce the built model?"""

    ok: bool
    n: int
    max_delta: float
    tolerance: float


class ReloadRequest(BaseModel):
    """Ask the service to swap in a candidate bundle (canary-gated hot reload)."""

    model_config = ConfigDict(extra="forbid")
    # Bundle filename or relative path; resolved under the configured models dir and
    # rejected if it escapes that directory (no arbitrary-path loads).
    bundle: str


class ReloadResponse(BaseModel):
    """Outcome of a hot-reload attempt: what is serving now, and the gate result."""

    reloaded: bool
    model_version: str
    canary: CanaryStatus | None = None
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_at: str
    # None when the bundle predates canaries; otherwise the load-time replay result,
    # so readiness probes can pull a pod whose runtime skews the model's behavior.
    canary: CanaryStatus | None = None
    # Version of the silently-scored shadow challenger, when one is configured.
    shadow_model_version: str | None = None
