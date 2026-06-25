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


class PredictionResponse(BaseModel):
    predicted_class: str
    is_attack: bool
    attack_probability: float
    anomaly_score: float | None = None
    is_anomaly: bool | None = None
    top_features: list[FeatureContribution]
    model_version: str
    threshold_profile: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_at: str
