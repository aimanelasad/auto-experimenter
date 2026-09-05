"""Experiment specifications and the planner output contract.

ExperimentSpec is what the runner executes. PlannerOutput is the structured
output the Claude planner must produce; it uses explicit, nullable parameter
fields so the JSON schema is fully specified.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import DEFAULT_PARAMS, INTEGER_PARAMS, PARAM_RANGES, ModelName

FeatureSetName = Literal["raw", "engineered", "minimal"]
ClassWeight = Literal["none", "balanced"]
Calibration = Literal["none", "sigmoid", "isotonic"]
Planner = Literal["rules", "claude"]

CONFIRM_MARGIN = 0.002  # PR-AUC gain over the incumbent needed for a "confirmed" verdict


class ExperimentSpec(BaseModel):
    """One fully specified, validated experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    model: ModelName
    feature_set: FeatureSetName
    params: dict[str, float] = Field(default_factory=dict)
    class_weight: ClassWeight = "none"
    calibration: Calibration = "none"
    hypothesis: str = ""
    proposed_by: Planner = "rules"
    round: int = 0

    @model_validator(mode="after")
    def _fill_and_check_params(self) -> "ExperimentSpec":
        ranges = PARAM_RANGES[self.model]
        unknown = sorted(set(self.params) - set(ranges))
        if unknown:
            raise ValueError(f"{self.model} does not accept params {unknown}")
        merged = {**DEFAULT_PARAMS[self.model], **self.params}
        for key, value in merged.items():
            lo, hi = ranges[key]
            if not lo <= value <= hi:
                raise ValueError(f"{self.model}.{key}={value} outside [{lo}, {hi}]")
            merged[key] = int(round(value)) if key in INTEGER_PARAMS else float(value)
        self.params = merged
        return self

    def key(self) -> str:
        """Stable hash of everything that changes the fitted model."""
        payload = {
            "model": self.model,
            "feature_set": self.feature_set,
            "params": self.params,
            "class_weight": self.class_weight,
            "calibration": self.calibration,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def short(self) -> str:
        params = ", ".join(f"{k}={v:g}" for k, v in sorted(self.params.items()))
        extras = []
        if self.class_weight != "none":
            extras.append("balanced")
        if self.calibration != "none":
            extras.append(f"cal={self.calibration}")
        tail = f" [{', '.join(extras)}]" if extras else ""
        return f"{self.model} / {self.feature_set} / {params}{tail}"


class Params(BaseModel):
    """All hyperparameters as nullable fields; null means 'use the default'."""

    model_config = ConfigDict(extra="forbid")

    C: float | None = Field(description="logreg only")
    n_estimators: int | None = Field(description="random_forest and lightgbm")
    max_depth: int | None = Field(description="random_forest and hist_gb, 0 = unlimited")
    min_samples_leaf: int | None = Field(description="random_forest only")
    learning_rate: float | None = Field(description="hist_gb and lightgbm")
    max_iter: int | None = Field(description="hist_gb only")
    l2_regularization: float | None = Field(description="hist_gb only")
    num_leaves: int | None = Field(description="lightgbm only")
    min_child_samples: int | None = Field(description="lightgbm only")
    reg_lambda: float | None = Field(description="lightgbm only")
    colsample_bytree: float | None = Field(description="lightgbm only")

    def as_dict(self, model: str) -> dict[str, float]:
        allowed = PARAM_RANGES[model]
        return {k: v for k, v in self.model_dump().items() if v is not None and k in allowed}


class PlannedExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="short unique name, e.g. 'lgbm_engineered_shallow'")
    model: ModelName
    feature_set: FeatureSetName
    params: Params
    class_weight: ClassWeight
    calibration: Calibration
    hypothesis: str = Field(description="one sentence: why this should beat the incumbent")

    def to_spec(self, round_no: int) -> ExperimentSpec:
        return ExperimentSpec(
            name=self.name,
            model=self.model,
            feature_set=self.feature_set,
            params=self.params.as_dict(self.model),
            class_weight=self.class_weight,
            calibration=self.calibration,
            hypothesis=self.hypothesis,
            proposed_by="claude",
            round=round_no,
        )


class PlannerOutput(BaseModel):
    """What a planner returns for one round."""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(description="two to four sentences on what the ledger shows and why these experiments")
    experiments: list[PlannedExperiment]
    stop: bool = Field(description="true if further rounds are unlikely to gain more than noise")
