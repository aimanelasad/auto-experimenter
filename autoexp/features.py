"""Feature sets and the preprocessing pipeline.

All preprocessing is a scikit-learn transformer, so it is fit inside each
cross-validation fold and never sees validation rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

FEATURE_SETS = ("raw", "engineered", "minimal")

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Order matters: it must match the order in which add_engineered() appends columns.
ENGINEERED_CATEGORICAL = ["tenure_bucket"]
ENGINEERED_NUMERIC = [
    "n_services",
    "is_month_to_month",
    "has_no_support",
    "avg_monthly",
    "charges_ratio",
    "new_customer",
]
ENGINEERED_COLUMNS = ENGINEERED_CATEGORICAL + ENGINEERED_NUMERIC

MINIMAL_CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
MINIMAL_NUMERIC = ["tenure", "MonthlyCharges"]

FEATURE_SET_DESCRIPTIONS = {
    "raw": "all 19 columns, one-hot encoded categoricals, numeric as-is",
    "engineered": "raw plus tenure bucket, add-on count, month-to-month flag, no-support flag, "
    "average monthly charge, charge ratio (current vs. average) and new-customer flag",
    "minimal": "seven columns only: Contract, InternetService, PaymentMethod, TechSupport, "
    "OnlineSecurity, tenure, MonthlyCharges",
}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Append the engineered columns. Stateless, so it needs no fitting."""
    out = df.copy()
    tenure = out["tenure"].astype(float)
    out["tenure_bucket"] = pd.cut(
        tenure, bins=[-1, 6, 12, 24, 48, 10_000], labels=["0-6", "7-12", "13-24", "25-48", "49+"]
    ).astype(str)
    out["n_services"] = sum((out[c] == "Yes").astype(int) for c in ADDON_COLUMNS)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["has_no_support"] = ((out["TechSupport"] != "Yes") & (out["OnlineSecurity"] != "Yes")).astype(int)
    avg_monthly = np.where(tenure > 0, out["TotalCharges"] / tenure.clip(lower=1), out["MonthlyCharges"])
    out["avg_monthly"] = avg_monthly
    ratio = out["MonthlyCharges"] / pd.Series(avg_monthly, index=out.index).replace(0, np.nan)
    out["charges_ratio"] = ratio.fillna(1.0).clip(0, 10)
    out["new_customer"] = (tenure <= 6).astype(int)  # same boundary as the first tenure bucket
    return out


def _engineered_feature_names(transformer, input_features):
    return np.asarray(list(input_features) + ENGINEERED_COLUMNS, dtype=object)


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return (categorical, numeric) column names that a feature set uses."""
    if feature_set == "raw":
        return list(CATEGORICAL_COLUMNS), list(NUMERIC_COLUMNS)
    if feature_set == "engineered":
        return CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL, NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    if feature_set == "minimal":
        return list(MINIMAL_CATEGORICAL), list(MINIMAL_NUMERIC)
    raise ValueError(f"unknown feature set {feature_set!r}; choose one of {FEATURE_SETS}")


def make_preprocessor(feature_set: str, scale_numeric: bool) -> Pipeline:
    """Build the preprocessing pipeline for a feature set.

    scale_numeric should be True for linear models and False for tree models
    (scaling does not change tree splits, and leaving it out keeps the
    one-hot columns and numeric columns readable in importances).
    """
    categorical, numeric = feature_columns(feature_set)
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
            ("num", numeric_step, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps = []
    if feature_set == "engineered":
        steps.append(
            ("engineer", FunctionTransformer(add_engineered, feature_names_out=_engineered_feature_names))
        )
    steps.append(("encode", encoder))
    return Pipeline(steps)


def encoded_feature_names(preprocessor: Pipeline) -> list[str]:
    """Names of the columns that leave a fitted preprocessor."""
    return [str(n) for n in preprocessor.named_steps["encode"].get_feature_names_out()]
