"""Dataset loading, cleaning and the one-time holdout split.

Dataset: IBM Telco Customer Churn (7,043 customers, 21 columns).
Kaggle page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Loaded from the committed copy in data/; downloaded from the IBM GitHub mirror
if the copy is missing.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"

TARGET = "Churn"
ID_COL = "customerID"
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV; download and cache it if it is not present."""
    if path.exists():
        return pd.read_csv(path)
    raw = urllib.request.urlopen(DATA_URL, timeout=60).read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quirks of the file and encode the target.

    TotalCharges is stored as text and is blank for the 11 customers with
    tenure 0; those become 0.0. Churn becomes 1 for "Yes" and 0 otherwise.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    out[TARGET] = (out[TARGET] == "Yes").astype(int)
    out = out.drop(columns=[ID_COL])
    return out[FEATURE_COLUMNS + [TARGET]].reset_index(drop=True)


@dataclass
class Dataset:
    """Train part (used by the loop) and holdout part (used once, at the end)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    seed: int

    def card(self) -> dict:
        n_train, n_holdout = len(self.X_train), len(self.X_holdout)
        return {
            "name": "IBM Telco Customer Churn",
            "source_kaggle": KAGGLE_URL,
            "source_file": DATA_URL,
            "n_customers": n_train + n_holdout,
            "n_features": len(FEATURE_COLUMNS),
            "n_categorical": len(CATEGORICAL_COLUMNS),
            "n_numeric": len(NUMERIC_COLUMNS),
            "churn_rate": round(float(pd.concat([self.y_train, self.y_holdout]).mean()), 4),
            "n_train": n_train,
            "n_holdout": n_holdout,
            "churn_rate_train": round(float(self.y_train.mean()), 4),
            "churn_rate_holdout": round(float(self.y_holdout.mean()), 4),
            "holdout_fraction": HOLDOUT_FRACTION,
            "split_seed": self.seed,
        }


def load_dataset(seed: int = SPLIT_SEED, path: Path = DATA_PATH) -> Dataset:
    """Load, clean and split once. The holdout is stratified on the target."""
    df = clean(load_raw(path))
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, stratify=y, random_state=seed
    )
    return Dataset(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_holdout=X_holdout.reset_index(drop=True),
        y_holdout=y_holdout.reset_index(drop=True),
        seed=seed,
    )
