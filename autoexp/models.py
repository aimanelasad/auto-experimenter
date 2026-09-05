"""Model families, their hyperparameter ranges and defaults.

The ranges are the contract between the planner (which proposes values) and
the runner (which refuses anything outside them). They are also shown to the
Claude planner verbatim.
"""

from __future__ import annotations

from typing import Literal

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ModelName = Literal["logreg", "random_forest", "hist_gb", "lightgbm"]
MODEL_FAMILIES: tuple[str, ...] = ("logreg", "random_forest", "hist_gb", "lightgbm")

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gb": "Histogram gradient boosting (sklearn)",
    "lightgbm": "LightGBM",
}

# (low, high) inclusive. A max_depth of 0 means "unlimited".
PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "logreg": {"C": (0.001, 100.0)},
    "random_forest": {
        "n_estimators": (100, 800),
        "max_depth": (0, 30),
        "min_samples_leaf": (1, 50),
    },
    "hist_gb": {
        "learning_rate": (0.01, 0.3),
        "max_iter": (50, 800),
        "max_depth": (0, 12),
        "l2_regularization": (0.0, 10.0),
    },
    "lightgbm": {
        "num_leaves": (4, 128),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (50, 1000),
        "min_child_samples": (5, 100),
        "reg_lambda": (0.0, 10.0),
        "colsample_bytree": (0.4, 1.0),
    },
}

INTEGER_PARAMS = {
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_iter",
    "num_leaves",
    "min_child_samples",
}

DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "logreg": {"C": 1.0},
    "random_forest": {"n_estimators": 300, "max_depth": 0, "min_samples_leaf": 1},
    "hist_gb": {"learning_rate": 0.1, "max_iter": 200, "max_depth": 0, "l2_regularization": 0.0},
    "lightgbm": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 20,
        "reg_lambda": 0.0,
        "colsample_bytree": 1.0,
    },
}

PARAM_NOTES = {
    "C": "inverse L2 strength; smaller = stronger regularisation",
    "n_estimators": "number of trees",
    "max_depth": "0 = unlimited",
    "min_samples_leaf": "minimum rows per leaf",
    "learning_rate": "shrinkage per boosting step",
    "max_iter": "boosting rounds",
    "l2_regularization": "L2 penalty on leaf values",
    "num_leaves": "leaves per tree (LightGBM complexity knob)",
    "min_child_samples": "minimum rows per leaf",
    "reg_lambda": "L2 penalty",
    "colsample_bytree": "fraction of features sampled per tree",
}


def search_space() -> dict:
    """The full search space as plain data (used in prompts and the UI)."""
    return {
        "feature_sets": ["raw", "engineered", "minimal"],
        "class_weight": ["none", "balanced"],
        "calibration": ["none", "sigmoid", "isotonic"],
        "models": {
            name: {
                "label": MODEL_LABELS[name],
                "params": {
                    key: {
                        "min": lo,
                        "max": hi,
                        "default": DEFAULT_PARAMS[name][key],
                        "integer": key in INTEGER_PARAMS,
                        "note": PARAM_NOTES[key],
                    }
                    for key, (lo, hi) in ranges.items()
                },
            }
            for name, ranges in PARAM_RANGES.items()
        },
    }


def needs_scaling(model: str) -> bool:
    return model == "logreg"


def build_estimator(model: str, params: dict[str, float], class_weight: str, seed: int, n_jobs: int = 2):
    """Instantiate an unfitted estimator from a validated spec."""
    cw = "balanced" if class_weight == "balanced" else None
    p = {**DEFAULT_PARAMS[model], **params}
    if model == "logreg":
        return LogisticRegression(C=p["C"], class_weight=cw, max_iter=3000, random_state=seed)
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(p["n_estimators"]),
            max_depth=None if int(p["max_depth"]) == 0 else int(p["max_depth"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            class_weight=cw,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            max_iter=int(p["max_iter"]),
            max_depth=None if int(p["max_depth"]) == 0 else int(p["max_depth"]),
            l2_regularization=p["l2_regularization"],
            class_weight=cw,
            early_stopping=False,
            random_state=seed,
        )
    if model == "lightgbm":
        return LGBMClassifier(
            num_leaves=int(p["num_leaves"]),
            learning_rate=p["learning_rate"],
            n_estimators=int(p["n_estimators"]),
            min_child_samples=int(p["min_child_samples"]),
            reg_lambda=p["reg_lambda"],
            colsample_bytree=p["colsample_bytree"],
            class_weight=cw,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    raise ValueError(f"unknown model {model!r}; choose one of {MODEL_FAMILIES}")
