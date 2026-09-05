"""Classification metrics for a churn model scored as probabilities.

PR-AUC (average precision) is the primary metric: the positive class is what a
retention team acts on, and 26 % of customers churn, so ROC-AUC alone hides
differences in the top of the ranking. The top-decile metrics mirror the
business use ("call the 10 % most likely to churn").
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

PRIMARY_METRIC = "pr_auc"
TOP_FRACTION = 0.10

# (key, label, higher_is_better)
METRIC_INFO: list[tuple[str, str, bool]] = [
    ("pr_auc", "PR-AUC", True),
    ("roc_auc", "ROC-AUC", True),
    ("lift_at_10", "Lift@10%", True),
    ("precision_at_10", "Precision@10%", True),
    ("recall_at_10", "Recall@10%", True),
    ("best_f1", "Best F1", True),
    ("best_threshold", "Threshold (best F1)", True),
    ("f1_at_0_5", "F1@0.5", True),
    ("brier", "Brier", False),
    ("ece", "ECE", False),
]
METRIC_LABELS = {key: label for key, label, _ in METRIC_INFO}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: weighted mean of |observed rate - mean predicted|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def compute_metrics(y_true, y_prob, top_fraction: float = TOP_FRACTION) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    base_rate = float(y_true.mean())

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    # The last precision/recall pair has no threshold; ignore it.
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    best_f1 = float(f1[best_idx]) if len(thresholds) else 0.0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5

    k = max(1, int(np.ceil(top_fraction * n)))
    top = np.argsort(-y_prob, kind="stable")[:k]
    precision_at_k = float(y_true[top].mean())
    recall_at_k = float(y_true[top].sum() / max(1, y_true.sum()))

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "lift_at_10": float(precision_at_k / base_rate) if base_rate > 0 else 0.0,
        "precision_at_10": precision_at_k,
        "recall_at_10": recall_at_k,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "f1_at_0_5": float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of every metric across folds."""
    keys = fold_metrics[0].keys()
    out = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return out
