"""Run one experiment under the fixed protocol, judge it, and keep the ledger.

Protocol: 5-fold stratified cross-validation on the train part, identical fold
assignment for every experiment (same seed), so two experiments can be compared
fold by fold. The holdout is touched only by fit_final(), once, for the winner.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .data import Dataset
from .features import make_preprocessor
from .metrics import METRIC_INFO, PRIMARY_METRIC, aggregate_folds, compute_metrics
from .models import build_estimator, needs_scaling
from .spec import CONFIRM_MARGIN, ExperimentSpec

N_FOLDS = 5
T_CONFIRM = 2.0  # paired t statistic across folds required for a "confirmed" verdict

VERDICT_LABELS = {
    "baseline": "baseline (first result)",
    "confirmed": "confirmed: beat the incumbent beyond fold noise",
    "inconclusive": "inconclusive: small gain within fold noise",
    "no_gain": "no gain: did not beat the incumbent",
    "failed": "failed: error or time limit",
}


def build_pipeline(spec: ExperimentSpec, seed: int, n_jobs: int = 2) -> Pipeline:
    estimator = build_estimator(spec.model, spec.params, spec.class_weight, seed, n_jobs)
    if spec.calibration != "none":
        # ensemble=False: refit the base estimator on the full fold and fit only the
        # calibrator on cross-validated predictions, so the ranking stays comparable.
        estimator = CalibratedClassifierCV(estimator, method=spec.calibration, cv=3, ensemble=False)
    return Pipeline(
        [
            ("prep", make_preprocessor(spec.feature_set, needs_scaling(spec.model))),
            ("model", estimator),
        ]
    )


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    cv: dict[str, dict[str, float]]
    folds: list[dict[str, float]]
    fit_seconds: float
    n_folds: int
    seed: int
    timestamp: str
    verdict: str = "pending"
    delta_vs_incumbent: float | None = None
    incumbent_name: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.cv)

    @property
    def primary(self) -> float:
        return self.cv[PRIMARY_METRIC]["mean"] if self.ok else float("nan")

    @property
    def primary_std(self) -> float:
        return self.cv[PRIMARY_METRIC]["std"] if self.ok else float("nan")

    def fold_primary(self) -> list[float]:
        return [f[PRIMARY_METRIC] for f in self.folds]

    def metric(self, key: str) -> float:
        return self.cv[key]["mean"] if self.ok else float("nan")

    def to_record(self) -> dict:
        return {
            "key": self.spec.key(),
            "spec": self.spec.model_dump(),
            "cv": self.cv,
            "folds": self.folds,
            "fit_seconds": self.fit_seconds,
            "n_folds": self.n_folds,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "delta_vs_incumbent": self.delta_vs_incumbent,
            "incumbent_name": self.incumbent_name,
            "error": self.error,
        }

    @classmethod
    def from_record(cls, rec: dict) -> "ExperimentResult":
        return cls(
            spec=ExperimentSpec(**rec["spec"]),
            cv=rec.get("cv", {}),
            folds=rec.get("folds", []),
            fit_seconds=rec.get("fit_seconds", 0.0),
            n_folds=rec.get("n_folds", 0),
            seed=rec.get("seed", 0),
            timestamp=rec.get("timestamp", ""),
            verdict=rec.get("verdict", "pending"),
            delta_vs_incumbent=rec.get("delta_vs_incumbent"),
            incumbent_name=rec.get("incumbent_name"),
            error=rec.get("error"),
        )


def run_experiment(
    spec: ExperimentSpec,
    ds: Dataset,
    seed: int = 42,
    n_folds: int = N_FOLDS,
    n_jobs: int = 2,
    time_limit: float = 90.0,
) -> ExperimentResult:
    """Cross-validate one spec. Errors and time-outs become a failed result, not an exception."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, float]] = []
    t0 = time.perf_counter()
    try:
        splits = list(skf.split(ds.X_train, ds.y_train))
        for i, (train_idx, val_idx) in enumerate(splits):
            pipe = build_pipeline(spec, seed, n_jobs)
            pipe.fit(ds.X_train.iloc[train_idx], ds.y_train.iloc[train_idx])
            prob = pipe.predict_proba(ds.X_train.iloc[val_idx])[:, 1]
            folds.append(compute_metrics(ds.y_train.iloc[val_idx].to_numpy(), prob))
            elapsed = time.perf_counter() - t0
            if i + 1 < len(splits) and elapsed > time_limit:
                raise TimeoutError(f"exceeded the {time_limit:.0f} s limit after {i + 1} folds")
    except Exception as exc:  # noqa: BLE001 - a failed experiment is data, not a crash
        return ExperimentResult(
            spec=spec,
            cv={},
            folds=folds,
            fit_seconds=round(time.perf_counter() - t0, 2),
            n_folds=len(folds),
            seed=seed,
            timestamp=timestamp,
            verdict="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    return ExperimentResult(
        spec=spec,
        cv=aggregate_folds(folds),
        folds=folds,
        fit_seconds=round(time.perf_counter() - t0, 2),
        n_folds=n_folds,
        seed=seed,
        timestamp=timestamp,
    )


def judge(result: ExperimentResult, incumbent: ExperimentResult | None) -> ExperimentResult:
    """Attach a verdict by comparing fold-by-fold against the incumbent."""
    if not result.ok:
        result.verdict = "failed"
        return result
    if incumbent is None or not incumbent.ok:
        result.verdict = "baseline"
        return result
    a = np.asarray(result.fold_primary(), dtype=float)
    b = np.asarray(incumbent.fold_primary(), dtype=float)
    result.incumbent_name = incumbent.spec.name
    if len(a) == len(b) and len(a) > 1:
        d = a - b
        mean = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        t_stat = mean / se if se > 0 else float("inf") if mean > 0 else 0.0
    else:
        mean = float(result.primary - incumbent.primary)
        t_stat = float("inf") if mean > 0 else 0.0
    result.delta_vs_incumbent = round(mean, 5)
    if mean >= CONFIRM_MARGIN and t_stat >= T_CONFIRM:
        result.verdict = "confirmed"
    elif mean > 0:
        result.verdict = "inconclusive"
    else:
        result.verdict = "no_gain"
    return result


class Ledger:
    """Append-only record of every experiment, optionally persisted as JSON lines."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.results: list[ExperimentResult] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.results.append(ExperimentResult.from_record(json.loads(line)))

    def __len__(self) -> int:
        return len(self.results)

    def append(self, result: ExperimentResult) -> None:
        self.results.append(result)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result.to_record()) + "\n")

    def keys(self) -> set[str]:
        return {r.spec.key() for r in self.results}

    def names(self) -> set[str]:
        return {r.spec.name for r in self.results}

    def ok_results(self) -> list[ExperimentResult]:
        return [r for r in self.results if r.ok]

    def incumbent(self) -> ExperimentResult | None:
        """The established model: the baseline or the last confirmed gain over it.

        A gain inside fold noise (verdict inconclusive) does not move the
        incumbent, so selection, judging and the stop rule use one criterion.
        """
        established = [r for r in self.results if r.ok and r.verdict in ("baseline", "confirmed")]
        return established[-1] if established else None

    def best_mean(self) -> ExperimentResult | None:
        """The result with the highest CV mean, which may differ from the incumbent."""
        ok = self.ok_results()
        return max(ok, key=lambda r: r.primary) if ok else None

    def by_round(self) -> dict[int, list[ExperimentResult]]:
        out: dict[int, list[ExperimentResult]] = {}
        for r in self.results:
            out.setdefault(r.spec.round, []).append(r)
        return out

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {
                "name": r.spec.name,
                "model": r.spec.model,
                "feature_set": r.spec.feature_set,
                "params": ", ".join(f"{k}={v:g}" for k, v in sorted(r.spec.params.items())),
                "class_weight": r.spec.class_weight,
                "calibration": r.spec.calibration,
                "round": r.spec.round,
                "proposed_by": r.spec.proposed_by,
            }
            for key, _label, _hib in METRIC_INFO:
                row[key] = r.metric(key)
            row["pr_auc_std"] = r.primary_std
            row["fit_seconds"] = r.fit_seconds
            row["verdict"] = r.verdict
            row["delta_vs_incumbent"] = r.delta_vs_incumbent
            row["hypothesis"] = r.spec.hypothesis
            row["error"] = r.error
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(PRIMARY_METRIC, ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    def planner_table(self, max_rows: int = 30) -> str:
        """Compact markdown table for the planner prompt, best first."""
        lines = [
            "| round | name | model | features | params | cw | cal | PR-AUC (mean +- std) | ROC-AUC | lift@10% | verdict | hypothesis |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        ordered = sorted(self.results, key=lambda r: (-(r.primary if r.ok else -1.0), r.spec.round))
        for r in ordered[:max_rows]:
            params = ", ".join(f"{k}={v:g}" for k, v in sorted(r.spec.params.items()))
            if r.ok:
                metrics = f"{r.primary:.4f} +- {r.primary_std:.4f} | {r.metric('roc_auc'):.4f} | {r.metric('lift_at_10'):.2f}"
            else:
                metrics = f"failed ({r.error}) | - | -"
            lines.append(
                f"| {r.spec.round} | {r.spec.name} | {r.spec.model} | {r.spec.feature_set} | {params} | "
                f"{r.spec.class_weight} | {r.spec.calibration} | {metrics} | {r.verdict} | {r.spec.hypothesis} |"
            )
        if len(ordered) > max_rows:
            lines.append(f"| ... | {len(ordered) - max_rows} more rows omitted | | | | | | | | | | |")
        return "\n".join(lines)

    def summary(self) -> dict:
        inc = self.incumbent()
        return {
            "n_experiments": len(self.results),
            "n_ok": len(self.ok_results()),
            "n_failed": sum(1 for r in self.results if not r.ok),
            "n_confirmed": sum(1 for r in self.results if r.verdict == "confirmed"),
            "incumbent": inc.spec.name if inc else None,
            "incumbent_pr_auc": round(inc.primary, 4) if inc else None,
        }


def fit_final(spec: ExperimentSpec, ds: Dataset, seed: int = 42, n_jobs: int = 2) -> tuple[Pipeline, dict[str, float]]:
    """Refit on the full train part and score the holdout once."""
    pipe = build_pipeline(spec, seed, n_jobs)
    pipe.fit(ds.X_train, ds.y_train)
    prob = pipe.predict_proba(ds.X_holdout)[:, 1]
    return pipe, compute_metrics(ds.y_holdout.to_numpy(), prob)


def permutation_drivers(pipe: Pipeline, ds: Dataset, seed: int = 42, n_repeats: int = 5) -> pd.DataFrame:
    """Column-level permutation importance on the holdout, scored by PR-AUC."""
    res = permutation_importance(
        pipe,
        ds.X_holdout,
        ds.y_holdout,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(ds.X_holdout.columns),
                "importance": res.importances_mean,
                "std": res.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
