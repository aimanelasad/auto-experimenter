"""Finalize a loop run: holdout score, drivers, retention, narration, and files.

The same FinalReport object feeds the CLI (which writes it to disk) and the app
(which either builds it live or loads a saved one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import Dataset
from .experiments import fit_final, permutation_drivers
from .llm import estimate_cost_usd
from .loop import LoopOutcome
from .metrics import METRIC_INFO
from .narrator import narrate
from .retention import churn_by_segment, retention_table
from .spec import ExperimentSpec

PROTOCOL = (
    "One stratified 80/20 holdout split (seed 42) made once and never used by the loop. Every "
    "experiment is scored by 5-fold stratified cross-validation on the train part with identical "
    "folds, preprocessing fit inside each fold. Verdicts compare fold by fold against the incumbent "
    "(confirmed: mean gain >= 0.002 PR-AUC and paired t >= 2). The incumbent only changes on a confirmed "
    "verdict, so the leaderboard's rank 1 by raw mean can differ from the winner. Only the winner is refit on "
    "the full train part and scored once on the holdout."
)


@dataclass
class FinalReport:
    card: dict
    run: dict
    leaderboard: pd.DataFrame
    trail: pd.DataFrame
    winner: ExperimentSpec
    winner_cv: dict
    holdout: dict
    drivers: pd.DataFrame
    segments: dict[str, pd.DataFrame]
    retention: pd.DataFrame
    retention_meta: dict
    narration: str = ""
    narration_source: str = ""
    usage: dict = field(default_factory=dict)

    def narrator_payload(self, top_n: int = 12) -> dict:
        lb = self.leaderboard.head(top_n)
        rows = []
        for _, r in lb.iterrows():
            rows.append(
                {
                    "rank": int(r["rank"]),
                    "name": r["name"],
                    "spec": f"{r['model']} / {r['feature_set']} / {r['params']}"
                    + (f" / class_weight={r['class_weight']}" if r["class_weight"] != "none" else "")
                    + (f" / calibration={r['calibration']}" if r["calibration"] != "none" else ""),
                    "round": int(r["round"]),
                    "proposed_by": r["proposed_by"],
                    "pr_auc": round(float(r["pr_auc"]), 4) if pd.notna(r["pr_auc"]) else None,
                    "pr_auc_std": round(float(r["pr_auc_std"]), 4) if pd.notna(r["pr_auc_std"]) else None,
                    "roc_auc": round(float(r["roc_auc"]), 4) if pd.notna(r["roc_auc"]) else None,
                    "lift_at_10": round(float(r["lift_at_10"]), 3) if pd.notna(r["lift_at_10"]) else None,
                    "brier": round(float(r["brier"]), 4) if pd.notna(r["brier"]) else None,
                    "ece": round(float(r["ece"]), 4) if pd.notna(r["ece"]) else None,
                    "delta_vs_incumbent": r["delta_vs_incumbent"],
                    "verdict": r["verdict"],
                    "hypothesis": r["hypothesis"],
                    "error": r["error"],
                }
            )
        return {
            "dataset": self.card,
            "protocol": PROTOCOL,
            "primary_metric": "pr_auc (average precision); higher is better. brier and ece: lower is better.",
            "run": {
                "planner": self.run["config"]["planner"],
                "n_rounds": self.run["n_rounds"],
                "n_experiments": self.run["n_experiments"],
                "n_confirmed": self.run["n_confirmed"],
                "stop_reason": self.run["stop_reason"],
                "rounds": [
                    {"round": rd["round_no"], "planner": rd["planner"], "rationale": rd["rationale"], "note": rd["note"]}
                    for rd in self.run["rounds"]
                ],
            },
            "leaderboard": rows,
            "winner": {"name": self.winner.name, "spec": self.winner.short(), "hypothesis": self.winner.hypothesis,
                       "note": "the established incumbent: the baseline or the last confirmed gain; gains inside "
                       "fold noise do not replace it"},
            "leaderboard_rank_1": rows[0]["name"] if rows else None,
            "winner_cv": self.winner_cv,
            "holdout": {k: round(float(v), 4) for k, v in self.holdout.items()},
            "top_drivers": [
                {"feature": r["feature"], "pr_auc_drop_when_permuted": round(float(r["importance"]), 4)}
                for _, r in self.drivers.head(8).iterrows()
            ],
            "churn_rate_by_segment": {
                dim: [
                    {"segment": str(r["segment"]), "customers": int(r["customers"]), "churn_rate": float(r["churn_rate"])}
                    for _, r in df.iterrows()
                ]
                for dim, df in self.segments.items()
            },
            "retention": self.retention.to_dict(orient="records"),
            "retention_meta": self.retention_meta,
            "next_suggestions": next_suggestions(self),
        }


def next_suggestions(report: FinalReport) -> list[str]:
    """Rule-based proposals for the template narrator (the model writes its own)."""
    lb = report.leaderboard
    tried_models = set(lb["model"])
    tried_features = set(lb["feature_set"])
    tried_cal = set(lb["calibration"])
    winner = report.winner
    out = []
    for model in ("lightgbm", "hist_gb", "logreg", "random_forest"):
        if model not in tried_models:
            out.append(f"Run {model} on the {winner.feature_set} features to close the gap in family coverage.")
    if "engineered" not in tried_features:
        out.append("Try the engineered feature set on the winner; tenure buckets and charge ratio may add signal.")
    if "isotonic" not in tried_cal and "sigmoid" not in tried_cal:
        out.append(
            f"Calibrate {winner.name} (isotonic, 3-fold); expected to lower Brier and ECE with unchanged PR-AUC."
        )
    if winner.model in ("lightgbm", "hist_gb"):
        out.append("Sweep learning rate against number of trees on the winner with early stopping to check the plateau.")
    else:
        out.append(
            "Add pairwise interaction terms (contract x tenure, internet service x monthly charges) to the "
            "linear model; a small PR-AUC gain is plausible."
        )
    out.append("Repeat the whole loop with two more holdout seeds to put an interval on the CV-holdout gap.")
    out.append("Try monotonic constraints on tenure and contract length in the boosting models to improve calibration.")
    return out[:5]


def run_summary(outcome: LoopOutcome) -> dict:
    usage = outcome.total_usage()
    return {
        "config": {
            "planner": outcome.config.planner,
            "rounds": outcome.config.rounds,
            "per_round": outcome.config.per_round,
            "seed": outcome.config.seed,
            "n_folds": outcome.config.n_folds,
            "patience": outcome.config.patience,
            "time_limit": outcome.config.time_limit,
        },
        "stop_reason": outcome.stop_reason,
        "seconds": outcome.seconds,
        "n_rounds": len(outcome.rounds),
        "n_experiments": len(outcome.ledger),
        "n_confirmed": sum(1 for r in outcome.ledger.results if r.verdict == "confirmed"),
        "rounds": [
            {
                "round_no": rl.round_no,
                "planner": rl.planner,
                "rationale": rl.rationale,
                "note": rl.note,
                "stop_requested": rl.stop_requested,
                "seconds": rl.seconds,
                "usage": rl.usage,
                "results": [
                    {
                        "name": r.spec.name,
                        "pr_auc": r.primary,
                        "delta_vs_incumbent": r.delta_vs_incumbent,
                        "verdict": r.verdict,
                    }
                    for r in rl.results
                ],
            }
            for rl in outcome.rounds
        ],
        "usage": usage,
        "cost_usd_estimate": estimate_cost_usd(usage) if usage else 0.0,
    }


def finalize(
    outcome: LoopOutcome,
    ds: Dataset,
    narrator: str = "template",
    seed: int = 42,
    n_jobs: int = 2,
) -> FinalReport:
    best = outcome.best()
    if best is None:
        raise RuntimeError("no successful experiment to finalize")
    pipe, holdout = fit_final(best.spec, ds, seed, n_jobs)
    drivers = permutation_drivers(pipe, ds, seed)
    prob = pipe.predict_proba(ds.X_holdout)[:, 1]
    retention = retention_table(prob, ds.X_holdout, ds.y_holdout)
    segments = churn_by_segment(
        pd.concat([ds.X_train, ds.X_holdout], ignore_index=True),
        pd.concat([ds.y_train, ds.y_holdout], ignore_index=True),
    )
    winner_cv = {key: round(best.metric(key), 4) for key, _l, _h in METRIC_INFO}
    winner_cv["pr_auc_std"] = round(best.primary_std, 4)
    report = FinalReport(
        card=ds.card(),
        run=run_summary(outcome),
        leaderboard=outcome.ledger.leaderboard(),
        trail=outcome.trail(),
        winner=best.spec,
        winner_cv=winner_cv,
        holdout=holdout,
        drivers=drivers,
        segments=segments,
        retention=retention,
        retention_meta=dict(retention.attrs),
    )
    text, source, usage = narrate(report.narrator_payload(), narrator)
    report.narration, report.narration_source, report.usage = text, source, usage
    if usage:
        report.run["usage"] = {k: report.run["usage"].get(k, 0) + v for k, v in usage.items()} | {
            k: v for k, v in report.run["usage"].items() if k not in usage
        }
        report.run["cost_usd_estimate"] = estimate_cost_usd(report.run["usage"])
    return report


def renarrate(report: FinalReport, narrator: str) -> FinalReport:
    text, source, usage = narrate(report.narrator_payload(), narrator)
    report.narration, report.narration_source, report.usage = text, source, usage
    return report


def save(report: FinalReport, out_dir: Path | str) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaderboard": out / "leaderboard.csv",
        "trail": out / "trail.csv",
        "retention": out / "retention.csv",
        "run": out / "run.json",
        "final": out / "final.json",
        "narration": out / "narration.md",
    }
    report.leaderboard.to_csv(paths["leaderboard"], index=False)
    report.trail.to_csv(paths["trail"], index=False)
    report.retention.to_csv(paths["retention"], index=False)
    paths["run"].write_text(json.dumps(report.run, indent=1), encoding="utf-8")
    final = {
        "card": report.card,
        "protocol": PROTOCOL,
        "winner": report.winner.model_dump(),
        "winner_cv": report.winner_cv,
        "holdout": report.holdout,
        "drivers": report.drivers.to_dict(orient="records"),
        "segments": {k: v.to_dict(orient="records") for k, v in report.segments.items()},
        "retention_meta": report.retention_meta,
        "narration_source": report.narration_source,
        "narration_usage": report.usage,
    }
    paths["final"].write_text(json.dumps(final, indent=1), encoding="utf-8")
    paths["narration"].write_text(report.narration + "\n", encoding="utf-8")
    return paths


def load(out_dir: Path | str) -> FinalReport:
    out = Path(out_dir)
    final = json.loads((out / "final.json").read_text(encoding="utf-8"))
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(out / "leaderboard.csv")
    for col in ("hypothesis", "error", "params"):
        if col in leaderboard:
            leaderboard[col] = leaderboard[col].fillna("")
    trail = pd.read_csv(out / "trail.csv")
    retention = pd.read_csv(out / "retention.csv")
    return FinalReport(
        card=final["card"],
        run=run,
        leaderboard=leaderboard,
        trail=trail,
        winner=ExperimentSpec(**final["winner"]),
        winner_cv=final["winner_cv"],
        holdout=final["holdout"],
        drivers=pd.DataFrame(final["drivers"]),
        segments={k: pd.DataFrame(v) for k, v in final["segments"].items()},
        retention=retention,
        retention_meta=final.get("retention_meta", {}),
        narration=(out / "narration.md").read_text(encoding="utf-8").strip(),
        narration_source=final.get("narration_source", ""),
        usage=final.get("narration_usage", {}),
    )


def trail_markdown(report: FinalReport) -> str:
    """The hypothesis trail as a markdown table (for the README)."""
    lines = [
        "| round | planner | experiment | hypothesis | PR-AUC | delta | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in report.trail.iterrows():
        delta = "" if pd.isna(r["delta_vs_incumbent"]) else f"{r['delta_vs_incumbent']:+.4f}"
        pr = "" if pd.isna(r["pr_auc"]) else f"{r['pr_auc']:.4f}"
        lines.append(
            f"| {int(r['round'])} | {r['planner']} | `{r['name']}` | {r['hypothesis']} | {pr} | {delta} | {r['verdict']} |"
        )
    return "\n".join(lines)
