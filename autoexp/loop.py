"""The autonomous loop: plan, run, judge, repeat until nothing improves."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .data import Dataset
from .experiments import ExperimentResult, Ledger, judge, run_experiment
from .planner import get_plan
from .spec import ExperimentSpec

Event = dict
EventHandler = Callable[[Event], None]


@dataclass
class LoopConfig:
    rounds: int = 4
    per_round: int = 3
    planner: str = "rules"
    seed: int = 42
    n_folds: int = 5
    n_jobs: int = 2
    time_limit: float = 90.0
    patience: int = 2
    ledger_path: Path | None = None


@dataclass
class RoundLog:
    round_no: int
    planner: str
    rationale: str
    note: str | None
    planned: list[ExperimentSpec]
    results: list[ExperimentResult] = field(default_factory=list)
    stop_requested: bool = False
    seconds: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return any(r.verdict == "confirmed" for r in self.results)


@dataclass
class LoopOutcome:
    rounds: list[RoundLog]
    ledger: Ledger
    stop_reason: str
    config: LoopConfig
    seconds: float

    def best(self) -> ExperimentResult | None:
        return self.ledger.incumbent()

    def trail(self) -> pd.DataFrame:
        """The hypothesis trail: what was tried, in order, and how it turned out."""
        rows = []
        for rl in self.rounds:
            for r in rl.results:
                rows.append(
                    {
                        "round": rl.round_no,
                        "planner": rl.planner,
                        "name": r.spec.name,
                        "spec": r.spec.short(),
                        "hypothesis": r.spec.hypothesis,
                        "pr_auc": r.primary,
                        "delta_vs_incumbent": r.delta_vs_incumbent,
                        "verdict": r.verdict,
                    }
                )
        return pd.DataFrame(rows)

    def total_usage(self) -> dict:
        totals: dict[str, float] = {}
        for rl in self.rounds:
            for k, v in rl.usage.items():
                if isinstance(v, (int, float)):
                    totals[k] = totals.get(k, 0) + v
        return totals


def run_loop(ds: Dataset, config: LoopConfig, on_event: EventHandler | None = None) -> LoopOutcome:
    emit = on_event or (lambda event: None)
    ledger = Ledger(config.ledger_path)
    card = ds.card()
    rounds: list[RoundLog] = []
    t_start = time.perf_counter()
    stop_reason = "reached the round limit"
    rounds_without_gain = 0

    for round_no in range(1, config.rounds + 1):
        t_round = time.perf_counter()
        plan = get_plan(config.planner, round_no, ledger, config.per_round, config.seed, card)
        rl = RoundLog(
            round_no=round_no,
            planner=plan.planner,
            rationale=plan.rationale,
            note=plan.note,
            planned=plan.specs,
            stop_requested=plan.stop,
            usage=plan.usage,
        )
        rounds.append(rl)
        emit({"type": "round_planned", "round": rl})

        if not plan.specs:
            stop_reason = f"planner proposed nothing new in round {round_no}"
            emit({"type": "stopped", "reason": stop_reason})
            break

        for spec in plan.specs:
            incumbent = ledger.incumbent()
            result = run_experiment(
                spec, ds, seed=config.seed, n_folds=config.n_folds, n_jobs=config.n_jobs, time_limit=config.time_limit
            )
            judge(result, incumbent)
            ledger.append(result)
            rl.results.append(result)
            emit({"type": "experiment_done", "round": rl, "result": result, "incumbent": ledger.incumbent()})

        rl.seconds = round(time.perf_counter() - t_round, 1)
        emit({"type": "round_done", "round": rl, "incumbent": ledger.incumbent()})

        if round_no == 1:
            rounds_without_gain = 0
        else:
            rounds_without_gain = 0 if rl.confirmed else rounds_without_gain + 1

        if plan.stop:
            stop_reason = f"planner asked to stop after round {round_no}"
            emit({"type": "stopped", "reason": stop_reason})
            break
        if rounds_without_gain >= config.patience and round_no < config.rounds:
            stop_reason = f"no confirmed gain in {config.patience} consecutive rounds"
            emit({"type": "stopped", "reason": stop_reason})
            break

    return LoopOutcome(
        rounds=rounds,
        ledger=ledger,
        stop_reason=stop_reason,
        config=config,
        seconds=round(time.perf_counter() - t_start, 1),
    )
