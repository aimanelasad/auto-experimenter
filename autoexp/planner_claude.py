"""The Claude planner: reads the ledger, proposes hypotheses as structured output.

The model proposes; the code validates. Every proposal passes through the same
pydantic ranges as the rules planner, duplicates are dropped, and any failure
raises PlannerError so the loop can fall back to rules for that round.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .data import ROOT
from .experiments import Ledger
from .features import FEATURE_SET_DESCRIPTIONS
from .llm import LLMUnavailable, call_structured
from .models import search_space
from .planner import PlannerError, PlanResult, dedupe
from .spec import PlannerOutput

PROMPT_PATH = ROOT / "prompts" / "planner.md"


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def user_prompt(round_no: int, ledger: Ledger, per_round: int, card: dict) -> str:
    incumbent = ledger.incumbent()
    if incumbent is None:
        ledger_block = "The ledger is empty: this is round 1."
        incumbent_block = "No incumbent yet."
    else:
        ledger_block = ledger.planner_table()
        incumbent_block = (
            f"Incumbent: {incumbent.spec.name} = {incumbent.spec.short()} with PR-AUC "
            f"{incumbent.primary:.4f} +- {incumbent.primary_std:.4f} (fold values "
            f"{', '.join(f'{v:.4f}' for v in incumbent.fold_primary())})."
        )
    summary = ledger.summary()
    return "\n\n".join(
        [
            f"Round {round_no}. Propose {per_round} experiments (or fewer with stop=true).",
            "Dataset card:\n" + json.dumps(card, indent=1),
            "Feature sets:\n" + "\n".join(f"- {k}: {v}" for k, v in FEATURE_SET_DESCRIPTIONS.items()),
            "Search space (ranges are inclusive, 'integer' parameters must be whole numbers):\n"
            + json.dumps(search_space(), indent=1),
            f"Ledger so far ({summary['n_experiments']} experiments, {summary['n_confirmed']} confirmed gains), "
            "best first:\n" + ledger_block,
            incumbent_block,
        ]
    )


def plan_claude(round_no: int, ledger: Ledger, per_round: int, card: dict) -> PlanResult:
    try:
        response = call_structured(
            system_prompt(), user_prompt(round_no, ledger, per_round, card), PlannerOutput, effort="medium"
        )
    except LLMUnavailable as exc:
        raise PlannerError(str(exc)) from exc
    output: PlannerOutput = response.parsed

    specs, rejected = [], []
    for planned in output.experiments:
        try:
            specs.append(planned.to_spec(round_no))
        except ValidationError as exc:
            rejected.append(f"{planned.name}: {exc.errors()[0].get('msg', 'invalid')}")
    kept, dropped = dedupe(specs, ledger)
    kept = kept[: max(per_round, 1)]

    notes = []
    if rejected:
        notes.append(f"rejected out-of-range proposals: {rejected}")
    if dropped:
        notes.append(f"dropped duplicates of earlier experiments: {dropped}")
    return PlanResult(
        planner="claude",
        rationale=output.rationale.strip(),
        specs=kept,
        stop=output.stop,
        note="; ".join(notes) if notes else None,
        usage=response.usage,
    )
