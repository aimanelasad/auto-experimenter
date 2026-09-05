"""Planners propose the next round of experiments from the ledger.

Two planners share one interface:
- rules: a deterministic staged strategy (baselines, features, weighting and
  calibration, then perturbation of the incumbent). Always available.
- claude: an LLM reads the ledger and proposes hypotheses (planner_claude.py).
  Its output is validated and deduplicated here; on any failure the loop
  falls back to the rules planner for that round and records why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .experiments import Ledger
from .models import DEFAULT_PARAMS, INTEGER_PARAMS, MODEL_FAMILIES, PARAM_RANGES
from .spec import ExperimentSpec


class PlannerError(RuntimeError):
    """Raised by a planner when it cannot produce a valid plan."""


@dataclass
class PlanResult:
    planner: str
    rationale: str
    specs: list[ExperimentSpec]
    stop: bool = False
    note: str | None = None
    usage: dict = field(default_factory=dict)


BASELINES: list[tuple[str, str, str]] = [
    (
        "logreg_raw",
        "logreg",
        "Linear baseline sets the floor; the main churn drivers (contract, tenure, charges) "
        "are largely additive, so it should already be competitive.",
    ),
    (
        "rf_raw",
        "random_forest",
        "Bagged trees capture interactions without tuning; expected to beat the linear model "
        "on ROC-AUC, less clearly on PR-AUC.",
    ),
    (
        "hgb_raw",
        "hist_gb",
        "Boosting usually wins on tabular data of this size; expected best PR-AUC of the baselines.",
    ),
    (
        "lgbm_raw",
        "lightgbm",
        "Leaf-wise boosting with the same defaults as hist_gb; checks whether the implementation "
        "matters more than the family.",
    ),
]

REGULARISED_VARIANTS: dict[str, tuple[dict[str, float], str]] = {
    "logreg": ({"C": 0.1}, "Stronger L2 regularisation should reduce variance of the one-hot coefficients."),
    "random_forest": (
        {"min_samples_leaf": 5, "max_depth": 12},
        "Larger leaves and a depth cap should smooth the probabilities and lift PR-AUC.",
    ),
    "hist_gb": (
        {"learning_rate": 0.03, "max_iter": 400, "max_depth": 4, "l2_regularization": 1.0},
        "Slower learning with shallow trees usually generalises better on a 5.6k-row table.",
    ),
    "lightgbm": (
        {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 400, "min_child_samples": 40, "reg_lambda": 1.0},
        "Fewer leaves and larger minimum leaves should reduce overfitting seen in the defaults.",
    ),
}


def dedupe(specs: list[ExperimentSpec], ledger: Ledger) -> tuple[list[ExperimentSpec], list[str]]:
    """Drop specs already in the ledger or repeated within the round; unique names are enforced."""
    seen_keys = ledger.keys()
    seen_names = ledger.names()
    kept: list[ExperimentSpec] = []
    dropped: list[str] = []
    for spec in specs:
        if spec.key() in seen_keys:
            dropped.append(spec.name)
            continue
        base, name, i = spec.name, spec.name, 2
        while name in seen_names:
            name = f"{base}_{i}"
            i += 1
        if name != spec.name:
            spec = spec.model_copy(update={"name": name})
        seen_keys.add(spec.key())
        seen_names.add(spec.name)
        kept.append(spec)
    return kept, dropped


def _family_ranking(ledger: Ledger) -> list[str]:
    best: dict[str, float] = {}
    for r in ledger.ok_results():
        best[r.spec.model] = max(best.get(r.spec.model, -1.0), r.primary)
    ranked = sorted(best, key=best.get, reverse=True)
    return ranked + [m for m in MODEL_FAMILIES if m not in ranked]


def _perturb(spec: ExperimentSpec, rng: np.random.Generator, round_no: int, idx: int) -> ExperimentSpec:
    ranges = PARAM_RANGES[spec.model]
    keys = list(ranges)
    n_change = 1 if len(keys) == 1 else int(rng.integers(1, min(3, len(keys)) + 1))
    chosen = rng.choice(keys, size=n_change, replace=False)
    params = dict(spec.params)
    changes = []
    for key in chosen:
        lo, hi = ranges[key]
        current = params.get(key, DEFAULT_PARAMS[spec.model][key])
        if key == "max_depth" and current == 0:
            new = float(rng.choice([4, 6, 8, 12]))
        else:
            factor = float(rng.choice([0.5, 0.7, 1.5, 2.0]))
            new = current * factor if current != 0 else lo + factor
        new = float(np.clip(new, lo, hi))
        if key in INTEGER_PARAMS:
            new = float(int(round(new)))
        params[key] = new
        changes.append(f"{key} {current:g} -> {new:g}")
    return ExperimentSpec(
        name=f"{spec.model}_r{round_no}_v{idx}",
        model=spec.model,
        feature_set=spec.feature_set,
        params=params,
        class_weight=spec.class_weight,
        calibration=spec.calibration,
        hypothesis=f"Perturb the incumbent ({'; '.join(changes)}) to test whether it sits on a plateau.",
        proposed_by="rules",
        round=round_no,
    )


def plan_rules(round_no: int, ledger: Ledger, per_round: int, seed: int) -> PlanResult:
    """Deterministic staged strategy. Round 1 always runs the four baselines."""
    incumbent = ledger.incumbent()
    if incumbent is None:
        specs = [
            ExperimentSpec(name=name, model=model, feature_set="raw", hypothesis=hyp, round=round_no)
            for name, model, hyp in BASELINES
        ]
        kept, dropped = dedupe(specs, ledger)
        return PlanResult(
            planner="rules",
            rationale="No results yet: run one baseline per model family on the raw features to see "
            "where each family stands before spending compute on features or tuning.",
            specs=kept,
            note=f"dropped duplicates: {dropped}" if dropped else None,
        )

    ranking = _family_ranking(ledger)
    inc = incumbent.spec
    candidates: list[ExperimentSpec] = []
    rationale: str

    if round_no == 2:
        rationale = (
            f"Round 1 ranked the families {', '.join(ranking)} (incumbent {inc.name}, PR-AUC "
            f"{incumbent.primary:.4f}). Next: give the two best families the engineered features, "
            "and check whether a seven-column model keeps most of the signal."
        )
        for model in ranking[:2]:
            candidates.append(
                ExperimentSpec(
                    name=f"{model}_engineered",
                    model=model,
                    feature_set="engineered",
                    hypothesis="Explicit tenure buckets, add-on count and charge ratio encode known churn "
                    f"patterns that {model} otherwise has to learn from one-hot columns.",
                    round=round_no,
                )
            )
        candidates.append(
            ExperimentSpec(
                name=f"{ranking[0]}_minimal",
                model=ranking[0],
                feature_set="minimal",
                hypothesis="If seven columns hold most of the signal, the simpler model is easier to "
                "deploy and explain; expected small PR-AUC loss.",
                round=round_no,
            )
        )
    elif round_no == 3:
        runner_up = next((m for m in ranking if m != inc.model), ranking[0])
        rationale = (
            f"Incumbent is {inc.name} (PR-AUC {incumbent.primary:.4f}); best other family is "
            f"{runner_up}. Next: a regularised variant of the incumbent, a strongly regularised "
            f"{runner_up} on the incumbent's feature set (tests whether the family gap is only "
            "overfitting at default settings), and class weighting on the incumbent."
        )
        params, hyp = REGULARISED_VARIANTS[inc.model]
        candidates.append(
            inc.model_copy(
                update={
                    "name": f"{inc.name}_regularised",
                    "params": {**inc.params, **params},
                    "hypothesis": hyp,
                    "proposed_by": "rules",
                    "round": round_no,
                }
            )
        )
        ru_params, ru_hyp = REGULARISED_VARIANTS[runner_up]
        candidates.append(
            ExperimentSpec(
                name=f"{runner_up}_{inc.feature_set}_regularised",
                model=runner_up,
                feature_set=inc.feature_set,
                params=ru_params,
                hypothesis=f"{ru_hyp} If so, the gap to {inc.model} at default settings was overfitting, "
                "not a family limit.",
                round=round_no,
            )
        )
        candidates.append(
            inc.model_copy(
                update={
                    "name": f"{inc.name}_balanced",
                    "class_weight": "balanced",
                    "hypothesis": "Balanced class weights push the model to rank rare churners higher; "
                    "may raise recall@10% at some cost in calibration.",
                    "proposed_by": "rules",
                    "round": round_no,
                }
            )
        )
    elif round_no == 4:
        rationale = (
            f"Incumbent is {inc.name} (PR-AUC {incumbent.primary:.4f}). The ranking is settled enough "
            "to test probability calibration (expected to change Brier and ECE, not PR-AUC) and to "
            "perturb the incumbent's hyperparameters."
        )
        candidates.append(
            inc.model_copy(
                update={
                    "name": f"{inc.name}_isotonic",
                    "calibration": "isotonic",
                    "hypothesis": "Isotonic calibration should cut Brier and ECE while leaving the ranking "
                    "(PR-AUC) unchanged up to ties.",
                    "proposed_by": "rules",
                    "round": round_no,
                }
            )
        )
        rng = np.random.default_rng(seed * 1000 + round_no)
        attempts = 0
        while len(candidates) < per_round and attempts < 40:
            attempts += 1
            cand = _perturb(inc, rng, round_no, len(candidates))
            if cand.key() in ledger.keys() or any(cand.key() == c.key() for c in candidates):
                continue
            candidates.append(cand)
    else:
        rationale = (
            f"Incumbent is {inc.name} (PR-AUC {incumbent.primary:.4f}). Later rounds perturb its "
            "hyperparameters with seeded random factors to see whether the optimum is flat."
        )
        rng = np.random.default_rng(seed * 1000 + round_no)
        attempts = 0
        while len(candidates) < per_round and attempts < 40:
            attempts += 1
            cand = _perturb(inc, rng, round_no, len(candidates) + 1)
            if cand.key() in ledger.keys() or any(cand.key() == c.key() for c in candidates):
                continue
            candidates.append(cand)

    kept, dropped = dedupe(candidates[: max(per_round, 1)], ledger)
    rounds = ledger.by_round()
    recent = [n for n in sorted(rounds) if n >= round_no - 2]
    recent_confirmed = any(r.verdict == "confirmed" for n in recent for r in rounds[n])
    stop = round_no > 4 and not recent_confirmed
    return PlanResult(
        planner="rules",
        rationale=rationale,
        specs=kept,
        stop=stop,
        note=f"dropped duplicates: {dropped}" if dropped else None,
    )


def get_plan(
    planner: str,
    round_no: int,
    ledger: Ledger,
    per_round: int,
    seed: int,
    dataset_card: dict,
) -> PlanResult:
    """Dispatch to the requested planner; fall back to rules on failure."""
    if planner == "rules":
        return plan_rules(round_no, ledger, per_round, seed)
    if planner == "claude":
        from .planner_claude import plan_claude

        try:
            result = plan_claude(round_no, ledger, per_round, dataset_card)
        except PlannerError as exc:
            fallback = plan_rules(round_no, ledger, per_round, seed)
            fallback.note = f"claude planner unavailable ({exc}); used rules for this round"
            return fallback
        if not result.specs and not result.stop:
            fallback = plan_rules(round_no, ledger, per_round, seed)
            fallback.note = "claude planner returned no new experiments; used rules for this round"
            fallback.usage = result.usage
            return fallback
        return result
    raise ValueError(f"unknown planner {planner!r}; choose 'rules' or 'claude'")
