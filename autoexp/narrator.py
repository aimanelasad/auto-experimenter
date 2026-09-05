"""The narrator: metrics in, a short data-driven conclusion out.

Two implementations with the same input payload: the Claude narrator uses the
prompt in prompts/narrator.md, the template narrator builds the same five
sections from the numbers directly. The template is the fallback and also the
reference for what the model must not contradict.
"""

from __future__ import annotations

import json

from .data import ROOT
from .llm import LLMUnavailable, MODEL, call_text

PROMPT_PATH = ROOT / "prompts" / "narrator.md"


def narrate(payload: dict, narrator: str = "template") -> tuple[str, str, dict]:
    """Return (markdown, source, usage). source is the model id or 'template'."""
    if narrator == "claude":
        try:
            response = call_text(
                PROMPT_PATH.read_text(encoding="utf-8"),
                "Results JSON:\n\n" + json.dumps(payload, indent=1, default=str),
            )
            return response.text, MODEL, response.usage
        except LLMUnavailable as exc:
            text = narrate_template(payload)
            return text + f"\n\n(Template narration: the model was unavailable: {exc}.)", "template", {}
    return narrate_template(payload), "template", {}


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def narrate_template(p: dict) -> str:
    winner = p["winner"]
    cv = p["winner_cv"]
    ho = p["holdout"]
    lb = p["leaderboard"]
    n_exp = p["run"]["n_experiments"]
    n_conf = p["run"]["n_confirmed"]
    baseline = next((r for r in lb if r["verdict"] == "baseline"), None)
    gains = [r for r in lb if r["verdict"] in ("confirmed", "inconclusive")]
    losses = [r for r in lb if r["verdict"] == "no_gain"]
    failed = [r for r in lb if r["verdict"] == "failed"]

    lines = []
    lines.append("### Headline")
    lines.append(
        f"Deploy {winner['name']} ({winner['spec']}): cross-validated PR-AUC {_f(cv['pr_auc'])} "
        f"(fold std {_f(cv['pr_auc_std'])}), holdout PR-AUC {_f(ho['pr_auc'])}, "
        f"lift {_f(ho['lift_at_10'], 2)} in the top 10 %."
    )
    rank1 = p.get("leaderboard_rank_1")
    if rank1 and rank1 != winner["name"]:
        top = lb[0]
        lines.append(
            f"{rank1} has the highest CV mean ({_f(top['pr_auc'])}, {_f(top['delta_vs_incumbent'])} above the "
            f"incumbent) but the gain is inside fold noise, so the established model stays."
        )
    lines.append("")
    lines.append("### What the experiments showed")
    lines.append(
        f"- {n_exp} experiments ran over {p['run']['n_rounds']} rounds; {n_conf} beat the incumbent beyond "
        f"fold noise. Stop reason: {p['run']['stop_reason']}."
    )
    if baseline:
        lines.append(
            f"- The first baseline {baseline['name']} reached PR-AUC {_f(baseline['pr_auc'])}; the winner "
            f"improved on it by {_f(cv['pr_auc'] - baseline['pr_auc'])}."
        )
    for r in gains[:2]:
        lines.append(
            f"- {r['name']} gained {_f(r['delta_vs_incumbent'])} PR-AUC over its incumbent "
            f"({r['verdict']}): {r['hypothesis']}"
        )
    for r in losses[:2]:
        lines.append(
            f"- Refuted: {r['name']} scored {_f(r['pr_auc'])} ({_f(r['delta_vs_incumbent'])} versus its "
            f"incumbent). Hypothesis was: {r['hypothesis']}"
        )
    if failed:
        lines.append(f"- {len(failed)} experiment(s) failed or timed out: {', '.join(r['name'] for r in failed)}.")
    lines.append("")
    lines.append("### Recommendation and caveats")
    gap = cv["pr_auc"] - ho["pr_auc"]
    lines.append(
        f"Use {winner['name']}. The holdout PR-AUC differs from the CV mean by {_f(gap)}, "
        f"{'within' if abs(gap) <= 2 * cv['pr_auc_std'] else 'outside'} two fold standard deviations. "
        f"Holdout Brier score {_f(ho['brier'])} and ECE {_f(ho['ece'])}: "
        f"{'probabilities are usable as-is' if ho['ece'] < 0.05 else 'recalibrate before using the probabilities as risk scores'}. "
        f"Precision in the top 10 % is {_f(ho['precision_at_10'])} against a base rate of "
        f"{_f(p['dataset']['churn_rate_holdout'])}. Limits: {p['dataset']['n_customers']} customers, one "
        "holdout split, and only the options inside the search space were tested."
    )
    lines.append("")
    lines.append("### Next three experiments")
    for s in p["next_suggestions"][:3]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("### Retention actions")
    for r in p["retention"][:4]:
        lines.append(
            f"- {r['segment']}: {r['customers']} customers in the top decile, {r['expected_churners']} expected "
            f"churners (observed rate {_f(r['observed_churn_rate'])}). {r['actions']}."
        )
    return "\n".join(lines)
