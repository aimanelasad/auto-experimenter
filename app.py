"""Streamlit front end for auto-experimenter.

Shows a saved run instantly and can run a fresh loop live (rules planner always,
Claude planner and narrator when an API key is configured in the environment).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from autoexp.data import KAGGLE_URL, load_dataset
from autoexp.experiments import VERDICT_LABELS
from autoexp.llm import DAILY_CALL_CAP, MODEL, api_key_present, calls_today
from autoexp.loop import LoopConfig, run_loop
from autoexp.metrics import METRIC_INFO, METRIC_LABELS
from autoexp.models import MODEL_LABELS
from autoexp.report import PROTOCOL, FinalReport, finalize, load

ROOT = Path(__file__).resolve().parent
REPO_URL = os.environ.get("AUTOEXP_REPO_URL", "https://github.com/aimanelasad/auto-experimenter")
FAMILY_COLORS = {"logreg": "#2a78d6", "random_forest": "#eb6834", "hist_gb": "#1baf7a", "lightgbm": "#eda100"}
SAVED_RUNS = {
    "Claude planner and narrator": ROOT / "results" / "claude",
    "Rules planner, template narrator": ROOT / "results" / "rules",
}
PER_ROUND = 3
SEED = 42
MAX_CLAUDE_RUNS_PER_SESSION = 1

st.set_page_config(page_title="auto-experimenter", layout="wide")


@st.cache_resource(show_spinner=False)
def dataset():
    return load_dataset()


@st.cache_data(show_spinner=False)
def saved_report(path: str, stamp: float) -> FinalReport:
    """stamp is the results file's mtime, so regenerated results are picked up."""
    return load(path)


def fmt(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def leaderboard_chart(lb: pd.DataFrame, random_pr_auc: float) -> go.Figure:
    d = lb.dropna(subset=["pr_auc"]).sort_values("pr_auc")
    fig = go.Figure()
    for model in FAMILY_COLORS:
        part = d[d["model"] == model]
        if part.empty:
            continue
        fig.add_bar(
            y=part["name"],
            x=part["pr_auc"],
            orientation="h",
            name=MODEL_LABELS[model],
            marker_color=FAMILY_COLORS[model],
            error_x={"type": "data", "array": part["pr_auc_std"], "color": "#6b6b66", "thickness": 1},
            customdata=part[["verdict", "round", "roc_auc", "lift_at_10", "hypothesis", "proposed_by"]].values,
            hovertemplate="<b>%{y}</b><br>PR-AUC %{x:.4f}<br>ROC-AUC %{customdata[2]:.4f} | lift@10%% "
            "%{customdata[3]:.2f}<br>round %{customdata[1]} (%{customdata[5]}) | %{customdata[0]}"
            "<br><i>%{customdata[4]}</i><extra></extra>",
        )
    lo = max(0.0, d["pr_auc"].min() - 0.05)
    hi = min(1.0, d["pr_auc"].max() + 0.03)
    fig.update_layout(
        barmode="overlay",
        height=max(320, 28 * len(d) + 130),
        margin={"l": 10, "r": 10, "t": 30, "b": 40},
        title={"text": f"PR-AUC per experiment (5-fold CV mean, error bar = fold std); random ranking scores {random_pr_auc:.3f}",
               "font": {"size": 13}},
        xaxis={"title": "PR-AUC", "range": [lo, hi]},
        yaxis={"categoryorder": "array", "categoryarray": list(d["name"])},
        legend={"orientation": "h", "y": -0.16},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def progress_chart(trail: pd.DataFrame, lb: pd.DataFrame) -> go.Figure:
    t = trail.reset_index(drop=True)
    idx = list(range(1, len(t) + 1))
    model_of = lb.set_index("name")["model"]
    std_of = lb.set_index("name")["pr_auc_std"]
    best = t["pr_auc"].cummax()
    best_name = []
    current = None
    for _, r in t.iterrows():
        if current is None or (pd.notna(r["pr_auc"]) and r["pr_auc"] >= best.loc[_]):
            current = r["name"]
        best_name.append(current)
    band = [std_of.get(n, 0.0) for n in best_name]
    fig = go.Figure()
    fig.add_scatter(x=idx + idx[::-1], y=list(best + pd.Series(band)) + list((best - pd.Series(band))[::-1]),
                    fill="toself", fillcolor="rgba(31,31,30,0.08)", line={"width": 0}, hoverinfo="skip",
                    name="fold std of the best mean")
    fig.add_scatter(x=idx, y=best, mode="lines", line={"shape": "hv", "color": "#1f1f1e", "width": 2},
                    name="best CV mean so far", hoverinfo="skip")
    for model in FAMILY_COLORS:
        mask = [model_of.get(n) == model for n in t["name"]]
        if not any(mask):
            continue
        part = t[mask]
        fig.add_scatter(
            x=[i for i, m in zip(idx, mask) if m],
            y=part["pr_auc"],
            mode="markers",
            marker={"color": FAMILY_COLORS[model], "size": 10, "line": {"color": "#fcfcfb", "width": 1}},
            name=MODEL_LABELS[model],
            customdata=part[["name", "verdict", "round"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>PR-AUC %{y:.4f}<br>round %{customdata[2]} | %{customdata[1]}<extra></extra>",
        )
    rounds = t["round"].tolist()
    for i in range(1, len(rounds)):
        if rounds[i] != rounds[i - 1]:
            fig.add_vline(x=i + 0.5, line={"color": "#e6e6e2", "dash": "dash"})
    fig.update_layout(
        height=max(320, 28 * len(t) + 130) if len(t) > 12 else 360,
        margin={"l": 10, "r": 10, "t": 30, "b": 40},
        title={"text": "Best CV mean so far; band = its fold std", "font": {"size": 13}},
        xaxis={"title": "experiment number in run order", "dtick": 1},
        yaxis={"title": "PR-AUC (CV mean)"},
        legend={"orientation": "h", "y": -0.25},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def drivers_chart(drivers: pd.DataFrame, model: str, top: int = 10) -> go.Figure:
    d = drivers.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            y=d["feature"],
            x=d["importance"],
            orientation="h",
            marker_color=FAMILY_COLORS.get(model, "#2a78d6"),
            error_x={"type": "data", "array": d["std"], "color": "#6b6b66", "thickness": 1},
            hovertemplate="%{y}: PR-AUC drop %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=30 * len(d) + 100,
        margin={"l": 10, "r": 10, "t": 10, "b": 40},
        xaxis={"title": "PR-AUC drop on the holdout when the column is permuted (5 repeats)"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def run_live(planner: str, narrator: str, rounds: int) -> FinalReport:
    ds = dataset()
    config = LoopConfig(rounds=rounds, per_round=PER_ROUND, planner=planner, seed=SEED, n_jobs=2, time_limit=90.0)
    with st.status("Running the autonomous loop", expanded=True) as status:
        box = st.container()

        def on_event(event: dict) -> None:
            kind = event["type"]
            if kind == "round_planned":
                rl = event["round"]
                box.markdown(f"**Round {rl.round_no}** ({rl.planner} planner): {rl.rationale}")
                if rl.note:
                    box.caption(rl.note)
            elif kind == "experiment_done":
                r = event["result"]
                if r.ok:
                    delta = f" ({r.delta_vs_incumbent:+.4f})" if r.delta_vs_incumbent is not None else ""
                    box.markdown(f"- `{r.spec.name}`: PR-AUC {r.primary:.4f} +- {r.primary_std:.4f} in {r.fit_seconds:.1f} s, **{r.verdict}**{delta}")
                else:
                    box.markdown(f"- `{r.spec.name}` failed: {r.error}")
            elif kind == "stopped":
                box.markdown(f"Stopped: {event['reason']}")

        outcome = run_loop(ds, config, on_event)
        status.update(label="Scoring the winner on the holdout and writing the conclusion", state="running")
        report = finalize(outcome, ds, narrator=narrator, seed=SEED, n_jobs=2)
        status.update(label=f"Done in {outcome.seconds:.0f} s: {outcome.stop_reason}", state="complete", expanded=False)
    return report


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Run a new loop")
    key_ok = api_key_present()
    claude_runs = st.session_state.get("claude_runs", 0)
    claude_allowed = key_ok and claude_runs < MAX_CLAUDE_RUNS_PER_SESSION
    planner = st.radio(
        "Planner",
        ["rules", "claude"],
        format_func=lambda p: "Rules (deterministic)" if p == "rules" else f"Claude ({MODEL})",
        disabled=not claude_allowed,
        help="The Claude planner needs an ANTHROPIC_API_KEY in the environment of this app and is limited to one run per session.",
    )
    narrator = st.radio(
        "Narrator",
        ["template", "claude"],
        format_func=lambda p: "Template (deterministic)" if p == "template" else f"Claude ({MODEL})",
        disabled=not claude_allowed,
    )
    if not claude_allowed:
        planner, narrator = "rules", "template"
        if not key_ok:
            st.caption("No API key configured here: a live run uses the rules planner and the template narrator. "
                       "The saved runs below include the committed results.")
        else:
            st.caption("One Claude run per session; further live runs use the rules planner.")
    rounds = st.slider("Rounds", 1, 5, 5)
    uses_claude = planner == "claude" or narrator == "claude"
    if uses_claude:
        st.caption(f"Model calls today: {calls_today()} of {DAILY_CALL_CAP}. This run makes at most {rounds + 1} calls, "
                   "roughly 5 to 15 US cents in total.")
    duration = "about 20 s" if not uses_claude else "2 to 4 minutes"
    go_live = st.button(f"Run the loop now ({duration})", type="primary", width="stretch")
    st.caption(f"{PER_ROUND} experiments per round, seed {SEED}, same protocol as the saved runs.")
    st.divider()
    available = {k: v for k, v in SAVED_RUNS.items() if (v / "final.json").exists()}
    choice = st.selectbox("Saved run", list(available) or ["none"], disabled=not available)
    st.caption("Saved runs were produced by the CLI and committed with the repository.")

# ----------------------------------------------------------------------------- header
st.title("auto-experimenter")
st.markdown(
    "An autonomous loop for churn models. A planner reads the ledger of everything tried so far and proposes the "
    "next experiments as testable hypotheses; the runner scores them under one fixed cross-validation protocol; "
    "a judge marks each hypothesis confirmed or refuted against the incumbent; a narrator turns the numbers into "
    "a conclusion with next steps and retention actions."
)

if st.query_params.get("autorun") == "rules" and "live_report" not in st.session_state and not st.session_state.get("autorun_done"):
    st.session_state["autorun_done"] = True
    go_live, planner, narrator, uses_claude = True, "rules", "template", False

if go_live:
    if uses_claude:
        st.session_state["claude_runs"] = claude_runs + 1
    st.session_state["live_report"] = run_live(planner, narrator, rounds)
    st.session_state["live_label"] = f"live run ({planner} planner, {narrator} narrator)"

if "live_report" in st.session_state:
    report: FinalReport = st.session_state["live_report"]
    source_label = st.session_state["live_label"]
    if st.button("Back to the saved run"):
        del st.session_state["live_report"]
        st.rerun()
elif available:
    report = saved_report(str(available[choice]), (available[choice] / "final.json").stat().st_mtime)
    source_label = f"saved run: {choice}"
else:
    st.warning("No saved results found. Run `python -m autoexp run` first or start a live run from the sidebar.")
    st.stop()

card = report.card
random_pr_auc = card["churn_rate_train"]

# ----------------------------------------------------------------------------- headline
st.caption(f"Showing {source_label}. Planner: {report.run['config']['planner']}; narrator: {report.narration_source}.")
st.markdown(f"**Winner: `{report.winner.name}`**, {report.winner.short()}")
top_row = report.leaderboard.iloc[0] if len(report.leaderboard) else None
if top_row is not None and top_row["name"] != report.winner.name:
    st.caption(
        f"Leaderboard rank 1 is `{top_row['name']}` with CV PR-AUC {fmt(top_row['pr_auc'])}, "
        f"{top_row['pr_auc'] - report.winner_cv['pr_auc']:+.4f} above the winner: inside fold noise, so the "
        "established model was kept. The incumbent only changes on a confirmed verdict."
    )
m1, m2, m3, m4 = st.columns(4)
m1.metric("CV PR-AUC", fmt(report.winner_cv["pr_auc"]))
m1.caption(f"fold std {fmt(report.winner_cv['pr_auc_std'])}; ROC-AUC {fmt(report.winner_cv['roc_auc'])}; "
           f"random ranking {fmt(random_pr_auc)}")
m2.metric("Holdout PR-AUC", fmt(report.holdout["pr_auc"]))
m2.caption(f"{report.holdout['pr_auc'] - report.winner_cv['pr_auc']:+.3f} versus the CV mean; ROC-AUC {fmt(report.holdout['roc_auc'])}")
m3.metric("Holdout lift in the top 10 %", fmt(report.holdout["lift_at_10"], 2))
m3.caption(f"precision {fmt(report.holdout['precision_at_10'])} against a base rate of {fmt(card['churn_rate_holdout'])}")
m4.metric("Experiments", f"{report.run['n_experiments']}")
m4.caption(f"{report.run['n_rounds']} rounds, {report.run['n_confirmed']} confirmed gains, {report.run['seconds']:.0f} s")

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(leaderboard_chart(report.leaderboard, random_pr_auc), width="stretch")
with right:
    st.plotly_chart(progress_chart(report.trail, report.leaderboard), width="stretch")

# ----------------------------------------------------------------------------- narration
st.subheader("Conclusion")
st.caption(f"Written by {report.narration_source} from the numbers below.")
st.markdown("\n".join(f"**{line[4:].strip()}**" if line.startswith("### ") else line for line in report.narration.splitlines()))

# ----------------------------------------------------------------------------- trail
st.subheader("Hypothesis trail")
st.caption(f"Stop reason: {report.run['stop_reason']}. Verdicts: " + "; ".join(f"{k} = {v}" for k, v in VERDICT_LABELS.items()))
for rd in report.run["rounds"]:
    n_conf = sum(1 for r in rd["results"] if r["verdict"] == "confirmed")
    with st.expander(f"Round {rd['round_no']} ({rd['planner']} planner): {len(rd['results'])} experiments, {n_conf} confirmed", expanded=rd["round_no"] <= 2):
        st.markdown(f"**Planner rationale:** {rd['rationale']}")
        if rd.get("note"):
            st.caption(f"Note: {rd['note']}")
        part = report.trail[report.trail["round"] == rd["round_no"]][["name", "hypothesis", "pr_auc", "delta_vs_incumbent", "verdict"]]
        st.dataframe(
            part,
            hide_index=True,
            width="stretch",
            column_config={
                "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
                "delta_vs_incumbent": st.column_config.NumberColumn("delta vs incumbent", format="%+.4f"),
                "hypothesis": st.column_config.TextColumn("hypothesis", width="large"),
            },
        )

# ----------------------------------------------------------------------------- leaderboard table
st.subheader("Leaderboard")
cols = ["rank", "name", "model", "feature_set", "params", "class_weight", "calibration", "round", "proposed_by",
        "pr_auc", "pr_auc_std", "roc_auc", "lift_at_10", "recall_at_10", "brier", "ece", "fit_seconds", "verdict"]
st.dataframe(
    report.leaderboard[cols],
    hide_index=True,
    width="stretch",
    column_config={k: st.column_config.NumberColumn(METRIC_LABELS.get(k, k), format="%.4f")
                   for k in ("pr_auc", "roc_auc", "recall_at_10", "brier", "ece")}
    | {"pr_auc_std": st.column_config.NumberColumn("fold std", format="%.4f"),
       "lift_at_10": st.column_config.NumberColumn("Lift@10%", format="%.2f"),
       "fit_seconds": st.column_config.NumberColumn("fit s", format="%.1f")},
)

# ----------------------------------------------------------------------------- winner
st.subheader(f"Winner on the holdout: {report.winner.name}")
w1, w2 = st.columns([2, 3])
with w1:
    rows = [{"metric": label, "CV mean": report.winner_cv.get(key), "holdout": report.holdout.get(key)} for key, label, _ in METRIC_INFO]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"CV mean": st.column_config.NumberColumn(format="%.4f"), "holdout": st.column_config.NumberColumn(format="%.4f")})
    st.caption("The holdout was scored once, after the loop finished. Brier and ECE: lower is better.")
with w2:
    st.plotly_chart(drivers_chart(report.drivers, report.winner.model), width="stretch")

# ----------------------------------------------------------------------------- retention
st.subheader("Retention actions for the top decile")
meta = report.retention_meta
st.caption(
    f"Top 10 % of the holdout by predicted risk = {meta.get('top_decile_size', 'n/a')} customers, observed churn rate "
    f"{fmt(meta.get('top_decile_observed_churn_rate'))} against a base rate of {fmt(meta.get('base_churn_rate'))}. "
    "Groups with fewer than 12 customers are pooled in the last row."
)
st.dataframe(
    report.retention,
    hide_index=True,
    width="stretch",
    column_config={
        "share_of_top_decile": st.column_config.NumberColumn("share of top decile", format="%.0f%%"),
        "mean_churn_prob": st.column_config.NumberColumn("mean churn prob", format="%.3f"),
        "observed_churn_rate": st.column_config.NumberColumn("observed churn rate", format="%.3f"),
        "electronic_check_share": st.column_config.NumberColumn("e-check share", format="%.2f"),
        "no_support_share": st.column_config.NumberColumn("no support share", format="%.2f"),
        "new_customer_share": st.column_config.NumberColumn("new customer share", format="%.2f"),
        "actions": st.column_config.TextColumn("actions", width="large"),
    },
)

# ----------------------------------------------------------------------------- details
with st.expander("How the loop works, the dataset and the protocol"):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Plan**  \nrules or Claude proposes experiments from the ledger")
    c2.markdown("**2. Run**  \n5-fold CV, identical folds, preprocessing inside the fold")
    c3.markdown("**3. Judge**  \nfold-paired comparison against the incumbent")
    c4.markdown("**4. Narrate**  \nmetrics in, conclusion and actions out")
    st.markdown(f"**Protocol.** {PROTOCOL}")
    st.markdown(
        f"**Dataset.** {card['name']}: {card['n_customers']:,} customers, {card['n_features']} columns "
        f"({card['n_categorical']} categorical, {card['n_numeric']} numeric), churn rate {card['churn_rate']:.1%}; "
        f"train {card['n_train']:,} / holdout {card['n_holdout']:,}. Random ranking scores PR-AUC {random_pr_auc:.3f} "
        "(the churn rate); published models on this dataset reach ROC-AUC around 0.84 to 0.85. "
        f"[Kaggle page]({KAGGLE_URL})."
    )
    seg_cols = st.columns(len(report.segments))
    for col, (dim, df) in zip(seg_cols, report.segments.items()):
        col.markdown(f"**Churn rate by {dim}**")
        col.dataframe(df[["segment", "customers", "churn_rate"]], hide_index=True, width="stretch",
                      column_config={"churn_rate": st.column_config.NumberColumn(format="%.3f")})

st.divider()
st.markdown(f"Code, design notes and results: [{REPO_URL}]({REPO_URL}). Dataset: [IBM Telco Customer Churn on Kaggle]({KAGGLE_URL}).")
