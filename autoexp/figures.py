"""Static PNG figures for the README, rendered with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .models import MODEL_LABELS  # noqa: E402
from .report import FinalReport  # noqa: E402

# Fixed hue per model family; never reassigned when a family is missing.
FAMILY_COLORS = {
    "logreg": "#2a78d6",
    "random_forest": "#eb6834",
    "hist_gb": "#1baf7a",
    "lightgbm": "#eda100",
}
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_MUTED = "#6b6b66"
GRID = "#e6e6e2"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def leaderboard_figure(report: FinalReport, path: Path) -> Path:
    lb = report.leaderboard.dropna(subset=["pr_auc"]).sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(lb) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    colors = [FAMILY_COLORS[m] for m in lb["model"]]
    ax.barh(lb["name"], lb["pr_auc"], color=colors, height=0.55, xerr=lb["pr_auc_std"],
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
    xmin = max(0.0, lb["pr_auc"].min() - 0.05)
    ax.set_xlim(xmin, min(1.0, lb["pr_auc"].max() + 0.04))
    for y, (_, r) in enumerate(lb.iterrows()):
        ax.text(r["pr_auc"] + r["pr_auc_std"] + 0.002, y, f"{r['pr_auc']:.3f}  {r['verdict']}",
                va="center", fontsize=8, color=INK)
    ax.set_xlabel("PR-AUC, 5-fold CV mean with fold standard deviation")
    ax.set_title("Every experiment, best at the top", loc="left", fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for m, c in FAMILY_COLORS.items() if m in set(lb["model"])]
    labels = [MODEL_LABELS[m] for m in FAMILY_COLORS if m in set(lb["model"])]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def progress_figure(report: FinalReport, path: Path) -> Path:
    trail = report.trail.reset_index(drop=True)
    idx = list(range(1, len(trail) + 1))
    best = trail["pr_auc"].cummax()
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    lb = report.leaderboard.set_index("name")
    colors = [FAMILY_COLORS[lb.loc[n, "model"]] for n in trail["name"]]
    band = []
    current = None
    for i, (_, r) in enumerate(trail.iterrows()):
        if current is None or (pd.notna(r["pr_auc"]) and r["pr_auc"] >= best.iloc[i]):
            current = r["name"]
        band.append(float(lb.loc[current, "pr_auc_std"]))
    band = pd.Series(band, index=best.index)
    ax.fill_between(idx, best - band, best + band, step="post", color=INK, alpha=0.07, linewidth=0,
                    label="fold std of the best mean")
    ax.step(idx, best, where="post", color=INK, linewidth=2, label="best CV mean so far")
    ax.scatter(idx, trail["pr_auc"], c=colors, s=42, zorder=3, edgecolor=SURFACE, linewidth=1)
    rounds = trail["round"].tolist()
    for i in range(1, len(rounds)):
        if rounds[i] != rounds[i - 1]:
            ax.axvline(i + 0.5, color=GRID, linewidth=1, linestyle="--")
    for rnd in sorted(set(rounds)):
        first = rounds.index(rnd) + 1
        ax.text(first, ax.get_ylim()[1] if False else trail["pr_auc"].max() + 0.012, f"round {rnd}",
                fontsize=8, color=INK_MUTED, ha="left")
    for i, (_, r) in enumerate(trail.iterrows()):
        if r["verdict"] in ("confirmed", "baseline"):
            ax.annotate(r["name"], (idx[i], r["pr_auc"]), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7, color=INK)
    ax.set_xticks(idx)
    ax.set_xlabel("experiment number (in the order the loop ran them)")
    ax.set_ylabel("PR-AUC (CV mean)")
    ax.set_ylim(trail["pr_auc"].min() - 0.02, trail["pr_auc"].max() + 0.025)
    ax.set_title("Best CV mean over the run; dots are single experiments coloured by model family",
                 loc="left", fontsize=11, color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def drivers_figure(report: FinalReport, path: Path, top: int = 10) -> Path:
    d = report.drivers.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(d) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.barh(d["feature"], d["importance"], color=FAMILY_COLORS[report.winner.model], height=0.55,
            xerr=d["std"], error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_xlabel("PR-AUC drop on the holdout when the column is permuted (mean of 5 repeats)")
    ax.set_title(f"What drives the winner ({report.winner.name})", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def save_figures(report: FinalReport, out_dir: Path | str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        leaderboard_figure(report, out / "leaderboard.png"),
        progress_figure(report, out / "progress.png"),
        drivers_figure(report, out / "drivers.png"),
    ]
