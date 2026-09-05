"""Command line entry point.

    python -m autoexp run --planner rules --rounds 5 --out results/rules
    python -m autoexp run --planner claude --narrator claude --out results/claude
    python -m autoexp narrate --results results/claude --narrator claude
    python -m autoexp figures --results results/claude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import load_dataset
from .loop import LoopConfig, run_loop
from .report import finalize, load, renarrate, save


def _print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "round_planned":
        rl = event["round"]
        print(f"\nRound {rl.round_no} [{rl.planner}] {len(rl.planned)} experiments planned")
        print(f"  rationale: {rl.rationale}")
        if rl.note:
            print(f"  note: {rl.note}")
    elif kind == "experiment_done":
        r = event["result"]
        if r.ok:
            print(
                f"  {r.spec.name:32s} PR-AUC {r.primary:.4f} +- {r.primary_std:.4f}  "
                f"ROC-AUC {r.metric('roc_auc'):.4f}  lift@10% {r.metric('lift_at_10'):.2f}  "
                f"{r.fit_seconds:5.1f}s  -> {r.verdict}"
                + (f" ({r.delta_vs_incumbent:+.4f})" if r.delta_vs_incumbent is not None else "")
            )
        else:
            print(f"  {r.spec.name:32s} FAILED: {r.error}")
    elif kind == "stopped":
        print(f"\nStopped: {event['reason']}")


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    ds = load_dataset(seed=args.seed)
    config = LoopConfig(
        rounds=args.rounds,
        per_round=args.per_round,
        planner=args.planner,
        seed=args.seed,
        n_folds=args.folds,
        n_jobs=args.n_jobs,
        time_limit=args.time_limit,
        patience=args.patience,
        ledger_path=ledger_path,
    )
    print(f"Dataset: {ds.card()['n_customers']} customers, churn rate {ds.card()['churn_rate']:.3f}; planner={args.planner}")
    outcome = run_loop(ds, config, _print_event)
    print(f"\nLoop finished in {outcome.seconds}s: {outcome.stop_reason}; {len(outcome.ledger)} experiments")
    report = finalize(outcome, ds, narrator=args.narrator, seed=args.seed, n_jobs=args.n_jobs)
    paths = save(report, out)
    if args.figures:
        from .figures import save_figures

        save_figures(report, out / "figures")
    print(f"\nWinner: {report.winner.name} = {report.winner.short()}")
    print(f"  CV PR-AUC {report.winner_cv['pr_auc']:.4f} +- {report.winner_cv['pr_auc_std']:.4f}; holdout PR-AUC "
          f"{report.holdout['pr_auc']:.4f}, ROC-AUC {report.holdout['roc_auc']:.4f}, lift@10% {report.holdout['lift_at_10']:.2f}")
    print(f"Narration by {report.narration_source}; usage {report.run.get('usage')} "
          f"(about ${report.run.get('cost_usd_estimate', 0):.3f})")
    print("Written:", ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_narrate(args: argparse.Namespace) -> int:
    report = load(args.results)
    report = renarrate(report, args.narrator)
    save(report, args.results)
    print(report.narration)
    print(f"\n(narrated by {report.narration_source}; usage {report.usage})")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from .figures import save_figures

    report = load(args.results)
    paths = save_figures(report, Path(args.results) / "figures")
    print("Written:", ", ".join(str(p) for p in paths))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoexp", description="Autonomous experiment loop for churn models.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the loop, finalize and write results")
    run.add_argument("--planner", choices=["rules", "claude"], default="rules")
    run.add_argument("--narrator", choices=["template", "claude"], default="template")
    run.add_argument("--rounds", type=int, default=5)
    run.add_argument("--per-round", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--folds", type=int, default=5)
    run.add_argument("--n-jobs", type=int, default=2)
    run.add_argument("--time-limit", type=float, default=90.0, help="seconds per experiment")
    run.add_argument("--patience", type=int, default=2, help="rounds without a confirmed gain before stopping")
    run.add_argument("--out", default="results/latest")
    run.add_argument("--no-figures", dest="figures", action="store_false")
    run.set_defaults(func=cmd_run)

    narrate = sub.add_parser("narrate", help="re-narrate saved results")
    narrate.add_argument("--results", default="results/latest")
    narrate.add_argument("--narrator", choices=["template", "claude"], default="claude")
    narrate.set_defaults(func=cmd_narrate)

    figures = sub.add_parser("figures", help="render PNG figures from saved results")
    figures.add_argument("--results", default="results/latest")
    figures.set_defaults(func=cmd_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
