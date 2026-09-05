import json

import numpy as np
import pandas as pd
import pytest

from autoexp.data import FEATURE_COLUMNS, load_dataset
from autoexp.experiments import ExperimentResult, Ledger, judge, run_experiment
from autoexp.features import FEATURE_SETS, encoded_feature_names, make_preprocessor
from autoexp.loop import LoopConfig, run_loop
from autoexp.metrics import compute_metrics
from autoexp.models import PARAM_RANGES
from autoexp.planner import dedupe, plan_rules
from autoexp.report import finalize, load, save
from autoexp.spec import ExperimentSpec, PlannedExperiment, Params, PlannerOutput


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_dataset_is_clean_and_split(ds):
    assert list(ds.X_train.columns) == FEATURE_COLUMNS
    assert ds.X_train.isna().sum().sum() == 0 and ds.X_holdout.isna().sum().sum() == 0
    assert len(ds.X_train) + len(ds.X_holdout) == 7043
    assert abs(ds.y_train.mean() - ds.y_holdout.mean()) < 0.01
    assert set(ds.y_train.unique()) == {0, 1}


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_preprocessor_names_match_columns(ds, feature_set):
    prep = make_preprocessor(feature_set, scale_numeric=True)
    Xt = prep.fit_transform(ds.X_train.head(300))
    assert Xt.shape[1] == len(encoded_feature_names(prep))


def test_metrics_perfect_and_bounded():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    m = compute_metrics(y, y.astype(float))
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    rng = np.random.default_rng(1)
    m = compute_metrics(rng.integers(0, 2, 500), rng.random(500))
    assert all(0.0 <= v <= 1.0 for k, v in m.items() if k != "lift_at_10")


def test_spec_fills_defaults_and_rejects_out_of_range():
    spec = ExperimentSpec(name="a", model="lightgbm", feature_set="raw", params={"num_leaves": 15.4})
    assert spec.params["num_leaves"] == 15 and "learning_rate" in spec.params
    with pytest.raises(ValueError):
        ExperimentSpec(name="b", model="lightgbm", feature_set="raw", params={"num_leaves": 999})
    with pytest.raises(ValueError):
        ExperimentSpec(name="c", model="logreg", feature_set="raw", params={"num_leaves": 8})
    same = ExperimentSpec(name="other_name", model="lightgbm", feature_set="raw", params={"num_leaves": 15})
    assert same.key() == spec.key()


def test_planned_experiment_converts_to_spec():
    planned = PlannedExperiment(
        name="hgb_shallow",
        model="hist_gb",
        feature_set="engineered",
        params=Params(C=None, n_estimators=None, max_depth=3, min_samples_leaf=None, learning_rate=0.05,
                      max_iter=300, l2_regularization=None, num_leaves=None, min_child_samples=None,
                      reg_lambda=None, colsample_bytree=None),
        class_weight="none",
        calibration="none",
        hypothesis="shallow trees generalise better",
    )
    spec = planned.to_spec(2)
    assert spec.params["max_depth"] == 3 and spec.params["max_iter"] == 300 and spec.proposed_by == "claude"
    schema = PlannerOutput.model_json_schema()
    assert set(schema["required"]) == {"rationale", "experiments", "stop"}


def test_judge_uses_paired_folds():
    base = ExperimentSpec(name="base", model="logreg", feature_set="raw")
    inc = ExperimentResult(spec=base, cv={"pr_auc": {"mean": 0.60, "std": 0.01}},
                           folds=[{"pr_auc": v} for v in (0.59, 0.60, 0.61, 0.60, 0.60)],
                           fit_seconds=1, n_folds=5, seed=0, timestamp="")
    better = ExperimentResult(spec=base.model_copy(update={"name": "better"}),
                              cv={"pr_auc": {"mean": 0.61, "std": 0.01}},
                              folds=[{"pr_auc": v + 0.01} for v in (0.59, 0.60, 0.61, 0.60, 0.60)],
                              fit_seconds=1, n_folds=5, seed=0, timestamp="")
    noisy = ExperimentResult(spec=base.model_copy(update={"name": "noisy"}),
                             cv={"pr_auc": {"mean": 0.602, "std": 0.03}},
                             folds=[{"pr_auc": v} for v in (0.65, 0.55, 0.64, 0.56, 0.61)],
                             fit_seconds=1, n_folds=5, seed=0, timestamp="")
    assert judge(better, inc).verdict == "confirmed"
    assert judge(noisy, inc).verdict == "inconclusive"
    assert judge(inc, better).verdict == "no_gain"
    assert judge(inc, None).verdict == "baseline"


def test_run_experiment_returns_metrics(ds):
    spec = ExperimentSpec(name="quick", model="logreg", feature_set="minimal", params={"C": 0.5})
    result = run_experiment(spec, ds, n_folds=3)
    assert result.ok and 0.5 < result.primary < 0.8 and len(result.folds) == 3
    failing = ExperimentSpec(name="slow", model="random_forest", feature_set="raw", params={"n_estimators": 800})
    result = run_experiment(failing, ds, n_folds=3, time_limit=0.0)
    assert not result.ok and result.verdict == "failed" and "limit" in result.error


def test_rules_planner_never_repeats_and_stays_in_range(ds):
    ledger = Ledger()
    plan = plan_rules(1, ledger, per_round=3, seed=1)
    assert [s.model for s in plan.specs] == ["logreg", "random_forest", "hist_gb", "lightgbm"]
    for spec in plan.specs:
        result = run_experiment(spec.model_copy(update={"params": {}}), ds, n_folds=2, time_limit=60)
        ledger.append(judge(result, ledger.incumbent()))
    for round_no in (2, 3, 4, 5):
        plan = plan_rules(round_no, ledger, per_round=3, seed=1)
        assert plan.specs, round_no
        for spec in plan.specs:
            assert spec.key() not in ledger.keys()
            for key, value in spec.params.items():
                lo, hi = PARAM_RANGES[spec.model][key]
                assert lo <= value <= hi
            # Record a synthetic result so the next round sees this spec in the ledger.
            fake = ExperimentResult(spec=spec, cv={"pr_auc": {"mean": 0.6, "std": 0.01}},
                                    folds=[{"pr_auc": 0.6}] * 2, fit_seconds=0.0, n_folds=2, seed=1, timestamp="")
            ledger.append(judge(fake, ledger.incumbent()))
    kept, dropped = dedupe([plan.specs[0]], ledger)
    assert not kept and dropped == [plan.specs[0].name]


def test_loop_finalize_save_load_roundtrip(ds, tmp_path):
    config = LoopConfig(rounds=2, per_round=2, planner="rules", seed=3, n_folds=2, ledger_path=tmp_path / "ledger.jsonl")
    outcome = run_loop(ds, config)
    assert len(outcome.ledger) >= 4 and outcome.best() is not None
    report = finalize(outcome, ds, narrator="template", seed=3)
    for heading in ("### Headline", "### What the experiments showed", "### Retention actions"):
        assert heading in report.narration
    save(report, tmp_path)
    loaded = load(tmp_path)
    assert loaded.winner.key() == report.winner.key()
    assert loaded.leaderboard.shape[0] == report.leaderboard.shape[0]
    assert loaded.holdout["pr_auc"] == pytest.approx(report.holdout["pr_auc"])
    ledger_lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(ledger_lines) == len(outcome.ledger)
    assert json.loads(ledger_lines[0])["verdict"] == "baseline"
