"""Tiny synthetic end-to-end QA for the bounded multi-family modeling v2 API."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from daylight_workbench.core import (
    Cancelled,
    DataConfig,
    Scale,
    _leakage_components,
    prepare_data,
    validate_splits,
)
from daylight_workbench.modeling import (
    LayerSpec,
    ModelSpec,
    SearchConfig,
    make_search_plan,
    run_model_search,
)
from daylight_workbench.project import load_project, save_project


def synthetic_grouped_data():
    """Thirty small groups; rows 1 and 3 duplicate X across different groups."""
    rows = []
    for group in range(40):
        for replicate in range(3):
            x1 = group / 5 + replicate * 0.13
            x2 = ((group * 7) % 31) / 31 + replicate * 0.05
            if group == 1 and replicate == 0:
                x1, x2 = 0.13, 0.05
            y = 1.7 * x1 - 0.4 * x2 + np.sin(group / 3) * 0.1 + replicate * 0.02
            rows.append({"x1": x1, "x2": x2, "y": y, "group_id": f"g{group}"})
    data = prepare_data(pd.DataFrame(rows), DataConfig(
        inputs=["x1", "x2"], targets=["y"], group="group_id", seed=11, drop_duplicates=False
    ))
    # Construct a deterministic group-safe 70/15/15 split where the highest
    # support values occur in validation, proving the final refit expands from
    # the train-only range to the full 85% development range.
    group = data.frame["group_id"].to_numpy()
    data.splits = {
        "train": np.flatnonzero(np.isin(group, [f"g{i}" for i in range(28)])).tolist(),
        "validation": np.flatnonzero(np.isin(group, [f"g{i}" for i in range(34, 40)])).tolist(),
        "test": np.flatnonzero(np.isin(group, [f"g{i}" for i in range(28, 34)])).tolist(),
    }
    validate_splits(data)
    return data


def candidates_for(family):
    if family == "ann":
        return [
            ModelSpec("ann", {"epochs": 2, "patience": 1, "learning_rate": 0.01, "batch_size": 32},
                      [LayerSpec(4, "relu")]),
            ModelSpec("ann", {"epochs": 2, "patience": 1, "learning_rate": 0.01, "batch_size": 32},
                      [LayerSpec(3, "tanh")]),
        ]
    if family == "rf":
        return [
            ModelSpec("rf", {"n_estimators": 3, "max_depth": 3, "min_samples_leaf": 1, "max_features": 1.0}, []),
            ModelSpec("rf", {"n_estimators": 4, "max_depth": 4, "min_samples_leaf": 1, "max_features": 1.0}, []),
        ]
    if family == "xgboost":
        return [
            ModelSpec("xgboost", {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.1,
                                  "patience": 1, "subsample": 1.0, "colsample_bytree": 1.0}, []),
            ModelSpec("xgboost", {"n_estimators": 4, "max_depth": 2, "learning_rate": 0.1,
                                  "patience": 1, "subsample": 1.0, "colsample_bytree": 1.0}, []),
        ]
    raise AssertionError(family)


def config_for(family, *, max_fits=7, mode="cv"):
    return SearchConfig(candidates_for(family), mode=mode, folds=3, max_fits=max_fits,
                        seed=13, cpu_threads=1)


def assert_group_safe_plan(data, plan):
    components = _leakage_components(data.frame, data.config)
    test = plan["test"]
    dev = plan["development"]
    assert set(components[test]).isdisjoint(components[dev])
    assert len(plan["folds"]) == 3
    all_outer_valid = []
    for fold in plan["folds"]:
        train, valid = fold["train"], fold["validation"]
        inner_train, inner_valid = fold["inner_train"], fold["inner_validation"]
        assert set(train).isdisjoint(valid)
        assert sorted(train + valid) == sorted(dev)
        assert set(components[train]).isdisjoint(components[valid])
        assert set(inner_train).isdisjoint(inner_valid)
        assert sorted(inner_train + inner_valid) == sorted(train)
        assert set(components[inner_train]).isdisjoint(components[inner_valid])
        for part in (train, valid, inner_train, inner_valid):
            assert set(components[test]).isdisjoint(components[part])
        all_outer_valid.extend(valid)
    assert sorted(all_outer_valid) == sorted(dev)
    assert len(all_outer_valid) == len(set(all_outer_valid))


@pytest.fixture(scope="module")
def data():
    return synthetic_grouped_data()


def test_three_model_families_search_scaling_metrics_and_safe_roundtrip(data, tmp_path):
    # This duplicate feature vector occurs across different group labels. It
    # must nevertheless bind both rows, and both whole groups, into one unit.
    components = _leakage_components(data.frame, data.config)
    assert components[1] == components[3]

    evidence = {"rows": len(data.frame), "families": {}, "synthetic_only": True}
    for family in ("ann", "rf", "xgboost"):
        cfg = config_for(family)
        roundtripped = SearchConfig.from_dict(asdict(cfg))
        roundtripped.validate(data)
        assert asdict(roundtripped) == asdict(cfg)

        plan = make_search_plan(data, cfg)
        assert plan["candidate_count"] == 2
        assert plan["fold_count"] == 3
        assert plan["candidate_fold_jobs"] == 6
        assert plan["per_candidate_fits"] == [3, 3]
        assert plan["fit_budget"] == 7  # six CV fits plus one final refit
        assert plan["test"] == data.splits["test"]
        assert plan["test_fraction_actual"] == len(data.splits["test"]) / len(data.frame)
        assert_group_safe_plan(data, plan)

        events = []
        result = run_model_search(data, cfg, snapshot=events.append)
        selected = result.selected
        assert result.plan == plan
        assert result.selected_index in (0, 1)
        assert selected.family == family
        assert len(result.candidates) == 2
        assert all(len(candidate["folds"]) == 3 for candidate in result.candidates)
        assert sum(e.get("kind") == "search" and e.get("stage") == "fit" for e in events) == 6

        development = plan["development"]
        assert selected.fit_indices == development
        assert len(selected.fit_indices) == len(data.splits["train"]) + len(data.splits["validation"])
        assert len(selected.fit_indices) > len(data.splits["train"])  # refit uses dev, not stale 70% only
        expected_x, expected_y = Scale.fit(data.X[development]), Scale.fit(data.Y[development])
        np.testing.assert_allclose(selected.x_scale.mean, expected_x.mean)
        np.testing.assert_allclose(selected.x_scale.scale, expected_x.scale)
        np.testing.assert_allclose(selected.y_scale.mean, expected_y.mean)
        np.testing.assert_allclose(selected.y_scale.scale, expected_y.scale)
        assert selected.support_bounds == {
            name: (float(data.X[development, j].min()), float(data.X[development, j].max()))
            for j, name in enumerate(data.config.inputs)
        }
        assert selected.support_bounds != {
            name: (float(data.X[data.splits["train"], j].min()), float(data.X[data.splits["train"], j].max()))
            for j, name in enumerate(data.config.inputs)
        }

        test_indices = data.splits["test"]
        expected_mse = mean_squared_error(data.Y[test_indices], selected.predict(data.X[test_indices]))
        assert selected.metrics["test"]["y"]["MSE"] == pytest.approx(expected_mse, rel=1e-12, abs=1e-12)
        train_mse = mean_squared_error(data.Y[development], selected.predict(data.X[development]))
        assert selected.metrics["train"]["y"]["MSE"] == pytest.approx(train_mse, rel=1e-12, abs=1e-12)

        path = tmp_path / f"{family}.dwb"
        probe = data.X[[0, 2, 13, 44, 89]]
        before = selected.predict(probe)
        save_project(path, selected)
        loaded, optimization = load_project(path)
        assert optimization is None
        assert loaded.family == family
        np.testing.assert_allclose(loaded.predict(probe), before, rtol=1e-10, atol=1e-10)
        evidence["families"][family] = {
            "selected_index": result.selected_index,
            "fold_jobs": plan["candidate_fold_jobs"],
            "fit_budget": plan["fit_budget"],
            "development_rows": len(development),
            "locked_test_rows": len(test_indices),
            "test_mse": float(selected.metrics["test"]["y"]["MSE"]),
            "roundtrip_max_abs_error": float(np.max(np.abs(loaded.predict(probe) - before))),
        }

    out = Path(__file__).resolve().parent / "evidence" / "modeling-v2-qa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")


def test_locked_test_target_mutation_cannot_change_rf_selection_or_predictions(data):
    cfg = config_for("rf")
    baseline = run_model_search(data, cfg)
    changed = deepcopy(data)
    test = changed.splits["test"]
    original_test_y = changed.Y[test].copy()
    changed.frame.loc[test, "y"] = changed.frame.loc[test, "y"].to_numpy() + 1_000_000.0
    changed.fingerprint = "synthetic-test-target-mutated"

    tampered = run_model_search(changed, cfg)
    assert tampered.plan["test"] == baseline.plan["test"]
    assert tampered.selected_index == baseline.selected_index
    np.testing.assert_allclose(
        [c["score_mean"] for c in tampered.candidates],
        [c["score_mean"] for c in baseline.candidates], rtol=0, atol=0,
    )
    np.testing.assert_allclose(tampered.selected.predict(data.X), baseline.selected.predict(data.X), rtol=0, atol=0)
    assert tampered.selected.metrics["test"]["y"]["MSE"] != baseline.selected.metrics["test"]["y"]["MSE"]
    assert np.any(changed.Y[test] != original_test_y)


def test_fit_budget_preflight_and_cancelled_incomplete_candidate_never_return_winner(data):
    too_small = config_for("rf", max_fits=6)
    with pytest.raises(ValueError, match="7次实际拟合"):
        make_search_plan(data, too_small)

    cancel = threading.Event()
    scored = []

    def snapshot(event):
        if event.get("kind") == "search" and event.get("stage") == "scored":
            scored.append(event)
            cancel.set()

    # Stop after candidate 1's first of three folds: it is incomplete, so
    # run_model_search must raise, never fabricate a selected winner.
    with pytest.raises(Cancelled):
        run_model_search(data, config_for("rf"), cancel=cancel, snapshot=snapshot)
    assert len(scored) == 1
