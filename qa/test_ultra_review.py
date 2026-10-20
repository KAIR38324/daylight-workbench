"""Bounded independent regressions from the high-rigor core/interaction review.

All inputs are tiny deterministic synthetic fixtures. No network, real dataset,
or browser-rendering claim is involved in these core regressions.
"""
from copy import deepcopy
from types import SimpleNamespace
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from daylight_workbench.core import (
    DataConfig, OptimizeConfig, TrainConfig, Variable, optimize_model, prepare_data,
    train_model,
)
from daylight_workbench.modeling import FittedSurrogate, LayerSpec, ModelSpec, SearchConfig, run_model_search
from daylight_workbench.project import load_project, save_project


def tiny_data():
    x = np.arange(48, dtype=float)
    return prepare_data(pd.DataFrame({"x": x, "z": np.sin(x),
                                      "a": x * .1, "b": np.cos(x) + x * .03}),
                        DataConfig(["x", "z"], ["a", "b"], seed=9))


def tiny_spec(family):
    if family == "ann":
        return ModelSpec(family, {"epochs": 2, "patience": 1, "batch_size": 16}, [LayerSpec(3, "tanh")])
    if family == "rf":
        return ModelSpec(family, {"n_estimators": 3, "max_depth": 2}, [])
    return ModelSpec(family, {"n_estimators": 2, "max_depth": 1, "patience": 1}, [])


@pytest.mark.parametrize("mode", ["cv", "fixed"])
def test_ann_locked_test_labels_outside_float32_cannot_affect_search(mode):
    data = tiny_data()
    config = SearchConfig([tiny_spec("ann")], mode=mode, folds=2, max_fits=3, cpu_threads=1)
    baseline = run_model_search(data, config)
    changed = deepcopy(data)
    # Finite float64 labels and finite squared errors, but not representable by
    # float32. Test labels need never become network training tensors.
    changed.frame.loc[changed.splits["test"], "a"] = 1e40
    mutated = run_model_search(changed, config)
    assert mutated.selected_index == baseline.selected_index
    assert mutated.candidates == baseline.candidates
    assert mutated.selected.history == baseline.selected.history
    np.testing.assert_array_equal(mutated.selected.predict(data.X), baseline.selected.predict(data.X))
    assert mutated.selected.metrics["test"]["a"]["MSE"] > 1e79
    assert np.isfinite(mutated.selected.metrics["test"]["a"]["MSE"])


def test_legacy_ann_locked_test_labels_cannot_affect_early_stopping():
    data = tiny_data()
    config = TrainConfig([3], epochs=2, patience=1, batch_size=16, seed=4)
    baseline = train_model(data, config)
    changed = deepcopy(data)
    changed.frame.loc[changed.splits["test"], "a"] = 1e40
    mutated = train_model(changed, config)
    assert mutated.best_epoch == baseline.best_epoch
    assert mutated.history == baseline.history
    np.testing.assert_array_equal(mutated.predict(data.X), baseline.predict(data.X))
    assert np.isfinite(mutated.metrics["test"]["a"]["MSE"])
    assert mutated.metrics["test"]["a"]["MSE"] > 1e79


@pytest.mark.parametrize("family", ["ann", "rf", "xgboost"])
def test_multitarget_score_and_reload_match_independent_numeric_oracle(family, tmp_path):
    data = tiny_data()
    config = SearchConfig([tiny_spec(family)], folds=2, max_fits=6, cpu_threads=1)
    result = run_model_search(data, config)
    selected = result.selected
    assert selected.fit_indices == sorted(data.splits["train"] + data.splits["validation"])
    for fold in result.candidates[0]["folds"]:
        # Rebuild aggregate normalized score from the independently checked
        # original-unit per-target MSE and the explicitly recorded denominator.
        rmse = np.sqrt([fold["metrics"][name]["MSE"] for name in data.config.targets])
        expected = float(np.mean(rmse / np.asarray(fold["normalization_std"])))
        assert fold["score"] == pytest.approx(expected, rel=1e-12)
    predicted = selected.predict(data.X[data.splits["test"]])
    assert predicted.shape == (len(data.splits["test"]), 2)
    error = predicted - data.Y[data.splits["test"]]
    for j, name in enumerate(data.config.targets):
        assert selected.metrics["test"][name]["MAE"] == pytest.approx(np.abs(error[:, j]).mean())
        assert selected.metrics["test"][name]["MSE"] == pytest.approx(np.square(error[:, j]).mean())
    path = tmp_path / (family + ".dwb")
    save_project(path, selected)
    loaded, optimization = load_project(path)
    assert optimization is None
    assert loaded.predict(data.X).shape == (len(data.frame), 2)
    np.testing.assert_allclose(loaded.predict(data.X), selected.predict(data.X), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", ["ann", "rf", "xgboost"])
def test_every_family_rejects_backend_output_dimension_mismatch(family):
    from daylight_workbench.core import Scale
    data = tiny_data()
    if family == "ann":
        import torch
        backend = torch.nn.Linear(2, 3)
    elif family == "rf":
        backend = SimpleNamespace(predict=lambda x: np.zeros((len(x), 3)))
    else:
        # Two declared scalar targets, but one backend unexpectedly emits two
        # outputs. This must not be accepted via scaler broadcasting.
        backend = [SimpleNamespace(predict=lambda dm: np.zeros((dm.num_row(), 2))),
                   SimpleNamespace(predict=lambda dm: np.zeros(dm.num_row()))]
    fit = data.splits["train"]
    model = FittedSurrogate(data, tiny_spec(family), backend,
                            Scale.fit(data.X[fit]), Scale.fit(data.Y[fit]), fit)
    with pytest.raises(ValueError, match="输出维度"):
        model.predict(data.X[:3])


def test_optimizer_snapshots_and_global_front_match_exact_mixed_type_oracle():
    values = np.column_stack([np.arange(9.), np.linspace(0, 1, 9), np.full(9, 7.)])
    data = SimpleNamespace(config=SimpleNamespace(inputs=["i", "c", "f"], targets=["a", "b"]),
                           X=values, splits={"train": list(range(9))})

    def predict(x):
        return np.column_stack([x[:, 0] + x[:, 1], (x[:, 0] - 4)**2 + x[:, 1]])

    trained = SimpleNamespace(data=data, predict=predict)
    cfg = OptimizeConfig([Variable("i", 0, 8, "integer"), Variable("c", 0, 1),
                          Variable("f", 7, 7, "integer", True)],
                         {"a": "max", "b": "min"}, population=8, generations=3, seed=5)
    events = []
    result = optimize_model(trained, cfg, snapshot=events.append)
    assert events and events[-1]["generation"] == 3
    assert events[-1]["evaluations"] == result.evaluations <= 24
    for event in events:
        xx, yy = np.asarray(event["X"]), np.asarray(event["Y"])
        np.testing.assert_array_equal(xx[:, 0], np.rint(xx[:, 0]))
        np.testing.assert_array_equal(xx[:, 2], np.full(len(xx), 7))
        np.testing.assert_allclose(yy, predict(xx), rtol=0, atol=0)
    frame = result.candidates
    objective = frame[["predicted::a", "predicted::b"]].to_numpy() * [-1, 1]
    expected = [not any(np.all(q <= p) and np.any(q < p) for q in objective) for p in objective]
    assert frame.pareto.tolist() == expected
    assert frame.candidate_id.is_unique


def test_goal_only_external_rows_retain_values_and_true_source(tmp_path):
    from daylight_workbench.external_results import prepare_external_results, export_external
    raw = pd.DataFrame({"a": [1., 2., 3., 3.], "b": [2., 3., 2., 2.]})
    context = prepare_external_results(raw, [], ["a", "b"], {"a": "max", "b": "min"}, "synthetic_external.csv")
    pd.testing.assert_frame_equal(context.frame[["a", "b"]], raw)
    assert context.inputs == []
    assert context.frame.pareto.tolist() == [False, False, True, True]
    path = tmp_path / "external.csv"
    export_external(path, context.frame, context)
    provenance = json.loads(path.with_suffix(".csv.provenance.json").read_text())
    assert provenance["kind"] == "external_results"
    assert provenance["source"] == "synthetic_external.csv"
    assert provenance["inputs"] == []
    assert not any(name.startswith("predicted::") for name in context.frame)
