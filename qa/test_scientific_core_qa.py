"""Independent, tiny-fixture QA for the scientific core and project round trips."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from daylight_workbench.core import (
    DataConfig,
    OptimizeConfig,
    TrainConfig,
    Variable,
    optimize_model,
    prepare_data,
    read_dataset,
    train_model,
    validate_splits,
)
from daylight_workbench.project import load_project, save_project


def tiny_frame(n=24):
    x = np.arange(n, dtype=float)
    return pd.DataFrame({"x": x, "z": (x * 3 + 1), "y": 2 * x + 0.5 * (x % 3)})


def fitted(frame=None, *, constant=False):
    frame = tiny_frame() if frame is None else frame.copy()
    if constant:
        frame["y"] = 7.0
    data = prepare_data(frame, DataConfig(inputs=["x", "z"], targets=["y"], seed=23))
    model = train_model(data, TrainConfig(hidden=[3], epochs=3, patience=3, batch_size=32, seed=9))
    return model


def test_duplicate_inputs_and_group_chains_never_cross_splits():
    # Rows 0 and 1 share an input vector; rows 1 and 2 share a group. The
    # transitive component must stay intact even though the data differ.
    rows = [{"x": 0.0, "y": 10.0, "batch": "A"},
            {"x": 0.0, "y": 11.0, "batch": "B"},
            {"x": 1.0, "y": 12.0, "batch": "B"}]
    rows += [{"x": float(i), "y": float(100 + i), "batch": f"g{i}"} for i in range(2, 14)]
    data = prepare_data(pd.DataFrame(rows), DataConfig(inputs=["x"], targets=["y"], group="batch", seed=4))
    validate_splits(data)
    owner = {idx: split for split, indexes in data.splits.items() for idx in indexes}
    assert owner[0] == owner[1] == owner[2]
    # Repeated X with conflicting outcomes is retained and reported, not
    # silently dropped or treated as independent observations.
    assert data.report["conflicting_input_rows"] == 2
    assert data.report["retained_rows"] == len(rows)


def test_training_scalers_are_fit_only_on_training_rows():
    model = fitted()
    train = model.data.splits["train"]
    np.testing.assert_allclose(model.x_scale.mean, model.data.X[train].mean(axis=0))
    np.testing.assert_allclose(model.x_scale.scale, model.data.X[train].std(axis=0))
    np.testing.assert_allclose(model.y_scale.mean, model.data.Y[train].mean(axis=0))
    np.testing.assert_allclose(model.y_scale.scale, model.data.Y[train].std(axis=0))
    assert not np.allclose(model.x_scale.mean, model.data.X.mean(axis=0))


def test_constant_target_has_finite_errors_and_explicitly_undefined_r2():
    model = fitted(constant=True)
    assert model.y_scale.mean.tolist() == [7.0]
    assert model.y_scale.scale.tolist() == [1.0]
    for split in ("train", "validation", "test"):
        scores = model.metrics[split]["y"]
        assert scores["R2"] is None
        assert np.isfinite([scores["MAE"], scores["RMSE"]]).all()


def test_objective_directions_and_nondominance_match_independent_oracle():
    # Avoid a trained network here: exact monotonic outputs make a compact,
    # reproducible oracle for minimize/maximize sign handling.
    frame = pd.DataFrame({"x": np.arange(11, dtype=float), "a": 0.0, "b": 0.0})
    data = SimpleNamespace(config=SimpleNamespace(inputs=["x"], targets=["a", "b"]),
                           X=frame[["x"]].to_numpy(), splits={"train": list(range(11))})
    trained = SimpleNamespace(data=data, predict=lambda x: np.column_stack([x[:, 0], x[:, 0]]))

    def check(directions):
        cfg = OptimizeConfig([Variable("x", 0, 10)], directions, population=6, generations=2, seed=3)
        result = optimize_model(trained, cfg)
        candidates = result.candidates
        vals = candidates[["predicted::a", "predicted::b"]].to_numpy()
        signs = np.array([1 if directions[t] == "min" else -1 for t in ("a", "b")])
        objective = vals * signs
        expected = []
        for i, point in enumerate(objective):
            dominated = any(np.all(other <= point) and np.any(other < point)
                            for j, other in enumerate(objective) if i != j)
            expected.append(not dominated)
        assert candidates["pareto"].tolist() == expected
        return candidates

    aligned = check({"a": "max", "b": "max"})
    assert aligned["pareto"].sum() == 1
    assert aligned.loc[aligned["pareto"], "x"].iloc[0] == aligned["x"].max()
    opposing = check({"a": "max", "b": "min"})
    assert opposing["pareto"].all()


def test_integer_and_fixed_candidate_variables_are_respected():
    model = fitted(pd.DataFrame({"x": np.arange(24, dtype=float), "z": np.full(24, 42.0),
                                 "y": np.arange(24, dtype=float)}))
    cfg = OptimizeConfig([Variable("x", 0, 23, "integer"), Variable("z", 42, 42, "integer", True)],
                         {"y": "min"}, population=6, generations=2, seed=7, allow_extrapolation=True)
    out = optimize_model(model, cfg).candidates
    assert np.equal(out["x"], np.rint(out["x"])).all()
    assert (out["x"].between(0, 23)).all()
    assert (out["z"] == 42).all()

    # With only fixed inputs, the no-active-variable path emits that one point.
    fixed = OptimizeConfig([Variable("x", 4, 4, "integer", True), Variable("z", 42, 42, "continuous", True)],
                           {"y": "min"}, population=4, generations=1, seed=1)
    fixed_out = optimize_model(model, fixed)
    assert fixed_out.evaluations == 1
    assert fixed_out.candidates[["x", "z"]].to_numpy().tolist() == [[4.0, 42.0]]


def test_project_save_reload_preserves_predictions_and_split_scaling(tmp_path):
    model = fitted()
    path = tmp_path / "roundtrip.dwb"
    before = model.predict(np.array([[0.25, 1.75], [12.5, 38.5]], dtype=float))
    save_project(path, model)
    loaded, optimization = load_project(path)
    assert optimization is None
    np.testing.assert_allclose(loaded.predict(np.array([[0.25, 1.75], [12.5, 38.5]], dtype=float)), before,
                               rtol=0, atol=1e-7)
    assert loaded.data.splits == model.data.splits
    np.testing.assert_allclose(loaded.x_scale.mean, model.x_scale.mean)
    np.testing.assert_allclose(loaded.y_scale.scale, model.y_scale.scale)


@pytest.mark.parametrize("mutate", [
    lambda f: f.assign(y=np.where(np.arange(len(f)) == 2, np.inf, f.y)),
    lambda f: f.assign(y=np.where(np.arange(len(f)) == 2, "not-a-number", f.y)),
])
def test_nonfinite_and_non_numeric_targets_are_rejected(mutate):
    with pytest.raises(ValueError, match="数值列"):
        prepare_data(mutate(tiny_frame()), DataConfig(inputs=["x"], targets=["y"]))


def test_invalid_ratios_and_objective_specifications_are_rejected():
    with pytest.raises(ValueError, match="比例"):
        prepare_data(tiny_frame(), DataConfig(inputs=["x"], targets=["y"], train_ratio=0.7,
                                               val_ratio=0.2, test_ratio=0.2))
    model = fitted()
    with pytest.raises(ValueError, match="每个目标"):
        optimize_model(model, OptimizeConfig([Variable("x", 0, 23), Variable("z", 1, 70)],
                                             {"y": "maximize"}, population=4, generations=1,
                                             allow_extrapolation=True))
    with pytest.raises(ValueError, match="整数变量"):
        optimize_model(model, OptimizeConfig([Variable("x", 0.2, 20.5, "integer"), Variable("z", 1, 70)],
                                             {"y": "min"}, population=4, generations=1,
                                             allow_extrapolation=True))


def test_csv_duplicate_headers_are_rejected_not_silently_renamed(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("x,x,y\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        read_dataset(path)


@pytest.mark.parametrize("reserved", ["candidate_id", "generation", "pareto", "predicted::y"])
def test_optimizer_rejects_reserved_output_name_collisions(reserved):
    # Output names are part of the exported/result schema. These user-selected
    # inputs must be rejected, never overwritten or mistaken for generated fields.
    data = SimpleNamespace(config=SimpleNamespace(inputs=[reserved], targets=["y"]),
                           X=np.arange(11, dtype=float).reshape(-1, 1), splits={"train": list(range(11))})
    trained = SimpleNamespace(data=data, predict=lambda x: x[:, :1])
    config = OptimizeConfig([Variable(reserved, 0, 10)], {"y": "max"}, population=4, generations=1)
    with pytest.raises(ValueError, match="保留|冲突"):
        optimize_model(trained, config)
