"""Independent, synthetic-only v0.3 acceptance checks.

Run explicitly: QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q qa/v03_acceptance.py
No live DeepSeek request, API credential, real building data, or 3D WebGL claim.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
import threading
import time
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/daylight-v03-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/daylight-v03-cache")

import numpy as np
import pandas as pd
import pytest

from daylight_workbench.core import (Cancelled, DataConfig, TrainConfig,
                                     _leakage_components, prepare_data, train_model)
from daylight_workbench.external_results import prepare_external_results
from daylight_workbench.modeling import (
    CompositeSurrogate, LayerSpec, ModelSpec, SearchConfig, make_search_plan,
    run_model_search, target_data,
)
from daylight_workbench.preference_engine import (
    ObjectiveSpec, PlanError, build_preference_preview, default_preference_plan,
    make_normalization_reference, preference_request_body, rank_candidates,
    request_preference_plan, validate_preference_plan,
)
from daylight_workbench.llm_filter import RequestSettings
from daylight_workbench.project import load_project, save_project


def grouped_data():
    rows = []
    for group in range(30):
        for replicate in range(2):
            x = group / 29
            z = replicate * .2 + ((group * 13) % 23) / 23
            rows.append({"x": x, "z": z, "daylight": 4 * x + .1 * z,
                         "energy": (x > .48) * 2 + .2 * z, "group": f"g{group}"})
    data = prepare_data(pd.DataFrame(rows), DataConfig(
        ["x", "z"], ["daylight", "energy"], group="group", seed=11,
        directions={"daylight": "max", "energy": "min"}))
    return data


def search_config(*, max_fits=20, candidates=None):
    return SearchConfig(candidates or [
        ModelSpec("ann", {"epochs": 2, "patience": 1, "learning_rate": .01,
                          "batch_size": 16}, [LayerSpec(4, "relu")]),
        ModelSpec("rf", {"n_estimators": 4, "max_depth": 3}, []),
        ModelSpec("xgboost", {"n_estimators": 3, "max_depth": 2,
                               "learning_rate": .2, "patience": 1}, []),
    ], folds=3, max_fits=max_fits, cpu_threads=1, seed=17)


def test_independent_default_common_group_safe_plan_and_budgets():
    data = grouped_data()
    cfg = search_config()
    assert cfg.target_mode == "independent"
    plan = make_search_plan(data, cfg)
    assert plan["target_order"] == data.config.targets
    assert plan["test"] == data.splits["test"]
    assert plan["candidate_fold_jobs"] == 18
    assert plan["fit_budget"] == 20
    assert plan["per_target_fit_budget"] == 10
    components = _leakage_components(data.frame, data.config)
    test_groups = set(components[plan["test"]])
    outer_validation = []
    for fold in plan["folds"]:
        assert sorted(fold["train"] + fold["validation"]) == plan["development"]
        assert sorted(fold["inner_train"] + fold["inner_validation"]) == fold["train"]
        for a, b in ((fold["train"], fold["validation"]),
                     (fold["inner_train"], fold["inner_validation"])):
            assert set(components[a]).isdisjoint(components[b])
        assert test_groups.isdisjoint(components[fold["train"]])
        outer_validation.extend(fold["validation"])
    assert sorted(outer_validation) == plan["development"]
    with pytest.raises(ValueError, match="预算"):
        make_search_plan(data, search_config(max_fits=19))
    legacy = asdict(cfg)
    legacy.pop("target_mode")
    assert SearchConfig.from_dict(legacy).target_mode == "shared"


@pytest.fixture(scope="module")
def trained():
    data = grouped_data()
    events = []
    result = run_model_search(data, search_config(), snapshot=events.append)
    return data, result, events


def test_independent_three_families_metrics_progress_and_v3_roundtrip(trained, tmp_path):
    data, result, events = trained
    model = result.selected
    assert isinstance(model, CompositeSurrogate)
    assert model.fit_indices == result.plan["development"]
    assert list(result.target_results) == data.config.targets
    assert set(result.selected_index) == set(data.config.targets)
    assert len([e for e in events if e.get("stage") == "fit"]) == 18
    assert len([e for e in events if e.get("stage") == "target_selected"]) == 2
    assert all(e.get("target") in data.config.targets for e in events if e.get("stage") == "fit")
    assert events[-1]["stage"] == "complete"
    assert set(events[-1]["test_metrics"]) == set(data.config.targets)
    assert len(result.candidates) == 6
    for target in data.config.targets:
        record = result.target_results[target]
        assert [c["family"] for c in record["candidates"]] == ["ann", "rf", "xgboost"]
        assert record["selected_index"] == min(
            range(3), key=lambda i: (record["candidates"][i]["score_mean"], i))
        for candidate in record["candidates"]:
            assert len(candidate["folds"]) == 3
            assert set(candidate["metric_spread"]) == {target}
        assert model.target_models[target].fit_indices == result.plan["development"]
        assert set(model.metrics["test"][target]) == {"MAE", "MSE", "RMSE", "R2"}
    X = data.X[[1, 9, 27]]
    expected = np.column_stack([model.target_models[t].predict(X)[:, 0] for t in data.config.targets])
    np.testing.assert_allclose(model.predict(X), expected)
    # Dict insertion order is not the prediction contract.
    model.target_models = dict(reversed(list(model.target_models.items())))
    np.testing.assert_allclose(model.predict(X), expected)
    path = tmp_path / "v03-synthetic.dwb"
    save_project(path, model)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["schema_version"] == 3
        assert not any("pickle" in name or "chat" in name or "key" in name for name in archive.namelist())
    loaded, optimization = load_project(path)
    assert optimization is None and isinstance(loaded, CompositeSurrogate)
    np.testing.assert_allclose(loaded.predict(X), expected, rtol=1e-8, atol=1e-8)
    assert loaded.search_metadata["target_order"] == data.config.targets


def test_v3_semantic_validation_rejects_rehashed_target_order_tamper(trained, tmp_path):
    source = tmp_path / "good.dwb"
    save_project(source, trained[1].selected)
    with zipfile.ZipFile(source) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(blobs["metadata.json"])
    metadata["target_order"].reverse()
    blobs["metadata.json"] = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode()
    manifest = json.loads(blobs["manifest.json"])
    manifest["sha256"]["metadata.json"] = hashlib.sha256(blobs["metadata.json"]).hexdigest()
    blobs["manifest.json"] = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    tampered = tmp_path / "rehashed-tamper.dwb"
    with zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr(name, blob)
    with pytest.raises(ValueError, match="顺序"):
        load_project(tampered)


def test_locked_test_labels_cannot_influence_selection():
    data = grouped_data()
    cfg = search_config(candidates=[ModelSpec("rf", {"n_estimators": 3, "max_depth": 3}, []),
                                    ModelSpec("rf", {"n_estimators": 4, "max_depth": 2}, [])], max_fits=14)
    original = run_model_search(data, cfg)
    changed = deepcopy(data)
    changed.frame.loc[changed.splits["test"], "daylight"] += 10000
    changed.frame.loc[changed.splits["test"], "energy"] -= 10000
    changed.fingerprint = "synthetic-test-label-mutation"
    repeated = run_model_search(changed, cfg)
    assert repeated.selected_index == original.selected_index
    for target in data.config.targets:
        a = [c["score_mean"] for c in original.target_results[target]["candidates"]]
        b = [c["score_mean"] for c in repeated.target_results[target]["candidates"]]
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(original.selected.predict(data.X), repeated.selected.predict(data.X), rtol=0, atol=0)
    assert original.selected.metrics["test"] != repeated.selected.metrics["test"]


def test_mixed_family_prediction_order_and_shared_legacy_project(tmp_path):
    data = grouped_data()
    ann = ModelSpec("ann", {"epochs": 2, "patience": 1, "learning_rate": .01,
                            "batch_size": 16}, [LayerSpec(4, "relu")])
    rf = ModelSpec("rf", {"n_estimators": 3, "max_depth": 2}, [])
    daylight = run_model_search(target_data(data, "daylight"), SearchConfig(
        [ann], folds=3, max_fits=4, cpu_threads=1, seed=17)).selected
    energy = run_model_search(target_data(data, "energy"), SearchConfig(
        [rf], folds=3, max_fits=4, cpu_threads=1, seed=17)).selected
    mixed = CompositeSurrogate(data, {"energy": energy, "daylight": daylight},
                               daylight.fit_indices)
    assert [mixed.target_models[t].family for t in data.config.targets] == ["ann", "rf"]
    probe = data.X[[2, 11, 44]]
    np.testing.assert_allclose(mixed.predict(probe), np.column_stack([
        daylight.predict(probe)[:, 0], energy.predict(probe)[:, 0]]))
    assert make_search_plan(data, SearchConfig([rf], target_mode="shared", folds=3,
                                              max_fits=4, cpu_threads=1))["fit_budget"] == 4
    shared = run_model_search(data, SearchConfig([rf], target_mode="shared", folds=3,
                                                  max_fits=4, cpu_threads=1))
    path = tmp_path / "v02-shared-legacy.dwb"
    save_project(path, shared.selected)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["schema_version"] == 2
    loaded, _ = load_project(path)
    np.testing.assert_allclose(loaded.predict(probe), shared.selected.predict(probe), rtol=1e-8, atol=1e-8)


def test_legacy_v1_project_still_loads(tmp_path):
    data = grouped_data()
    trained = train_model(data, TrainConfig([4], epochs=2, patience=1))
    path = tmp_path / "v01-legacy.dwb"
    save_project(path, trained)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["schema_version"] == 1
    loaded, _ = load_project(path)
    probe = data.X[[1, 19, 48]]
    np.testing.assert_allclose(loaded.predict(probe), trained.predict(probe), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("family", ["ann", "xgboost"])
def test_v3_safe_artifacts_for_non_rf_families(family, tmp_path):
    data = grouped_data()
    spec = (ModelSpec("ann", {"epochs": 2, "patience": 1, "batch_size": 16},
                      [LayerSpec(4, "relu")]) if family == "ann"
            else ModelSpec("xgboost", {"n_estimators": 3, "max_depth": 2,
                                        "learning_rate": .2, "patience": 1}, []))
    result = run_model_search(data, SearchConfig([spec], folds=3, max_fits=8,
                                                 cpu_threads=1, seed=17))
    assert all(m.family == family for m in result.selected.target_models.values())
    path = tmp_path / f"v03-{family}.dwb"
    save_project(path, result.selected)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert json.loads(archive.read("manifest.json"))["schema_version"] == 3
        assert sum(n.startswith("target_") for n in names) == 2
        expected_suffix = "weights.pt" if family == "ann" else "booster_0.json"
        assert all(n.endswith(expected_suffix) for n in names if n.startswith("target_"))
    loaded, _ = load_project(path)
    probe = data.X[[1, 17, 53]]
    np.testing.assert_allclose(loaded.predict(probe), result.selected.predict(probe), rtol=1e-8, atol=1e-8)


def test_cancel_never_returns_partial_winner():
    event = threading.Event()
    scored = []
    def observe(snapshot):
        if snapshot.get("stage") == "scored":
            scored.append(snapshot)
            event.set()
    with pytest.raises(Cancelled):
        run_model_search(grouped_data(), search_config(
            candidates=[ModelSpec("rf", {"n_estimators": 3}, [])], max_fits=8),
            cancel=event, snapshot=observe)
    assert len(scored) == 1


def preference_frame():
    return pd.DataFrame({
        "candidate_id": ["A", "B", "C", "D", "E"],
        "daylight": [40., 80., 60., 80., 20.],
        "energy": [10., 30., 20., 30., 50.],
        "glare": [20., 10., 30., 10., np.nan],
        "constant": [7., 7., 7., 7., 7.],
        "pareto": [True, True, True, True, False],
    })


def objectives():
    return [ObjectiveSpec("daylight", "采光", "max", ("UDI",), "%"),
            ObjectiveSpec("energy", "能耗", "min", ("EUI",), "kWh"),
            ObjectiveSpec("glare", "眩光", "min", ("DGP",), "%"),
            ObjectiveSpec("constant", "常量", "min")]


def preference_plan(weights, **changes):
    plan = default_preference_plan(objectives(), scope="all")
    plan["weights"] = [{"column": col, "weight": value} for col, value in weights.items()]
    plan.update(changes)
    return plan


def test_preference_direction_fixed_reference_filter_missing_constant_ties():
    frame, specs = preference_frame(), objectives()
    original = frame.copy(deep=True)
    reference = make_normalization_reference(frame, specs)
    daylight = preference_plan({"daylight": 1, "energy": 0, "glare": 0, "constant": 0})
    ranked = rank_candidates(frame, daylight, specs, reference)
    assert ranked.frame.candidate_id.tolist()[:2] == ["B", "D"]
    assert ranked.metrics.preference_score.iloc[0] == pytest.approx(0)
    # Missing glare is safe when glare has zero weight, but excluded when active.
    assert "E" in ranked.frame.candidate_id.tolist()
    balanced = preference_plan({"daylight": 0, "energy": 1, "glare": 1, "constant": 0})
    ranked = rank_candidates(frame, balanced, specs, reference)
    assert "E" not in ranked.frame.candidate_id.tolist()
    assert ranked.excluded_nonfinite == 1
    assert ranked.frame.candidate_id.tolist()[0] == "A"
    assert ranked.frame.candidate_id.tolist().index("B") < ranked.frame.candidate_id.tolist().index("D")
    fixed = rank_candidates(frame, preference_plan({"daylight": 1, "energy": 0, "glare": 0, "constant": 0},
                                                    constraints=[{"column": "daylight", "operator": "<=", "value": 60.}]), specs, reference)
    assert fixed.bounds["daylight"] == (20., 80.)
    assert fixed.frame.candidate_id.tolist() == ["C", "A", "E"]
    assert rank_candidates(frame, preference_plan({"daylight": 0, "energy": 0, "glare": 0, "constant": 1}), specs, reference).constant_objectives == ["constant"]
    empty = rank_candidates(frame, preference_plan({"daylight": 1, "energy": 0, "glare": 0, "constant": 0},
                                                    constraints=[{"column": "energy", "operator": "<", "value": 0.}]), specs, reference)
    assert empty.frame.empty and empty.eligible_count == 0
    pd.testing.assert_frame_equal(frame, original)


def test_preference_mock_transport_preview_validation_and_privacy():
    frame, specs = preference_frame(), objectives()
    previous = preference_plan({"daylight": .6, "energy": .4, "glare": 0, "constant": 0})
    preview = build_preference_preview(frame, "帮我选更考虑采光的", specs, previous)
    assert preview["preference_state"] == validate_preference_plan(previous, preview)
    body = preference_request_body(preview)
    payload = json.dumps(body, ensure_ascii=False)
    assert "candidate_id" not in payload
    assert '"A"' not in payload and '"B"' not in payload
    assert "test-key-do-not-send" not in payload
    assert "帮我选更考虑采光的" in payload
    assert "我要能耗和眩光平衡的方案" in preference_request_body(
        build_preference_preview(frame, "我要能耗和眩光平衡的方案", specs))["messages"][1]["content"]
    proposal = preference_plan({"daylight": 0, "energy": .5, "glare": .5, "constant": 0},
                               intent="能耗与眩光平衡")
    captured = []
    def fake(req, timeout, maximum):
        captured.append(req)
        return json.dumps({"choices": [{"message": {"content": json.dumps(proposal)}}]}).encode()
    result = request_preference_plan(RequestSettings(), "test-key-do-not-send", preview, fake)
    assert result["intent"] == "能耗与眩光平衡"
    assert len(captured) == 1 and captured[0].full_url == "https://api.deepseek.com/chat/completions"
    assert b"candidate_id" not in captured[0].data and b"test-key-do-not-send" not in captured[0].data
    assert captured[0].get_header("Authorization") == "Bearer test-key-do-not-send"
    with pytest.raises(PlanError):
        validate_preference_plan({**proposal, "code": "import os"}, preview)
    with pytest.raises(PlanError):
        validate_preference_plan({**proposal, "weights": [{"column": "candidate_id", "weight": 1}]}, preview)


def test_dedicated_page_explicit_actions_and_stale_async():
    from PySide6.QtWidgets import QApplication
    from daylight_workbench.decision_page import PreferenceDecisionPage
    app = QApplication.instance() or QApplication([])
    page = PreferenceDecisionPage()
    frame, specs = preference_frame(), objectives()
    page.set_candidates(frame, specs, input_columns=())
    assert not page.send_button.isEnabled()
    assert not page.apply_button.isEnabled()
    page.confirm_mapping()
    assert page.apply_button.isEnabled()
    page.query_edit.setPlainText("帮我选更考虑采光的")
    page.make_preview()
    assert page.send_button.isEnabled()
    assert "candidate_id" not in page.preview_edit.toPlainText()
    selected = []
    page.selectionChanged.connect(selected.append)
    page.apply_reviewed_plan()
    assert page.ranking is not None
    assert set(page.selected_ids()) == set(page.ranking.frame.candidate_id.head(3))
    assert selected[-1] == page.selected_ids()
    page.set_selection(["D"])
    assert page.selected_ids() == ["D"]
    old_id = page._request_id
    old_rank = page.ranking
    page.cancel_request()
    page._request_done(old_id, preference_plan({"daylight": 0, "energy": 1, "glare": 0, "constant": 0}), None)
    assert page.ranking is old_rank and page.selected_ids() == ["D"]
    page.shutdown()


def test_dedicated_page_mock_two_turn_review_before_apply(monkeypatch):
    """Mock the provider boundary only; exercise the real Qt conversation flow."""
    from PySide6.QtWidgets import QApplication
    from daylight_workbench import decision_page
    app = QApplication.instance() or QApplication([])
    calls = []
    responses = [
        preference_plan({"daylight": .6, "energy": .2, "glare": .2, "constant": 0},
                        intent="更重视采光"),
        preference_plan({"daylight": 0, "energy": .5, "glare": .5, "constant": 0},
                        intent="能耗与眩光平衡"),
    ]
    def mocked_request(settings, key, preview):
        calls.append((key, deepcopy(preview)))
        return responses[len(calls) - 1]
    monkeypatch.setattr(decision_page, "request_preference_plan", mocked_request)
    page = decision_page.PreferenceDecisionPage()
    page.set_candidates(preference_frame(), objectives())
    page.confirm_mapping()
    page.key_edit.setText("SYNTHETIC-KEY-NOT-SENT")
    for number, phrase in enumerate(("帮我选更考虑采光的", "我要能耗和眩光平衡的方案"), 1):
        page.query_edit.setPlainText(phrase)
        page.make_preview()
        assert phrase in page.preview_edit.toPlainText()
        if number == 1:
            assert page.ranking is None
        else:
            assert page.ranking is not None
        page.send()
        deadline = time.monotonic() + 5
        while page._busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        app.processEvents()
        assert not page._busy and len(calls) == number
        assert page._state["intent"] == responses[number - 1]["intent"]
        assert "SYNTHETIC-KEY-NOT-SENT" not in page.chat.toPlainText()
        assert "SYNTHETIC-KEY-NOT-SENT" not in page.preview_edit.toPlainText()
        if number == 2:
            assert calls[1][1]["preference_state"]["intent"] == responses[0]["intent"]
        page.apply_reviewed_plan()
        assert page.ranking is not None and page._applied["intent"] == responses[number - 1]["intent"]
    page.shutdown()
    assert not page.key_edit.text()


def test_integrated_main_window_fifth_page_and_close():
    """Run only after implementation freeze; assert the real close path exits cleanly."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from daylight_workbench.ui import MainWindow
    if QApplication.instance() is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1280, 900)
    window.show()
    context = prepare_external_results(pd.DataFrame({
        "daylight": [40., 80., 60., 75., 25.],
        "energy": [10., 30., 20., 25., 50.],
        "glare": [20., 10., 30., 15., 40.],
    }), [], ["daylight", "energy", "glare"],
        {"daylight": "max", "energy": "min", "glare": "min"}, "SYNTHETIC TEST ONLY")
    window.external_ready(context)
    assert window.tabs.count() == 5
    page = window.decision_page
    assert len(page._frame) == len(context.frame)
    assert page._reference is not None and not page._mapping_confirmed
    assert page._visible_ids == window.results_model.frame.candidate_id.tolist()
    original_view_ids = window.results_model.frame.candidate_id.tolist()
    window.tabs.setCurrentWidget(page)
    page.detail_tabs.setCurrentIndex(2)
    deadline = time.monotonic() + 10
    while not page.interactive_plot._bridge.ready_for_data and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert page.interactive_plot._bridge.ready_for_data
    assert len(page.interactive_plot._payload["rows"]) == len(context.frame)
    page.detail_tabs.setCurrentIndex(0)
    page.confirm_mapping()
    assert page._mapping_confirmed
    assert all(o.direction == context.directions[o.column] for o in page._objectives)
    page.query_edit.setPlainText("我要能耗和眩光平衡的方案")
    page.make_preview()
    assert page.send_button.isEnabled() and not page._busy
    assert "candidate_id" not in page.preview_edit.toPlainText()
    page.weight_table.cellWidget(0, 2).setValue(0)
    page.weight_table.cellWidget(1, 2).setValue(.5)
    page.weight_table.cellWidget(2, 2).setValue(.5)
    page.scope_combo.setCurrentIndex(page.scope_combo.findData("current_view"))
    assert not page.send_button.isEnabled()  # edited plan requires new preview
    page.apply_reviewed_plan()
    assert page.ranking is not None
    assert page.ranking.scope_count == len(original_view_ids)
    assert window.last_filter_plan["kind"] == "preference_v1"
    assert window.results_model.frame.candidate_id.tolist() == original_view_ids
    assert page._visible_ids == original_view_ids
    assert set(page.selected_ids()) == set(window.selected_frame().candidate_id)
    assert set(page.selected_ids()) == set(window.interactive_plot.selected_ids())
    window.results_table.sortByColumn(0, __import__("PySide6").QtCore.Qt.SortOrder.DescendingOrder)
    assert set(page.selected_ids()) == set(window.selected_frame().candidate_id)
    window.close()
    app.processEvents()
    assert not window.isVisible()

