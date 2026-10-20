"""Real Qt controls and mocked provider traffic, never a live API request.

Offscreen checks verify payload/selection state and 2D comparison bars. They do
not assert that a GPU-rendered interactive 3D scene was visually verified.
"""
import copy
import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

import pandas as pd
import pytest
from PySide6 import QtCore as C, QtWidgets as W

from daylight_workbench.decision_page import PreferenceDecisionPage
from daylight_workbench.external_results import prepare_external_results
from daylight_workbench.preference_engine import ObjectiveSpec, default_preference_plan, request_preference_plan
from daylight_workbench.ui import MainWindow


_APP = None

def application():
    global _APP
    _APP = W.QApplication.instance() or W.QApplication([])
    return _APP


def pump(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert predicate(), "Mocked Qt response did not complete in time"


@pytest.fixture
def page():
    app = application()
    widget = PreferenceDecisionPage()
    frame = pd.DataFrame({"candidate_id": [11, 12, 13], "UDI": [90., 70., 40.],
                          "EUI": [20., 44., 100.], "DGP": [.8, .38, .1],
                          "private_note": ["row-data-never-sent"]*3,
                          "pareto": [True, True, True]})
    objectives = [ObjectiveSpec("UDI", "UDI", "max", ("采光",), "%"),
                  ObjectiveSpec("EUI", "EUI", "min", ("能耗",), "kWh/m²·a"),
                  ObjectiveSpec("DGP", "DGP", "min", ("眩光",), "1")]
    widget.set_candidates(frame, objectives)
    yield app, widget
    widget.shutdown()
    widget.deleteLater()
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)
    app.processEvents()


def mock_answer(widget, weights, **changes):
    p = default_preference_plan(widget._objectives)
    p["weights"] = [{"column": o.column, "weight": w} for o, w in zip(widget._objectives, weights)]
    p.update(changes)
    return p


def install_mock(monkeypatch, plan, captured):
    def fake_send(settings, key, preview):
        def transport(req, timeout, max_bytes):
            captured.append(req)
            return json.dumps({"choices": [{"message": {"content": json.dumps(plan, ensure_ascii=False)}}]}).encode()
        return request_preference_plan(settings, key, preview, transport)
    monkeypatch.setattr("daylight_workbench.decision_page.request_preference_plan", fake_send)


def test_mapping_once_preview_exact_explicit_send_then_apply(page, monkeypatch):
    app, widget = page
    assert not widget._mapping_confirmed
    assert not widget.preview_button.isEnabled()
    widget.confirm_mapping()
    assert widget._mapping_confirmed
    plan = mock_answer(widget, [0, .5, .5], intent="能耗与眩光等权折中", assumptions=["仅对这两个目标等权，其余目标为零。"])
    captured, applied, selected = [], [], []
    install_mock(monkeypatch, plan, captured)
    widget.planApplied.connect(lambda f, p: applied.append((f, p)))
    widget.selectionChanged.connect(selected.append)
    widget.query_edit.setPlainText("我要能耗和眩光平衡的方案")
    widget.make_preview()
    exact = widget.preview_edit.toPlainText()
    assert captured == [] and applied == []
    assert "row-data-never-sent" not in exact
    assert '"candidate_id"' not in exact
    widget.key_edit.setText("mock-session-only")
    widget.send()
    pump(app, lambda: not widget._busy)
    assert len(captured) == 1
    assert captured[0].data.decode() == exact
    assert applied == [] and selected == [] and widget.ranking is None
    assert widget.apply_button.isEnabled()
    assert [widget.weight_table.cellWidget(i, 2).value() for i in range(3)] == [0, .5, .5]
    widget.apply_reviewed_plan()
    assert len(applied) == 1
    assert applied[0][0].candidate_id.tolist() == [12, 11, 13]
    pd.testing.assert_frame_equal(applied[0][0], widget._frame.loc[[1, 0, 2]])
    assert set(widget.selected_ids()) == {11, 12, 13}
    widget.recommendations.clearSelection()
    widget.recommendations.selectRow(0)
    assert widget.selected_ids() == [12]
    assert widget.interactive_plot.selected_ids() == [12]
    assert widget.comparison_figure.axes[0].patches
    assert "DeepSeek（建议，未应用）" in widget.chat.toPlainText()
    widget.shutdown()
    assert widget.key_edit.text() == ""


def test_multiturn_manual_weights_scope_constraints_preserved_until_reset(page, monkeypatch):
    app, widget = page
    widget.confirm_mapping()
    widget.weight_table.cellWidget(0, 2).setValue(.8)
    widget.weight_table.cellWidget(1, 2).setValue(.1)
    widget.weight_table.cellWidget(2, 2).setValue(.1)
    widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("all"))
    widget.add_constraint({"column": "DGP", "operator": "<=", "value": .6})
    widget.query_edit.setPlainText("保持约束，能耗和眩光平衡")
    widget.make_preview()
    first_state = copy.deepcopy(widget._preview["preference_state"])
    assert first_state["scope"] == "all"
    assert first_state["constraints"] == [{"column": "DGP", "operator": "<=", "value": .6}]
    assert [w["weight"] for w in first_state["weights"]] == [.8, .1, .1]
    answer = {**first_state, "weights": [{"column": c, "weight": w} for c, w in zip(["UDI", "EUI", "DGP"], [0, .5, .5])]}
    install_mock(monkeypatch, answer, [])
    widget.key_edit.setText("mock-only")
    widget.send()
    pump(app, lambda: not widget._busy)
    assert widget._mapping_confirmed  # no repeated semantic confirmation each turn
    assert widget.reviewed_plan()["constraints"] == first_state["constraints"]
    widget.apply_reviewed_plan()
    assert widget.ranking.frame.candidate_id.tolist() == [12, 13]
    before = widget.ranking.metrics.preference_score.tolist()
    widget.set_selection([13])
    widget.set_visible_ids([13])
    assert widget.ranking.metrics.preference_score.tolist() == before  # no invisible re-rank
    widget.query_edit.setPlainText("推荐两个，保持刚才的平衡")
    widget.make_preview()
    assert [w["weight"] for w in widget._preview["preference_state"]["weights"]] == [0, .5, .5]
    widget.reset_preferences()
    assert widget.ranking is None and widget._state is None
    assert widget._mapping_confirmed
    assert widget.constraint_table.rowCount() == 0


def test_cancel_ignores_late_response_and_repeated_send_is_single_post(page, monkeypatch):
    app, widget = page
    widget.confirm_mapping()
    widget.query_edit.setPlainText("帮我选更考虑采光的")
    widget.make_preview()
    widget.key_edit.setText("mock-only")
    release, started, calls = threading.Event(), threading.Event(), []
    answer = mock_answer(widget, [.6, .2, .2])
    def delayed(*args):
        calls.append(1)
        started.set()
        release.wait(3)
        return answer
    monkeypatch.setattr("daylight_workbench.decision_page.request_preference_plan", delayed)
    widget.send()
    assert started.wait(1)
    widget.send()
    assert calls == [1]
    assert widget._busy and not widget.apply_button.isEnabled()
    job = widget._request_id
    widget.cancel_request()
    assert not widget._busy and widget.query_edit.isEnabled()
    assert "可能仍完成并计费" in widget.status_label.text()
    release.set()
    # A deterministically delivered stale callback exercises the same ID guard.
    widget._request_done(job, answer, None)
    app.processEvents()
    assert widget.ranking is None
    assert [widget.weight_table.cellWidget(i, 2).value() for i in range(3)] == [1, 1, 1]
    assert widget._preview is None


def test_invalid_mapping_and_all_filtered_out_do_not_invent_rows(page):
    app, widget = page
    widget.mapping_table.item(0, 1).setText("")
    widget.confirm_mapping()
    assert not widget._mapping_confirmed
    widget.mapping_table.item(0, 1).setText("UDI")
    widget.confirm_mapping()
    widget.add_constraint({"column": "EUI", "operator": "<", "value": -1})
    widget.apply_reviewed_plan()
    assert widget.ranking.frame.empty
    assert widget.recommendations.rowCount() == 0
    assert "无符合项" in widget.chat.toPlainText()
    widget.clear_candidates()
    assert widget._frame.empty and not widget.apply_button.isEnabled()


def test_main_window_dedicated_fifth_page_and_bidirectional_ids():
    app = application()
    window = MainWindow()
    raw = pd.DataFrame({"UDI": [90., 70., 40.], "EUI": [20., 44., 100.], "DGP": [.8, .38, .1]})
    context = prepare_external_results(raw, [], list(raw.columns), {"UDI": "max", "EUI": "min", "DGP": "min"}, "synthetic")
    window.external_ready(context)
    page = window.decision_page
    assert window.tabs.count() == 5
    window.natural_language_filter()
    assert window.tabs.currentWidget() is page
    page.confirm_mapping()
    page._load_plan_editor(mock_answer(page, [0, .5, .5], limit=1))
    page.apply_reviewed_plan()
    assert window.selected_frame().candidate_id.tolist() == [2]
    assert window.interactive_plot.selected_ids() == [2]
    assert page.selected_ids() == [2]
    # Click an original full-context point outside the shortlist: reveal its
    # original row, preserve ID across both plots and result table.
    page._plot_selection([3])
    assert window.selected_frame().candidate_id.tolist() == [3]
    assert window.interactive_plot.selected_ids() == page.selected_ids() == [3]
    window.results_table.selectRow(0)
    assert page.selected_ids() == window.selected_frame().candidate_id.tolist()
    pd.testing.assert_frame_equal(context.frame[raw.columns], raw)
    window.close()
    window.deleteLater()
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_nonfinite_is_missing_in_graph_not_replaced_in_original_rows(page):
    app, widget = page
    frame = widget._frame.copy()
    frame.loc[0, "EUI"] = float("inf")
    widget.set_candidates(frame, widget._objectives)
    assert widget._frame.loc[0, "EUI"] == float("inf")
    assert widget.interactive_plot._payload["rows"][0]["values"]["EUI"] is None
    widget.confirm_mapping()
    widget.apply_reviewed_plan()
    assert 11 not in widget.ranking.frame.candidate_id.tolist()


def test_apply_current_view_does_not_redefine_its_own_top_k_scope():
    app = application()
    window = MainWindow()
    raw = pd.DataFrame({"UDI": [90., 70., 40.], "EUI": [20., 44., 100.], "DGP": [.8, .38, .1]})
    window.external_ready(prepare_external_results(raw, [], list(raw.columns), {"UDI": "max", "EUI": "min", "DGP": "min"}))
    page = window.decision_page
    page.confirm_mapping()
    page._load_plan_editor(mock_answer(page, [0, .5, .5], scope="current_view", limit=1))
    original_view = window.results_model.frame.candidate_id.tolist()
    page.apply_reviewed_plan()
    assert page.ranking.scope_count == 3
    assert window.results_model.frame.candidate_id.tolist() == original_view
    page.apply_reviewed_plan()
    assert page.ranking.scope_count == 3
    assert page.ranking.frame.candidate_id.tolist() == [2]
    window.close()
    window.deleteLater()
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)
    app.processEvents()


def test_project_restoration_replays_ids_to_both_plots_before_and_after_bridge_ready():
    """Regress the reopen bug: table had restored IDs while results plot had [].

    Uses real Qt/Plotly SVG and bridge reload, not a fake selected-header label.
    Scientific fitting is irrelevant to the UI state contract, so the trained
    context is a minimal fixture with exactly the same analysis-state fields.
    """
    from types import SimpleNamespace
    app = application()
    window = MainWindow()
    window.resize(1100, 760)
    window.show()
    source = pd.DataFrame({"candidate_id": [9, 13, 21],
                           "predicted::UDI": [80., 65., 40.],
                           "predicted::DGP": [.4, .2, .1],
                           "pareto": [True, True, True]})
    state = {"scatter_x": "UDI", "scatter_y": "DGP", "result_source": 1,
             "visible_ids": [9, 13, 21], "selected_ids": [9, 13], "ranges": {},
             "filter_plan": {"kind": "manual_ranges", "ranges": {}, "pareto_only": False}}
    window.trained = SimpleNamespace(data=SimpleNamespace(config=SimpleNamespace(inputs=[], targets=["UDI", "DGP"])),
                                     search_metadata={"analysis_state": state})
    window.optimization = SimpleNamespace(candidates=source, evaluations=3,
                                          config=SimpleNamespace(directions={"UDI": "max", "DGP": "min"}))
    window.show_results()
    window.tabs.setCurrentIndex(3)
    # Reopen restores immediately, before waiting for the bridge/browser. The
    # Python-side caches must already contain these exact original IDs.
    window.restore_analysis_state()
    assert window.selected_frame().candidate_id.tolist() == [9, 13]
    assert window.interactive_plot.selected_ids() == [9, 13]
    assert window.decision_page.selected_ids() == [9, 13]
    assert window.decision_page.interactive_plot.selected_ids() == [9, 13]

    def js(plot, expression):
        results = []
        plot.view.page().runJavaScript(expression, results.append)
        pump(app, lambda: bool(results), timeout=15)
        return results[0]

    pump(app, lambda: window.interactive_plot._bridge.ready_for_data, timeout=20)
    pump(app, lambda: js(window.interactive_plot, "document.getElementById('selected').textContent") == "2", timeout=20)
    pump(app, lambda: js(window.interactive_plot, "document.querySelector('#scatter')?.data?.at(-1)?.x?.length") == 2, timeout=20)
    if js(window.interactive_plot, "document.querySelector('#parallel svg') !== null"):
        assert js(window.interactive_plot, "document.querySelectorAll('#parallel path[stroke=\"#f2a430\"]').length") == 2
    else:
        assert js(window.interactive_plot, "document.querySelector('#parallel')?.data?.[0]?.line?.color?.filter(v=>v===2).length") == 2
    # A bridge/browser reload after restoration must replay the cached selection,
    # rather than resetting the chart to 0 selected while the table remains set.
    window.interactive_plot.view.reload()
    pump(app, lambda: not window.interactive_plot._bridge.ready_for_data, timeout=10)
    pump(app, lambda: window.interactive_plot._bridge.ready_for_data, timeout=20)
    pump(app, lambda: js(window.interactive_plot, "document.getElementById('selected').textContent") == "2", timeout=20)
    pump(app, lambda: js(window.interactive_plot, "document.querySelector('#scatter')?.data?.at(-1)?.customdata?.join(',')") == "9,13", timeout=20)
    window.tabs.setCurrentIndex(4)
    window.decision_page.detail_tabs.setCurrentIndex(2)
    pump(app, lambda: window.decision_page.interactive_plot._bridge.ready_for_data, timeout=20)
    pump(app, lambda: js(window.decision_page.interactive_plot, "document.getElementById('selected').textContent") == "2", timeout=20)
    assert window.selected_frame().candidate_id.tolist() == [9, 13]
    window.close()
    window.deleteLater()
    C.QCoreApplication.sendPostedEvents(None, C.QEvent.Type.DeferredDelete)
    app.processEvents()
