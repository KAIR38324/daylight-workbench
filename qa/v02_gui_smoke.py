"""Bounded, entirely synthetic v0.2 GUI integration; no provider request.

Run with QT_QPA_PLATFORM=offscreen python qa/v02_gui_smoke.py. The screenshots
show real rendered widgets, but are not a substitute for Windows acceptance.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'daylight-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'daylight-cache'))

import numpy as np
import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daylight_workbench.core import DataConfig, prediction_column
from daylight_workbench.demo import make_demo
from daylight_workbench.external_results import ExternalMappingDialog
from daylight_workbench.llm_filter import apply_plan, validate_plan
from daylight_workbench.llm_dialog import NaturalLanguageFilterDialog
from daylight_workbench.ui import MainWindow


def wait_idle(window, app, timeout=50):
    deadline = time.monotonic() + timeout
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert window.thread is None, 'Background task exceeded bounded smoke timeout'


def shot(window, name):
    app = W.QApplication.instance()
    for _ in range(8):
        app.processEvents()
        time.sleep(.03)
    path = ROOT / 'qa' / 'evidence' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    assert window.grab().save(str(path))
    return str(path.relative_to(ROOT))


def main():
    app = W.QApplication.instance() or W.QApplication([])
    from daylight_workbench.theme import apply_theme
    apply_theme(app)
    window = MainWindow()
    window.resize(1240, 900)
    window.show()
    raw = make_demo(120)
    columns = list(raw)
    data_path = ROOT / 'qa' / 'evidence' / 'v02-synthetic-input.csv'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(data_path, index=False)
    # Exercise the import button path. Only the test chooser is replaced.
    original_open = W.QFileDialog.getOpenFileName
    W.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(data_path), 'CSV'))
    try:
        window.import_data()
    finally:
        W.QFileDialog.getOpenFileName = original_open
    config = DataConfig(columns[:10], columns[10:], directions=dict(zip(columns[10:], ['max', 'min', 'min'])))
    window.set_raw(window.raw, 'SYNTHETIC IMPORT ONLY', config, config.directions)
    assert len(window.prepare().frame) == 120

    window.tabs.setCurrentIndex(1)
    window.train_tabs.setCurrentIndex(0)
    window.layer_builder.set_layers([__import__('daylight_workbench.modeling', fromlist=['LayerSpec']).LayerSpec(4, 'tanh')])
    window.epochs.setValue(12)
    window.patience.setValue(5)
    window.batch.setValue(64)
    shots = [shot(window, 'v02-ann-editor-1240x900.png')]
    window.start_training()
    wait_idle(window, app)
    assert window.trained is not None and window.trained.history
    shots.append(shot(window, 'v02-ann-training-1240x900.png'))
    window.epochs.setValue(2)
    window.patience.setValue(1)

    controls = window.search_controls
    controls.compare_all.setChecked(True)
    controls.fold_count.setValue(3)
    controls.max_fits.setValue(18)
    controls.threads.setValue(1)
    controls.ann_rates.setText('0.01')
    controls.rf_trees.setText('3')
    controls.rf_depth.setText('3')
    controls.xgb_rounds.setText('3')
    controls.xgb_depth.setText('2')
    controls.xgb_rates.setText('0.1')
    window.train_tabs.setCurrentIndex(1)
    assert window.preview_search() is not None
    shots.append(shot(window, 'v02-grid-cv-1240x900.png'))
    window.start_search()
    wait_idle(window, app)
    assert window.trained is not None
    meta = window.trained.search_metadata
    assert len(meta['candidates']) == 3
    assert [c['family'] for c in meta['candidates']] == ['ann', 'rf', 'xgboost']
    assert meta['plan']['fit_budget'] == 18
    assert len(window.trained.fit_indices) == len(meta['plan']['development'])
    assert {'MAE', 'MSE', 'RMSE', 'R2'} <= set(window.metrics_model.frame)
    assert window.bounds.rowCount() == 10
    assert window.bounds.item(0, 5).text()
    shots.append(shot(window, 'v02-model-metrics-1240x900.png'))

    window.population.setValue(4)
    window.generations.setValue(2)
    floor_row = columns.index('floor_level')
    kind = window.bounds.cellWidget(floor_row, 3)
    kind.setCurrentIndex(kind.findData('integer'))
    window.start_optimization()
    wait_idle(window, app)
    assert window.optimization is not None
    assert len(window.results_model.frame) > 0
    window.result_source.setCurrentIndex(1)
    window.apply_filter()
    frame = window.results_model.frame
    chosen_id = frame.candidate_id.iloc[0]
    window.plot_selection_changed([chosen_id])
    assert window.selected_frame().candidate_id.tolist() == [chosen_id]
    window.results_table.clearSelection()
    window.results_table.selectRow(0)
    assert window.interactive_plot.selected_ids() == [chosen_id]
    window.tabs.setCurrentIndex(3)
    window.result_plots.setCurrentIndex(0)
    deadline = time.monotonic() + 8
    while not window.interactive_plot._bridge.ready_for_data and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    assert window.interactive_plot._bridge.ready_for_data, 'Offline plot page did not load'
    for _ in range(25):
        app.processEvents()
        time.sleep(.02)
    shots.append(shot(window, 'v02-results-1240x900.png'))

    objective = prediction_column(columns[10])
    minimum = float(frame[objective].min())
    plan = validate_plan({'filters': [{'column': objective, 'operator': '>=', 'value': minimum}],
                          'sort': [], 'limit': 2, 'needs_clarification': False,
                          'explanation': 'Mocked local smoke plan.'}, frame)
    # The dialog's transport is not invoked. Exercise the MainWindow handoff
    # with the same reviewed, deterministic plan and retain candidate IDs.
    original_exec = NaturalLanguageFilterDialog.exec
    def mocked_exec(dialog):
        dialog.plan = plan
        dialog.result_frame = apply_plan(dialog._frame, plan)
        return W.QDialog.DialogCode.Accepted
    NaturalLanguageFilterDialog.exec = mocked_exec
    try:
        window.natural_language_filter()
    finally:
        NaturalLanguageFilterDialog.exec = original_exec
    assert window.results_model.frame.candidate_id.tolist() == frame.candidate_id.tolist()[:2]
    assert window.last_filter_plan == plan

    project = ROOT / 'qa' / 'evidence' / 'v02-roundtrip.dwb'
    original_save = W.QFileDialog.getSaveFileName
    W.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(project), 'Daylight project (*.dwb)'))
    try:
        window.save_project()
    finally:
        W.QFileDialog.getSaveFileName = original_save
    assert project.is_file()
    W.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(project), 'Daylight project (*.dwb)'))
    try:
        window.open_project()
    finally:
        W.QFileDialog.getOpenFileName = original_open
    wait_idle(window, app)
    assert window.trained is not None and window.optimization is not None
    assert window.results_model.frame.candidate_id.tolist() == frame.candidate_id.tolist()[:2]

    external_path = ROOT / 'qa' / 'evidence' / 'v02-external-objectives-only.csv'
    pd.DataFrame({'score_a': [2., 1., 3.], 'score_b': [1., 2., 3.]}).to_csv(external_path, index=False)
    original_mapping_exec = ExternalMappingDialog.exec
    def mapped_exec(dialog):
        dialog.mapping = ([], ['score_a', 'score_b'], {'score_a': 'min', 'score_b': 'min'})
        return W.QDialog.DialogCode.Accepted
    ExternalMappingDialog.exec = mapped_exec
    W.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(external_path), 'CSV'))
    try:
        window.import_external_results()
    finally:
        ExternalMappingDialog.exec = original_mapping_exec
        W.QFileDialog.getOpenFileName = original_open
    wait_idle(window, app)
    assert window.external_results is not None
    assert window.external_results.inputs == []
    assert list(window.external_results.frame.candidate_id) == [1, 2, 3]
    assert window.results_model.frame.columns.isin(['score_a', 'score_b']).sum() == 2

    report = {'source': 'Synthetic-only Linux offscreen Qt, not Windows manual acceptance',
              'candidate_models': [c['family'] for c in meta['candidates']],
              'selected_model': meta['candidates'][meta['selected_index']]['family'],
              'fit_budget': meta['plan']['fit_budget'],
              'locked_test_rows': len(meta['plan']['test']),
              'development_rows': len(meta['plan']['development']),
              'screenshots': shots,
              'steps': ['CSV import', 'group-safe CV ANN/RF/XGBoost', 'NSGA-II', 'table/plot stable-ID selection',
                        'mocked DeepSeek plan locally applied', 'project save/reopen', 'objectives-only external CSV import']}
    (ROOT / 'qa' / 'evidence' / 'v02-gui-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    window.close()
    app.processEvents()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
