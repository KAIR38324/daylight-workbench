"""Offscreen native Qt regressions, no browser rendering or online claims."""
from dataclasses import asdict
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pandas as pd
from PySide6 import QtWidgets as W

from daylight_workbench.core import DataConfig, TrainConfig, prepare_data, train_model
from daylight_workbench.modeling import CompositeSurrogate, LayerSpec, ModelSpec, SearchConfig, run_model_search
from daylight_workbench.project import save_project
from daylight_workbench.ui import MainWindow


def application():
    return W.QApplication.instance() or W.QApplication([])


def small_data():
    x = np.linspace(-3, 3, 48)
    return prepare_data(pd.DataFrame({'x': x, 'z': np.cos(x), '采光': 2*x, '能耗': x*x}),
                        DataConfig(['x', 'z'], ['采光', '能耗'], directions={'采光': 'max', '能耗': 'min'}))


def wait_done(app, window):
    deadline = time.monotonic() + 15
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert window.thread is None


def test_ui_defaults_independent_and_network_preview_has_scalar_output():
    app, window = application(), MainWindow()
    window.load_demo()
    window.prepare()
    assert window.search_controls.target_mode.currentData() == 'independent'
    assert window.search_controls.compare_all.isChecked()
    assert window.layer_builder.io == (10, 1)
    data, config = window.preview_search()
    assert config.target_mode == 'independent'
    assert '3 目标' in window.search_label.text()
    assert '总拟合预算 48/200' in window.search_label.text()
    window.search_controls.target_mode.setCurrentIndex(window.search_controls.target_mode.findData('shared'))
    assert window.layer_builder.io == (10, 3)
    assert window.search_controls.configuration(2, 1).target_mode == 'shared'
    window.close()
    app.processEvents()


def test_per_target_display_real_forest_records_and_invalidation_clears_decisions():
    app, window = application(), MainWindow()
    data = small_data()
    model = run_model_search(data, SearchConfig([ModelSpec('rf', {'n_estimators': 3, 'max_depth': 2}, [])],
                             folds=2, max_fits=6, cpu_threads=1)).selected
    window.set_raw(data.frame.copy(), 'synthetic', data.config, data.config.directions)
    window.training_ready(model)
    assert window.metrics_model.rowCount() == 4
    assert window.search_model.rowCount() == 2
    assert '逐目标独立模型' in window.selected_model_summary.text()
    assert all(target in window.selected_model_summary.text() for target in data.config.targets)
    for index, target in enumerate(data.config.targets):
        window.target_plot.setCurrentIndex(index)
        assert target in window.training_stats['phase'].toolTip()
        assert window.training_stats['loss'].text() == '不适用'
        assert not window.training_figure.axes[1].lines  # real trees, never fake loss
        assert len(window.training_figure.axes[1].patches) == 1
    cleared = []
    window.decision_page.clear_candidates = lambda: cleared.append(True)
    window.invalidate_training()
    assert cleared
    assert window.trained is None and window.search_model.rowCount() == 0
    window.close()
    app.processEvents()


def test_independent_quick_ann_uses_scalar_models_and_common_test():
    app, window = application(), MainWindow()
    data = small_data()
    window.set_raw(data.frame.copy(), 'synthetic', data.config, data.config.directions)
    window.layer_builder.set_layers([LayerSpec(3, 'tanh')])
    window.epochs.setValue(2)
    window.patience.setValue(1)
    window.start_training()
    wait_done(app, window)
    assert isinstance(window.trained, CompositeSurrogate)
    assert window.trained.search_metadata['plan']['mode'] == 'fixed'
    assert window.trained.search_metadata['plan']['fit_budget'] == 4
    assert all(model.data.config.targets == [target] for target, model in window.trained.target_models.items())
    window.close()
    app.processEvents()


def test_ui_reloads_composite_and_preserves_exact_saved_candidate_specs(tmp_path, monkeypatch):
    app, window = application(), MainWindow()
    data = small_data()
    # Include a valid parameter not exposed in the compact grid editor. Reopen
    # and preview must preserve it, not silently regenerate a different search.
    config = SearchConfig([ModelSpec('rf', {'n_estimators': 3, 'max_depth': 2, 'min_samples_leaf': 2}, [])],
                           folds=2, max_fits=6, cpu_threads=1)
    model = run_model_search(data, config).selected
    path = tmp_path/'independent.dwb'
    save_project(path, model)
    monkeypatch.setattr(W.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(path), ''))
    window.open_project()
    wait_done(app, window)
    assert isinstance(window.trained, CompositeSurrogate)
    assert window.search_controls.target_mode.currentData() == 'independent'
    restored = window.search_controls.configuration(window.epochs.value(), window.patience.value())
    assert asdict(restored) == asdict(config)
    np.testing.assert_array_equal(window.trained.predict(data.X), model.predict(data.X))
    assert '已恢复项目完整' in window.search_controls.loaded_notice.text()
    window.search_controls.seed.setValue(99)
    reseeded = window.search_controls.configuration(window.epochs.value(), window.patience.value())
    assert reseeded.candidates[0].params['min_samples_leaf'] == 2
    assert reseeded.seed == 99
    window.search_controls.rf_trees.setText('4')
    edited = window.search_controls.configuration(2, 1)
    assert edited.candidates[0].params['n_estimators'] == 4
    assert 'min_samples_leaf' not in edited.candidates[0].params
    window.close()
    app.processEvents()


def test_ui_reloads_legacy_ann_as_shared_not_new_default(tmp_path, monkeypatch):
    app, window = application(), MainWindow()
    data = small_data()
    model = train_model(data, TrainConfig([3], epochs=2, patience=1, batch_size=16))
    path = tmp_path/'legacy.dwb'
    save_project(path, model)
    monkeypatch.setattr(W.QFileDialog, 'getOpenFileName', lambda *args, **kwargs: (str(path), ''))
    window.open_project()
    wait_done(app, window)
    assert window.trained is not None and not isinstance(window.trained, CompositeSurrogate)
    assert window.search_controls.target_mode.currentData() == 'shared'
    assert window.layer_builder.io == (2, 2)
    assert not window.search_controls.compare_all.isChecked()
    window.close()
    app.processEvents()
