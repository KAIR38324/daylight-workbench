"""Reproducible, synthetic-only, real Qt end-to-end getting-started journey.

Run from repository root: QT_QPA_PLATFORM=offscreen .venv/bin/python qa/user_journey.py
The only patched UI elements are native file choosers and the external mapping
modal's timed human-equivalent selections. Scientific workers, plots, exports,
and save/load use their actual application paths. No provider request occurs.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'daylight-manual-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'daylight-manual-cache'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W
from daylight_workbench.core import prediction_column
from daylight_workbench.demo import make_demo
from daylight_workbench.external_results import ExternalMappingDialog
from daylight_workbench.llm_dialog import NaturalLanguageFilterDialog
from daylight_workbench.modeling import LayerSpec
from daylight_workbench.theme import apply_theme
from daylight_workbench.ui import MainWindow

OUT = ROOT / 'qa/evidence/user-journey'
SHOTS = ROOT / 'docs/getting-started'
DATA = ROOT / 'demo/manual-synthetic.csv'
SEED = 20261020


def pump(app, seconds=.4):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(.008)


def wait(app, predicate, seconds=100):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.008)
    raise AssertionError('Timed out waiting for real application worker or chart')


def js(app, window, expression):
    result = []
    window.interactive_plot.view.page().runJavaScript(expression, result.append)
    wait(app, lambda: bool(result), 15)
    return result[0]


def with_open(path, func):
    old = W.QFileDialog.getOpenFileName
    W.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(path), 'CSV'))
    try:
        return func()
    finally:
        W.QFileDialog.getOpenFileName = old


def with_save(path, func):
    old = W.QFileDialog.getSaveFileName
    W.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(path), 'CSV'))
    try:
        return func()
    finally:
        W.QFileDialog.getSaveFileName = old


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    raw = make_demo(240, seed=SEED)
    raw.to_csv(DATA, index=False, float_format='%.17g')
    source_sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    app = W.QApplication.instance() or W.QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.resize(1440, 1000)
    window.show()
    shots = []

    def shot(number, slug, caption, widget=None, metadata=None, settle=True):
        if settle:
            pump(app, .7)
        path = SHOTS / f'{number:02d}-{slug}.png'
        assert (widget or window).grab().save(str(path))
        item = {'number': number, 'file': str(path.relative_to(ROOT)), 'caption': caption,
                'width': (widget or window).width(), 'height': (widget or window).height(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if metadata:
            item.update(metadata)
        shots.append(item)
        print(f'CAPTURE {number:02d} {slug}', flush=True)
        return item

    with_open(DATA, window.import_data)
    assert window.raw is not None and len(window.raw) == 240
    shot(1, 'import-csv', '点击“导入 CSV / XLSX”，选择明确标注为合成的 CSV。')
    for i, name in enumerate(raw.columns):
        role = 'input' if i < 10 else 'target'
        combo = window.roles.cellWidget(i, 1)
        combo.setCurrentIndex(combo.findData(role))
        if role == 'target':
            direction = ['max', 'min', 'min'][i-10]
            combo = window.roles.cellWidget(i, 2)
            combo.setCurrentIndex(combo.findData(direction))
    assert window.data_config().inputs == list(raw.columns[:10])
    assert window.data_config().targets == list(raw.columns[10:])
    shot(2, 'assign-roles', '前 10 列设为自变量 X，后 3 列设为目标 Y；依次选择最大、最小、最小。')
    prepared = window.prepare()
    assert sum(prepared.report['split_counts'].values()) == len(prepared.frame)
    shot(3, 'validate-split', '点击“校验数据与划分”，确认有效行数与训练、验证、测试拆分。')

    window.tabs.setCurrentIndex(1)
    window.train_tabs.setCurrentIndex(0)
    window.layer_builder.set_layers([LayerSpec(64, 'relu'), LayerSpec(32, 'tanh')])
    window.epochs.setValue(180)
    window.patience.setValue(24)
    window.lr.setText('0.003')
    window.batch.setValue(32)
    shot(4, 'ann-settings', '在 ANN 结构与参数中设两层 64 ReLU、32 Tanh；最多 180 轮、耐心 24、学习率 0.003、批量 32。')

    live_training_captured = [False]
    class TrainingObserver(C.QObject):
        @C.Slot(object)
        def on_snapshot(self, event):
            if live_training_captured[0] or event.get('kind') != 'loss' or event.get('epoch', 0) < 15:
                return
            assert C.QThread.currentThread() is app.thread()
            if window.thread is None or not window.thread.isRunning() or '训练中' not in window.training_stats['phase'].text():
                return
            window.training_canvas.draw()
            displayed = int(window.training_stats['epoch'].text())
            plotted = int(window.training_figure.axes[0].lines[0].get_xdata()[-1])
            if displayed == plotted == event['epoch']:
                live_training_captured[0] = True
                shot(5, 'ann-live-loss', '点击“单次 ANN（快速）”；训练中查看实时轮次与真实训练、验证损失。',
                     metadata={'epoch': displayed, 'worker_active': True, 'plotted_through_epoch': plotted}, settle=False)
    observer = TrainingObserver()
    window.start_training()
    window.worker.snapshot.connect(observer.on_snapshot, C.Qt.ConnectionType.QueuedConnection)
    wait(app, lambda: window.thread is None, 130)
    assert any(s['number'] == 5 for s in shots)
    assert window.trained is not None and window.trained.history
    quick_ann_epochs = len(window.trained.history)
    window.train_tabs.setCurrentIndex(2)
    assert {'MAE', 'MSE', 'R2'} <= set(window.metrics_model.frame)
    quick_metrics = window.metrics_model.frame.to_dict(orient='records')
    shot(6, 'ann-test-metrics', '训练完成后，在“指标 / 实时曲线”查看独立测试集各目标误差与实测、预测图。')

    # Reconfigure the actual search controls. This invalidates the quick ANN as intended.
    window.epochs.setValue(30)
    window.patience.setValue(8)
    controls = window.search_controls
    controls.compare_all.setChecked(True)
    controls.mode.setCurrentIndex(controls.mode.findData('cv'))
    controls.fold_count.setValue(5)
    controls.max_fits.setValue(40)
    controls.threads.setValue(1)
    controls.ann_rates.setText('0.003')
    controls.ann_batch.setText('32')
    controls.rf_trees.setText('40')
    controls.rf_depth.setText('8')
    controls.xgb_rounds.setText('30')
    controls.xgb_depth.setText('3')
    controls.xgb_rates.setText('0.1')
    window.train_tabs.setCurrentIndex(1)
    shot(7, 'cv-grid-settings', '勾选跨模型比较，保留严格模式 5 折 CV，设置有上限的 ANN、RF、XGBoost 网格。')
    assert window.preview_search() is not None
    shot(8, 'cv-budget-preview', '点击“预览候选与拟合预算”；确认候选、折数、预算和锁定测试行数后启动。')
    window.start_search()
    wait(app, lambda: window.thread is None, 210)
    assert window.trained is not None
    metadata = window.trained.search_metadata
    assert [c['family'] for c in metadata['candidates']] == ['ann', 'rf', 'xgboost']
    assert metadata['plan']['fold_count'] == 5
    assert len(window.trained.fit_indices) == len(metadata['plan']['development'])
    window.train_tabs.setCurrentIndex(2)
    window.search_toggle_button.setChecked(True)
    shot(9, 'cv-model-selection', '在“指标 / 实时曲线”查看验证候选排名及最终独立测试集逐目标指标。')
    selected_family = window.trained.family
    heldout_metrics = window.metrics_model.frame.to_dict(orient='records')
    assert {'MAE','MSE','R2'} <= set(window.metrics_model.frame)

    window.tabs.setCurrentIndex(2)
    floor = list(raw.columns).index('floor_level')
    integer = window.bounds.cellWidget(floor, 3)
    integer.setCurrentIndex(integer.findData('integer'))
    window.bounds.scrollToItem(window.bounds.item(floor, 0), W.QAbstractItemView.ScrollHint.PositionAtCenter)
    window.population.setValue(32)
    window.generations.setValue(8)
    window.opt_seed.setValue(42)
    assert window.bounds.rowCount() == 10
    shot(10, 'opt-bounds', '检查 10 个变量的训练观测范围与搜索边界；将 floor_level 明确设为整数，并设置 NSGA-II 种群 32、代数 8。')

    live_optimization_captured = [False]
    class OptimizationObserver(C.QObject):
        @C.Slot(object)
        def on_snapshot(self, event):
            if live_optimization_captured[0] or event.get('kind') != 'optimization' or event.get('generation', 0) < 2:
                return
            if window.thread is None or not window.thread.isRunning():
                return
            window.live_opt_canvas.draw()
            live_optimization_captured[0] = True
            shot(11, 'nsga-live', '点击“运行 NSGA-II（仅代理预测）”，查看实时代数、评估次数及预测前沿。',
                 metadata={'generation': event['generation'], 'worker_active': True}, settle=False)
    opt_observer = OptimizationObserver()
    window.start_optimization()
    window.worker.snapshot.connect(opt_observer.on_snapshot, C.Qt.ConnectionType.QueuedConnection)
    wait(app, lambda: window.thread is None, 120)
    assert window.optimization is not None and any(s['number'] == 11 for s in shots)
    window.tabs.setCurrentIndex(2)
    shot(12, 'nsga-complete', '任务完成后核对实际评估数；优化候选仍只是代理模型预测。')

    window.tabs.setCurrentIndex(3)
    window.result_source.setCurrentIndex(1)
    window.apply_filter()
    wait(app, lambda: window.interactive_plot._bridge.ready_for_data, 20)
    wait(app, lambda: js(app, window, "document.querySelector('#scatter')?.data?.length") == 3, 20)
    assert len(window.interactive_plot._payload['dimensions']) == 13
    full = window.results_model.frame.copy()
    shot(13, 'results-all', '切到“结果分析与导出”，选择“全部候选”，查看二维目标空间与 13 轴平行坐标。')
    window.toggle_results_table()
    chosen = full.candidate_id.iloc[[0, len(full)//2, len(full)-1]].tolist()
    window.plot_selection_changed(chosen)
    window.interactive_plot.set_selection(chosen)
    wait(app, lambda: js(app, window, "document.getElementById('selected').textContent") == '3', 15)
    assert set(window.selected_frame().candidate_id) == set(chosen)
    window.results_table.sortByColumn(0, C.Qt.SortOrder.DescendingOrder)
    assert set(window.selected_frame().candidate_id) == set(chosen)
    shot(14, 'linked-selection', '显示表格，在图中选 3 个稳定候选 ID；表格排序后选择仍与图同步。', metadata={'selected_ids': chosen})
    window.toggle_results_table()
    for number, view, slug, caption in [
        (15,'matrix','objective-matrix','切换目标两两散点矩阵，检查已选方案的关联高亮。'),
        (16,'distribution','objective-distribution','切换各目标分布，查看已选方案所处位置。'),
        (17,'comparison','selected-comparison','切换已选方案比较，以原始单位对照三项目标。'),
        (18,'correlation','input-target-correlation','切换输入与目标相关性热图；只作相关描述，不推断因果。')]:
        js(app, window, f"document.querySelector('[data-analysis-view=\"{view}\"]').click()")
        assert set(window.interactive_plot.selected_ids()) == set(chosen)
        shot(number, slug, caption)
    js(app, window, "document.querySelector('[data-analysis-view=\"frontier\"]').click()")
    assert js(app, window, "document.getElementById('mode3').disabled") is True

    # Manual numeric range filtering, in the same source units as the displayed objective.
    window.toggle_filters()
    objective = prediction_column(raw.columns[10])
    threshold = float(full[objective].quantile(.60))
    row = next(i for i in range(window.filters.rowCount()) if window.filters.item(i, 0).text() == objective)
    window.filters.cellWidget(row, 1).setText(format(threshold, '.12g'))
    window.filters.scrollToItem(window.filters.item(row, 0), W.QAbstractItemView.ScrollHint.PositionAtCenter)
    shot(19, 'manual-filter-settings', '展开“区间筛选”，在预测采光分数下限输入演示阈值；留空的边界不限制。',
         metadata={'column': objective, 'lower_bound': threshold})
    window.apply_filter()
    filtered = window.results_model.frame.copy()
    assert 0 < len(filtered) < len(full)
    assert (filtered[objective] >= float(window.filters.cellWidget(row, 1).text())).all()
    shot(20, 'manual-filter-result', '点击“应用筛选”，确认当前行数减少且数值与 Pareto 原标记未被改写。')
    chosen_filtered = filtered.candidate_id.iloc[[0, min(1,len(filtered)-1)]].drop_duplicates().tolist()
    window.plot_selection_changed(chosen_filtered)
    window.interactive_plot.set_selection(chosen_filtered)
    exported = OUT / 'selected-candidates.csv'
    with_save(exported, lambda: window.export_rows('selected'))
    exported_frame = pd.read_csv(exported, encoding='utf-8-sig')
    original_selected = full.set_index('candidate_id').loc[chosen_filtered].reset_index()
    assert exported_frame.candidate_id.tolist() == chosen_filtered
    for col in original_selected.select_dtypes(include='number').columns:
        assert np.allclose(exported_frame[col], original_selected[col], rtol=1e-12, atol=1e-12), col
    provenance = json.loads((OUT/'selected-candidates.csv.provenance.json').read_text(encoding='utf-8'))
    assert provenance['csv_sha256'] == hashlib.sha256(exported.read_bytes()).hexdigest()
    shot(21, 'export-selected', '点击“导出已选方案”，生成 CSV 与来源 JSON；按候选 ID 和各数值列逐项核对导出。',
         metadata={'exported_ids': chosen_filtered})

    project = OUT/'manual-roundtrip.dwb'
    with_save(project, window.save_project)
    assert project.is_file()
    shot(22, 'project-saved', '点击顶端“保存项目”，将模型、划分和优化结果保存为 .dwb；项目文件未加密。')
    with_open(project, window.open_project)
    wait(app, lambda: window.thread is None, 60)
    assert window.trained is not None and window.optimization is not None
    assert window.results_model.frame.candidate_id.tolist() == filtered.candidate_id.tolist()
    assert set(window.selected_frame().candidate_id) == set(chosen_filtered)
    shot(23, 'project-reopened', '点击“打开项目”重开本次 .dwb，确认候选 ID、筛选视图与已选方案恢复。')

    # Actual mapping dialog interactions, with a timer standing in for mouse clicks.
    external_path = OUT/'external-objectives-only.csv'
    pd.DataFrame({'score_a':[2.,1.,3.], 'score_b':[1.,2.,3.]}).to_csv(external_path,index=False)
    old_exec = ExternalMappingDialog.exec
    def configured_exec(dialog):
        def configure():
            for i in range(2):
                role = dialog.table.cellWidget(i,1)
                role.setCurrentIndex(role.findData('target'))
                direction = dialog.table.cellWidget(i,2)
                direction.setCurrentIndex(direction.findData('min'))
            shot(24, 'external-mapping', '点击“导入外部结果”，将仅有的两列数值映射为最小化目标，不虚构输入。', widget=dialog)
            dialog.validate_accept()
        C.QTimer.singleShot(100, configure)
        return old_exec(dialog)
    ExternalMappingDialog.exec = configured_exec
    try:
        with_open(external_path, window.import_external_results)
    finally:
        ExternalMappingDialog.exec = old_exec
    wait(app, lambda: window.thread is None, 30)
    assert window.external_results is not None and window.external_results.inputs == []
    assert window.results_model.frame.candidate_id.tolist() == [1,2,3] or len(window.results_model.frame) > 0
    window.result_source.setCurrentIndex(1)
    window.apply_filter()
    assert window.results_model.frame.candidate_id.tolist() == [1,2,3]
    shot(25, 'external-results', '导入后显示外部数值结果；保留源行 ID 和原目标列名，不标称代理预测或物理验证。')

    # Real opt-in dialog, preview only. It is intentionally never sent online.
    def preview_dialog():
        dialog = next(w for w in app.topLevelWidgets() if isinstance(w, NaturalLanguageFilterDialog) and w.isVisible())
        dialog.query_edit.setPlainText('筛选 score_a 大于 1 的行，按 score_b 从小到大排序')
        dialog._make_preview()
        assert dialog._preview is not None and dialog.send_button.isEnabled() and not dialog.apply_button.isEnabled()
        assert 'api.deepseek.com' in dialog.status_label.text()
        assert 'candidate_id' not in dialog.preview_edit.toPlainText()
        shot(26, 'llm-outbound-preview', '可选 DeepSeek：输入问题并点击“生成发送预览”；检查目标主机与完整请求体。此处不填写密钥、不发送。', widget=dialog)
        dialog.reject()
    C.QTimer.singleShot(100, preview_dialog)
    window.natural_language_filter()
    assert window.last_filter_plan['kind'] == 'manual_ranges' or window.last_filter_plan is None

    source_files = ['daylight_workbench/ui.py', 'daylight_workbench/theme.py',
                    'daylight_workbench/modeling_widgets.py',
                    'daylight_workbench/plot_assets/index.html',
                    'daylight_workbench/plot_assets/style.css',
                    'daylight_workbench/plot_assets/plot.js']
    source_fingerprint = hashlib.sha256(b''.join((ROOT/file).read_bytes() for file in source_files)).hexdigest()
    project_sha = hashlib.sha256(project.read_bytes()).hexdigest()
    result = {
        'kind': 'synthetic-only-real-qt-user-journey', 'seed': SEED, 'data_file': str(DATA.relative_to(ROOT)),
        'data_sha256': source_sha, 'source_files': source_files, 'source_fingerprint_sha256': source_fingerprint,
        'source_commit': os.environ.get('DAYLIGHT_SOURCE_COMMIT'),
        'row_count': len(raw), 'input_columns': list(raw.columns[:10]),
        'targets': list(raw.columns[10:]), 'directions': window.directions(),
        'split_report': prepared.report, 'quick_ann_epochs': quick_ann_epochs,
        'quick_ann_metrics': [row for row in quick_metrics if row['划分'] == 'test'],
        'search_summary': {key:metadata['plan'][key] for key in ['mode','seed','candidate_count','fold_count','candidate_fold_jobs','fit_budget','test_fraction_actual','selection_score']},
        'search_locked_test_rows': len(metadata['plan']['test']),
        'search_development_rows': len(metadata['plan']['development']),
        'candidate_models': [c['family'] for c in metadata['candidates']],
        'selected_model': selected_family,
        'heldout_metrics': [row for row in heldout_metrics if row['划分'] == 'test'],
        'optimization_evaluations': window.optimization.evaluations,
        'optimization_candidates': len(window.optimization.candidates),
        'selected_ids_before_filter': chosen, 'filtered_rows': len(filtered),
        'exported_ids': chosen_filtered, 'export_csv_sha256': hashlib.sha256(exported.read_bytes()).hexdigest(),
        'project_sha256': project_sha, 'project_saved_reopened': True,
        'external_only_targets': window.external_results.targets,
        'webgl_3d_executed': False, 'online_llm_executed': False, 'browser_sandbox_disabled': False,
        'screenshots': shots,
    }
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    project.unlink()  # Do not commit a trained-model binary; the SHA records the verified round-trip.
    window.close()
    pump(app, .1)
    print(json.dumps({'selected_model': selected_family, 'candidate_count':len(full),'exported_ids':chosen_filtered,'shots':len(shots)}, ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
