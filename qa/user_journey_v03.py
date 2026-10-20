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
from daylight_workbench.modeling import LayerSpec, CompositeSurrogate
import daylight_workbench.decision_page as decision_page_module
from daylight_workbench.preference_engine import default_preference_plan
from daylight_workbench.theme import apply_theme
from daylight_workbench.ui import MainWindow

OUT = ROOT / 'qa/evidence/user-journey-v03'
SHOTS = ROOT / 'docs/getting-started-v03'
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
    assert window.search_controls.target_mode.currentData() == 'independent'
    window.layer_builder.set_layers([LayerSpec(64, 'relu'), LayerSpec(32, 'tanh')])
    window.epochs.setValue(180)
    window.patience.setValue(24)
    window.lr.setText('0.003')
    window.batch.setValue(32)
    shot(4, 'ann-settings', '保持默认逐目标独立模式；在 ANN 结构与参数中设两层 64 ReLU、32 Tanh；最多 180 轮、耐心 24、学习率 0.003、批量 32。')

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
    assert isinstance(window.trained, CompositeSurrogate)
    assert set(window.trained.target_models) == set(raw.columns[10:])
    assert all(m.history for m in window.trained.target_models.values())
    quick_ann_epochs = {target: len(model.history) for target, model in window.trained.target_models.items()}
    window.train_tabs.setCurrentIndex(2)
    assert {'MAE', 'MSE', 'R2'} <= set(window.metrics_model.frame)
    quick_metrics = window.metrics_model.frame.to_dict(orient='records')
    shot(6, 'ann-test-metrics', '训练完成后，在“指标 / 实时曲线”查看三个独立 ANN 的逐目标测试误差；目标选择框切换对应模型记录。')

    # Reconfigure the actual search controls. This invalidates the quick ANN as intended.
    window.epochs.setValue(30)
    window.patience.setValue(8)
    controls = window.search_controls
    controls.compare_all.setChecked(True)
    controls.mode.setCurrentIndex(controls.mode.findData('cv'))
    controls.fold_count.setValue(5)
    controls.max_fits.setValue(50)
    controls.threads.setValue(1)
    controls.ann_rates.setText('0.003')
    controls.ann_batch.setText('32')
    controls.rf_trees.setText('40')
    controls.rf_depth.setText('8')
    controls.xgb_rounds.setText('30')
    controls.xgb_depth.setText('3')
    controls.xgb_rates.setText('0.1')
    window.train_tabs.setCurrentIndex(1)
    assert controls.target_mode.currentData() == 'independent'
    shot(7, 'cv-grid-settings', '确认默认“逐目标独立建模”，勾选跨模型比较，保留严格模式 5 折 CV，设置有上限的 ANN、RF、XGBoost 候选。')
    assert window.preview_search() is not None
    shot(8, 'cv-budget-preview', '点击“预览候选与拟合预算”；确认 3 目标 × 3 候选 × 5 折 + 3 次重拟合、预算和锁定测试行数后启动。')
    window.start_search()
    wait(app, lambda: window.thread is None, 210)
    assert window.trained is not None
    metadata = window.trained.search_metadata
    assert isinstance(window.trained, CompositeSurrogate)
    assert metadata['target_mode'] == 'independent'
    assert [c['family'] for c in metadata['config']['candidates']] == ['ann', 'rf', 'xgboost']
    assert set(metadata['target_results']) == set(raw.columns[10:])
    assert metadata['plan']['fit_budget'] == 48
    assert metadata['plan']['fold_count'] == 5
    assert len(window.trained.fit_indices) == len(metadata['plan']['development'])
    window.train_tabs.setCurrentIndex(2)
    window.search_toggle_button.setChecked(True)
    shot(9, 'cv-model-selection', '在“指标 / 实时曲线”查看各目标分别验证选优的候选排名，及最终独立测试集逐目标指标。')
    selected_families = {target: model.family for target, model in window.trained.target_models.items()}
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
    assert isinstance(window.trained, CompositeSurrogate) and window.optimization is not None
    assert window.search_controls.target_mode.currentData() == 'independent'
    assert {target: model.family for target, model in window.trained.target_models.items()} == selected_families
    assert window.trained.search_metadata['selected_index'] == metadata['selected_index']
    assert window.results_model.frame.candidate_id.tolist() == filtered.candidate_id.tolist()
    assert set(window.selected_frame().candidate_id) == set(chosen_filtered)
    wait(app, lambda: window.interactive_plot._bridge.ready_for_data, 20)
    wait(app, lambda: set(window.interactive_plot.selected_ids()) == set(chosen_filtered), 20)
    wait(app, lambda: js(app, window, "document.getElementById('selected').textContent") == str(len(chosen_filtered)), 20)
    wait(app, lambda: js(app, window, "document.querySelector('#scatter')?.data?.at(-1)?.x?.length") == len(chosen_filtered), 20)
    assert set(window.decision_page.interactive_plot.selected_ids()) == set(chosen_filtered)
    shot(23, 'project-reopened', '重开 .dwb 后核对模型、102 行筛选结果与已选 ID；二维散点和图表计数恢复 2 个高亮。')

    # The fifth page uses real UI preview, asynchronous Send and explicit local
    # Apply. Only the provider transport is replaced with two canned plans.
    # This must never be represented as a live DeepSeek service validation.
    window.tabs.setCurrentIndex(4)
    page = window.decision_page
    assert len(page._frame) == len(full)
    assert len(page._objectives) == 3
    for i, (label, alias) in enumerate([('采光演示分数', '采光,daylight'),
                                        ('眩光演示分数', '眩光,glare'),
                                        ('能耗演示分数', '能耗,energy')]):
        page.mapping_table.item(i, 1).setText(label)
        page.mapping_table.item(i, 3).setText(alias)
    page.confirm_mapping()
    assert page._mapping_confirmed
    assert [s.direction for s in page._objectives] == ['max', 'min', 'min']
    page.detail_tabs.setCurrentIndex(3)
    shot(24, 'decision-confirm-mapping', '进入第五页，按合成数据定义确认采光最大化、眩光和能耗最小化，并手动确认中文别名与未知单位。')

    mocked_calls = []
    original_request = decision_page_module.request_preference_plan
    def local_mock_plan(settings, key, preview):
        assert key == 'LOCAL_MOCK_NO_NETWORK'
        query = preview['query']
        assert query in ('帮我选更考虑采光的', '我要能耗和眩光平衡的方案')
        mocked_calls.append({'query': query, 'prior_state': preview.get('preference_state')})
        proposal = default_preference_plan(preview['objectives']) if preview.get('preference_state') is None else dict(preview['preference_state'])
        proposal['assumptions'] = ['本地模拟的结构化建议；不是 DeepSeek 在线响应。', '按完整候选固定归一化；权重可在应用前编辑。']
        if query == '帮我选更考虑采光的':
            weights = [.6, .2, .2]
            proposal['intent'] = '更考虑采光的方案（本地模拟建议）'
            proposal['explanation'] = '建议采光权重 60%，眩光和能耗各 20%；均为可审阅默认值。'
        else:
            weights = [0., .5, .5]
            proposal['intent'] = '能耗与眩光平衡的方案（本地模拟建议）'
            proposal['explanation'] = '建议能耗、眩光各 50%，采光 0%；不产生物理最优保证。'
        proposal['weights'] = [{'column': obj['column'], 'weight': weight}
                               for obj, weight in zip(preview['objectives'], weights)]
        return proposal

    decision_page_module.request_preference_plan = local_mock_plan
    try:
        page.query_edit.setPlainText('帮我选更考虑采光的')
        page.make_preview()
        assert page._preview is not None and page.send_button.isEnabled()
        assert 'api.deepseek.com' in page.status_label.text()
        assert 'candidate_id' not in page.preview_edit.toPlainText()
        shot(25, 'daylight-outbound-preview', '输入“帮我选更考虑采光的”，点击“预览发送内容”；本次只检查传出正文，不使用真实密钥。')
        page.key_edit.setText('LOCAL_MOCK_NO_NETWORK')
        page.send()
        wait(app, lambda: not page._busy and page._proposed is not None and '更考虑采光' in page._proposed['intent'], 20)
        assert page.ranking is None
        shot(26, 'daylight-mock-proposal', '本地模拟结构化响应：审核采光 60%、眩光 20%、能耗 20%；此时尚未应用，未联网。')
        page.apply_reviewed_plan()
        assert page.ranking is not None and len(page.ranking.frame) > 0
        first_recommended = page.ranking.frame.candidate_id.tolist()
        assert set(first_recommended[:3]) <= set(window.results_model.frame.candidate_id)
        assert set(page.selected_ids()) == set(first_recommended[:3])
        shot(27, 'daylight-applied', '点击“应用审核后的偏好并联动选择”后，本地计算排名；核对候选 ID、损失和原始三目标值。')
        page.detail_tabs.setCurrentIndex(2)
        wait(app, lambda: page.interactive_plot._bridge.ready_for_data, 20)
        shot(28, 'decision-linked-plots', '在第五页“交互图表”查看完整候选二维散点与平行坐标；推荐 ID 与结果页联动。')

        page.query_edit.setPlainText('我要能耗和眩光平衡的方案')
        page.make_preview()
        assert page._preview['preference_state']['weights'][0]['weight'] == .6
        assert 'candidate_id' not in page.preview_edit.toPlainText()
        shot(29, 'balance-outbound-preview', '第二轮输入“我要能耗和眩光平衡的方案”，重新预览；上下文含前轮权重，不含候选明细行。')
        page.send()
        wait(app, lambda: not page._busy and page._proposed is not None and '能耗与眩光平衡' in page._proposed['intent'], 20)
        assert page.ranking.frame.candidate_id.tolist() == first_recommended
        shot(30, 'balance-mock-proposal', '本地模拟第二轮结构化响应：能耗和眩光各 50%，采光 0%；旧排名未暗中改变。')
        page.apply_reviewed_plan()
        second_recommended = page.ranking.frame.candidate_id.tolist()
        assert set(second_recommended[:3]) <= set(window.results_model.frame.candidate_id)
        assert set(page.selected_ids()) == set(second_recommended[:3])
        assert len(mocked_calls) == 2
        shot(31, 'balance-applied', '显式应用第二轮建议，查看本地重新计算的权衡、推荐 ID 与稳定联动；不把权重结果称为物理最优。')
    finally:
        decision_page_module.request_preference_plan = original_request
        page.key_edit.clear()

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
            shot(32, 'external-mapping', '点击“导入外部结果”，将仅有的两列数值映射为最小化目标，不虚构输入。', widget=dialog)
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
    shot(33, 'external-results', '导入后显示外部数值结果；保留源行 ID 和原目标列名，不标称代理预测或物理验证。')

    source_files = ['daylight_workbench/ui.py', 'daylight_workbench/theme.py',
                    'daylight_workbench/modeling_widgets.py',
                    'daylight_workbench/modeling.py',
                    'daylight_workbench/project.py',
                    'daylight_workbench/decision_page.py',
                    'daylight_workbench/preference_engine.py',
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
        'candidate_models': [c['family'] for c in metadata['config']['candidates']],
        'target_mode': metadata['target_mode'],
        'per_target_winners': selected_families,
        'per_target_selected_index': metadata['selected_index'],
        'heldout_metrics': [row for row in heldout_metrics if row['划分'] == 'test'],
        'optimization_evaluations': window.optimization.evaluations,
        'optimization_candidates': len(window.optimization.candidates),
        'selected_ids_before_filter': chosen, 'filtered_rows': len(filtered),
        'exported_ids': chosen_filtered, 'export_csv_sha256': hashlib.sha256(exported.read_bytes()).hexdigest(),
        'project_sha256': project_sha, 'project_saved_reopened': True,
        'project_schema_version': 3,
        'preference_mock_queries': [call['query'] for call in mocked_calls],
        'preference_first_ids': first_recommended,
        'preference_second_ids': second_recommended,
        'preference_first_weights': [.6, .2, .2],
        'preference_second_weights': [0., .5, .5],
        'preference_mock_transport': True,
        'external_only_targets': window.external_results.targets,
        'webgl_3d_executed': False, 'online_llm_executed': False, 'browser_sandbox_disabled': False,
        'screenshots': shots,
    }
    (OUT/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    # Retain the local archive until final evidence has been reviewed. Never
    # include this trained-model binary in the public documentation upload.
    window.close()
    pump(app, .1)
    print(json.dumps({'per_target_winners': selected_families, 'candidate_count':len(full),'exported_ids':chosen_filtered,'shots':len(shots)}, ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
