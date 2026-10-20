"""Lean Chinese/bilingual Qt widgets. Expensive work runs on a cancellable QThread."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import threading
import traceback

import numpy as np
import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .core import (DataConfig, TrainConfig, Variable, OptimizeConfig, Cancelled,
                   read_dataset, prepare_data, train_model, optimize_model,
                   filter_candidates, export_candidates, prediction_column)
from .project import save_project, load_project
from .demo import make_demo

NOTICE = '仅代理模型预测 / Surrogate predictions。不是 Radiance / EnergyPlus 验证结果；轴向范围内也不代表物理可行。'


class FrameModel(C.QAbstractTableModel):
    def __init__(self, frame=None, parent=None):
        super().__init__(parent)
        self.frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame):
        self.beginResetModel()
        self.frame = frame.copy()
        self.endResetModel()

    def rowCount(self, parent=C.QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=C.QModelIndex()):
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=C.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == C.Qt.ItemDataRole.DisplayRole:
            value = self.frame.iloc[index.row(), index.column()]
            if isinstance(value, (float, np.floating)):
                return f'{value:.7g}'
            return str(value)
        if role == C.Qt.ItemDataRole.TextAlignmentRole:
            return int(C.Qt.AlignmentFlag.AlignRight | C.Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=C.Qt.ItemDataRole.DisplayRole):
        if role == C.Qt.ItemDataRole.DisplayRole:
            return str(self.frame.columns[section]) if orientation == C.Qt.Orientation.Horizontal else str(section + 1)
        return None

    def sort(self, column, order):
        if column < 0 or column >= len(self.frame.columns):
            return
        self.layoutAboutToBeChanged.emit()
        self.frame = self.frame.sort_values(self.frame.columns[column], ascending=order == C.Qt.SortOrder.AscendingOrder, kind='stable').reset_index(drop=True)
        self.layoutChanged.emit()


class Worker(C.QObject):
    result = C.Signal(object)
    error = C.Signal(str)
    progress = C.Signal(str)
    finished = C.Signal()

    def __init__(self, task, cancel):
        super().__init__()
        self.task, self.cancel = task, cancel

    @C.Slot()
    def run(self):
        try:
            self.result.emit(self.task(self.cancel, self.progress.emit))
        except Cancelled as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(f'{type(exc).__name__}: {exc}')
        finally:
            self.finished.emit()


def spin(value, lo, hi):
    widget = W.QSpinBox()
    widget.setRange(lo, hi)
    widget.setValue(value)
    return widget


def line(value=''):
    return W.QLineEdit(str(value))


def button(text, handler):
    widget = W.QPushButton(text)
    widget.clicked.connect(handler)
    return widget


def label(text):
    widget = W.QLabel(text)
    widget.setWordWrap(True)
    return widget


def table_view(model):
    widget = W.QTableView()
    widget.setModel(model)
    widget.setAlternatingRowColors(True)
    widget.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
    widget.setSortingEnabled(True)
    return widget


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Daylight Workbench P0 · 离线代理模型多目标优化')
        self.resize(1240, 860)
        self.raw = None
        self.prepared = None
        self.trained = None
        self.optimization = None
        self.thread = None
        self.worker = None
        self.cancel_event = None
        self._loading = False
        self._close_pending = False
        central = W.QWidget()
        outer = W.QVBoxLayout(central)
        top = W.QHBoxLayout()
        self.open_project_button = button('打开项目 .dwb', self.open_project)
        self.save_project_button = button('保存项目 .dwb', self.save_project)
        top.addWidget(self.open_project_button)
        top.addWidget(self.save_project_button)
        top.addWidget(label('P0 · 无账号/密钥 · 不访问网络'))
        outer.addLayout(top)
        outer.addWidget(label(NOTICE))
        self.tabs = W.QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.make_data_tab()
        self.make_train_tab()
        self.make_opt_tab()
        self.make_results_tab()
        footer = W.QHBoxLayout()
        self.status = label('请导入数据，或加载明确标记的合成演示。')
        self.cancel_button = button('取消当前任务', self.cancel_task)
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.cancel_button)
        outer.addLayout(footer)
        self.setCentralWidget(central)
        self.update_enabled()

    def make_data_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        row = W.QHBoxLayout()
        row.addWidget(button('导入 CSV / XLSX', self.import_data))
        row.addWidget(button('加载合成演示（非物理仿真）', self.load_demo))
        self.data_label = label('尚无数据。ID、类别和无关列请保留为“忽略”；P0不支持类别编码。')
        row.addWidget(self.data_label, 1)
        layout.addLayout(row)
        self.roles = W.QTableWidget(0, 3)
        self.roles.setHorizontalHeaderLabels(['列名', '角色：忽略 / 自变量 / 目标 / 分组', '目标方向（优化前必须明确选择）'])
        self.roles.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.roles, 2)
        row = W.QHBoxLayout()
        self.ratios = [line('0.70'), line('0.15'), line('0.15')]
        for text, field in zip(('训练', '验证', '测试'), self.ratios):
            row.addWidget(W.QLabel(text))
            field.setMaximumWidth(80)
            field.textChanged.connect(self.invalidate_data)
            row.addWidget(field)
        self.split_seed = spin(42, 0, 2**31 - 1)
        self.split_seed.valueChanged.connect(self.invalidate_data)
        row.addWidget(W.QLabel('随机种子'))
        row.addWidget(self.split_seed)
        self.drop_duplicates = W.QCheckBox('删除完全重复行（可取消；无论如何防止跨集重复输入）')
        self.drop_duplicates.setChecked(False)
        self.drop_duplicates.toggled.connect(self.invalidate_data)
        row.addWidget(self.drop_duplicates)
        layout.addLayout(row)
        layout.addWidget(button('校验数据并预览划分', self.prepare_clicked))
        self.data_report = label('默认70/15/15、种子42；同组及相同自变量绑定划分，比例可能偏离。只用训练子集拟合X/Y缩放器。')
        layout.addWidget(self.data_report)
        self.preview_model = FrameModel()
        self.preview = table_view(self.preview_model)
        layout.addWidget(self.preview, 1)
        self.tabs.addTab(tab, '1 数据与列定义')

    def make_train_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        form = W.QHBoxLayout()
        self.hidden = line('64,64')
        self.epochs = spin(500, 1, 100000)
        self.patience = spin(35, 1, 10000)
        self.lr = line('0.001')
        self.batch = spin(64, 1, 100000)
        self.train_seed = spin(42, 0, 2**31 - 1)
        for text, field in [('隐藏层', self.hidden), ('最多轮数', self.epochs), ('早停', self.patience), ('Adam学习率', self.lr), ('批量', self.batch), ('种子', self.train_seed)]:
            form.addWidget(W.QLabel(text))
            form.addWidget(field)
            if isinstance(field, W.QLineEdit):
                field.textChanged.connect(self.invalidate_training)
            else:
                field.valueChanged.connect(self.invalidate_training)
        layout.addLayout(form)
        self.train_button = button('训练 ANN（验证集早停，恢复最佳权重）', self.start_training)
        layout.addWidget(self.train_button)
        self.train_label = label('测试集仅用于最终报告，不参与缩放、优化器或早停。反复查看测试集并调参会形成间接泄漏。')
        layout.addWidget(self.train_label)
        self.metrics_model = FrameModel()
        layout.addWidget(table_view(self.metrics_model), 1)
        row = W.QHBoxLayout()
        self.target_plot = W.QComboBox()
        self.target_plot.currentIndexChanged.connect(self.draw_training)
        row.addWidget(W.QLabel('实测-预测图目标（测试集）'))
        row.addWidget(self.target_plot)
        layout.addLayout(row)
        self.training_figure = Figure(figsize=(9, 3), constrained_layout=True)
        self.training_canvas = FigureCanvasQTAgg(self.training_figure)
        layout.addWidget(self.training_canvas, 2)
        self.tabs.addTab(tab, '2 ANN训练与验证')

    def make_opt_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        layout.addWidget(label('上下界默认取训练集观测值。固定输入：勾选“固定”并将上下界设为相同值。整数变量边界须为整数。范围盒并不保证样本密度、相关性约束或物理可行性。'))
        self.bounds = W.QTableWidget(0, 6)
        self.bounds.setHorizontalHeaderLabels(['自变量', '下限', '上限', '类型', '固定', '训练观测范围'])
        self.bounds.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.bounds, 2)
        self.directions_label = label('请先训练模型，并在数据页明确每个目标的最小/最大方向。')
        layout.addWidget(self.directions_label)
        row = W.QHBoxLayout()
        self.population = spin(80, 4, 2000)
        self.generations = spin(60, 1, 10000)
        self.opt_seed = spin(42, 0, 2**31 - 1)
        for text, field in [('种群', self.population), ('代数', self.generations), ('种子', self.opt_seed)]:
            row.addWidget(W.QLabel(text))
            row.addWidget(field)
            field.valueChanged.connect(self.invalidate_optimization)
        layout.addLayout(row)
        self.opt_button = button('运行 NSGA-II（仅代理预测）', self.start_optimization)
        layout.addWidget(self.opt_button)
        layout.addWidget(label('不含LLM、不假装AI自然语言筛选；P0使用确定性的数值区间筛选，完全离线。暂不包含仿真调用、物理约束求解、类别变量或模型不确定性量化。'))
        layout.addStretch()
        self.tabs.addTab(tab, '3 NSGA-II设置')

    def make_results_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        row = W.QHBoxLayout()
        self.result_source = W.QComboBox()
        self.result_source.addItems(['仅Pareto候选', '全部已评估唯一候选'])
        row.addWidget(self.result_source)
        row.addWidget(button('应用区间筛选', self.apply_filter))
        row.addWidget(button('清空区间', self.clear_filter))
        row.addWidget(button('导出当前筛选', lambda: self.export_rows('view')))
        row.addWidget(button('导出选中行', lambda: self.export_rows('selected')))
        row.addWidget(button('导出全部候选', lambda: self.export_rows('all')))
        layout.addLayout(row)
        self.filters = W.QTableWidget(0, 3)
        self.filters.setHorizontalHeaderLabels(['数值列', '下限（空白=不限）', '上限（空白=不限）'])
        self.filters.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        self.filters.setMaximumHeight(170)
        layout.addWidget(self.filters)
        self.results_label = label('尚无优化结果。表格可按列排序、多选行；导出同时生成来源JSON。')
        layout.addWidget(self.results_label)
        self.results_model = FrameModel()
        self.results_table = table_view(self.results_model)
        self.results_table.selectionModel().selectionChanged.connect(self.draw_results)
        self.results_model.layoutAboutToBeChanged.connect(self.results_table.clearSelection)
        layout.addWidget(self.results_table, 2)
        row = W.QHBoxLayout()
        self.scatter_x, self.scatter_y = W.QComboBox(), W.QComboBox()
        for text, widget in [('散点横轴目标', self.scatter_x), ('纵轴目标', self.scatter_y)]:
            row.addWidget(W.QLabel(text))
            row.addWidget(widget)
            widget.currentIndexChanged.connect(self.draw_results)
        row.addStretch()
        layout.addLayout(row)
        self.results_figure = Figure(figsize=(10, 3), constrained_layout=True)
        self.results_canvas = FigureCanvasQTAgg(self.results_figure)
        layout.addWidget(self.results_canvas, 2)
        self.tabs.addTab(tab, '4 候选筛选与导出')

    def update_enabled(self):
        busy = self.thread is not None
        self.tabs.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.save_project_button.setEnabled(self.trained is not None and not busy)
        self.train_button.setEnabled(self.raw is not None and not busy)
        self.opt_button.setEnabled(self.trained is not None and not busy)
        self.cancel_button.setEnabled(busy)

    def invalidate_data(self, *args):
        if self._loading:
            return
        self.prepared = None
        self.invalidate_training()
        self.data_report.setText('定义已修改，请重新校验；之前的训练与优化结果已失效。')

    def invalidate_training(self, *args):
        if self._loading:
            return
        self.trained, self.optimization = None, None
        self.metrics_model.set_frame(pd.DataFrame())
        self.results_model.set_frame(pd.DataFrame())
        self.training_figure.clear()
        self.training_canvas.draw_idle()
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.bounds.setRowCount(0)
        self.train_label.setText('当前定义/训练配置已修改，需要重新训练。')
        self.results_label.setText('当前无有效优化结果。')
        self.update_enabled()

    def invalidate_optimization(self, *args):
        if self._loading:
            return
        self.optimization = None
        self.results_model.set_frame(pd.DataFrame())
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.results_label.setText('优化设置已修改，旧优化结果已失效；模型仍有效。')

    def role_changed(self):
        self.invalidate_data()
        self.refresh_directions_label()

    def direction_changed(self):
        if self._loading:
            return
        self.optimization = None
        if self.prepared is not None:
            self.prepared.config.directions = self.directions()
        if self.trained is not None:
            self.trained.data.config.directions = self.directions()
        self.results_model.set_frame(pd.DataFrame())
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.results_label.setText('目标方向已修改，旧优化结果已失效。')
        self.refresh_directions_label()

    def refresh_directions_label(self):
        text = []
        for row in range(self.roles.rowCount()):
            if self.roles.cellWidget(row, 1).currentData() == 'target':
                text.append(f'{self.roles.item(row, 0).text()}: {self.roles.cellWidget(row, 2).currentText()}')
        self.directions_label.setText('目标方向：' + '；'.join(text))

    def set_raw(self, raw, caption, config=None, directions=None):
        self._loading = True
        self.raw = raw
        self.roles.setRowCount(len(raw.columns))
        for row, col in enumerate(raw.columns):
            item = W.QTableWidgetItem(str(col))
            item.setFlags(item.flags() & ~C.Qt.ItemFlag.ItemIsEditable)
            self.roles.setItem(row, 0, item)
            role = W.QComboBox()
            for name, value in [('忽略（含ID）', 'ignore'), ('自变量 X', 'input'), ('目标 Y', 'target'), ('分组 Group', 'group')]:
                role.addItem(name, value)
            if config:
                value = 'input' if col in config.inputs else 'target' if col in config.targets else 'group' if col == config.group else 'ignore'
                role.setCurrentIndex(role.findData(value))
            self.roles.setCellWidget(row, 1, role)
            role.currentIndexChanged.connect(self.role_changed)
            direction = W.QComboBox()
            for text, value in [('请选择', None), ('最小化 min', 'min'), ('最大化 max', 'max')]:
                direction.addItem(text, value)
            if directions and col in directions:
                direction.setCurrentIndex(direction.findData(directions[col]))
            self.roles.setCellWidget(row, 2, direction)
            direction.currentIndexChanged.connect(self.direction_changed)
        if config:
            for widget, value in zip(self.ratios, [config.train_ratio, config.val_ratio, config.test_ratio]):
                widget.setText(str(value))
            self.split_seed.setValue(config.seed)
            self.drop_duplicates.setChecked(config.drop_duplicates)
        self.preview_model.set_frame(raw.head(100))
        self.data_label.setText(f'{caption} · {len(raw)}行 × {len(raw.columns)}列；预览前100行')
        self._loading = False
        self.invalidate_data()
        self.refresh_directions_label()
        self.status.setText('数据已加载。请显式定义X/Y、排除ID；目标方向在优化前必须选择。')
        self.update_enabled()

    def import_data(self):
        path, _ = W.QFileDialog.getOpenFileName(self, '导入数据', '', 'CSV / Excel (*.csv *.xlsx)')
        if not path:
            return
        self.safe(lambda: self.set_raw(read_dataset(path), Path(path).name))

    def load_demo(self):
        raw = make_demo()
        cols = list(raw.columns)
        config = DataConfig(cols[:10], cols[10:])
        self.set_raw(raw, '完全合成演示 / SYNTHETIC ONLY', config, dict(zip(cols[10:], ['max', 'min', 'min'])))

    def data_config(self):
        inputs, targets, groups = [], [], []
        for row in range(self.roles.rowCount()):
            name, role = self.roles.item(row, 0).text(), self.roles.cellWidget(row, 1).currentData()
            if role == 'input':
                inputs.append(name)
            elif role == 'target':
                targets.append(name)
            elif role == 'group':
                groups.append(name)
        if len(groups) > 1:
            raise ValueError('P0只支持一个分组列；多重组请先合并为一个标识。')
        return DataConfig(inputs, targets, groups[0] if groups else None,
                          *(float(w.text()) for w in self.ratios), self.split_seed.value(), self.drop_duplicates.isChecked(), self.directions())

    def directions(self):
        return {self.roles.item(row, 0).text(): self.roles.cellWidget(row, 2).currentData()
                for row in range(self.roles.rowCount()) if self.roles.cellWidget(row, 1).currentData() == 'target'}

    def prepare_clicked(self):
        self.safe(self.prepare)

    def prepare(self):
        if self.raw is None:
            raise ValueError('请先导入数据。')
        prepared = prepare_data(self.raw, self.data_config())
        self.prepared = prepared
        counts = prepared.report['split_counts']
        self.data_report.setText(f"有效{len(prepared.frame)}行，完全重复{prepared.report['exact_duplicates']}行，已删{prepared.report['duplicates_removed']}行；训练/验证/测试 = {counts['train']}/{counts['validation']}/{counts['test']}；相同输入相关行{prepared.report['conflicting_input_rows']}。\n{prepared.report['warning']}\n指纹 {prepared.fingerprint}")
        self.status.setText('数据校验通过，划分无组/重复输入跨集泄漏。')
        return prepared

    def training_config(self):
        hidden = [int(s.strip()) for s in self.hidden.text().split(',') if s.strip()]
        return TrainConfig(hidden, self.epochs.value(), self.patience.value(), float(self.lr.text()), self.batch.value(), self.train_seed.value())

    def start_training(self):
        def start():
            data = self.prepared or self.prepare()
            config = self.training_config()
            config.validate(len(data.config.inputs), len(data.config.targets))
            self.start_worker(lambda cancel, progress: train_model(data, config, cancel, progress), self.training_ready)
        self.safe(start)

    def start_worker(self, task, handler):
        if self.thread is not None:
            return
        self.cancel_event = threading.Event()
        self.thread = C.QThread(self)
        self.worker = Worker(task, self.cancel_event)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self._result_handler = handler
        self.worker.result.connect(self.deliver_result, C.Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.task_error, C.Qt.ConnectionType.QueuedConnection)
        self.worker.progress.connect(self.task_progress, C.Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.worker_finished, C.Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.thread.deleteLater)
        self.update_enabled()
        self.status.setText('正在计算，可随时取消；之前有效结果会保留到新任务成功。')
        self.thread.start()

    def cancel_task(self):
        if self.cancel_event:
            self.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status.setText('正在取消；将在当前小批次/代结束时停止。')

    @C.Slot(object)
    def deliver_result(self, result):
        # Python closures are not QObject receivers; never connect workers directly
        # to them. This QObject slot always marshals completion onto the GUI thread.
        self._result_handler(result)

    @C.Slot(str)
    def task_progress(self, text):
        self.status.setText(text)

    @C.Slot(str)
    def task_error(self, text):
        self.status.setText(text)
        if not text.startswith('操作已取消'):
            W.QMessageBox.warning(self, '计算未完成', text)

    @C.Slot()
    def worker_finished(self):
        self.thread, self.worker, self.cancel_event = None, None, None
        self.update_enabled()
        if self._close_pending:
            self.close()

    def training_ready(self, trained):
        if self.cancel_event and self.cancel_event.is_set():
            self.status.setText('操作已取消；新结果未采用。')
            return
        self.trained, self.prepared, self.optimization = trained, trained.data, None
        self.show_training()
        self.populate_bounds()
        self.results_model.set_frame(pd.DataFrame())
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.results_label.setText('新模型已生效，请重新运行优化。')
        self.status.setText('训练完成，已恢复验证集最佳权重。请检查逐目标测试指标和散点图。')
        self.update_enabled()

    def show_training(self):
        rows = []
        data = self.trained.data
        for split, metrics in self.trained.metrics.items():
            actual = data.Y[data.splits[split]]
            mean = data.Y[data.splits['train']].mean(axis=0)
            for j, (target, values) in enumerate(metrics.items()):
                rows.append({'划分': split, '目标': target, 'R2': values['R2'] if values['R2'] is not None else '未定义（常量目标）',
                             'MAE': values['MAE'], 'RMSE': values['RMSE'],
                             '训练均值基线MAE': float(np.abs(actual[:, j] - mean[j]).mean()),
                             '训练均值基线RMSE': float(np.sqrt(np.square(actual[:, j] - mean[j]).mean()))})
        self.metrics_model.set_frame(pd.DataFrame(rows))
        self.train_label.setText(f'最佳轮次 {self.trained.best_epoch}，实际训练 {len(self.trained.history)}轮。指标为原始目标单位；R²不适用于常量目标。测试集不参与训练/早停；请勿根据测试集反复调参。')
        self.target_plot.blockSignals(True)
        self.target_plot.clear()
        self.target_plot.addItems(data.config.targets)
        self.target_plot.blockSignals(False)
        self.draw_training()

    def draw_training(self, *args):
        if self.trained is None:
            return
        figure = self.training_figure
        figure.clear()
        ax, loss_ax = figure.subplots(1, 2)
        j = max(0, self.target_plot.currentIndex())
        idx = self.trained.data.splits['test']
        actual = self.trained.data.Y[idx, j]
        pred = self.trained.predict(self.trained.data.X[idx])[:, j]
        ax.scatter(actual, pred, s=16, alpha=.65)
        low, high = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([low, high], [low, high], '--', color='black')
        ax.set_xlabel('Actual (held-out test)')
        ax.set_ylabel('Surrogate predicted')
        ax.set_title(self.trained.data.config.targets[j])
        for key in ['train_loss', 'validation_loss']:
            loss_ax.plot([h['epoch'] for h in self.trained.history], [h[key] for h in self.trained.history], label=key)
        loss_ax.axvline(self.trained.best_epoch, linestyle=':', color='gray')
        loss_ax.set_xlabel('Epoch')
        loss_ax.set_ylabel('Scaled MSE')
        loss_ax.legend(fontsize=8)
        self.training_canvas.draw_idle()

    def populate_bounds(self, opt=None):
        data = self.trained.data
        X = data.X[data.splits['train']]
        self.bounds.setRowCount(len(data.config.inputs))
        for row, name in enumerate(data.config.inputs):
            v = opt.variables[row] if opt else Variable(name, float(X[:, row].min()), float(X[:, row].max()))
            item = W.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~C.Qt.ItemFlag.ItemIsEditable)
            self.bounds.setItem(row, 0, item)
            lower, upper = line(format(v.lower, '.17g')), line(format(v.upper, '.17g'))
            self.bounds.setCellWidget(row, 1, lower)
            self.bounds.setCellWidget(row, 2, upper)
            lower.textChanged.connect(self.invalidate_optimization)
            upper.textChanged.connect(self.invalidate_optimization)
            kind = W.QComboBox()
            kind.addItem('连续 continuous', 'continuous')
            kind.addItem('整数 integer', 'integer')
            kind.setCurrentIndex(kind.findData(v.kind))
            self.bounds.setCellWidget(row, 3, kind)
            kind.currentIndexChanged.connect(self.invalidate_optimization)
            fixed = W.QCheckBox()
            fixed.setChecked(v.fixed)
            self.bounds.setCellWidget(row, 4, fixed)
            fixed.toggled.connect(self.invalidate_optimization)
            observation = W.QTableWidgetItem(f'{X[:, row].min():.6g} … {X[:, row].max():.6g}')
            observation.setFlags(observation.flags() & ~C.Qt.ItemFlag.ItemIsEditable)
            self.bounds.setItem(row, 5, observation)
        self.refresh_directions_label()

    def optimization_config(self):
        variables = []
        for row in range(self.bounds.rowCount()):
            variables.append(Variable(self.bounds.item(row, 0).text(), float(self.bounds.cellWidget(row, 1).text()),
                                      float(self.bounds.cellWidget(row, 2).text()), self.bounds.cellWidget(row, 3).currentData(),
                                      self.bounds.cellWidget(row, 4).isChecked()))
        return OptimizeConfig(variables, self.directions(), self.population.value(), self.generations.value(), self.opt_seed.value())

    def start_optimization(self):
        def start():
            if self.trained is None:
                raise ValueError('请先训练有效模型。')
            config = self.optimization_config()
            config.validate(self.trained)
            trained = self.trained
            self.start_worker(lambda cancel, progress: optimize_model(trained, config, cancel, progress), self.optimization_ready)
        self.safe(start)

    def optimization_ready(self, result):
        if self.cancel_event and self.cancel_event.is_set():
            self.status.setText('操作已取消；新结果未采用。')
            return
        self.optimization = result
        self.show_results()
        self.tabs.setCurrentIndex(3)
        self.status.setText('优化完成。请记住：这是代理模型预测Pareto集，不是已验证物理方案。')

    def show_results(self):
        columns = self.trained.data.config.inputs + [prediction_column(t) for t in self.trained.data.config.targets]
        self.filters.setRowCount(len(columns))
        for row, col in enumerate(columns):
            item = W.QTableWidgetItem(col)
            item.setFlags(item.flags() & ~C.Qt.ItemFlag.ItemIsEditable)
            self.filters.setItem(row, 0, item)
            self.filters.setCellWidget(row, 1, line())
            self.filters.setCellWidget(row, 2, line())
        for combo in [self.scatter_x, self.scatter_y]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.trained.data.config.targets)
            combo.blockSignals(False)
        self.scatter_y.setCurrentIndex(min(1, self.scatter_y.count() - 1))
        self.apply_filter()

    def apply_filter(self):
        def apply():
            if self.optimization is None:
                return
            ranges = {}
            for row in range(self.filters.rowCount()):
                low, high = [self.filters.cellWidget(row, col).text().strip() for col in [1, 2]]
                ranges[self.filters.item(row, 0).text()] = (float(low) if low else None, float(high) if high else None)
            frame = filter_candidates(self.optimization.candidates, ranges, self.result_source.currentIndex() == 0)
            self.results_model.set_frame(frame.reset_index(drop=True))
            self.results_table.clearSelection()
            self.results_label.setText(f'评估 {self.optimization.evaluations} 次；唯一候选 {len(self.optimization.candidates)}，Pareto {int(self.optimization.candidates.pareto.sum())}，当前筛选 {len(frame)}。Pareto标记针对所有已评估唯一候选，不是仅当前筛选。')
            self.draw_results()
        self.safe(apply)

    def clear_filter(self):
        for row in range(self.filters.rowCount()):
            for col in [1, 2]:
                self.filters.cellWidget(row, col).clear()
        self.apply_filter()

    def selected_frame(self):
        indices = sorted({index.row() for index in self.results_table.selectionModel().selectedRows()})
        return self.results_model.frame.iloc[indices].copy()

    def draw_results(self, *args):
        if self.optimization is None or self.scatter_x.currentIndex() < 0 or self.scatter_y.currentIndex() < 0:
            return
        figure = self.results_figure
        figure.clear()
        ax, parallel = figure.subplots(1, 2)
        frame = self.results_model.frame
        if not len(frame):
            ax.set_title('No candidates in current filter')
            self.results_canvas.draw_idle()
            return
        x, y = prediction_column(self.scatter_x.currentText()), prediction_column(self.scatter_y.currentText())
        front = frame[frame.pareto]
        ax.scatter(frame[x], frame[y], s=12, alpha=.3, color='gray', label='All in filter')
        ax.scatter(front[x], front[y], s=18, color='tab:orange', label='Predicted Pareto')
        selected = self.selected_frame()
        if len(selected):
            ax.scatter(selected[x], selected[y], s=38, color='tab:blue', label='Selected')
        ax.set_xlabel(self.scatter_x.currentText(), fontsize=8)
        ax.set_ylabel(self.scatter_y.currentText(), fontsize=8)
        ax.legend(fontsize=7)
        cols = self.trained.data.config.inputs + [prediction_column(t) for t in self.trained.data.config.targets]
        all_values = self.optimization.candidates[cols]
        lo, hi = all_values.min().to_numpy(), all_values.max().to_numpy()
        scale = np.where(hi - lo > 0, hi - lo, 1)
        for subset, color, alpha, limit in [(frame, 'gray', .10, 500), (front, 'tab:orange', .25, 300), (selected, 'tab:blue', .8, 100)]:
            subset = subset.iloc[np.linspace(0, len(subset) - 1, min(len(subset), limit), dtype=int)] if len(subset) else subset
            if len(subset):
                parallel.plot(np.arange(len(cols)), ((subset[cols].to_numpy() - lo) / scale).T, color=color, alpha=alpha, linewidth=.7)
        parallel.set_xticks(np.arange(len(cols)), [c.replace('predicted::', 'Y:') for c in cols], rotation=70, fontsize=6)
        parallel.set_ylabel('Normalized over all candidates (0–1)', fontsize=8)
        parallel.set_title('Parallel coordinates (display sampled only)', fontsize=9)
        self.results_canvas.draw_idle()

    def export_rows(self, which):
        def export():
            if self.optimization is None:
                raise ValueError('没有可导出的有效候选。')
            frame = self.optimization.candidates if which == 'all' else self.selected_frame() if which == 'selected' else self.results_model.frame
            if not len(frame):
                raise ValueError('没有符合条件/选中的行。')
            path, _ = W.QFileDialog.getSaveFileName(self, '导出预测候选及来源JSON', 'predicted_candidates.csv', 'CSV (*.csv)')
            if not path:
                return
            if not path.lower().endswith('.csv'):
                path += '.csv'
            export_candidates(path, frame, self.trained, self.optimization)
            self.status.setText(f'已导出{len(frame)}行及 .provenance.json：{path}')
        self.safe(export)

    def save_project(self):
        def save():
            if self.trained is None:
                raise ValueError('请先训练模型。')
            path, _ = W.QFileDialog.getSaveFileName(self, '保存完整项目', 'project.dwb', 'Daylight project (*.dwb)')
            if not path:
                return
            if not path.lower().endswith('.dwb'):
                path += '.dwb'
            save_project(path, self.trained, self.optimization)
            self.status.setText(f'项目已保存（含数据、划分、权重、缩放器和候选）：{path}')
        self.safe(save)

    def open_project(self):
        path, _ = W.QFileDialog.getOpenFileName(self, '打开项目（仅接受已知来源文件）', '', 'Daylight project (*.dwb)')
        if not path:
            return
        def ready(value):
            trained, optimization = value
            if self.cancel_event and self.cancel_event.is_set():
                self.status.setText('打开已取消。')
                return
            self.set_raw(trained.data.frame.copy(), Path(path).name, trained.data.config, optimization.config.directions if optimization else trained.data.config.directions)
            self._loading = True
            self.hidden.setText(','.join(map(str, trained.config.hidden)))
            self.epochs.setValue(trained.config.epochs)
            self.patience.setValue(trained.config.patience)
            self.lr.setText(str(trained.config.learning_rate))
            self.batch.setValue(trained.config.batch_size)
            self.train_seed.setValue(trained.config.seed)
            self._loading = False
            self.trained, self.prepared, self.optimization = trained, trained.data, optimization
            self.show_training()
            self.populate_bounds(optimization.config if optimization else None)
            self.data_report.setText(json.dumps(trained.data.report, ensure_ascii=False))
            if optimization:
                self._loading = True
                self.population.setValue(optimization.config.population)
                self.generations.setValue(optimization.config.generations)
                self.opt_seed.setValue(optimization.config.seed)
                self._loading = False
                self.show_results()
            self.status.setText('项目校验与重载完成。')
            self.update_enabled()
        self.start_worker(lambda cancel, progress: load_project(path), ready)

    def safe(self, func):
        try:
            return func()
        except Exception as exc:
            traceback.print_exc()
            self.status.setText(str(exc))
            W.QMessageBox.warning(self, '请检查设置', str(exc))

    def closeEvent(self, event):
        if self.thread is not None:
            self._close_pending = True
            self.cancel_task()
            event.ignore()
        else:
            event.accept()
