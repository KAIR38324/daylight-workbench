"""Lean Chinese/bilingual Qt widgets. Expensive work runs on a cancellable QThread."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import threading
import traceback

import numpy as np
import pandas as pd
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
if C.QCoreApplication.instance() is None:
    C.QCoreApplication.setAttribute(C.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .core import (DataConfig, TrainConfig, Variable, OptimizeConfig, Cancelled,
                   read_dataset, prepare_data, train_model, optimize_model,
                   filter_candidates, export_candidates, prediction_column)
from .project import save_project, load_project
from .demo import make_demo
from .modeling import LayerSpec, ModelSpec, SearchConfig, make_search_plan, run_model_search, FittedSurrogate
from .modeling_widgets import ANNLayerBuilder, SearchControls
from .external_results import ExternalMappingDialog, prepare_external_results, export_external
from .branding import APP_NAME, APP_SUBTITLE, app_icon

NOTICE = '结果为代理模型预测，不是 Radiance、EnergyPlus 或实测验证；搜索范围内也不保证物理可行。'


def help_label(short, detail):
    widget = label(short)
    widget.setProperty('role', 'muted')
    widget.setToolTip(detail)
    return widget


def section_title(title, detail=''):
    row = W.QHBoxLayout()
    text = W.QLabel(title)
    text.setProperty('role', 'sectionTitle')
    row.addWidget(text)
    if detail:
        hint = W.QLabel(detail)
        hint.setProperty('role', 'muted')
        row.addWidget(hint)
    row.addStretch()
    return row


def style_axis(ax):
    """Shared scientific plot styling; never changes plotted data or limits."""
    ax.set_facecolor('#ffffff')
    ax.set_axisbelow(True)
    ax.grid(True, color='#e9eef3', linewidth=.7)
    ax.spines[['top', 'right']].set_visible(False)
    for side in ['bottom', 'left']:
        ax.spines[side].set_color('#ced8e1')
    ax.tick_params(axis='both', colors='#697b8b', labelsize=8, width=.6, length=3)
    ax.xaxis.label.set_color('#536b7d')
    ax.yaxis.label.set_color('#536b7d')
    ax.title.set_color('#233e53')


def empty_figure(figure, title, detail):
    figure.clear()
    figure.text(.5, .56, title, ha='center', color='#4e677b', fontsize=11)
    figure.text(.5, .45, detail, ha='center', color='#8795a3', fontsize=9)


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
                return f'{value:.6g}'
            return str(value)
        if role == C.Qt.ItemDataRole.TextAlignmentRole:
            value = self.frame.iloc[index.row(), index.column()]
            horizontal = C.Qt.AlignmentFlag.AlignRight if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) else C.Qt.AlignmentFlag.AlignLeft
            return int(horizontal | C.Qt.AlignmentFlag.AlignVCenter)
        if role == C.Qt.ItemDataRole.ToolTipRole:
            return str(self.frame.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=C.Qt.ItemDataRole.DisplayRole):
        if role in (C.Qt.ItemDataRole.DisplayRole, C.Qt.ItemDataRole.ToolTipRole):
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
    snapshot = C.Signal(object)

    def __init__(self, task, cancel, structured=False):
        super().__init__()
        self.task, self.cancel, self.structured = task, cancel, structured

    @C.Slot()
    def run(self):
        try:
            self.result.emit(self.task(self.cancel, self.progress.emit, self.snapshot.emit) if self.structured else self.task(self.cancel, self.progress.emit))
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
    widget.setShowGrid(False)
    widget.verticalHeader().setDefaultSectionSize(33)
    widget.verticalHeader().setMinimumSectionSize(30)
    widget.horizontalHeader().setMinimumSectionSize(86)
    widget.horizontalHeader().setDefaultSectionSize(135)
    widget.horizontalHeader().setTextElideMode(C.Qt.TextElideMode.ElideRight)
    widget.setTextElideMode(C.Qt.TextElideMode.ElideRight)
    widget.setFrameShape(W.QFrame.Shape.NoFrame)
    return widget


class MainWindow(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1440, 960)
        self.setMinimumSize(1100, 760)
        self.raw = None
        self.prepared = None
        self.trained = None
        self.optimization = None
        self.external_results = None
        self.last_filter_plan = None
        self._sync_selection = False
        self.thread = None
        self.worker = None
        self.cancel_event = None
        self._loading = False
        self._close_pending = False
        central = W.QWidget()
        central.setObjectName('workbenchRoot')
        outer = W.QVBoxLayout(central)
        outer.setContentsMargins(16, 0, 16, 0)
        outer.setSpacing(8)
        header = W.QFrame()
        header.setObjectName('appHeader')
        top = W.QHBoxLayout(header)
        top.setContentsMargins(12, 16, 12, 14)
        identity = W.QVBoxLayout()
        identity.setSpacing(2)
        title = W.QLabel(APP_NAME)
        title.setObjectName('appTitle')
        identity.addWidget(title)
        subtitle = W.QLabel(APP_SUBTITLE)
        subtitle.setObjectName('appSubtitle')
        identity.addWidget(subtitle)
        top.addLayout(identity)
        top.addStretch()
        self.dataset_badge = W.QLabel('尚未载入数据')
        self.dataset_badge.setProperty('role', 'badge')
        top.addWidget(self.dataset_badge, 0, C.Qt.AlignmentFlag.AlignVCenter)
        offline = W.QLabel('核心功能离线')
        offline.setProperty('role', 'badge')
        offline.setToolTip('建模、优化与图表在本地运行。可选 DeepSeek 筛选需主动预览并发送联网请求。')
        top.addWidget(offline, 0, C.Qt.AlignmentFlag.AlignVCenter)
        top.addSpacing(12)
        self.open_project_button = button('打开项目', self.open_project)
        self.save_project_button = button('保存项目', self.save_project)
        self.open_project_button.setToolTip('打开可信来源的 .dwb 项目')
        self.save_project_button.setToolTip('保存数据、划分、模型及分析状态到 .dwb 项目')
        top.addWidget(self.open_project_button)
        top.addWidget(self.save_project_button)
        outer.addWidget(header)
        notice = help_label('研究边界：代理模型预测不等于物理仿真或实测验证。', NOTICE)
        notice.setProperty('role', 'notice')
        outer.addWidget(notice)
        self.tabs = W.QTabWidget()
        self.tabs.setObjectName('workflowTabs')
        outer.addWidget(self.tabs, 1)
        self.make_data_tab()
        self.make_train_tab()
        self.make_opt_tab()
        self.make_results_tab()
        footer_widget = W.QFrame()
        footer_widget.setObjectName('statusBar')
        footer = W.QHBoxLayout(footer_widget)
        footer.setContentsMargins(2, 7, 2, 8)
        self.status = label('请导入数据，或加载明确标记的合成演示。')
        self.cancel_button = button('取消当前任务', self.cancel_task)
        self.cancel_button.setEnabled(False)
        self.status.setProperty('role', 'muted')
        footer.addWidget(self.status, 1)
        footer.addWidget(self.cancel_button)
        outer.addWidget(footer_widget)
        self.setCentralWidget(central)
        self.update_enabled()

    def make_data_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(12)
        layout.addLayout(section_title('数据与列定义', '明确 X、Y 与优化方向，不自动推断物理含义'))
        row = W.QHBoxLayout()
        import_button = button('导入 CSV / XLSX', self.import_data)
        import_button.setProperty('role', 'primary')
        row.addWidget(import_button)
        row.addWidget(button('加载合成演示（非物理仿真）', self.load_demo))
        self.data_label = help_label('尚无数据。请先导入文件或加载合成演示。', 'ID、类别和无关列请设为“忽略”；本版不支持类别编码。')
        row.addWidget(self.data_label, 1)
        layout.addLayout(row)
        data_splitter = W.QSplitter(C.Qt.Orientation.Horizontal)
        data_splitter.setChildrenCollapsible(False)
        roles_page, preview_page = W.QWidget(), W.QWidget()
        roles_layout, preview_layout = W.QVBoxLayout(roles_page), W.QVBoxLayout(preview_page)
        for column_layout in [roles_layout, preview_layout]:
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.setSpacing(9)
        roles_layout.addLayout(section_title('列角色', '每个目标单独设定方向'))
        preview_layout.addLayout(section_title('原始数据预览', '前 100 行 · 不改变原始值'))
        self.roles = W.QTableWidget(0, 3)
        self.roles.setHorizontalHeaderLabels(['列名', '角色', '目标方向'])
        self.roles.setShowGrid(False)
        self.roles.verticalHeader().hide()
        self.roles.verticalHeader().setDefaultSectionSize(32)
        self.roles.horizontalHeader().setSectionResizeMode(0, W.QHeaderView.ResizeMode.Stretch)
        self.roles.setColumnWidth(1, 132)
        self.roles.setColumnWidth(2, 124)
        self.roles.setMinimumWidth(480)
        roles_layout.addWidget(self.roles, 1)
        self.preview_model = FrameModel()
        self.preview = table_view(self.preview_model)
        self.preview.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
        self.preview.horizontalHeader().setMaximumSectionSize(300)
        preview_layout.addWidget(self.preview, 1)
        data_splitter.addWidget(roles_page)
        data_splitter.addWidget(preview_page)
        data_splitter.setStretchFactor(0, 4)
        data_splitter.setStretchFactor(1, 5)
        data_splitter.setSizes([570, 700])
        layout.addWidget(data_splitter, 1)
        split_box = W.QGroupBox('固定划分设置 · 快速模式')
        split_layout = W.QVBoxLayout(split_box)
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
        self.split_seed.setMaximumWidth(110)
        row.addSpacing(14)
        self.drop_duplicates = W.QCheckBox('删除完全重复行')
        self.drop_duplicates.setToolTip('无论是否删除完全重复行，始终防止同组、重复输入跨集。')
        self.drop_duplicates.setChecked(False)
        self.drop_duplicates.toggled.connect(self.invalidate_data)
        row.addWidget(self.drop_duplicates)
        row.addStretch()
        prepare_button = button('校验数据与划分', self.prepare_clicked)
        prepare_button.setProperty('role', 'primary')
        row.addWidget(prepare_button)
        split_layout.addLayout(row)
        self.data_report = help_label('默认 70/15/15 划分；模型比较时锁定 15% 测试集。', '同组及相同自变量绑定划分，实际比例可能偏离。缩放器只在相应训练子集拟合。')
        split_layout.addWidget(self.data_report)
        layout.addWidget(split_box)
        self.tabs.addTab(tab, '01  数据与列定义')

    def make_train_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)
        self.train_tabs = W.QTabWidget()
        self.train_tabs.setObjectName('trainingTabs')
        settings = W.QWidget()
        settings_layout = W.QVBoxLayout(settings)
        settings_layout.setContentsMargins(6, 18, 6, 4)
        settings_layout.setSpacing(16)
        self.layer_builder = ANNLayerBuilder()
        settings_layout.addWidget(self.layer_builder)
        parameter_box = W.QGroupBox('训练参数 · 单次 ANN')
        form = W.QGridLayout(parameter_box)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(12)
        self.hidden = line('64,64')
        self.hidden.setReadOnly(True)
        self.hidden.hide()  # Retain the compatibility field, not a duplicate editor.
        self.epochs = spin(300, 1, 3000)
        self.patience = spin(30, 1, 1000)
        self.lr = line('0.001')
        self.batch = spin(64, 1, 100000)
        self.train_seed = spin(42, 0, 2**31 - 1)
        for i, (text, field) in enumerate([('最多轮数', self.epochs), ('早停耐心', self.patience), ('学习率', self.lr), ('批量大小', self.batch), ('随机种子', self.train_seed)]):
            row, col = divmod(i, 3)
            form.addWidget(W.QLabel(text), row, col * 2)
            form.addWidget(field, row, col * 2 + 1)
            form.setColumnStretch(col * 2 + 1, 1)
        self.patience.setToolTip('验证损失连续不改善的最大轮数；测试集不参与早停。')
        self.hidden.textChanged.connect(self.legacy_hidden_changed)
        for field in [self.epochs, self.patience, self.batch, self.train_seed]:
            field.valueChanged.connect(self.invalidate_training)
        self.lr.textChanged.connect(self.invalidate_training)
        self.layer_builder.changed.connect(self.builder_changed)
        settings_layout.addWidget(parameter_box)
        settings_layout.addWidget(help_label('逐层设置网络结构；更多候选在“模型选择 / 网格 / CV”中配置。', '可添加/删除隐藏层，调整宽度与激活函数。图示与训练使用同一规格；不支持自由连线。搜索将整个网络结构作为候选，不做逐层笛卡尔积。'))
        settings_layout.addStretch()
        self.train_tabs.addTab(settings, 'ANN 结构与参数')
        search_page = W.QWidget()
        search_layout = W.QVBoxLayout(search_page)
        self.search_controls = SearchControls(self.layer_builder)
        self.search_controls.changed.connect(self.invalidate_training)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.search_controls)
        search_layout.addWidget(scroll)
        self.train_tabs.addTab(search_page, '模型选择 / 网格 / CV')
        report_page = W.QWidget()
        report_layout = W.QVBoxLayout(report_page)
        report_layout.setContentsMargins(4, 14, 4, 0)
        report_layout.setSpacing(9)
        self.training_summary = W.QFrame()
        self.training_summary.setProperty('role', 'softCard')
        summary_layout = W.QHBoxLayout(self.training_summary)
        summary_layout.setContentsMargins(15, 10, 15, 10)
        self.training_stats = {}
        for key, caption in [('phase', '任务阶段'), ('epoch', '当前轮次 / 树数'), ('best', '最佳轮次'), ('loss', '最新验证损失')]:
            col = W.QVBoxLayout()
            col.setSpacing(2)
            value, caption_label = W.QLabel('—'), W.QLabel(caption)
            value.setProperty('role', 'statValue')
            caption_label.setProperty('role', 'statLabel')
            col.addWidget(value)
            col.addWidget(caption_label)
            summary_layout.addLayout(col, 1)
            self.training_stats[key] = value
        report_layout.addWidget(self.training_summary)
        self.metrics_model = FrameModel()
        self.metrics_view = table_view(self.metrics_model)
        self.metrics_view.setMinimumHeight(145)
        self.metrics_view.setMaximumHeight(205)
        self.metrics_view.verticalHeader().hide()
        self.metrics_view.setToolTip('逐目标、逐划分指标。悬停可查看完整目标名与数值；MSE 使用平方单位，R² 可为负数。')
        report_layout.addWidget(self.metrics_view, 1)
        self.search_model = FrameModel()
        self.search_hint = help_label('候选只按验证集排名，测试集不参与选优。', '选择分数为各目标 RMSE 除以对应训练折目标标准差后等权平均，越低越好。目标常量时标准差按 1 处理。')
        self.search_view = table_view(self.search_model)
        self.search_view.setMaximumHeight(105)
        self.search_hint.hide()
        self.search_view.hide()
        search_row = W.QHBoxLayout()
        search_row.addWidget(self.search_hint, 1)
        self.search_toggle_button = W.QPushButton('展开候选排名')
        self.search_toggle_button.setCheckable(True)
        self.search_toggle_button.setProperty('role', 'quiet')
        self.search_toggle_button.hide()
        self.search_toggle_button.toggled.connect(self.toggle_search_summary)
        search_row.addWidget(self.search_toggle_button)
        report_layout.addLayout(search_row)
        report_layout.addWidget(self.search_view, 1)
        self.training_axis_bar = W.QWidget()
        row = W.QHBoxLayout(self.training_axis_bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.target_plot = W.QComboBox()
        self.target_plot.setMinimumWidth(310)
        self.target_plot.setSizeAdjustPolicy(W.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.target_plot.currentIndexChanged.connect(self.draw_training)
        row.addWidget(W.QLabel('独立测试集目标'))
        row.addWidget(self.target_plot)
        row.addStretch()
        chart_note = W.QLabel('左：实测 / 预测  ·  右：真实训练记录')
        chart_note.setProperty('role', 'muted')
        row.addWidget(chart_note)
        report_layout.addWidget(self.training_axis_bar)
        self.training_figure = Figure(figsize=(9, 3), constrained_layout=True)
        self.training_canvas = FigureCanvasQTAgg(self.training_figure)
        self.training_canvas.setMinimumHeight(180)
        empty_figure(self.training_figure, 'No training record yet', 'Run a model to inspect held-out test predictions and real fit progress.')
        report_layout.addWidget(self.training_canvas, 2)
        self.train_tabs.addTab(report_page, '指标 / 实时曲线')
        layout.addWidget(self.train_tabs, 1)
        row = W.QHBoxLayout()
        self.preview_search_button = button('预览候选与拟合预算', self.preview_search)
        self.search_button = button('开始网格验证 / 模型比较', self.start_search)
        self.search_button.setProperty('role', 'primary')
        self.train_button = button('单次 ANN（快速）', self.start_training)
        row.addWidget(self.preview_search_button)
        row.addWidget(self.search_button)
        row.addStretch()
        row.addWidget(self.train_button)
        layout.addLayout(row)
        self.search_label = help_label('默认锁定 15% 测试集，开发集做 5 折交叉验证。', '同组和相同输入不会跨折。ANN/XGBoost 使用外层训练折内部留出的早停集；选优后用全部开发集重拟合，最后仅评估一次测试集。')
        self.train_label = help_label('尚未训练。', 'ANN 显示真实训练/验证损失；RF 显示已拟合树数；XGBoost 显示逐目标迭代 RMSE，不生成伪造曲线。')
        layout.addWidget(self.search_label)
        layout.addWidget(self.train_label)
        self.tabs.addTab(tab, '02  模型与训练')

    def builder_changed(self):
        self.hidden.blockSignals(True)
        self.hidden.setText(','.join(str(l.width) for l in self.layer_builder.specification()))
        self.hidden.blockSignals(False)
        self.invalidate_training()

    def toggle_search_summary(self, checked):
        self.search_view.setVisible(checked)
        self.search_toggle_button.setText('收起候选排名' if checked else '展开候选排名')

    def legacy_hidden_changed(self):
        if self._loading:
            return
        try:
            layers = [LayerSpec(int(v.strip())) for v in self.hidden.text().split(',') if v.strip()]
            if layers and all(1 <= l.width <= 2048 for l in layers):
                self.layer_builder.set_layers(layers)
        except ValueError:
            pass
        self.invalidate_training()

    def preview_search(self):
        def preview():
            data = self.prepared or self.prepare()
            config = self.search_controls.configuration(self.epochs.value(), self.patience.value())
            plan = make_search_plan(data, config)
            self.search_label.setText(f"{plan['candidate_count']} 候选 × {plan['fold_count']} 折 = {plan['candidate_fold_jobs']} 次验证；拟合预算 {plan['fit_budget']}/{config.max_fits}。锁定测试 {len(plan['test'])} 行。")
            self.search_label.setToolTip('拟合预算包含 XGBoost 每目标独立拟合及最终重拟合。开发集 ' + str(len(plan['development'])) + ' 行；内部早停不查看外层验证集和锁定测试集。')
            return data, config
        return self.safe(preview)

    def start_search(self):
        value = self.preview_search()
        if value is None:
            return
        data, config = value
        self.train_tabs.setCurrentIndex(2)
        self.begin_training_display()
        self.start_worker(lambda cancel, progress, snapshot: run_model_search(data, config, cancel, progress, snapshot), self.search_ready, structured=True)

    def search_ready(self, result):
        if self.cancel_event and self.cancel_event.is_set():
            self.status.setText('搜索已取消，未采用新结果。')
            return
        self.training_ready(result.selected)
        self.show_search_summary()
        self.status.setText('模型比较完成：仅按开发集验证选优，已重拟合开发集；测试结果仅作为最终报告。')

    def show_search_summary(self):
        metadata = getattr(self.trained, 'search_metadata', {})
        rows = []
        for candidate in metadata.get('candidates', []):
            row = {'候选': candidate['candidate'], '模型': candidate['family'], '验证分数均值↓': candidate['score_mean'],
                   '折间分数标准差': candidate['score_std'], '选中': candidate['candidate'] == metadata.get('selected_index', -1) + 1,
                   '规格': json.dumps(candidate['spec'], ensure_ascii=False)}
            for target, metrics in candidate['metric_spread'].items():
                for metric, values in metrics.items():
                    row[f'{target} {metric}均值'] = values['mean']
                    row[f'{target} {metric}折间std'] = values['std']
            rows.append(row)
        self.search_model.set_frame(pd.DataFrame(rows))
        self.search_hint.setVisible(bool(rows))
        self.search_toggle_button.setVisible(bool(rows))
        self.search_view.setVisible(bool(rows) and self.search_toggle_button.isChecked())
        if rows:
            plan = metadata['plan']
            self.search_label.setText(f"验证选中候选 {metadata['selected_index']+1}（{self.trained.family}）；开发集 {len(self.trained.fit_indices)} 行重拟合，锁定测试 {len(plan['test'])} 行。")
            self.search_label.setToolTip('共 ' + str(plan['candidate_count']) + ' 个候选、' + str(plan['fold_count']) + ' 折。分数为各目标 RMSE 除以外层训练折目标标准差后等权平均；常量标准差按 1 处理。表格含原单位指标及折间波动。')

    def make_opt_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)
        layout.addLayout(section_title('NSGA-II 搜索设置', '在训练数据支持的范围内搜索'))
        layout.addWidget(help_label('搜索边界默认取模型实际训练数据的观测范围。', '可缩小范围、固定变量或选择整数类型。固定输入需上下界相同，整数边界需为整数。边界盒不代表样本密度、相关性约束或物理可行性。'))
        self.bounds = W.QTableWidget(0, 6)
        self.bounds.setHorizontalHeaderLabels(['自变量', '下限', '上限', '类型', '固定', '训练观测范围'])
        self.bounds.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        self.bounds.setShowGrid(False)
        self.bounds.verticalHeader().hide()
        self.bounds.verticalHeader().setDefaultSectionSize(40)
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
        self.opt_button.setProperty('role', 'primary')
        layout.addWidget(self.opt_button)
        layout.addWidget(help_label('仅使用变量边界和目标方向；完成后可筛选结果。', '不提供额外物理硬约束编辑器。可手动筛选，也可审阅 DeepSeek 生成的筛选计划；离线优化不需要密钥。'))
        self.live_opt_figure = Figure(figsize=(8, 2), constrained_layout=True)
        self.live_opt_canvas = FigureCanvasQTAgg(self.live_opt_figure)
        empty_figure(self.live_opt_figure, 'Live optimization frontier', 'Generation, evaluation count and predicted frontier will appear here.')
        layout.addWidget(self.live_opt_canvas, 1)
        self.tabs.addTab(tab, '03  NSGA-II 优化')

    def make_results_tab(self):
        tab = W.QWidget()
        layout = W.QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(9)
        row = W.QHBoxLayout()
        row.addWidget(button('导入外部结果', self.import_external_results))
        row.addWidget(button('DeepSeek 筛选', self.natural_language_filter))
        self.result_source = W.QComboBox()
        self.result_source.addItems(['仅Pareto候选', '全部候选 / 外部数据行'])
        self.result_source.setMinimumWidth(195)
        self.result_source.currentIndexChanged.connect(self.apply_filter)
        row.addSpacing(8)
        row.addWidget(self.result_source)
        row.addStretch()
        self.table_toggle_button = button('显示表格', self.toggle_results_table)
        row.addWidget(self.table_toggle_button)
        self.filter_toggle_button = button('区间筛选', self.toggle_filters)
        row.addWidget(self.filter_toggle_button)
        export_button = W.QToolButton()
        export_button.setText('导出当前')
        export_button.setPopupMode(W.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        export_button.clicked.connect(lambda: self.export_rows('view'))
        export_menu = W.QMenu(export_button)
        for caption, which in [('导出当前筛选', 'view'), ('导出已选方案', 'selected'), ('导出全部候选', 'all')]:
            action = export_menu.addAction(caption)
            action.triggered.connect(lambda checked=False, value=which: self.export_rows(value))
        export_button.setMenu(export_menu)
        row.addWidget(export_button)
        layout.addLayout(row)
        self.filter_panel = W.QWidget()
        filter_layout = W.QVBoxLayout(self.filter_panel)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_actions = W.QHBoxLayout()
        filter_actions.addWidget(help_label('按原始单位限定数值范围；留空表示不限。', '筛选只改变当前视图范围，不改写候选值或重新定义 Pareto 标记。'))
        filter_actions.addStretch()
        filter_actions.addWidget(button('清空区间', self.clear_filter))
        apply_button = button('应用筛选', self.apply_filter)
        apply_button.setProperty('role', 'primary')
        filter_actions.addWidget(apply_button)
        filter_layout.addLayout(filter_actions)
        self.filters = W.QTableWidget(0, 3)
        self.filters.setHorizontalHeaderLabels(['数值列', '下限（空白=不限）', '上限（空白=不限）'])
        self.filters.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        self.filters.setMaximumHeight(150)
        self.filters.verticalHeader().hide()
        self.filters.verticalHeader().setDefaultSectionSize(36)
        self.filters.setShowGrid(False)
        filter_layout.addWidget(self.filters)
        layout.addWidget(self.filter_panel)
        self.filter_panel.hide()
        self.results_label = help_label('尚无结果。', '表格可排序、多选；导出 CSV 时生成来源 JSON。代理预测不等于物理验证。')
        self.results_label.hide()
        self.results_model = FrameModel()
        self.results_table = table_view(self.results_model)
        self.results_table.setMinimumHeight(62)
        self.results_table.setToolTip('表格可排序、多选；拖动下方分隔线可展开表格。详情与进度显示在窗口底部。')
        self.results_table.selectionModel().selectionChanged.connect(self.table_selection_changed)
        self.results_model.layoutAboutToBeChanged.connect(self.remember_sorted_selection)
        self.results_model.layoutChanged.connect(self.restore_sorted_selection)
        self.static_axis_bar = W.QWidget()
        row = W.QHBoxLayout(self.static_axis_bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.scatter_x, self.scatter_y = W.QComboBox(), W.QComboBox()
        for text, widget in [('散点横轴目标', self.scatter_x), ('纵轴目标', self.scatter_y)]:
            row.addWidget(W.QLabel(text))
            row.addWidget(widget)
            widget.currentIndexChanged.connect(self.draw_results)
        row.addStretch()
        layout.addWidget(self.static_axis_bar)
        self.results_figure = Figure(figsize=(10, 3), constrained_layout=True)
        self.results_canvas = FigureCanvasQTAgg(self.results_figure)
        self.result_plots = W.QTabWidget()
        self.result_plots.setObjectName('resultTabs')
        from .interactive_plot import InteractiveParetoWidget
        self.interactive_plot = InteractiveParetoWidget()
        self.interactive_plot.selectionChanged.connect(self.plot_selection_changed)
        self.result_plots.addTab(self.interactive_plot, '交互分析')
        self.result_plots.setTabToolTip(0, '缩放、套索、框选与平行坐标刷选；同一候选 ID 在各图和结果表中联动。')
        self.result_plots.addTab(self.results_canvas, '静态概览（显示抽样）')
        self.result_plots.currentChanged.connect(lambda index: self.static_axis_bar.setVisible(index == 1))
        self.static_axis_bar.hide()
        self.result_splitter = W.QSplitter(C.Qt.Orientation.Vertical)
        self.result_splitter.setChildrenCollapsible(True)
        self.result_splitter.addWidget(self.results_table)
        self.result_splitter.addWidget(self.result_plots)
        self.result_splitter.setStretchFactor(0, 0)
        self.result_splitter.setStretchFactor(1, 1)
        self.result_splitter.setSizes([64, 600])
        self.results_table.hide()
        layout.addWidget(self.result_splitter, 1)
        self.tabs.addTab(tab, '04  结果分析与导出')

    def toggle_results_table(self):
        show = not self.results_table.isVisible()
        self.results_table.setVisible(show)
        if show:
            self.result_splitter.setSizes([145, 500])
        else:
            self.result_splitter.setSizes([0, self.result_splitter.height()])
        self.result_splitter.refresh()
        self.table_toggle_button.setText('隐藏表格' if show else '显示表格')

    def toggle_filters(self):
        show = not self.filter_panel.isVisible()
        self.filter_panel.setVisible(show)
        self.filter_toggle_button.setText('收起筛选' if show else '区间筛选')

    def remember_sorted_selection(self):
        self._sort_selected_ids = self.selected_frame().candidate_id.tolist() if 'candidate_id' in self.results_model.frame else []
        self._sync_selection = True
        self.results_table.clearSelection()

    def restore_sorted_selection(self):
        self._sync_selection = False
        ids = getattr(self, '_sort_selected_ids', [])
        self.plot_selection_changed(ids)
        self.interactive_plot.set_selection(ids)

    def update_enabled(self):
        busy = self.thread is not None
        self.tabs.setEnabled(not busy)
        self.open_project_button.setEnabled(not busy)
        self.save_project_button.setEnabled(self.trained is not None and not busy)
        self.train_button.setEnabled(self.raw is not None and not busy)
        self.search_button.setEnabled(self.raw is not None and not busy)
        self.preview_search_button.setEnabled(self.raw is not None and not busy)
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
        self.external_results = None
        self.metrics_model.set_frame(pd.DataFrame())
        self.search_model.set_frame(pd.DataFrame())
        self.search_hint.hide()
        self.search_view.hide()
        self.search_toggle_button.hide()
        self.results_model.set_frame(pd.DataFrame())
        self.training_figure.clear()
        empty_figure(self.training_figure, 'Training settings changed', 'Run the model again to refresh diagnostics.')
        self.training_canvas.draw_idle()
        for value in self.training_stats.values():
            value.setText('—')
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.bounds.setRowCount(0)
        self.train_label.setText('当前定义/训练配置已修改，需要重新训练。')
        self.results_label.setText('当前无有效优化结果。')
        self.interactive_plot.setVisible(False)
        self.update_enabled()

    def invalidate_optimization(self, *args):
        if self._loading:
            return
        self.optimization = None
        self.results_model.set_frame(pd.DataFrame())
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.results_label.setText('优化设置已修改，旧优化结果已失效；模型仍有效。')
        self.interactive_plot.setVisible(False)

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
        self.interactive_plot.setVisible(False)
        self.refresh_directions_label()

    def refresh_directions_label(self):
        text = []
        for row in range(self.roles.rowCount()):
            is_target = self.roles.cellWidget(row, 1).currentData() == 'target'
            self.roles.cellWidget(row, 2).setEnabled(is_target)
            if is_target:
                text.append(f'{self.roles.item(row, 0).text()}: {self.roles.cellWidget(row, 2).currentText()}')
        self.directions_label.setText('目标方向：' + '；'.join(text))

    def set_raw(self, raw, caption, config=None, directions=None):
        self._loading = True
        self.raw = raw
        self.roles.setRowCount(len(raw.columns))
        for row, col in enumerate(raw.columns):
            item = W.QTableWidgetItem(str(col))
            item.setToolTip(str(col))
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
            for text, value in [('请选择', None), ('最小化', 'min'), ('最大化', 'max')]:
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
        self.data_label.setText(f'{caption} · {len(raw):,} 行 × {len(raw.columns)} 列')
        is_synthetic = 'synthetic' in caption.casefold() or '合成' in caption
        self.dataset_badge.setText('合成演示 · 非物理仿真' if is_synthetic else f'{len(raw):,} 行 · {len(raw.columns)} 列')
        self.dataset_badge.setProperty('role', 'syntheticBadge' if is_synthetic else 'badge')
        self.dataset_badge.style().unpolish(self.dataset_badge)
        self.dataset_badge.style().polish(self.dataset_badge)
        self.dataset_badge.setToolTip(caption)
        self._loading = False
        self.invalidate_data()
        self.refresh_directions_label()
        self.status.setText('数据已加载。请显式定义X/Y、排除ID；目标方向在优化前必须选择。')
        if config:
            self.layer_builder.set_io(len(config.inputs), len(config.targets))
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
            raise ValueError('本版只支持一个分组列；多重组请先合并为一个标识。')
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
        self.layer_builder.set_io(len(prepared.config.inputs), len(prepared.config.targets))
        self.prepared = prepared
        counts = prepared.report['split_counts']
        self.data_report.setText(f"校验通过 · 有效 {len(prepared.frame):,} 行 · 训练 {counts['train']} / 验证 {counts['validation']} / 测试 {counts['test']} · 删除重复 {prepared.report['duplicates_removed']} 行" + (f" · {prepared.report['warning']}" if prepared.report['warning'] else ''))
        self.data_report.setToolTip(f"完全重复 {prepared.report['exact_duplicates']} 行；相同输入相关行 {prepared.report['conflicting_input_rows']}。\n{prepared.report['warning']}\n数据指纹：{prepared.fingerprint}\n同组与相同输入绑定划分，缩放器只拟合相应训练子集。")
        self.status.setText('数据校验通过，划分无组/重复输入跨集泄漏。')
        return prepared

    def training_config(self):
        hidden = [int(s.strip()) for s in self.hidden.text().split(',') if s.strip()]
        return TrainConfig(hidden, self.epochs.value(), self.patience.value(), float(self.lr.text()), self.batch.value(), self.train_seed.value(), [layer.activation for layer in self.layer_builder.specification()])

    def start_training(self):
        def start():
            data = self.prepared or self.prepare()
            config = self.training_config()
            config.validate(len(data.config.inputs), len(data.config.targets))
            self.train_tabs.setCurrentIndex(2)
            self.begin_training_display()
            self.start_worker(lambda cancel, progress, snapshot: train_model(data, config, cancel, progress, snapshot), self.training_ready, structured=True)
        self.safe(start)

    def begin_training_display(self):
        self._fitting_display = True
        self.metrics_view.hide()
        self.training_axis_bar.hide()
        self.search_hint.hide()
        self.search_view.hide()
        self.search_toggle_button.hide()
        self.training_stats['phase'].setText('准备训练')
        for name in ['epoch', 'best', 'loss']:
            self.training_stats[name].setText('—')

    def start_worker(self, task, handler, structured=False):
        if self.thread is not None:
            return
        self.cancel_event = threading.Event()
        self.thread = C.QThread(self)
        self.worker = Worker(task, self.cancel_event, structured)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self._result_handler = handler
        self.worker.result.connect(self.deliver_result, C.Qt.ConnectionType.QueuedConnection)
        self.worker.error.connect(self.task_error, C.Qt.ConnectionType.QueuedConnection)
        self.worker.progress.connect(self.task_progress, C.Qt.ConnectionType.QueuedConnection)
        self.worker.snapshot.connect(self.handle_snapshot, C.Qt.ConnectionType.QueuedConnection)
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

    @C.Slot(object)
    def handle_snapshot(self, event):
        kind = event.get('kind')
        if kind == 'loss':
            history = event.get('history', [])
            self.training_figure.clear()
            ax = self.training_figure.add_subplot(111)
            style_axis(ax)
            for key, name, color in [('train_loss', 'Training', '#427ca5'), ('validation_loss', 'Validation', '#dc9b32')]:
                points = [(h['epoch'], h.get(key)) for h in history if h.get(key) is not None]
                if points:
                    ax.plot([p[0] for p in points], [p[1] for p in points], label=name, color=color, linewidth=1.8)
            ax.set_xlabel('Epoch / boosting iteration')
            ax.set_ylabel(event.get('metric', 'Scaled MSE'))
            # Use a readable Latin title in Matplotlib on systems without its CJK
            # fallback; the full human-readable task context stays in native Qt.
            ax.set_title(f"{event.get('family','ANN').upper()} · real training progress" + (' · development refit' if event.get('refit') else ''), fontsize=11, loc='left', pad=14)
            ax.legend(fontsize=9, frameon=False, loc='upper right')
            if event.get('best_epoch'):
                ax.axvline(event['best_epoch'], color='#a3b2bd', linestyle=':', linewidth=1)
            self.training_canvas.draw_idle()
            last_val = history[-1].get('validation_loss') if history else None
            self.training_stats['phase'].setText(f"{event.get('family','ann').upper()} · {'早停' if event.get('early_stopped') else '训练中'}")
            self.training_stats['phase'].setToolTip(str(event.get('label', '')) + ' ' + str(event.get('target', '')))
            self.training_stats['epoch'].setText(str(event.get('epoch', '—')))
            self.training_stats['best'].setText(str(event.get('best_epoch', '—')))
            self.training_stats['loss'].setText(f'{last_val:.5g}' if last_val is not None else '无验证集' if event.get('refit') else '—')
            self.train_label.setText(f"真实训练进度：{event.get('family','ANN')}，轮次{event.get('epoch')}，最佳{event.get('best_epoch','见验证日志')}，早停{event.get('early_stopped',False)}；{'开发集固定轮数重拟合，无验证曲线' if event.get('refit') else '早停使用内部留出，不是外层验证/测试'}。")
        elif kind == 'fit':
            self.training_figure.clear()
            ax = self.training_figure.add_subplot(111)
            style_axis(ax)
            ax.barh(['Trees fitted'], [event.get('tree',0)], height=.35, color='#427ca5')
            ax.set_xlim(0,max(1,event.get('trees',1)))
            ax.set_title('Random forest · completed trees (not a loss curve)', loc='left', fontsize=11, pad=14)
            self.training_canvas.draw_idle()
            self.training_stats['phase'].setText('RF · 训练中')
            self.training_stats['epoch'].setText(f"{event.get('tree',0)} / {event.get('trees',0)}")
            self.training_stats['best'].setText('不适用')
            self.training_stats['loss'].setText('不适用')
            self.train_label.setText(f"RF真实进度：已拟合{event.get('tree')}/{event.get('trees')}棵树；不显示伪造损失。取消最迟在当前单棵树完成后响应。")
        elif kind == 'search':
            self.training_stats['phase'].setToolTip(str(event.get('label', '')))
            if event.get('stage') == 'fit':
                self.train_label.setText(f"候选 {event.get('candidate')}/{event.get('candidates')} · 第 {event.get('fold')}/{event.get('folds')} 折；仅开发集参与交叉验证。")
        elif kind == 'optimization':
            self.show_live_optimization(event)

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
        cancelled = bool(self.cancel_event and self.cancel_event.is_set())
        self.thread, self.worker, self.cancel_event = None, None, None
        if getattr(self, '_fitting_display', False):
            self._fitting_display = False
            if self.trained is not None:
                self.show_training()
                self.train_label.setText('保留上一次有效模型 · ' + self.train_label.text())
            else:
                self.training_stats['phase'].setText('已取消' if cancelled else '未完成')
        self.update_enabled()
        if self._close_pending:
            self.close()

    def training_ready(self, trained):
        if self.cancel_event and self.cancel_event.is_set():
            self.status.setText('操作已取消；新结果未采用。')
            return
        self.trained, self.prepared, self.optimization = trained, trained.data, None
        self.external_results = None
        self.show_training()
        self.populate_bounds()
        self.results_model.set_frame(pd.DataFrame())
        self.results_figure.clear()
        self.results_canvas.draw_idle()
        self.results_label.setText('新模型已生效，请重新运行优化。')
        self.interactive_plot.setVisible(False)
        self.status.setText('模型训练完成，请查看逐目标测试指标。')
        self.update_enabled()

    def show_training(self):
        self._fitting_display = False
        self.metrics_view.show()
        self.training_axis_bar.show()
        rows = []
        data = self.trained.data
        for split, metrics in self.trained.metrics.items():
            fit_indices = getattr(self.trained, 'fit_indices', data.splits['train'])
            actual = data.Y[fit_indices if split == 'train' else data.splits[split]]
            mean = data.Y[fit_indices].mean(axis=0)
            for j, (target, values) in enumerate(metrics.items()):
                rows.append({'划分': split, '目标': target, 'R2': values['R2'] if values['R2'] is not None else '未定义（常量目标）',
                             'MAE': values['MAE'], 'MSE': values.get('MSE', values['RMSE']**2), 'RMSE': values['RMSE'],
                             '训练均值基线MAE': float(np.abs(actual[:, j] - mean[j]).mean()),
                             '训练均值基线RMSE': float(np.sqrt(np.square(actual[:, j] - mean[j]).mean()))})
        rows.sort(key=lambda row: {'test': 0, 'validation': 1, 'train': 2}.get(row['划分'], 3))
        self.metrics_model.set_frame(pd.DataFrame(rows))
        self.metrics_view.setColumnWidth(0, 105)
        self.metrics_view.setColumnWidth(1, 285)
        for column in range(2, self.metrics_model.columnCount()):
            self.metrics_view.setColumnWidth(column, 120 if column < 6 else 158)
        family = getattr(self.trained, 'family', 'ann')
        fit_count = len(getattr(self.trained, 'fit_indices', data.splits['train']))
        detail = (f"{self.trained.spec.rf_params()['n_estimators']} 棵树" if family == 'rf'
                  else f'训练轮次 {self.trained.best_epoch}')
        self.train_label.setText(f'{family.upper()} · 实际拟合 {fit_count} 行 · {detail}。逐目标测试指标见上表。')
        self.train_label.setToolTip('指标为目标原单位，MSE 为平方单位；常量目标的 R² 未定义。测试集不参与模型选择或早停。')
        self.training_stats['phase'].setText(f'{family.upper()} · 已完成')
        self.training_stats['epoch'].setText(str(self.trained.spec.rf_params()['n_estimators']) if family == 'rf' else str(self.trained.history[-1]['epoch']) if self.trained.history else str(self.trained.best_epoch))
        self.training_stats['best'].setText('不适用' if family == 'rf' else str(self.trained.best_epoch))
        final_val = self.trained.history[-1].get('validation_loss') if self.trained.history else None
        self.training_stats['loss'].setText(f'{final_val:.5g}' if final_val is not None else '不适用' if family == 'rf' else '无验证集')
        self.show_search_summary()
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
        family = getattr(self.trained, 'family', 'ann')
        if family == 'rf':
            ax, loss_ax = figure.add_subplot(111), None
        else:
            ax, loss_ax = figure.subplots(1, 2)
        style_axis(ax)
        if loss_ax is not None:
            style_axis(loss_ax)
        j = max(0, self.target_plot.currentIndex())
        idx = self.trained.data.splits['test']
        actual = self.trained.data.Y[idx, j]
        pred = self.trained.predict(self.trained.data.X[idx])[:, j]
        ax.scatter(actual, pred, s=31, alpha=.8, color='#427ca5', edgecolors='#ffffff', linewidths=.6, zorder=3)
        low, high = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([low, high], [low, high], '--', color='#91a3b2', linewidth=1.1, label='Identity', zorder=2)
        ax.set_xlabel('Actual · held-out test', fontsize=9)
        ax.set_ylabel('Surrogate prediction', fontsize=9)
        ax.set_title(self.trained.data.config.targets[j], fontsize=10, loc='left', pad=13)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        if loss_ax is not None:
            if family == 'xgboost':
                for i, series in enumerate(self.trained.search_metadata.get('booster_history', [])):
                    points = [(h['epoch'], h['train_loss']) for h in series['history']]
                    if points:
                        loss_ax.plot([p[0] for p in points], [p[1] for p in points], label=series['target'], linewidth=1.6, color=['#427ca5', '#dc9b32', '#408f88', '#8d79a1'][i % 4])
                loss_ax.set_ylabel('Scaled training RMSE')
                loss_ax.set_title('Final refit · no validation set', fontsize=9)
            else:
                for key, name, color in [('train_loss', 'Training', '#427ca5'), ('validation_loss', 'Validation', '#dc9b32')]:
                    points = [(h['epoch'], h.get(key)) for h in self.trained.history if h.get(key) is not None]
                    if points:
                        loss_ax.plot([p[0] for p in points], [p[1] for p in points], label=name, color=color, linewidth=1.7)
                if self.trained.history:
                    loss_ax.axvline(self.trained.best_epoch, linestyle=':', color='#a3b2bd', linewidth=1, label=f'Best epoch {self.trained.best_epoch}')
                loss_ax.set_ylabel('Scaled MSE')
                loss_ax.set_title('Recorded training & validation loss', fontsize=10, loc='left', pad=13)
                if not any(h.get('validation_loss') is not None for h in self.trained.history):
                    loss_ax.set_title('Final refit · no validation set', fontsize=9)
            loss_ax.set_xlabel('Epoch / boosting iteration', fontsize=9)
            if loss_ax.get_legend_handles_labels()[1]:
                loss_ax.legend(fontsize=8, frameon=False, loc='best')
        self.training_canvas.draw_idle()

    def populate_bounds(self, opt=None):
        data = self.trained.data
        X = data.X[getattr(self.trained, 'fit_indices', data.splits['train'])]
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
            self.start_worker(lambda cancel, progress, snapshot: optimize_model(trained, config, cancel, progress, snapshot), self.optimization_ready, structured=True)
        self.safe(start)

    def show_live_optimization(self, event):
        self.live_opt_figure.clear()
        ax = self.live_opt_figure.add_subplot(111)
        style_axis(ax)
        values = np.asarray(event.get('Y', []), dtype=float)
        front = np.asarray(event.get('pareto_Y', []), dtype=float)
        targets = event.get('targets', [])
        if values.size:
            j = min(1, values.shape[1] - 1)
            ax.scatter(values[:, 0], values[:, j], s=18, alpha=.4, color='#98abb7', label='Current population')
            if front.size:
                ax.scatter(front[:, 0], front[:, j], s=28, color='#0b9b91', edgecolors='#fff', linewidths=.5, label='Live predicted frontier')
            ax.set_xlabel(targets[0], fontsize=8)
            ax.set_ylabel(targets[j], fontsize=8)
            ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"NSGA-II · generation {event.get('generation')}/{event.get('generations')} · {event.get('evaluations')} evaluations", fontsize=10, loc='left', pad=12)
        self.live_opt_canvas.draw_idle()

    def optimization_ready(self, result):
        if self.cancel_event and self.cancel_event.is_set():
            self.status.setText('操作已取消；新结果未采用。')
            return
        self.optimization = result
        self.external_results = None
        self.last_filter_plan = None
        self.show_results()
        self.tabs.setCurrentIndex(3)
        self.status.setText('优化完成。请记住：这是代理模型预测Pareto集，不是已验证物理方案。')

    def analysis_frame(self):
        if self.external_results is not None:
            return self.external_results.frame
        return self.optimization.candidates if self.optimization is not None else None

    def analysis_inputs(self):
        return self.external_results.inputs if self.external_results is not None else self.trained.data.config.inputs

    def analysis_targets(self):
        return self.external_results.targets if self.external_results is not None else self.trained.data.config.targets

    def objective_column(self, target):
        return target if self.external_results is not None else prediction_column(target)

    def import_external_results(self):
        path, _ = W.QFileDialog.getOpenFileName(self, '导入外部结果', '', 'CSV / Excel (*.csv *.xlsx)')
        if not path:
            return
        def start():
            raw = read_dataset(path)
            dialog = ExternalMappingDialog(raw, self)
            if dialog.exec() != W.QDialog.DialogCode.Accepted:
                return
            inputs, targets, directions = dialog.mapping
            self.start_worker(lambda cancel, progress: prepare_external_results(raw, inputs, targets, directions, Path(path).name, cancel), self.external_ready)
        self.safe(start)

    def external_ready(self, context):
        if self.cancel_event and self.cancel_event.is_set():
            return
        self.external_results = context
        self.last_filter_plan = None
        self.show_results()
        self.tabs.setCurrentIndex(3)
        self.status.setText('外部结果已载入。按用户映射和方向分析；未虚构输入，也未自动认证数据来源/物理有效性。')

    def natural_language_filter(self):
        def run():
            if self.analysis_frame() is None:
                raise ValueError('请先生成候选或导入外部结果。')
            from .llm_dialog import NaturalLanguageFilterDialog
            dialog = NaturalLanguageFilterDialog(self.results_model.frame.copy(), self)
            if dialog.exec() == W.QDialog.DialogCode.Accepted:
                self.results_model.set_frame(dialog.result_frame.reset_index(drop=True))
                self.last_filter_plan = dialog.plan
                self.results_label.setText(f'已应用用户审阅的结构化筛选计划；剩余{len(dialog.result_frame)}行。仅筛选/排序现有值，不生成或改写预测。')
                self.status.setText(self.results_label.text())
                self.draw_results()
        self.safe(run)

    def show_results(self):
        columns = self.analysis_inputs() + [self.objective_column(t) for t in self.analysis_targets()]
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
            combo.addItems(self.analysis_targets())
            combo.blockSignals(False)
        self.scatter_y.blockSignals(True)
        self.scatter_y.setCurrentIndex(min(1, self.scatter_y.count() - 1))
        self.scatter_y.blockSignals(False)
        self.apply_filter()

    def arrange_results_columns(self):
        """Put stable identity and objectives first without reordering the data."""
        columns = list(self.results_model.frame.columns)
        preferred = ['candidate_id', *[self.objective_column(t) for t in self.analysis_targets()], 'pareto', *self.analysis_inputs()]
        preferred += [name for name in columns if name not in preferred]
        header = self.results_table.horizontalHeader()
        header.setSectionsMovable(True)
        for position, name in enumerate(preferred):
            if name not in columns:
                continue
            logical = columns.index(name)
            header.moveSection(header.visualIndex(logical), position)
            width = 112 if name == 'candidate_id' else 96 if name == 'pareto' else min(265, max(145, self.results_table.fontMetrics().horizontalAdvance(name) + 28))
            self.results_table.setColumnWidth(logical, width)

    def apply_filter(self):
        def apply():
            source = self.analysis_frame()
            if source is None:
                return
            ranges = {}
            for row in range(self.filters.rowCount()):
                low, high = [self.filters.cellWidget(row, col).text().strip() for col in [1, 2]]
                ranges[self.filters.item(row, 0).text()] = (float(low) if low else None, float(high) if high else None)
            frame = filter_candidates(source, ranges, self.result_source.currentIndex() == 0)
            self.results_model.set_frame(frame.reset_index(drop=True))
            self.arrange_results_columns()
            self.results_table.clearSelection()
            kind = '用户外部数据（未认证来源）' if self.external_results is not None else f'代理模型预测：评估{self.optimization.evaluations}次'
            self.results_label.setText(f'{kind}；全部{len(source)}行，Pareto {int(source.pareto.sum())}，当前筛选{len(frame)}。Pareto标记针对全表，不随筛选重新定义。')
            self.results_table.setToolTip(self.results_label.text() + ' 拖动下方分隔线可展开表格。')
            self.last_filter_plan = {'kind':'manual_ranges','ranges':ranges,'pareto_only':self.result_source.currentIndex() == 0}
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

    def table_selection_changed(self, *args):
        if self._sync_selection:
            return
        self.draw_results(refresh_interactive=False)
        frame = self.selected_frame()
        self.interactive_plot.set_selection(frame.candidate_id.tolist() if 'candidate_id' in frame else [])

    @C.Slot(list)
    def plot_selection_changed(self, ids):
        if self._sync_selection or 'candidate_id' not in self.results_model.frame:
            return
        self._sync_selection = True
        selection = C.QItemSelection()
        for row in np.flatnonzero(self.results_model.frame.candidate_id.isin(ids).to_numpy()):
            selection.select(self.results_model.index(int(row),0), self.results_model.index(int(row),self.results_model.columnCount()-1))
        self.results_table.selectionModel().select(selection, C.QItemSelectionModel.SelectionFlag.ClearAndSelect | C.QItemSelectionModel.SelectionFlag.Rows)
        self._sync_selection = False
        self.draw_results(refresh_interactive=False)
        self.results_label.setText(f'图表与表格按稳定候选ID联动：选中{len(ids)}行；可直接导出选中行。')
        self.status.setText(self.results_label.text())

    def draw_results(self, *args, refresh_interactive=True):
        if self.analysis_frame() is None or self.scatter_x.currentIndex() < 0 or self.scatter_y.currentIndex() < 0:
            return
        required = self.analysis_inputs() + [self.objective_column(t) for t in self.analysis_targets()] + ['candidate_id','pareto']
        if not set(required) <= set(self.results_model.frame.columns):
            return
        if refresh_interactive:
            objectives = [self.objective_column(t) for t in self.analysis_targets()]
            if len(objectives) >= 2:
                self.interactive_plot.setVisible(True)
                self.interactive_plot.set_data(self.results_model.frame, objectives, self.analysis_inputs())
                self.interactive_plot.set_selection(self.selected_frame().candidate_id.tolist() if len(self.selected_frame()) else [])
            else:
                self.interactive_plot.setVisible(False)
                self.result_plots.setCurrentIndex(1)
        figure = self.results_figure
        figure.clear()
        ax, parallel = figure.subplots(1, 2)
        style_axis(ax)
        style_axis(parallel)
        frame = self.results_model.frame
        if not len(frame):
            ax.set_title('No candidates in current filter')
            self.results_canvas.draw_idle()
            return
        x, y = self.objective_column(self.scatter_x.currentText()), self.objective_column(self.scatter_y.currentText())
        front = frame[frame.pareto]
        ax.scatter(frame[x], frame[y], s=15, alpha=.3, color='#98abb7', label='External candidates' if self.external_results is not None else 'Predicted candidates')
        ax.scatter(front[x], front[y], s=22, color='#0b9b91', label='Non-dominated candidates' if self.external_results is not None else 'Predicted Pareto')
        selected = self.selected_frame()
        if len(selected):
            ax.scatter(selected[x], selected[y], s=42, color='#f2a430', edgecolors='#8b621e', linewidths=.8, label='Selected', zorder=4)
        ax.set_xlabel(self.scatter_x.currentText(), fontsize=8)
        ax.set_ylabel(self.scatter_y.currentText(), fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        cols = self.analysis_inputs() + [self.objective_column(t) for t in self.analysis_targets()]
        all_values = self.analysis_frame()[cols]
        lo, hi = all_values.min().to_numpy(), all_values.max().to_numpy()
        scale = np.where(hi - lo > 0, hi - lo, 1)
        for subset, color, alpha, limit in [(frame, '#98abb7', .10, 500), (front, '#0b9b91', .25, 300), (selected, '#f2a430', .9, 100)]:
            subset = subset.iloc[np.linspace(0, len(subset) - 1, min(len(subset), limit), dtype=int)] if len(subset) else subset
            if len(subset):
                parallel.plot(np.arange(len(cols)), ((subset[cols].to_numpy() - lo) / scale).T, color=color, alpha=alpha, linewidth=.7)
        parallel.set_xticks(np.arange(len(cols)), [c.replace('predicted::', 'Y:') for c in cols], rotation=70, fontsize=6)
        parallel.set_ylabel('Normalized over all candidates (0–1)', fontsize=8)
        parallel.set_title('Parallel coordinates (display sampled only)', fontsize=9)
        self.results_canvas.draw_idle()

    def export_rows(self, which):
        def export():
            if self.analysis_frame() is None:
                raise ValueError('没有可导出的有效候选/外部数据。')
            frame = self.analysis_frame() if which == 'all' else self.selected_frame() if which == 'selected' else self.results_model.frame
            if not len(frame):
                raise ValueError('没有符合条件/选中的行。')
            path, _ = W.QFileDialog.getSaveFileName(self, '导出预测候选及来源JSON', 'predicted_candidates.csv', 'CSV (*.csv)')
            if not path:
                return
            if not path.lower().endswith('.csv'):
                path += '.csv'
            if self.external_results is not None:
                export_external(path, frame, self.external_results, self.last_filter_plan)
            else:
                export_candidates(path, frame, self.trained, self.optimization)
                provenance_path = Path(path).with_suffix(Path(path).suffix + '.provenance.json')
                provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
                provenance['filter_plan'] = self.last_filter_plan
                provenance_path.write_text(json.dumps(provenance,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            self.status.setText(f'已导出{len(frame)}行及 .provenance.json：{path}')
        self.safe(export)

    def capture_analysis_state(self):
        return {'scatter_x':self.scatter_x.currentText(), 'scatter_y':self.scatter_y.currentText(),
                'result_source':self.result_source.currentIndex(), 'filter_plan':self.last_filter_plan,
                'visible_ids':self.results_model.frame.candidate_id.tolist() if 'candidate_id' in self.results_model.frame else [],
                'selected_ids':self.selected_frame().candidate_id.tolist() if 'candidate_id' in self.results_model.frame else [],
                'ranges':{self.filters.item(r,0).text():[self.filters.cellWidget(r,1).text(),self.filters.cellWidget(r,2).text()] for r in range(self.filters.rowCount())}}

    def restore_analysis_state(self):
        state = getattr(self.trained,'search_metadata',{}).get('analysis_state',{})
        if not state or self.optimization is None:
            return
        for widget, key in [(self.scatter_x,'scatter_x'),(self.scatter_y,'scatter_y')]:
            index = widget.findText(state.get(key,''))
            if index >= 0:
                widget.setCurrentIndex(index)
        self.result_source.setCurrentIndex(int(state.get('result_source',0)))
        for r in range(self.filters.rowCount()):
            values = state.get('ranges',{}).get(self.filters.item(r,0).text(),['',''])
            self.filters.cellWidget(r,1).setText(values[0])
            self.filters.cellWidget(r,2).setText(values[1])
        ids = state.get('visible_ids',[])
        source = self.optimization.candidates
        if len(ids) == len(set(ids)) and set(ids) <= set(source.candidate_id):
            self.results_model.set_frame(source.set_index('candidate_id',drop=False).loc[ids].reset_index(drop=True))
        self.last_filter_plan = state.get('filter_plan')
        self.draw_results()
        self.plot_selection_changed(state.get('selected_ids',[]))

    def save_project(self):
        def save():
            if self.trained is None:
                raise ValueError('请先训练模型。')
            path, _ = W.QFileDialog.getSaveFileName(self, '保存模型项目（不含外部结果表；外部表请用CSV导出）', 'project.dwb', 'Daylight project (*.dwb)')
            if not path:
                return
            if not path.lower().endswith('.dwb'):
                path += '.dwb'
            if hasattr(self.trained, 'search_metadata') and self.external_results is None:
                self.trained.search_metadata['analysis_state'] = self.capture_analysis_state()
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
            if hasattr(trained, 'spec'):
                self.search_controls.family.setCurrentIndex(self.search_controls.family.findData(trained.family))
                if trained.family == 'ann':
                    self.layer_builder.set_layers(trained.spec.layers)
            else:
                self.layer_builder.set_layers([LayerSpec(w, (trained.config.activations[i] if trained.config.activations else 'relu')) for i,w in enumerate(trained.config.hidden)])
            self.hidden.setText(','.join(str(l.width) for l in self.layer_builder.specification()))
            self.epochs.setValue(trained.config.epochs)
            self.patience.setValue(trained.config.patience)
            self.lr.setText(str(trained.config.learning_rate))
            self.batch.setValue(trained.config.batch_size)
            self.train_seed.setValue(trained.config.seed)
            self._loading = False
            self.trained, self.prepared, self.optimization = trained, trained.data, optimization
            self.external_results = None
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
                self.restore_analysis_state()
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
