"""Dedicated, opt-in conversational preference workbench.

All recommendations and values are computed locally. Preview, network Send and
local Apply are independent explicit actions. No key or chat is persisted.
"""
from __future__ import annotations

import copy
import json
import threading

import numpy as np
import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .llm_filter import PlanError, RequestError, RequestSettings
from .preference_engine import (
    ObjectiveSpec, RankingResult, build_preference_preview, default_preference_plan,
    make_normalization_reference, preference_request_body, rank_candidates,
    request_preference_plan, validate_preference_plan,
)


class _Relay(C.QObject):
    completed = C.Signal(int, object, object)


def _text(text, role=None):
    label = W.QLabel(text)
    label.setTextFormat(C.Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if role:
        label.setProperty("role", role)
    return label


def _button(text, callback, primary=False):
    button = W.QPushButton(text)
    button.clicked.connect(callback)
    if primary:
        button.setProperty("role", "primary")
    return button


def _readonly_item(value):
    item = W.QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~C.Qt.ItemFlag.ItemIsEditable)
    return item


def _table(headers):
    table = W.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(33)
    table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class PreferenceDecisionPage(W.QWidget):
    """Session-only dialogue with a reviewable declarative plan and local plots.

    set_candidates is called only for a new analysis source. set_visible_ids and
    set_selection never recompute normalization or silently re-rank proposals.
    """
    selectionChanged = C.Signal(list)
    planApplied = C.Signal(object, object)  # original shortlist frame, validated plan

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._objectives: list[ObjectiveSpec] = []
        self._inputs = []
        self._visible_ids = []
        self._selected_ids = []
        self._reference = None
        self._mapping_confirmed = False
        self._preview = None
        self._preview_settings = None
        self._preview_body = None
        self._state = None
        self._proposed = None
        self._applied = None
        self.ranking: RankingResult | None = None
        self._request_id = 0
        self._busy = False
        self._sync = False
        self._editing = False
        self._relay = _Relay(self)
        self._relay.completed.connect(self._request_done, C.Qt.ConnectionType.QueuedConnection)
        self._build_ui()
        self._set_busy(False)

    def _build_ui(self):
        root = W.QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)
        head = W.QHBoxLayout()
        title = _text("偏好决策 / DeepSeek", "sectionTitle")
        head.addWidget(title)
        head.addWidget(_text("自然语言表达倾向，本地计算权衡", "muted"))
        head.addStretch()
        self.mapping_button = _button("核对目标映射", lambda: self.detail_tabs.setCurrentIndex(3))
        head.addWidget(self.mapping_button)
        self.reset_button = _button("重置本轮偏好", self.reset_preferences)
        head.addWidget(self.reset_button)
        root.addLayout(head)
        self.context_label = _text("请先在结果页生成候选或导入外部结果。", "muted")
        root.addWidget(self.context_label)
        split = W.QSplitter(C.Qt.Orientation.Horizontal)
        left = W.QWidget()
        left.setMinimumWidth(300)
        chat_layout = W.QVBoxLayout(left)
        chat_layout.setContentsMargins(0, 0, 8, 0)
        chat_layout.addWidget(_text("讨论偏好", "sectionTitle"))
        self.chat = W.QPlainTextEdit()
        self.chat.setReadOnly(True)
        self.chat.setObjectName("preferenceConversation")
        self.chat.setPlaceholderText("偏好可连续调整，例如先强调采光，再补充眩光上限。对话仅保留在当前会话。")
        chat_layout.addWidget(self.chat, 1)
        quick = W.QHBoxLayout()
        quick.addWidget(_button("更考虑采光", lambda: self.query_edit.setPlainText("帮我选更考虑采光的")))
        quick.addWidget(_button("能耗与眩光平衡", lambda: self.query_edit.setPlainText("我要能耗和眩光平衡的方案")))
        chat_layout.addLayout(quick)
        self.query_edit = W.QPlainTextEdit()
        self.query_edit.setPlaceholderText("例如：更重视采光；保持刚才的限制；能耗和眩光各占一半。")
        self.query_edit.setMaximumHeight(108)
        self.query_edit.setMinimumHeight(75)
        chat_layout.addWidget(self.query_edit)
        send_row = W.QHBoxLayout()
        self.preview_button = _button("预览发送内容", self.make_preview)
        self.send_button = _button("发送", self.send, True)
        self.cancel_button = _button("取消请求", self.cancel_request)
        send_row.addWidget(self.preview_button)
        send_row.addWidget(self.send_button)
        send_row.addWidget(self.cancel_button)
        chat_layout.addLayout(send_row)
        chat_layout.addWidget(_text("先预览全部传出内容，再发送。模型只提议，不直接选方案；右侧审核后应用。", "muted"))
        self.status_label = _text("尚未联网。API Key 只在当前页面会话内使用。", "muted")
        chat_layout.addWidget(self.status_label)
        split.addWidget(left)
        self.detail_tabs = W.QTabWidget()
        self.detail_tabs.setObjectName("decisionDetails")
        self.detail_tabs.addTab(self._review_pane(), "偏好审核")
        self.detail_tabs.addTab(self._recommendation_pane(), "推荐与权衡")
        self.detail_tabs.addTab(self._plot_pane(), "交互图表")
        self.detail_tabs.addTab(self._mapping_pane(), "映射与发送")
        self.detail_tabs.setTabToolTip(2, "完整候选集的 2D / 3D / 平行坐标；按原始候选 ID 联动。")
        split.addWidget(self.detail_tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)
        split.setSizes([355, 850])
        root.addWidget(split, 1)
        self.query_edit.textChanged.connect(self._invalidate_outbound)
        self.model_edit.textChanged.connect(self._invalidate_outbound)

    def _review_pane(self):
        page = W.QWidget()
        box = W.QVBoxLayout(page)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(7)
        self.intent_label = _text("待提出偏好。常见的倾向和平衡可先用透明的默认权重，无需先指定硬阈值。")
        box.addWidget(self.intent_label)
        self.weight_table = _table(["现有目标 / 已确认含义", "优化方向", "数值权重", "单位"])
        self.weight_table.setMinimumHeight(138)
        self.weight_table.setMaximumHeight(205)
        box.addWidget(self.weight_table)
        row = W.QHBoxLayout()
        self.scope_combo = W.QComboBox()
        for label, value in [("原始 Pareto 候选", "pareto"), ("全部候选", "all"), ("结果页当前视图", "current_view")]:
            self.scope_combo.addItem(label, value)
        self.normalization_combo = W.QComboBox()
        self.normalization_combo.addItem("固定全候选 min–max", "fixed_minmax")
        self.normalization_combo.addItem("固定全候选 P5–P95（截断）", "fixed_robust")
        self.limit_spin = W.QSpinBox()
        self.limit_spin.setRange(1, 100)
        self.limit_spin.setValue(5)
        row.addWidget(_text("范围"))
        row.addWidget(self.scope_combo)
        row.addWidget(_text("归一化"))
        row.addWidget(self.normalization_combo, 1)
        row.addWidget(_text("推荐数"))
        row.addWidget(self.limit_spin)
        box.addLayout(row)
        constraints_head = W.QHBoxLayout()
        constraints_head.addWidget(_text("硬约束（原单位，AND；不从“更好”臆造阈值）", "muted"))
        constraints_head.addStretch()
        self.add_constraint_button = _button("添加", self.add_constraint)
        self.remove_constraint_button = _button("删除所选", self.remove_constraints)
        constraints_head.addWidget(self.add_constraint_button)
        constraints_head.addWidget(self.remove_constraint_button)
        box.addLayout(constraints_head)
        self.constraint_table = _table(["现有数值 / 布尔列", "操作符", "阈值 / between 下限,上限"])
        self.constraint_table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.constraint_table.verticalHeader().show()
        self.constraint_table.verticalHeader().setToolTip("点击行号选择约束，再点击“删除所选”。")
        self.constraint_table.setMaximumHeight(105)
        self.constraint_table.setMinimumHeight(38)
        box.addWidget(self.constraint_table)
        self.assumptions_label = _text("权重是可编辑的偏好假设，不代表物理真值或唯一最佳。", "muted")
        box.addWidget(self.assumptions_label)
        action = W.QHBoxLayout()
        self.apply_button = _button("应用审核后的偏好并联动选择", self.apply_reviewed_plan, True)
        action.addWidget(self.apply_button)
        self.score_label = _text("尚未应用。", "muted")
        action.addWidget(self.score_label, 1)
        box.addLayout(action)
        for widget in (self.scope_combo, self.normalization_combo):
            widget.currentIndexChanged.connect(self._review_changed)
        self.limit_spin.valueChanged.connect(self._review_changed)
        box.addStretch(1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def _recommendation_pane(self):
        page = W.QWidget()
        box = W.QVBoxLayout(page)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)
        self.applied_summary = _text("尚未应用偏好。先在“偏好审核”检查权重、方向和硬约束。", "muted")
        box.addWidget(self.applied_summary)
        self.recommendations = _table(["候选 ID", "偏好损失 ↓", "原始 Pareto"])
        self.recommendations.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.recommendations.setSelectionMode(W.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.recommendations.setMinimumHeight(145)
        self.recommendations.itemSelectionChanged.connect(self._recommendation_selection)
        box.addWidget(self.recommendations, 1)
        self.comparison_figure = Figure(figsize=(7, 1.5), constrained_layout=True, facecolor="white")
        self.comparison_canvas = FigureCanvasQTAgg(self.comparison_figure)
        self.comparison_canvas.setMinimumHeight(165)
        self.comparison_canvas.setMaximumHeight(260)
        box.addWidget(self.comparison_canvas)
        self.tradeoff_label = _text("比较图使用同一个固定对照集。各目标原始值与单位在表内显示。", "muted")
        self.tradeoff_label.setMinimumHeight(35)
        box.addWidget(self.tradeoff_label)
        return page

    def _plot_pane(self):
        page = W.QWidget()
        box = W.QVBoxLayout(page)
        box.setContentsMargins(8, 8, 8, 8)
        box.addWidget(_text("完整候选始终保留作背景，原始 Pareto 标记不随权重改变。点击推荐、图形和结果表按同一候选 ID 联动。", "muted"))
        from .interactive_plot import InteractiveParetoWidget
        self.interactive_plot = InteractiveParetoWidget()
        self.interactive_plot.selectionChanged.connect(self._plot_selection)
        box.addWidget(self.interactive_plot, 1)
        return page

    def _mapping_pane(self):
        page = W.QWidget()
        box = W.QVBoxLayout(page)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(8)
        box.addWidget(_text("先核对一次目标含义。CSV 名称不是物理证据：保留原列名，可修改显示名、别名、方向和单位。方向改变仅用于偏好打分，不重算原始 Pareto。", "muted"))
        self.mapping_table = _table(["原始目标列（不改数据）", "显示含义", "方向", "自然语言别名（逗号分隔）", "单位"])
        self.mapping_table.setMinimumHeight(138)
        self.mapping_table.setMaximumHeight(205)
        self.mapping_table.itemChanged.connect(self._mapping_edited)
        box.addWidget(self.mapping_table)
        self.confirm_mapping_button = _button("确认这些目标含义和方向", self.confirm_mapping, True)
        box.addWidget(self.confirm_mapping_button)
        settings = W.QHBoxLayout()
        self.model_edit = W.QLineEdit("deepseek-flash")
        self.model_edit.setMaximumWidth(200)
        self.key_edit = W.QLineEdit()
        self.key_edit.setEchoMode(W.QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("API Key，仅会话内存，不保存")
        settings.addWidget(_text("模型"))
        settings.addWidget(self.model_edit)
        settings.addWidget(_text("API Key"))
        settings.addWidget(self.key_edit, 1)
        self.clear_key_button = _button("清除密钥", self.key_edit.clear)
        settings.addWidget(self.clear_key_button)
        box.addLayout(settings)
        box.addWidget(_text("仅向 api.deepseek.com 官方 HTTPS 地址发送一次；无自动重试或重定向。可能产生 API 费用。只发送问题、列/目标映射、汇总范围及当前偏好，不发送候选行、ID、训练数据或项目。你自己输入的问题也会原样发送，请勿填入秘密。", "muted"))
        box.addWidget(_text("实际发送的完整 JSON 请求体（不含 Authorization 密钥）", "sectionTitle"))
        self.preview_edit = W.QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setObjectName("preferenceOutboundPreview")
        self.preview_edit.setPlaceholderText("点击左侧“预览发送内容”；预览与后续发送使用同一正文。")
        self.preview_edit.setMinimumHeight(160)
        box.addWidget(self.preview_edit, 1)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(W.QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    def set_candidates(self, frame, objective_specs, input_columns=(), visible_ids=None):
        self.cancel_request(quiet=True)
        self._frame = frame.copy(deep=True) if frame is not None else pd.DataFrame()
        self._objectives = [o if isinstance(o, ObjectiveSpec) else ObjectiveSpec(**{**o, "aliases": tuple(o.get("aliases", ()))}) for o in objective_specs]
        self._inputs = list(input_columns)
        self._visible_ids = list(visible_ids) if visible_ids is not None else self._frame.get("candidate_id", pd.Series(dtype=object)).tolist()
        self._selected_ids = []
        self._mapping_confirmed = False
        self._reference = None
        self._state = self._proposed = self._applied = None
        self.ranking = None
        self._preview = self._preview_settings = self._preview_body = None
        self.preview_edit.clear()
        self._editing = True
        self.mapping_table.setRowCount(len(self._objectives))
        for row, spec in enumerate(self._objectives):
            self.mapping_table.setItem(row, 0, _readonly_item(spec.column))
            for col, value in [(1, spec.label), (3, ", ".join(spec.aliases)), (4, spec.unit)]:
                self.mapping_table.setItem(row, col, W.QTableWidgetItem(value))
            direction = W.QComboBox()
            direction.addItem("最小化 ↓", "min")
            direction.addItem("最大化 ↑", "max")
            direction.setCurrentIndex(direction.findData(spec.direction))
            direction.currentIndexChanged.connect(self._mapping_edited)
            self.mapping_table.setCellWidget(row, 2, direction)
        self._editing = False
        self.recommendations.setRowCount(0)
        self.weight_table.setRowCount(0)
        self.constraint_table.setRowCount(0)
        self.intent_label.setText("请核对当前目标映射，再开始偏好讨论。")
        self.score_label.setText("尚未应用。")
        self.applied_summary.setText("尚未应用当前候选集的偏好；旧对照集的建议已失效。")
        self.confirm_mapping_button.setText("确认这些目标含义和方向")
        self.context_label.setText(f"完整候选 {len(self._frame)} 行；归一化固定对照集，不随筛选/点击改变。目标含义待确认。" if len(self._frame) else "请先在结果页生成候选或导入外部结果。")
        if self._objectives and len(self._frame):
            try:
                self._reference = make_normalization_reference(self._frame, self._objectives)
                if len(self._objectives) >= 2:
                    # The graph represents unavailable/non-finite coordinates as
                    # missing, never as invented finite values. Original rows in
                    # the ranking/table remain unchanged and incomplete weighted
                    # metrics are excluded explicitly by the local engine.
                    plot_frame = self._frame.copy()
                    for spec in self._objectives:
                        numbers = pd.to_numeric(plot_frame[spec.column], errors="coerce")
                        plot_frame[spec.column] = numbers.where(np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan)))
                    self.interactive_plot.set_data(plot_frame, [o.column for o in self._objectives], self._inputs)
                    self.interactive_plot.show()
                else:
                    self.interactive_plot.hide()
                self.chat.appendPlainText("系统：载入新的候选对照集。请核对目标映射，旧建议不再应用。")
                self.detail_tabs.setCurrentIndex(3)
            except (PlanError, ValueError) as exc:
                self.status_label.setText(str(exc))
        self._draw_comparison()
        self._set_busy(False)

    def clear_candidates(self):
        if self._frame.empty and not self._objectives:
            return
        self.set_candidates(None, [])
        self.key_edit.clear()
        self.status_label.setText("原分析结果已失效，旧偏好建议和选择不再可用。尚未联网。")

    def set_visible_ids(self, ids):
        new = list(ids)
        changed = new != self._visible_ids
        self._visible_ids = new
        if changed and self._proposed and self._proposed["scope"] == "current_view":
            self._invalidate_outbound()
            self.status_label.setText("结果页当前视图已改变。原推荐不自动重排；请重新审核并应用。")

    def _mapping_edited(self, *args):
        if self._editing:
            return
        self._mapping_confirmed = False
        self._invalidate_outbound()
        self.apply_button.setEnabled(False)
        self.confirm_mapping_button.setText("映射已修改，请重新确认")
        self.status_label.setText("目标含义/方向已改动，需确认后再提议或应用；旧图与推荐仅供回看。")

    def confirm_mapping(self):
        if self._frame.empty:
            self.status_label.setText("请先生成候选或导入外部结果。")
            return
        try:
            objectives = []
            for row in range(self.mapping_table.rowCount()):
                def item(col):
                    return self.mapping_table.item(row, col).text().strip()
                aliases = tuple(a.strip() for a in item(3).replace("，", ",").split(",") if a.strip())
                objectives.append(ObjectiveSpec(item(0), item(1), self.mapping_table.cellWidget(row, 2).currentData(), aliases, item(4)))
            build_preference_preview(self._frame, "审核目标映射", objectives)
            reference = make_normalization_reference(self._frame, objectives)
        except PlanError as exc:
            self.status_label.setText(str(exc))
            return
        same_columns = [o.column for o in objectives] == [o.column for o in self._objectives]
        self._objectives = objectives
        self._reference = reference
        self._mapping_confirmed = True
        self.ranking = None
        self._applied = None
        self.recommendations.setRowCount(0)
        self.applied_summary.setText("目标映射已确认，尚未应用当前语义/方向下的偏好。")
        draft = self._state if self._state and same_columns else default_preference_plan(objectives, scope="pareto" if "pareto" in self._frame else "all")
        self._proposed = validate_preference_plan(draft, self._frame, objectives)
        self._load_plan_editor(self._proposed)
        self.confirm_mapping_button.setText("目标映射已确认（可修改）")
        self.mapping_button.setText("目标映射已确认")
        self.context_label.setText(f"完整候选 {len(self._frame)} 行；{len(objectives)} 个明确方向目标；全候选固定归一化。原始 Pareto 标记不变。")
        self.chat.appendPlainText("系统：目标含义已确认。可以自然表达倾向或平衡，也可直接修改右侧本地权重；所有变更需点击应用。")
        self.status_label.setText("可开始讨论，或审核右侧本地初始权重。尚未联网。")
        self.detail_tabs.setCurrentIndex(0)
        self._draw_comparison()
        self._set_busy(False)

    def _invalidate_outbound(self, *args):
        self._preview = self._preview_settings = self._preview_body = None
        self.preview_edit.clear()
        self.send_button.setEnabled(False)

    def _load_plan_editor(self, plan):
        self._editing = True
        self._proposed = copy.deepcopy(plan)
        self.intent_label.setText("待审核意图：" + plan["intent"])
        weights = {w["column"]: w["weight"] for w in plan["weights"]}
        total = sum(weights.values())
        self.weight_table.setRowCount(len(self._objectives))
        self.weight_table.setFixedHeight(min(205, 39 + 33 * len(self._objectives)))
        for row, spec in enumerate(self._objectives):
            self.weight_table.setItem(row, 0, _readonly_item(f"{spec.label} · {spec.column}"))
            self.weight_table.setItem(row, 1, _readonly_item("最大化 ↑" if spec.direction == "max" else "最小化 ↓"))
            spin = W.QDoubleSpinBox()
            spin.setRange(0, 100)
            spin.setDecimals(4)
            spin.setSingleStep(.05)
            spin.setValue(weights[spec.column])
            spin.setToolTip(f"应用时按总和归一化；当前占比 {weights[spec.column]/total:.1%}。零权重明确不参与打分。")
            spin.valueChanged.connect(self._review_changed)
            self.weight_table.setCellWidget(row, 2, spin)
            self.weight_table.setItem(row, 3, _readonly_item(spec.unit or "未指定，不推断"))
        self.scope_combo.setCurrentIndex(self.scope_combo.findData(plan["scope"]))
        self.normalization_combo.setCurrentIndex(self.normalization_combo.findData(plan["normalization"]))
        self.limit_spin.setValue(plan["limit"])
        self.constraint_table.setRowCount(0)
        for constraint in plan["constraints"]:
            self.add_constraint(constraint)
        self.assumptions_label.setText(plan["explanation"] + "\n" + "；".join(plan["assumptions"]))
        self._editing = False
        self.apply_button.setEnabled(self._mapping_confirmed and not plan["needs_clarification"] and not self._busy)

    def add_constraint(self, item=None):
        if isinstance(item, bool):
            item = None
        if self.constraint_table.rowCount() >= 12 or not len(self._frame):
            return
        schema = build_preference_preview(self._frame, "本地约束", self._objectives)["columns"]
        row = self.constraint_table.rowCount()
        self.constraint_table.insertRow(row)
        column = W.QComboBox()
        for spec in schema:
            column.addItem(spec["name"])
        operator = W.QComboBox()
        operator.addItems(["<=", ">=", "<", ">", "==", "!=", "between"])
        value = W.QLineEdit()
        value.setPlaceholderText("如 60 或 30,50；布尔 true/false")
        if item:
            column.setCurrentText(item["column"])
            operator.setCurrentText(item["operator"])
            value.setText(",".join(str(v) for v in item["value"]) if isinstance(item["value"], list) else str(item["value"]).lower())
        for col, widget in enumerate((column, operator, value)):
            self.constraint_table.setCellWidget(row, col, widget)
        column.currentIndexChanged.connect(self._review_changed)
        operator.currentIndexChanged.connect(self._review_changed)
        value.textChanged.connect(self._review_changed)
        self._review_changed()

    def remove_constraints(self):
        rows = sorted({i.row() for i in self.constraint_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.constraint_table.removeRow(row)
        self._review_changed()

    def _review_changed(self, *args):
        if self._editing:
            return
        self._invalidate_outbound()
        self.score_label.setText("参数已修改，原推荐未重排；应用后更新。")
        self.apply_button.setEnabled(self._mapping_confirmed and self._proposed is not None and not self._proposed["needs_clarification"] and not self._busy)

    def reviewed_plan(self):
        if not self._mapping_confirmed or self._proposed is None:
            raise PlanError("请先确认目标含义。")
        plan = copy.deepcopy(self._proposed)
        plan["weights"] = [{"column": spec.column, "weight": self.weight_table.cellWidget(row, 2).value()} for row, spec in enumerate(self._objectives)]
        plan["scope"] = self.scope_combo.currentData()
        plan["normalization"] = self.normalization_combo.currentData()
        plan["limit"] = self.limit_spin.value()
        plan["constraints"] = []
        schema = {c["name"]: c["type"] for c in build_preference_preview(self._frame, "本地约束", self._objectives)["columns"]}
        for row in range(self.constraint_table.rowCount()):
            col = self.constraint_table.cellWidget(row, 0).currentText()
            op = self.constraint_table.cellWidget(row, 1).currentText()
            text = self.constraint_table.cellWidget(row, 2).text().strip()
            try:
                if schema[col] == "boolean":
                    if text.lower() not in ("true", "false"):
                        raise ValueError()
                    value = text.lower() == "true"
                else:
                    value = [float(v.strip()) for v in text.replace("，", ",").split(",")] if op == "between" else float(text)
            except ValueError:
                raise PlanError("请为每条硬约束输入有效数值，between 使用两个逗号分隔值；布尔使用 true/false。") from None
            plan["constraints"].append({"column": col, "operator": op, "value": value})
        return validate_preference_plan(plan, self._frame, self._objectives)

    def make_preview(self):
        if self._busy:
            return
        try:
            state = self.reviewed_plan()
            preview = build_preference_preview(self._frame, self.query_edit.toPlainText(), self._objectives, state)
            settings = RequestSettings(model=self.model_edit.text().strip())
            body = preference_request_body(preview, settings)
            body_text = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (PlanError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._preview, self._preview_settings, self._preview_body = preview, settings, body_text
        self.preview_edit.setPlainText(body_text)
        self.send_button.setEnabled(True)
        self.status_label.setText("尚未联网。请核对完整预览；发送目标 api.deepseek.com，可能产生 API 费用。")
        self.detail_tabs.setCurrentIndex(3)

    def send(self):
        if self._busy or self._preview is None:
            return
        key = self.key_edit.text()
        if not key.strip():
            self.status_label.setText("请在“目标语义与发送预览”中输入会话 API Key。")
            self.detail_tabs.setCurrentIndex(3)
            return
        # Deep copies freeze the exact body and validated prior preferences even
        # if another part of the window changes while the request is in flight.
        preview, settings = copy.deepcopy(self._preview), self._preview_settings
        if json.dumps(preference_request_body(preview, settings), ensure_ascii=False, allow_nan=False, separators=(",", ":")) != self._preview_body:
            self._invalidate_outbound()
            self.status_label.setText("请求内容已改变，请重新预览。未发送。")
            return
        self._request_id += 1
        job_id = self._request_id
        self._state = copy.deepcopy(preview["preference_state"])
        self.chat.appendPlainText("你：" + preview["query"])
        self._set_busy(True)
        self.status_label.setText("正在等待 DeepSeek 的结构化建议；尚未改变任何方案选择。")
        relay = self._relay

        def run():
            try:
                outcome = request_preference_plan(settings, key, preview)
                error = None
            except Exception as exc:
                outcome = None
                error = str(exc) if isinstance(exc, (PlanError, RequestError)) else "请求失败或响应无效，请检查网络、密钥或模型。"
            try:
                relay.completed.emit(job_id, outcome, error)
            except RuntimeError:
                pass  # Widget was destroyed; never log the key or remote response.

        threading.Thread(target=run, daemon=True).start()

    def _set_busy(self, busy):
        self._busy = busy
        has_data = bool(len(self._frame)) and self._reference is not None
        self.send_button.setEnabled(not busy and self._preview is not None and self._mapping_confirmed)
        self.preview_button.setEnabled(not busy and has_data and self._mapping_confirmed)
        self.cancel_button.setEnabled(busy)
        self.query_edit.setEnabled(not busy)
        self.model_edit.setEnabled(not busy)
        self.mapping_table.setEnabled(not busy)
        self.confirm_mapping_button.setEnabled(not busy and has_data)
        self.reset_button.setEnabled(not busy)
        self.weight_table.setEnabled(not busy)
        self.constraint_table.setEnabled(not busy)
        self.add_constraint_button.setEnabled(not busy and has_data and self._mapping_confirmed)
        self.remove_constraint_button.setEnabled(not busy and has_data and self._mapping_confirmed)
        for widget in (self.scope_combo, self.normalization_combo, self.limit_spin):
            widget.setEnabled(not busy and has_data and self._mapping_confirmed)
        self.apply_button.setEnabled(not busy and has_data and self._mapping_confirmed and self._proposed is not None and not self._proposed["needs_clarification"])

    @C.Slot(int, object, object)
    def _request_done(self, job_id, outcome, error):
        if job_id != self._request_id:
            return
        self._set_busy(False)
        if error is not None:
            self.chat.appendPlainText("系统：" + str(error))
            self.status_label.setText(str(error) + " 旧选择未改变。可重新预览后再发送。")
            self._invalidate_outbound()
            return
        try:
            validated = validate_preference_plan(outcome, self._frame, self._objectives)
        except PlanError as exc:
            self.status_label.setText(str(exc) + " 未应用建议。")
            return
        self._state = copy.deepcopy(validated)
        self._load_plan_editor(validated)
        self._invalidate_outbound()
        self.chat.appendPlainText("DeepSeek（建议，未应用）：" + validated["intent"] + "\n" + validated["explanation"] + "\n" + "；".join(validated["assumptions"]))
        self.query_edit.clear()
        self.detail_tabs.setCurrentIndex(0)
        if validated["needs_clarification"]:
            self.status_label.setText("需要澄清目标含义：" + validated["explanation"])
        else:
            self.status_label.setText("已生成待审核建议。请核对目标映射、各权重、范围和硬约束，再应用。")

    def cancel_request(self, quiet=False):
        was_busy = self._busy
        self._request_id += 1
        self._set_busy(False)
        self._invalidate_outbound()
        if not quiet and was_busy:
            self.status_label.setText("请求已在界面取消，迟到响应不会采用；已发出的 HTTPS 请求可能仍完成并计费，最长受 20 秒超时约束。")
            self.chat.appendPlainText("系统：已取消等待，旧偏好与选择不变。")

    def reset_preferences(self):
        if not self._mapping_confirmed:
            return
        self.cancel_request(quiet=True)
        self._state = None
        self._applied = None
        self.ranking = None
        self.chat.clear()
        self.query_edit.clear()
        self.recommendations.setRowCount(0)
        self._load_plan_editor(default_preference_plan(self._objectives, scope="pareto" if "pareto" in self._frame else "all"))
        self.chat.appendPlainText("系统：偏好已重置。目标映射与固定完整候选对照集保持不变；尚未改变结果页选择。")
        self.status_label.setText("偏好已重置为待审核的本地等权起点。尚未联网。")
        self.applied_summary.setText("偏好已重置，旧结果不再作为当前推荐。结果页的已有选择未改变。")
        self._draw_comparison()

    def apply_reviewed_plan(self):
        if self._busy:
            return
        try:
            plan = self.reviewed_plan()
            ranking = rank_candidates(self._frame, plan, self._objectives, self._reference, self._visible_ids)
        except PlanError as exc:
            self.status_label.setText(str(exc))
            return
        self._state = self._applied = self._proposed = copy.deepcopy(plan)
        self.ranking = ranking
        self._show_recommendations()
        self.score_label.setText(f"范围 {ranking.scope_count} → 合格 {ranking.eligible_count} → 推荐 {len(ranking.frame)}；损失越低越符合偏好。")
        weight_sum = sum(w["weight"] for w in plan["weights"])
        weights = "；".join(f"{spec.label} {next(w['weight'] for w in plan['weights'] if w['column'] == spec.column)/weight_sum:.1%}" for spec in self._objectives)
        self.chat.appendPlainText("本地已应用：" + weights + f"。推荐 {len(ranking.frame)} 个现有方案。" + (" 无符合项，未自动放宽条件。" if ranking.frame.empty else ""))
        self.status_label.setText("已在本地应用并显示推荐。可继续对话或修改权重，旧结果不会悄悄重排。")
        self.applied_summary.setText("已应用权重：" + weights + f"。范围 {ranking.scope_count} → 合格 {ranking.eligible_count} → 推荐 {len(ranking.frame)}；" + ("固定完整候选 min–max。" if plan["normalization"] == "fixed_minmax" else "固定完整候选 P5–P95（截断）。"))
        self.tradeoff_label.setText("损失越低越符合当前偏好，不代表物理真值或唯一最佳。完整候选固定归一化，原始 Pareto 标记保留；同分按稳定 ID 排序。" + (f" 已排除 {ranking.excluded_nonfinite} 个正权重指标不完整候选。" if ranking.excluded_nonfinite else "") + (" 常量目标损失为 0，不区分候选。" if ranking.constant_objectives else ""))
        self.tradeoff_label.setToolTip("\n".join(ranking.notes))
        self.detail_tabs.setCurrentIndex(1)
        self.planApplied.emit(ranking.frame.copy(), copy.deepcopy(plan))
        self.set_selection(ranking.frame.candidate_id.head(3).tolist())
        self.selectionChanged.emit(self._selected_ids[:])
        self._invalidate_outbound()

    def _show_recommendations(self):
        self._sync = True
        frame = self.ranking.frame
        headers = ["候选 ID", "偏好损失 ↓", "原始 Pareto", *[f"{o.label} ({o.unit}) · {o.column}" if o.unit else f"{o.label} · {o.column}" for o in self._objectives]]
        self.recommendations.setColumnCount(len(headers))
        self.recommendations.setHorizontalHeaderLabels(headers)
        self.recommendations.setRowCount(len(frame))
        for i, (_, row) in enumerate(frame.iterrows()):
            values = [frame.candidate_id.iloc[i], self.ranking.metrics.iloc[i].preference_score, row.get("pareto", "未提供"), *[row[o.column] for o in self._objectives]]
            for col, value in enumerate(values):
                text = f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
                item = _readonly_item(text)
                item.setToolTip(str(value))
                self.recommendations.setItem(i, col, item)
        self._sync = False

    def _recommendation_selection(self):
        if self._sync or self.ranking is None:
            return
        rows = sorted({i.row() for i in self.recommendations.selectedIndexes()})
        ids = self.ranking.frame.iloc[rows].candidate_id.tolist()
        self.set_selection(ids)
        self.selectionChanged.emit(ids)

    def _plot_selection(self, ids):
        if self._sync:
            return
        self.set_selection(ids)
        self.selectionChanged.emit(self._selected_ids[:])

    def set_selection(self, ids):
        if "candidate_id" not in self._frame:
            return
        # Use the exact original ID values and types, never model-created values.
        from .preference_engine import _id_key
        requested = {_id_key(v) for v in ids}
        self._selected_ids = [v for v in self._frame.candidate_id.tolist() if _id_key(v) in requested]
        self._sync = True
        self.interactive_plot.set_selection(self._selected_ids)
        self.recommendations.clearSelection()
        if self.ranking is not None:
            for row, value in enumerate(self.ranking.frame.candidate_id.tolist()):
                if _id_key(value) in requested:
                    index = self.recommendations.model().index(row, 0)
                    self.recommendations.selectionModel().select(index, C.QItemSelectionModel.SelectionFlag.Select | C.QItemSelectionModel.SelectionFlag.Rows)
        self._sync = False
        self._draw_comparison()

    def selected_ids(self):
        return self._selected_ids[:]

    def _draw_comparison(self):
        figure = self.comparison_figure
        figure.clear()
        if not self._selected_ids or self._applied is None or self._reference is None:
            figure.text(.5, .55, "Select recommended candidates to compare trade-offs", ha="center", color="#7b8996", fontsize=9)
            self.comparison_canvas.draw_idle()
            return
        from .preference_engine import _id_key
        selected = {_id_key(v) for v in self._selected_ids[:4]}
        subset = self._frame.loc[[_id_key(v) in selected for v in self._frame.candidate_id]]
        ax = figure.subplots()
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        colors = ["#315f87", "#0b9b91", "#d99a38", "#9a79af"]
        count, x = max(1, len(subset)), np.arange(len(self._objectives))
        width = .72 / count
        for i, (_, row) in enumerate(subset.iterrows()):
            losses = []
            for spec in self._objectives:
                bounds = self._reference.bounds[spec.column][self._applied["normalization"]]
                value = row[spec.column]
                if bounds is None or pd.isna(value) or not np.isfinite(value):
                    losses.append(np.nan)
                elif bounds[1] == bounds[0]:
                    losses.append(0.)
                else:
                    fraction = float(np.clip((np.longdouble(value)-bounds[0]) / (np.longdouble(bounds[1])-bounds[0]), 0, 1))
                    losses.append(1-fraction if spec.direction == "max" else fraction)
            ax.bar(x + (i - (count-1)/2)*width, losses, width=width, color=colors[i % 4], label=str(row.candidate_id), zorder=3)
        weights = {w["column"]: w["weight"] for w in self._applied["weights"]}
        total = sum(weights.values())
        ax.set_xticks(x, [f"{o.column.replace('predicted::', '')}\n{'max' if o.direction == 'max' else 'min'} · w={weights[o.column]/total:.0%}" for o in self._objectives], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized loss ↓", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(title="Candidate ID" + (" (first 4 selected)" if len(self._selected_ids) > 4 else ""), fontsize=7, title_fontsize=7, frameon=False, loc="upper right")
        self.comparison_canvas.draw_idle()

    def shutdown(self):
        self.cancel_request(quiet=True)
        self.key_edit.clear()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
