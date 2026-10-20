"""Opt-in Qt dialog: inspect exact outbound JSON, send, review, then apply locally."""
from __future__ import annotations

import json
import threading

import pandas as pd
from PySide6 import QtCore as C, QtWidgets as W

from .llm_filter import (RequestSettings, PlanError, RequestError, apply_plan, build_request_preview,
                         provider_host, request_body, request_plan)


class _ResultRelay(C.QObject):
    completed = C.Signal(int, object, object)


class NaturalLanguageFilterDialog(W.QDialog):
    """The dialog never sends until Send is clicked; Apply is a separate action.

    After Accepted, `result_frame` is a subset of the supplied frame, and `plan`
    is the reviewed normalized plan. The API key is only held in this dialog and
    one transient request thread, never written to a project or log.
    """
    planApplied = C.Signal(object)

    def __init__(self, frame: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自然语言筛选 / DeepSeek")
        self.resize(780, 720)
        self._frame = frame.copy()
        self.result_frame: pd.DataFrame | None = None
        self.plan: dict | None = None
        self._preview: dict | None = None
        self._request_id = 0
        self._busy = False
        self._relay = _ResultRelay(self)
        self._relay.completed.connect(self._request_done, C.Qt.ConnectionType.QueuedConnection)

        layout = W.QVBoxLayout(self)
        intro = W.QLabel("可选在线功能。只发送你的问题、可筛选列名/类型及数值列的汇总范围；不发送候选行或 ID。预测并非物理验证。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(W.QLabel("筛选请求"))
        self.query_edit = W.QPlainTextEdit()
        self.query_edit.setPlaceholderText("例如：筛选预测采光指标大于 60 的方案，按预测能耗从低到高排序，最多 20 个。")
        self.query_edit.setMaximumHeight(90)
        layout.addWidget(self.query_edit)

        form = W.QFormLayout()
        self.model_edit = W.QLineEdit("deepseek-flash")
        self.model_edit.setToolTip("可填写 DeepSeek 后续支持的兼容模型名。服务地址固定为官方 DeepSeek。")
        self.key_edit = W.QLineEdit()
        self.key_edit.setEchoMode(W.QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("仅本次会话使用，不保存")
        form.addRow("模型", self.model_edit)
        form.addRow("API Key", self.key_edit)
        layout.addLayout(form)
        cost_notice = W.QLabel("发送前核对：目标主机 api.deepseek.com。该服务可能产生 API 费用；请查看你的 DeepSeek 账户计费。Authorization 密钥不会显示在下方。")
        cost_notice.setWordWrap(True)
        layout.addWidget(cost_notice)
        layout.addWidget(W.QLabel("实际发送的 JSON 请求体（预览与发送使用同一内容）"))
        self.preview_edit = W.QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setPlaceholderText("点击“生成发送预览”，检查即将传出的全部请求体。")
        layout.addWidget(self.preview_edit, 2)
        layout.addWidget(W.QLabel("模型建议（仅供审核，点击“应用筛选”才会在本地执行）"))
        self.plan_edit = W.QPlainTextEdit()
        self.plan_edit.setReadOnly(True)
        layout.addWidget(self.plan_edit, 1)
        self.status_label = W.QLabel("尚未联网。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = W.QHBoxLayout()
        self.preview_button = W.QPushButton("生成发送预览")
        self.send_button = W.QPushButton("发送给 DeepSeek")
        self.apply_button = W.QPushButton("应用筛选")
        self.close_button = W.QPushButton("关闭")
        self.send_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        for button in (self.preview_button, self.send_button, self.apply_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.preview_button.clicked.connect(self._make_preview)
        self.send_button.clicked.connect(self._send)
        self.apply_button.clicked.connect(self._apply)
        self.close_button.clicked.connect(self.reject)
        self.query_edit.textChanged.connect(self._invalidate_preview)
        self.model_edit.textChanged.connect(self._invalidate_preview)

    @C.Slot()
    def _invalidate_preview(self):
        self._preview = None
        self.plan = None
        self.preview_edit.clear()
        self.plan_edit.clear()
        self.send_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        if not self._busy:
            self.status_label.setText("请求或模型已改变。请重新生成发送预览。")

    @C.Slot()
    def _make_preview(self):
        try:
            settings = RequestSettings(model=self.model_edit.text().strip())
            preview = build_request_preview(self._frame, self.query_edit.toPlainText())
            body = request_body(preview, settings)
            host = provider_host(settings)
            # This is byte-for-byte the UTF-8 JSON body created by request_plan.
            body_text = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (PlanError, ValueError) as exc:
            self.status_label.setText(str(exc))
            return
        self._preview = preview
        self.preview_edit.setPlainText(body_text)
        self.status_label.setText(f"尚未联网。目标主机：{host}。请检查上方完整请求体及可能的 API 费用。")
        self.send_button.setEnabled(True)
        self.apply_button.setEnabled(False)

    @C.Slot()
    def _send(self):
        if self._preview is None or self._busy:
            return
        key = self.key_edit.text()
        if not key.strip():
            self.status_label.setText("请先输入 API Key。密钥不会保存。")
            return
        settings = RequestSettings(model=self.model_edit.text().strip())
        preview = self._preview
        self._request_id += 1
        job_id = self._request_id
        self._busy = True
        self.plan = None
        self.plan_edit.clear()
        self.send_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.query_edit.setEnabled(False)
        self.model_edit.setEnabled(False)
        self.status_label.setText("正在等待 DeepSeek 响应。尚未应用任何筛选。")
        relay = self._relay

        def run():
            try:
                outcome = request_plan(settings, key, preview)
                try:
                    relay.completed.emit(job_id, outcome, None)
                except RuntimeError:
                    pass  # Dialog was destroyed while the bounded request completed.
            except Exception as exc:
                # No raw third-party error/body, trace, or credentials in the UI.
                message = str(exc) if isinstance(exc, PlanError) else (
                    str(exc) if isinstance(exc, RequestError) else "请求失败或响应无效。请检查网络、密钥和模型。")
                try:
                    relay.completed.emit(job_id, None, message)
                except RuntimeError:
                    pass  # Dialog was destroyed while the bounded request completed.

        threading.Thread(target=run, daemon=True).start()

    @C.Slot(int, object, object)
    def _request_done(self, job_id: int, outcome: object, error: object):
        if job_id != self._request_id or not self.isVisible():
            return
        self._busy = False
        self.preview_button.setEnabled(True)
        self.query_edit.setEnabled(True)
        self.model_edit.setEnabled(True)
        self.send_button.setEnabled(self._preview is not None)
        if error is not None:
            self.status_label.setText(str(error))
            return
        self.plan = outcome
        self.plan_edit.setPlainText(json.dumps(outcome, ensure_ascii=False, indent=2))
        if outcome["needs_clarification"]:
            self.status_label.setText("模型请求澄清。请修改问题并重新生成预览；不会执行筛选。")
            return
        self.apply_button.setEnabled(True)
        self.status_label.setText("请检查建议的全部条件、排序和数量限制，再点击“应用筛选”。")

    @C.Slot()
    def _apply(self):
        if self.plan is None or self.plan["needs_clarification"] or self._busy:
            return
        try:
            self.result_frame = apply_plan(self._frame, self.plan)
        except PlanError as exc:
            self.status_label.setText(str(exc))
            return
        self.key_edit.clear()
        self.planApplied.emit(self.result_frame)
        self.accept()

    def reject(self):
        self._request_id += 1  # Ignore a response that arrives after Close.
        self.key_edit.clear()
        super().reject()
