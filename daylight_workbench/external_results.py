"""Explicit external-table analysis, independent of any surrogate model."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PySide6 import QtWidgets as W
from .core import check_cancel, frame_fingerprint


@dataclass
class ExternalResults:
    frame: pd.DataFrame
    inputs: list[str]
    targets: list[str]
    directions: dict[str, str]
    source: str
    fingerprint: str


def prepare_external_results(raw, inputs, targets, directions, source='external', cancel=None):
    if not targets or set(directions) != set(targets) or any(v not in ('min', 'max') for v in directions.values()):
        raise ValueError('至少选择一个目标，并明确每个目标的最小/最大方向。')
    columns = inputs + targets
    if len(columns) != len(set(columns)) or not set(columns) <= set(raw.columns) or any(c in {'candidate_id','pareto','generation'} for c in columns):
        raise ValueError('映射列重复、不存在或占用保留结果名称。')
    if not 1 <= len(raw) <= 20000:
        raise ValueError('外部结果分析支持1-20,000行。')
    frame = raw[columns].apply(pd.to_numeric, errors='coerce').astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('所选输入/目标列含缺失、非数值或无穷值。')
    fingerprint = frame_fingerprint(frame)
    frame = frame.reset_index(drop=True)
    # Stable source-row IDs are metadata, never invented design variables.
    frame.insert(0, 'candidate_id', np.arange(1, len(frame) + 1))
    objectives = frame[targets].to_numpy() * np.array([1 if directions[t] == 'min' else -1 for t in targets])
    front = []
    for i, point in enumerate(objectives):
        if i % 32 == 0:
            check_cancel(cancel)
        if front:
            current = objectives[front]
            if np.any(np.all(current <= point, axis=1) & np.any(current < point, axis=1)):
                continue
            removed = np.all(point <= current, axis=1) & np.any(point < current, axis=1)
            front = [idx for idx, remove in zip(front, removed) if not remove]
        front.append(i)
    frame['pareto'] = False
    frame.loc[front, 'pareto'] = True
    return ExternalResults(frame, list(inputs), list(targets), dict(directions), source, fingerprint)


def export_external(path, frame, context, filter_plan=None):
    path = Path(path)
    safe = frame.copy()
    safe.columns = [("'" + c) if str(c).startswith(('=','+','-','@','\t','\r')) else c for c in safe.columns]
    safe.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')
    provenance = {'schema_version': 2, 'kind': 'external_results', 'source': context.source,
                  'data_sha256': context.fingerprint, 'inputs': context.inputs, 'targets': context.targets,
                  'directions': context.directions, 'exported_rows': len(frame),
                  'notice': '用户导入的外部数值；平台不自动认证为仿真、实测或代理预测。未补造缺失自变量。',
                  'filter_plan': filter_plan, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(path.suffix + '.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class ExternalMappingDialog(W.QDialog):
    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle('外部结果列映射（允许仅目标列）')
        self.resize(780, 520)
        layout = W.QVBoxLayout(self)
        label = W.QLabel('显式选择输入/目标与目标方向。不要求自变量，不会虚构设计输入。原始目标值保持原名，不自动冠以“预测”标签。')
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = W.QTableWidget(len(frame.columns), 3)
        self.table.setHorizontalHeaderLabels(['列名','角色','目标方向'])
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.Stretch)
        for i, name in enumerate(frame.columns):
            self.table.setItem(i, 0, W.QTableWidgetItem(name))
            self.table.item(i, 0).setFlags(self.table.item(i, 0).flags() & ~__import__('PySide6').QtCore.Qt.ItemFlag.ItemIsEditable)
            role = W.QComboBox()
            for title, value in [('忽略','ignore'),('输入（可选）','input'),('目标','target')]:
                role.addItem(title,value)
            self.table.setCellWidget(i,1,role)
            direction = W.QComboBox()
            direction.addItem('请选择',None)
            direction.addItem('最小化','min')
            direction.addItem('最大化','max')
            self.table.setCellWidget(i,2,direction)
        layout.addWidget(self.table)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.validate_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.mapping = None

    def validate_accept(self):
        inputs, targets, directions = [], [], {}
        for i in range(self.table.rowCount()):
            name, role = self.table.item(i,0).text(), self.table.cellWidget(i,1).currentData()
            if role == 'input':
                inputs.append(name)
            elif role == 'target':
                targets.append(name)
                directions[name] = self.table.cellWidget(i,2).currentData()
        if not targets or any(v not in ('min','max') for v in directions.values()):
            W.QMessageBox.warning(self,'映射未完成','请至少选择一个目标，并指定每个目标的方向。')
            return
        self.mapping = inputs, targets, directions
        self.accept()
