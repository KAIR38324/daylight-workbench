"""Versioned ZIP project storage: JSON/CSV plus PyTorch tensor-only weights.

Never extracts archives, never loads pickle objects; checks exact member names,
member sizes, hashes, schema, numeric values, model dimensions and split leakage.
Hashes detect corruption, not the authenticity of untrusted authors.
"""
from __future__ import annotations
from dataclasses import asdict
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .core import (DataConfig, TrainConfig, TrainedModel, Scale, PreparedData,
                   Variable, OptimizeConfig, OptimizationResult, make_model,
                   prepare_data, validate_splits, frame_fingerprint, prediction_column)

SCHEMA_VERSION = 1
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
REQUIRED = {'manifest.json', 'data.csv', 'metadata.json', 'weights.pt'}
OPTIONAL = {'candidates.csv', 'optimization.json'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _csv(frame):
    return frame.to_csv(index=False, float_format='%.17g').encode('utf-8')


def save_project(path: str | Path, trained: TrainedModel, optimization: OptimizationResult | None = None):
    import torch
    path = Path(path)
    if path.suffix.lower() != '.dwb':
        raise ValueError('项目文件必须使用 .dwb 扩展名。')
    validate_splits(trained.data)
    weights = BytesIO()
    torch.save(trained.model.state_dict(), weights)
    metadata = {'data_config': asdict(trained.data.config), 'training_config': asdict(trained.config),
                'fingerprint': trained.data.fingerprint, 'splits': trained.data.splits,
                'report': trained.data.report, 'x_scale': trained.x_scale.to_dict(),
                'y_scale': trained.y_scale.to_dict(), 'history': trained.history,
                'metrics': trained.metrics, 'best_epoch': trained.best_epoch}
    payloads = {'metadata.json': _json(metadata), 'data.csv': _csv(trained.data.frame), 'weights.pt': weights.getvalue()}
    if optimization is not None:
        optimization.config.validate(trained)
        payloads['candidates.csv'] = _csv(optimization.candidates)
        payloads['optimization.json'] = _json({'config': asdict(optimization.config), 'evaluations': optimization.evaluations})
    payloads['manifest.json'] = _json({'schema_version': SCHEMA_VERSION, 'app_version': '0.1.0',
                                     'sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()}})
    if sum(map(len, payloads.values())) > MAX_ARCHIVE_BYTES:
        raise ValueError('项目超过P0的128MB大小限制。')
    fd, temp = tempfile.mkstemp(prefix='.dwb-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, blob in payloads.items():
                    archive.writestr(name, blob)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_project(path: str | Path) -> tuple[TrainedModel, OptimizationResult | None]:
    import torch
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('项目文件超过128MB。')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        if len(names) != len(entries) or not REQUIRED.issubset(names) or not names.issubset(REQUIRED | OPTIONAL):
            raise ValueError('项目成员不合法（拒绝额外路径、重复文件或缺失文件）。')
        if bool('candidates.csv' in names) != bool('optimization.json' in names):
            raise ValueError('优化数据不完整。')
        if sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES or any(e.flag_bits & 1 for e in entries):
            raise ValueError('项目解压大小过大或使用了加密。')
        blobs = {name: archive.read(name) for name in names}
    manifest = json.loads(blobs.pop('manifest.json'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('不支持该项目版本。')
    hashes = manifest.get('sha256', {})
    if set(hashes) != set(blobs) or any(hashlib.sha256(b).hexdigest() != hashes[n] for n, b in blobs.items()):
        raise ValueError('项目完整性校验失败。')
    meta = json.loads(blobs['metadata.json'])
    dc, tc = DataConfig(**meta['data_config']), TrainConfig(**meta['training_config'])
    dc.validate()
    tc.validate(len(dc.inputs), len(dc.targets))
    raw = pd.read_csv(BytesIO(blobs['data.csv']), float_precision='round_trip', dtype={dc.group: str} if dc.group else None)
    data = prepare_data(raw, dc)
    if data.fingerprint != meta['fingerprint']:
        raise ValueError('数据指纹不匹配。')
    splits = meta['splits']
    if set(splits) != {'train', 'validation', 'test'} or any(type(i) is not int for rows in splits.values() for i in rows):
        raise ValueError('划分格式无效。')
    data.splits = splits
    validate_splits(data)
    data.report = meta['report']
    xs = Scale.from_dict(meta['x_scale'], len(dc.inputs))
    ys = Scale.from_dict(meta['y_scale'], len(dc.targets))
    # A loaded scaler must really correspond to this project's TRAINING subset.
    for stored, expected in ((xs, Scale.fit(data.X[splits['train']])), (ys, Scale.fit(data.Y[splits['train']]))):
        if not np.allclose(stored.mean, expected.mean, rtol=1e-12, atol=1e-12) or not np.allclose(stored.scale, expected.scale, rtol=1e-12, atol=1e-12):
            raise ValueError('缩放器不是由该训练子集计算的。')
    model = make_model(len(dc.inputs), len(dc.targets), tc)
    state = torch.load(BytesIO(blobs['weights.pt']), map_location='cpu', weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('模型权重键与架构不匹配。')
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('模型权重维度、类型或数值无效。')
    model.load_state_dict(state, strict=True)
    history = meta['history']
    if not isinstance(history, list) or not 1 <= len(history) <= tc.epochs or not 1 <= meta['best_epoch'] <= len(history):
        raise ValueError('训练历史无效。')
    for entry in history:
        if set(entry) != {'epoch', 'train_loss', 'validation_loss'} or not np.isfinite(list(entry.values())).all():
            raise ValueError('训练历史含无效值。')
    trained = TrainedModel(data, tc, model, xs, ys, history, meta['metrics'], meta['best_epoch'])
    # Regenerate metrics instead of trusting supplied report values.
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    for name, idx in splits.items():
        actual, predicted = data.Y[idx], trained.predict(data.X[idx])
        trained.metrics[name] = {target: {'R2': float(r2_score(actual[:, j], predicted[:, j], force_finite=False)) if np.ptp(actual[:, j]) > 0 else None,
                                        'MAE': float(mean_absolute_error(actual[:, j], predicted[:, j])),
                                        'RMSE': float(np.sqrt(mean_squared_error(actual[:, j], predicted[:, j])))}
                                 for j, target in enumerate(dc.targets)}
    optimization = None
    if 'candidates.csv' in blobs:
        od = json.loads(blobs['optimization.json'])
        cfg = od['config']
        cfg['variables'] = [Variable(**v) for v in cfg['variables']]
        oc = OptimizeConfig(**cfg)
        oc.validate(trained)
        frame = pd.read_csv(BytesIO(blobs['candidates.csv']), float_precision='round_trip')
        required = ['candidate_id'] + dc.inputs + [prediction_column(t) for t in dc.targets] + ['generation', 'pareto']
        if list(frame.columns) != required or not len(frame) or len(frame) > 20000:
            raise ValueError('候选表结构无效。')
        numeric = frame.drop(columns='pareto').to_numpy(float)
        if not np.isfinite(numeric).all() or not frame.candidate_id.is_unique or frame.pareto.dtype != bool:
            raise ValueError('候选表包含无效值或重复编号。')
        for v in oc.variables:
            values = frame[v.name].to_numpy()
            if (values < v.lower - 1e-10).any() or (values > v.upper + 1e-10).any() or (v.kind == 'integer' and not np.equal(values, np.rint(values)).all()):
                raise ValueError('候选输入不满足变量定义。')
        y = trained.predict(frame[dc.inputs].to_numpy())
        if not np.allclose(y, frame[[prediction_column(t) for t in dc.targets]].to_numpy(), rtol=1e-5, atol=1e-7):
            raise ValueError('候选预测与模型不一致。')
        optimization = OptimizationResult(oc, frame, int(od['evaluations']))
    return trained, optimization
