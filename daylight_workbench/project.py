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


def _save_v1(path: str | Path, trained: TrainedModel, optimization: OptimizationResult | None = None):
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


def _load_v1(path: str | Path) -> tuple[TrainedModel, OptimizationResult | None]:
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
                                        'MSE': float(mean_squared_error(actual[:, j], predicted[:, j])),
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


def _write_archive(path, payloads, version):
    path = Path(path)
    if path.suffix.lower() != '.dwb':
        raise ValueError('项目文件必须使用 .dwb 扩展名。')
    payloads['manifest.json'] = _json({'schema_version': version, 'app_version': '0.2.0',
                                     'sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()}})
    if sum(map(len, payloads.values())) > MAX_ARCHIVE_BYTES:
        raise ValueError('项目超过128MB安全限制，请减小模型/候选规模。')
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


def save_project(path, trained, optimization=None):
    from .modeling import FittedSurrogate
    if not isinstance(trained, FittedSurrogate):
        return _save_v1(path, trained, optimization)
    validate_splits(trained.data)
    trained.spec.validate(len(trained.data.config.inputs), len(trained.data.config.targets))
    metadata = {'data_config': asdict(trained.data.config), 'spec': asdict(trained.spec),
                'fingerprint': trained.data.fingerprint, 'splits': trained.data.splits, 'report': trained.data.report,
                'fit_indices': trained.fit_indices, 'x_scale': trained.x_scale.to_dict(), 'y_scale': trained.y_scale.to_dict(),
                'history': trained.history, 'metrics': trained.metrics, 'best_epoch': trained.best_epoch,
                'search_metadata': trained.search_metadata, 'support_bounds': trained.support_bounds}
    payloads = {'data.csv': _csv(trained.data.frame), 'metadata.json': _json(metadata)}
    if trained.family == 'ann':
        import torch
        weights = BytesIO()
        torch.save(trained.backend.state_dict(), weights)
        payloads['weights.pt'] = weights.getvalue()
    elif trained.family == 'rf':
        arrays = {'tree_count': np.asarray([len(trained.backend.trees)], dtype=np.int64)}
        for i, tree in enumerate(trained.backend.trees):
            arrays.update({f't{i}_{name}': value for name, value in tree.items()})
        blob = BytesIO()
        np.savez_compressed(blob, **arrays)
        payloads['forest.npz'] = blob.getvalue()
    else:
        for j, booster in enumerate(trained.backend):
            payloads[f'booster_{j}.json'] = bytes(booster.save_raw(raw_format='json'))
    if optimization is not None:
        optimization.config.validate(trained)
        payloads['candidates.csv'] = _csv(optimization.candidates)
        payloads['optimization.json'] = _json({'config': asdict(optimization.config), 'evaluations': optimization.evaluations})
    _write_archive(path, payloads, 2)


def _validate_plan(data, metadata, fit_indices):
    from .core import _leakage_components
    plan = metadata.get('plan')
    if not isinstance(plan, dict):
        raise ValueError('版本2项目缺少CV/测试锁定清单。')
    components = _leakage_components(data.frame, data.config)
    if plan.get('components') != components.tolist() or plan.get('test') != data.splits['test']:
        raise ValueError('项目分组或锁定测试清单不一致。')
    development = sorted(data.splits['train'] + data.splits['validation'])
    if plan.get('development') != development or sorted(fit_indices) != development:
        raise ValueError('最终模型必须仅在完整开发集重拟合。')
    def check_partition(a, b, expected):
        if any(type(i) is not int for i in a + b) or sorted(a + b) != sorted(expected) or len(set(a + b)) != len(expected):
            raise ValueError('CV/内部早停索引缺失、越界或重叠。')
        if set(components[a]) & set(components[b]):
            raise ValueError('CV/内部早停存在重复输入或组泄漏。')
    folds = plan.get('folds', [])
    if not 1 <= len(folds) <= 10:
        raise ValueError('CV折数无效。')
    for fold in folds:
        check_partition(fold['train'], fold['validation'], development)
        check_partition(fold['inner_train'], fold['inner_validation'], fold['train'])
    if plan.get('mode') == 'cv' and sorted(sum([f['validation'] for f in folds], [])) != development:
        raise ValueError('外层CV验证折必须恰好覆盖开发集一次。')


def _load_v2(path):
    from .modeling import ModelSpec, FittedSurrogate, NumericForest, metric_report
    import re
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = {e.filename for e in entries}
        fixed = {'manifest.json', 'data.csv', 'metadata.json', 'weights.pt', 'forest.npz', 'candidates.csv', 'optimization.json'}
        if len(entries) != len(names) or not {'manifest.json', 'data.csv', 'metadata.json'} <= names or any(n not in fixed and not re.fullmatch(r'booster_[0-9]+\.json', n) for n in names):
            raise ValueError('版本2项目成员名称不合法。')
        if sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES or any(e.flag_bits & 1 for e in entries):
            raise ValueError('版本2项目超大或加密。')
        blobs = {n: archive.read(n) for n in names}
    manifest = json.loads(blobs.pop('manifest.json'))
    hashes = manifest.get('sha256', {})
    if manifest.get('schema_version') != 2 or set(hashes) != set(blobs) or any(hashlib.sha256(v).hexdigest() != hashes[n] for n, v in blobs.items()):
        raise ValueError('版本2项目完整性校验失败。')
    meta = json.loads(blobs['metadata.json'])
    dc, spec = DataConfig(**meta['data_config']), ModelSpec.from_dict(meta['spec'])
    dc.validate()
    spec.validate(len(dc.inputs), len(dc.targets))
    raw = pd.read_csv(BytesIO(blobs['data.csv']), float_precision='round_trip', dtype={dc.group: str} if dc.group else None)
    data = prepare_data(raw, dc)
    if data.fingerprint != meta['fingerprint']:
        raise ValueError('数据指纹不匹配。')
    data.splits, data.report = meta['splits'], meta['report']
    validate_splits(data)
    fit_indices = meta['fit_indices']
    _validate_plan(data, meta['search_metadata'], fit_indices)
    xs = Scale.from_dict(meta['x_scale'], len(dc.inputs))
    ys = Scale.from_dict(meta['y_scale'], len(dc.targets))
    for stored, expected in ((xs, Scale.fit(data.X[fit_indices])), (ys, Scale.fit(data.Y[fit_indices]))):
        if not np.allclose(stored.mean, expected.mean, rtol=1e-12, atol=1e-12) or not np.allclose(stored.scale, expected.scale, rtol=1e-12, atol=1e-12):
            raise ValueError('缩放器与实际模型拟合子集不一致。')
    artifact_names = set(blobs) - {'metadata.json', 'data.csv', 'candidates.csv', 'optimization.json'}
    if spec.family == 'ann':
        import torch
        if artifact_names != {'weights.pt'}:
            raise ValueError('ANN模型文件不匹配。')
        backend = make_model(len(dc.inputs), len(dc.targets), spec.train_config())
        state = torch.load(BytesIO(blobs['weights.pt']), map_location='cpu', weights_only=True)
        expected = backend.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected) or any(not isinstance(v, torch.Tensor) or v.shape != expected[n].shape or v.dtype != torch.float32 or not torch.isfinite(v).all() for n, v in state.items()):
            raise ValueError('ANN权重形状或数值无效。')
        backend.load_state_dict(state, strict=True)
    elif spec.family == 'rf':
        if artifact_names != {'forest.npz'}:
            raise ValueError('RF模型文件不匹配。')
        with zipfile.ZipFile(BytesIO(blobs['forest.npz'])) as nested:
            infos = nested.infolist()
            if sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES or len(infos) > 2501 or len(set(i.filename for i in infos)) != len(infos):
                raise ValueError('森林数值归档过大或重复。')
        with np.load(BytesIO(blobs['forest.npz']), allow_pickle=False, max_header_size=4096) as arr:
            count = arr['tree_count']
            if count.shape != (1,) or count.dtype.kind not in 'iu' or not 1 <= int(count[0]) <= 500:
                raise ValueError('森林树数无效。')
            expected = {'tree_count'} | {f't{i}_{k}' for i in range(int(count[0])) for k in ('left','right','feature','threshold','value')}
            if set(arr.files) != expected:
                raise ValueError('森林数值成员无效。')
            trees = [{k: arr[f't{i}_{k}'].copy() for k in ('left','right','feature','threshold','value')} for i in range(int(count[0]))]
        backend = NumericForest(trees, len(dc.inputs), len(dc.targets))
    else:
        import xgboost as xgb
        if artifact_names != {f'booster_{i}.json' for i in range(len(dc.targets))}:
            raise ValueError('XGBoost标量模型数量与目标数不一致。')
        backend = []
        for i in range(len(dc.targets)):
            booster = xgb.Booster(params={'nthread': 2, 'device': 'cpu'})
            booster.load_model(bytearray(blobs[f'booster_{i}.json']))
            if booster.num_features() != len(dc.inputs):
                raise ValueError('XGBoost输入维度不匹配。')
            try:
                configuration = json.loads(booster.save_config())['learner']
                shape = configuration['learner_model_param']
                scalar_regression = (int(shape['num_target']) == 1 and int(shape['num_class']) == 0
                                     and configuration['objective']['name'] == 'reg:squarederror')
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                scalar_regression = False
            if not scalar_regression:
                raise ValueError('XGBoost模型必须为每目标一个标量回归输出。')
            backend.append(booster)
    trained = FittedSurrogate(data, spec, backend, xs, ys, fit_indices, meta.get('history', []),
                               best_epoch=meta.get('best_epoch', 0), search_metadata=meta['search_metadata'])
    trained.metrics = {'train': metric_report(data.Y[fit_indices], trained.predict(data.X[fit_indices]), dc.targets),
                       'test': metric_report(data.Y[data.splits['test']], trained.predict(data.X[data.splits['test']]), dc.targets)}
    optimization = None
    if ('candidates.csv' in blobs) != ('optimization.json' in blobs):
        raise ValueError('候选与优化元数据不完整。')
    if 'candidates.csv' in blobs:
        value = json.loads(blobs['optimization.json'])
        cfg = value['config']
        cfg['variables'] = [Variable(**v) for v in cfg['variables']]
        oc = OptimizeConfig(**cfg)
        oc.validate(trained)
        frame = pd.read_csv(BytesIO(blobs['candidates.csv']), float_precision='round_trip')
        expected = ['candidate_id'] + dc.inputs + [prediction_column(t) for t in dc.targets] + ['generation','pareto']
        if list(frame.columns) != expected or not 1 <= len(frame) <= 20000 or frame.pareto.dtype != bool or not frame.candidate_id.is_unique or not np.isfinite(frame.drop(columns='pareto').to_numpy(float)).all():
            raise ValueError('优化候选结构或数值无效。')
        for v in oc.variables:
            values = frame[v.name].to_numpy()
            if np.any(values < v.lower - 1e-10) or np.any(values > v.upper + 1e-10) or (v.kind == 'integer' and not np.equal(values, np.rint(values)).all()):
                raise ValueError('候选不满足变量域。')
        if not np.allclose(trained.predict(frame[dc.inputs].to_numpy()), frame[[prediction_column(t) for t in dc.targets]].to_numpy(), rtol=1e-5, atol=1e-7):
            raise ValueError('候选预测与模型不一致。')
        optimization = OptimizationResult(oc, frame, int(value['evaluations']))
    return trained, optimization


def load_project(path):
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('项目超过128MB。')
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('manifest.json')
        if info.file_size > 65536:
            raise ValueError('项目清单过大。')
        version = json.loads(archive.read(info)).get('schema_version')
    if version == 1:
        return _load_v1(path)
    if version == 2:
        return _load_v2(path)
    raise ValueError('不支持该项目版本。')
