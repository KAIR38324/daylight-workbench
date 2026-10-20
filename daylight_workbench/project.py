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
MAX_MANIFEST_BYTES = 65536
REQUIRED = {'manifest.json', 'data.csv', 'metadata.json', 'weights.pt'}
OPTIONAL = {'candidates.csv', 'optimization.json'}


def _read_json(blob):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('项目JSON字段重复。')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('项目JSON包含非有限数值。')
    def finite_float(value):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('项目JSON数值超出有限范围。')
        return number
    return json.loads(blob, object_pairs_hook=unique_pairs, parse_constant=reject_constant, parse_float=finite_float)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _csv(frame):
    return frame.to_csv(index=False, float_format='%.17g').encode('utf-8')


def _check_archive_storage_budget(payloads):
    """Bound aggregate nested numeric storage, not only compressed ZIP bytes.

    A separate per-forest limit is insufficient for a composite containing
    many individually small compressed forests. This preflight precedes every
    native model loader and keeps their combined tensor/array storage bounded.
    """
    expanded = 0
    for name, blob in payloads.items():
        size = len(blob)
        if name.endswith(('.npz', '.pt')):
            try:
                with zipfile.ZipFile(BytesIO(blob)) as archive:
                    entries = archive.infolist()
                    if (len(entries) > 2501 or len({entry.filename for entry in entries}) != len(entries)
                            or any(entry.flag_bits & 1 for entry in entries)):
                        raise ValueError('嵌套数值归档成员重复、过多或加密。')
                    size = max(size, sum(entry.file_size for entry in entries))
            except zipfile.BadZipFile as exc:
                raise ValueError('嵌套数值归档格式无效。') from exc
        expanded += size
        if expanded > MAX_ARCHIVE_BYTES:
            raise ValueError('所有模型的嵌套数值存储展开总量超过128MB安全预算。')


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
    payloads['manifest.json'] = _json({'schema_version': SCHEMA_VERSION, 'app_version': '0.3.0',
                                     'sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()}})
    _check_archive_storage_budget(payloads)
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
    _check_archive_storage_budget(blobs)
    manifest = _read_json(blobs.pop('manifest.json'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('不支持该项目版本。')
    hashes = manifest.get('sha256', {})
    if set(hashes) != set(blobs) or any(hashlib.sha256(b).hexdigest() != hashes[n] for n, b in blobs.items()):
        raise ValueError('项目完整性校验失败。')
    meta = _read_json(blobs['metadata.json'])
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
    _validate_tensor_archive(blobs['weights.pt'])
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
        od = _read_json(blobs['optimization.json'])
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
    payloads['manifest.json'] = _json({'schema_version': version, 'app_version': '0.3.0',
                                     'sha256': {name: hashlib.sha256(blob).hexdigest() for name, blob in payloads.items()}})
    if len(payloads['manifest.json']) > MAX_MANIFEST_BYTES:
        raise ValueError('项目清单超过64KB安全限制；请减少独立目标或模型文件数量。')
    _check_archive_storage_budget(payloads)
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


def _encode_model_artifacts(trained):
    payloads = {}
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
    return payloads


def save_project(path, trained, optimization=None):
    from .modeling import FittedSurrogate, CompositeSurrogate
    if isinstance(trained, CompositeSurrogate):
        return _save_v3(path, trained, optimization)
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
    payloads.update(_encode_model_artifacts(trained))
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
    if (not isinstance(fit_indices, list) or any(type(i) is not int for i in fit_indices)
            or plan.get('development') != development or fit_indices != development):
        raise ValueError('最终模型必须仅在完整开发集重拟合。')
    def check_partition(a, b, expected):
        if not isinstance(a, list) or not isinstance(b, list) or any(type(i) is not int for i in a + b) or sorted(a + b) != sorted(expected) or len(set(a + b)) != len(expected):
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


def _validate_tensor_archive(blob):
    # weights_only restricts object construction; independently bound its
    # nested storage before PyTorch can allocate/decompress any tensor data.
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if (len(entries) > 256 or len(set(names)) != len(names)
                or sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES
                or any(entry.flag_bits & 1 for entry in entries)
                or any(name.startswith(('/', '\\')) or '..' in name.split('/') or '\\' in name for name in names)):
            raise ValueError('ANN数值权重归档超大、重复或路径无效。')


def _validate_xgb_json(blob, spec, n_inputs):
    """Bound native numeric trees before invoking the native XGBoost loader."""
    try:
        learner = _read_json(blob)['learner']
        shape, gradient = learner['learner_model_param'], learner['gradient_booster']
        if (int(shape['num_target']) != 1 or int(shape['num_class']) != 0
                or learner['objective']['name'] != 'reg:squarederror'):
            raise ValueError('XGBoost模型必须为每目标一个标量回归输出。')
        if int(shape['num_feature']) != n_inputs or gradient['name'] != 'gbtree':
            raise ValueError('XGBoost输入维度或树模型类型不一致。')
        base = _read_json(shape['base_score'])
        if isinstance(base, list):
            if len(base) != 1:
                raise ValueError('XGBoost标量模型基础分数维度无效。')
            base = base[0]
        if type(base) not in (float, int) or not np.isfinite(base):
            raise ValueError('XGBoost基础分数无效。')
        trees = gradient['model']['trees']
        count = int(gradient['model']['gbtree_model_param']['num_trees'])
        if (not isinstance(trees, list) or not 1 <= count <= spec.xgb_params()['n_estimators']
                or len(trees) != count or int(gradient['model']['gbtree_model_param']['num_parallel_tree']) != 1
                or gradient['model']['tree_info'] != [0] * count
                or gradient['model']['iteration_indptr'] != list(range(count + 1))):
            raise ValueError('XGBoost树数或标量迭代结构无效。')
        total_nodes = 0
        for i, tree in enumerate(trees):
            n = int(tree['tree_param']['num_nodes'])
            total_nodes += n
            if (not 1 <= n <= 8191 or total_nodes > 1_100_000 or tree['id'] != i
                    or int(tree['tree_param']['num_feature']) != n_inputs
                    or int(tree['tree_param']['num_deleted']) != 0
                    or int(tree['tree_param']['size_leaf_vector']) != 1):
                raise ValueError('XGBoost数值树大小或输出形状无效。')
            keys = ('base_weights', 'default_left', 'left_children', 'loss_changes', 'parents',
                    'right_children', 'split_conditions', 'split_indices', 'split_type', 'sum_hessian')
            if any(not isinstance(tree[k], list) or len(tree[k]) != n for k in keys):
                raise ValueError('XGBoost数值树数组长度无效。')
            if (any(tree.get(k) for k in ('categories', 'categories_nodes', 'categories_segments', 'categories_sizes'))
                    or any(type(value) not in (int, float) or not np.isfinite(value) for key in keys for value in tree[key])
                    or any(type(value) is not int for key in ('left_children', 'right_children', 'split_indices', 'parents') for value in tree[key])
                    or any(value != 0 for value in tree['split_type'])
                    or any(value not in (0, 1) for value in tree['default_left'])):
                raise ValueError('XGBoost仅允许有限、非分类的数值树。')
            left, right, features = map(np.asarray, (tree['left_children'], tree['right_children'], tree['split_indices']))
            internal = left != -1
            ids = np.arange(n)[internal]
            if (not np.array_equal(internal, right != -1) or np.any(left[internal] <= ids)
                    or np.any(right[internal] <= ids) or np.any(left[internal] >= n)
                    or np.any(right[internal] >= n) or np.any(features[internal] < 0)
                    or np.any(features[internal] >= n_inputs)):
                raise ValueError('XGBoost树路径存在环或越界。')
            children = np.concatenate((left[internal], right[internal]))
            if sorted(children.tolist()) != list(range(1, n)):
                raise ValueError('XGBoost树节点重复或不连通。')
            depth = np.zeros(n, dtype=int)
            for node in ids:
                depth[left[node]] = depth[right[node]] = depth[node] + 1
            if depth.max() > spec.xgb_params()['max_depth']:
                raise ValueError('XGBoost树深度超过所声明参数。')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('XGBoost原生数值JSON结构无效。') from exc


def _decode_model_artifacts(blobs, spec, n_inputs, n_outputs):
    from .modeling import NumericForest
    artifact_names = set(blobs)
    if spec.family == 'ann':
        import torch
        if artifact_names != {'weights.pt'}:
            raise ValueError('ANN模型文件不匹配。')
        _validate_tensor_archive(blobs['weights.pt'])
        backend = make_model(n_inputs, n_outputs, spec.train_config())
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
        backend = NumericForest(trees, n_inputs, n_outputs)
    else:
        import xgboost as xgb
        if artifact_names != {f'booster_{i}.json' for i in range(n_outputs)}:
            raise ValueError('XGBoost标量模型数量与目标数不一致。')
        backend = []
        for i in range(n_outputs):
            _validate_xgb_json(blobs[f'booster_{i}.json'], spec, n_inputs)
            booster = xgb.Booster(params={'nthread': 2, 'device': 'cpu'})
            booster.load_model(bytearray(blobs[f'booster_{i}.json']))
            if booster.num_features() != n_inputs:
                raise ValueError('XGBoost输入维度不匹配。')
            try:
                configuration = _read_json(booster.save_config())['learner']
                shape = configuration['learner_model_param']
                scalar_regression = (int(shape['num_target']) == 1 and int(shape['num_class']) == 0
                                     and configuration['objective']['name'] == 'reg:squarederror')
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                scalar_regression = False
            if not scalar_regression:
                raise ValueError('XGBoost模型必须为每目标一个标量回归输出。')
            backend.append(booster)
    return backend


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
    _check_archive_storage_budget(blobs)
    manifest = _read_json(blobs.pop('manifest.json'))
    hashes = manifest.get('sha256', {})
    if manifest.get('schema_version') != 2 or set(hashes) != set(blobs) or any(hashlib.sha256(v).hexdigest() != hashes[n] for n, v in blobs.items()):
        raise ValueError('版本2项目完整性校验失败。')
    meta = _read_json(blobs['metadata.json'])
    dc, spec = DataConfig(**meta['data_config']), ModelSpec.from_dict(meta['spec'])
    dc.validate()
    spec.validate(len(dc.inputs), len(dc.targets))
    raw = pd.read_csv(BytesIO(blobs['data.csv']), float_precision='round_trip', dtype={dc.group: str} if dc.group else None)
    data = prepare_data(raw, dc)
    if data.fingerprint != meta['fingerprint']:
        raise ValueError('数据指纹不匹配。')
    data.splits, data.report = meta['splits'], meta['report']
    _validate_split_types(data.splits)
    validate_splits(data)
    fit_indices = meta['fit_indices']
    _validate_plan(data, meta['search_metadata'], fit_indices)
    xs = Scale.from_dict(meta['x_scale'], len(dc.inputs))
    ys = Scale.from_dict(meta['y_scale'], len(dc.targets))
    for stored, expected in ((xs, Scale.fit(data.X[fit_indices])), (ys, Scale.fit(data.Y[fit_indices]))):
        if not np.allclose(stored.mean, expected.mean, rtol=1e-12, atol=1e-12) or not np.allclose(stored.scale, expected.scale, rtol=1e-12, atol=1e-12):
            raise ValueError('缩放器与实际模型拟合子集不一致。')
    artifact_blobs = {n: b for n, b in blobs.items() if n not in {'metadata.json', 'data.csv', 'candidates.csv', 'optimization.json'}}
    backend = _decode_model_artifacts(artifact_blobs, spec, len(dc.inputs), len(dc.targets))
    trained = FittedSurrogate(data, spec, backend, xs, ys, fit_indices, meta.get('history', []),
                               best_epoch=meta.get('best_epoch', 0), search_metadata=meta['search_metadata'])
    # Old configs explicitly mean shared selection when re-used by the UI.
    from .modeling import SearchConfig
    if 'config' in trained.search_metadata:
        trained.search_metadata['config'] = asdict(SearchConfig.from_dict(trained.search_metadata['config']))
        trained.search_metadata.setdefault('target_mode', trained.search_metadata['config']['target_mode'])
    _validate_final_history(trained)
    trained.metrics = {'train': metric_report(data.Y[fit_indices], trained.predict(data.X[fit_indices]), dc.targets),
                       'test': metric_report(data.Y[data.splits['test']], trained.predict(data.X[data.splits['test']]), dc.targets)}
    return trained, _load_optimization(blobs, trained)


def _load_optimization(blobs, trained):
    dc = trained.data.config
    optimization = None
    if ('candidates.csv' in blobs) != ('optimization.json' in blobs):
        raise ValueError('候选与优化元数据不完整。')
    if 'candidates.csv' in blobs:
        value = _read_json(blobs['optimization.json'])
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
    return optimization


def _validate_split_types(splits):
    if (not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}
            or any(not isinstance(rows, list) or any(type(i) is not int for i in rows) for rows in splits.values())):
        raise ValueError('项目划分索引格式无效。')


def _validate_final_history(model):
    """Only bounded real final-refit records are accepted, never executable state."""
    def history_records(history, maximum, validation_allowed=False):
        if not isinstance(history, list) or not 1 <= len(history) <= maximum:
            raise ValueError('最终模型训练历史长度无效。')
        for i, record in enumerate(history):
            if (not isinstance(record, dict) or set(record) != {'epoch', 'train_loss', 'validation_loss'}
                    or type(record['epoch']) is not int or record['epoch'] != i + 1
                    or type(record['train_loss']) not in (int, float) or not np.isfinite(record['train_loss'])
                    or record['train_loss'] < 0
                    or (record['validation_loss'] is not None and (not validation_allowed
                        or type(record['validation_loss']) not in (int, float)
                        or not np.isfinite(record['validation_loss']) or record['validation_loss'] < 0))):
                raise ValueError('最终模型训练历史包含无效轮次或损失。')
    if type(model.best_epoch) is not int:
        raise ValueError('最终模型轮数必须是整数。')
    if model.family == 'ann':
        history_records(model.history, model.spec.train_config().epochs)
        if model.best_epoch != len(model.history):
            raise ValueError('ANN最终固定重拟合轮数与历史不一致。')
    elif model.family == 'rf':
        if model.history != [] or model.best_epoch != 0 or len(model.backend.trees) != model.spec.rf_params()['n_estimators']:
            raise ValueError('RF树数或历史不一致；RF没有伪造损失记录。')
    else:
        if model.history != []:
            raise ValueError('XGBoost历史必须按目标存储。')
        rounds = model.search_metadata.get('best_rounds')
        history = model.search_metadata.get('booster_history')
        if (not isinstance(rounds, list) or len(rounds) != len(model.data.config.targets)
                or any(type(n) is not int or not 1 <= n <= model.spec.xgb_params()['n_estimators'] for n in rounds)
                or not isinstance(history, list) or len(history) != len(rounds) or model.best_epoch != max(rounds)):
            raise ValueError('XGBoost最终迭代记录无效。')
        for i, target in enumerate(model.data.config.targets):
            if (not isinstance(history[i], dict) or set(history[i]) != {'target', 'history'}
                    or history[i]['target'] != target or model.backend[i].num_boosted_rounds() != rounds[i]):
                raise ValueError('XGBoost迭代记录与目标/模型不一致。')
            history_records(history[i]['history'], rounds[i])
            if len(history[i]['history']) != rounds[i]:
                raise ValueError('XGBoost历史长度与实际迭代数不一致。')


def _validate_independent_metadata(data, metadata, fit_indices):
    from .modeling import SearchConfig, make_search_plan
    if not isinstance(metadata, dict) or metadata.get('schema_version') != 3 or metadata.get('target_mode') != 'independent':
        raise ValueError('组合项目必须包含逐目标独立选优元数据。')
    config = SearchConfig.from_dict(metadata['config'])
    if config.target_mode != 'independent' or metadata.get('target_order') != data.config.targets:
        raise ValueError('组合项目目标顺序或训练模式不一致。')
    expected = make_search_plan(data, config)
    if metadata.get('plan') != expected or metadata.get('seed') != config.seed:
        raise ValueError('组合项目公共CV/测试锁定清单与配置不一致。')
    _validate_plan(data, metadata, fit_indices)
    records, selected = metadata.get('target_results'), metadata.get('selected_index')
    if (not isinstance(records, dict) or set(records) != set(data.config.targets)
            or not isinstance(selected, dict) or set(selected) != set(data.config.targets)):
        raise ValueError('组合项目缺失或重复了逐目标选优记录。')
    for target in data.config.targets:
        record = records[target]
        if not isinstance(record, dict):
            raise ValueError('逐目标选优记录结构无效。')
        candidates = record.get('candidates')
        if (record.get('target') != target or not isinstance(candidates, list)
                or len(candidates) != len(config.candidates)):
            raise ValueError('逐目标候选数量/名称不一致。')
        for ci, (candidate, spec) in enumerate(zip(candidates, config.candidates)):
            if (candidate.get('candidate') != ci + 1 or candidate.get('family') != spec.family
                    or candidate.get('spec') != asdict(spec) or candidate.get('target') != target
                    or len(candidate.get('folds', [])) != expected['fold_count']):
                raise ValueError('逐目标候选规格或折数无效。')
            scores = []
            for fi, (fold, manifest) in enumerate(zip(candidate['folds'], expected['folds'])):
                indices = manifest['train'] if spec.family == 'rf' else manifest['inner_train']
                std = float(data.frame.iloc[manifest['train']][target].to_numpy(float).std())
                denom = 1. if std < 1e-12 else std
                if (fold.get('fold') != fi + 1 or fold.get('fit_indices') != indices
                        or fold.get('normalization_std') != [denom]
                        or fold.get('constant_training_targets') != [std < 1e-12]
                        or not isinstance(fold.get('metrics'), dict) or set(fold['metrics']) != {target}):
                    raise ValueError('逐目标CV实际拟合索引/缩放标准差不一致。')
                metrics = fold['metrics'][target]
                if (set(metrics) != {'MAE', 'MSE', 'RMSE', 'R2'}
                        or any(type(metrics[k]) not in (int, float) or not np.isfinite(metrics[k])
                               or metrics[k] < 0 for k in ('MAE', 'MSE', 'RMSE'))
                        or (metrics['R2'] is not None and (type(metrics['R2']) not in (int, float)
                            or not np.isfinite(metrics['R2'])))
                        or not np.isclose(metrics['RMSE']**2, metrics['MSE'], rtol=1e-10, atol=1e-12)):
                    raise ValueError('逐目标CV指标无效。')
                score = metrics['RMSE'] / denom
                if type(fold.get('score')) not in (int, float) or not np.isclose(fold['score'], score, rtol=1e-10, atol=1e-12):
                    raise ValueError('逐目标CV分数与原单位指标不一致。')
                scores.append(score)
                best, rounds = fold.get('best_epoch'), fold.get('best_rounds')
                if (type(best) is not int or (spec.family == 'rf' and (best != 0 or rounds is not None))
                        or (spec.family == 'ann' and (not 1 <= best <= spec.train_config().epochs or rounds is not None))
                        or (spec.family == 'xgboost' and (not isinstance(rounds, list) or len(rounds) != 1
                            or type(rounds[0]) is not int or not 1 <= rounds[0] <= spec.xgb_params()['n_estimators']
                            or best != rounds[0]))):
                    raise ValueError('逐目标CV早停轮数记录无效。')
            if (not np.isclose(candidate.get('score_mean', np.nan), np.mean(scores), rtol=1e-10, atol=1e-12)
                    or not np.isclose(candidate.get('score_std', np.nan), np.std(scores), rtol=1e-10, atol=1e-12)):
                raise ValueError('逐目标CV汇总分数无效。')
            spread = candidate.get('metric_spread')
            if not isinstance(spread, dict) or set(spread) != {target} or set(spread[target]) != {'MAE', 'MSE', 'RMSE', 'R2'}:
                raise ValueError('逐目标CV指标汇总结构无效。')
            for key in ('MAE', 'MSE', 'RMSE', 'R2'):
                values = [fold['metrics'][target][key] for fold in candidate['folds'] if fold['metrics'][target][key] is not None]
                expected_spread = {'mean': float(np.mean(values)) if values else None,
                                   'std': float(np.std(values)) if values else None}
                if spread[target][key] != expected_spread:
                    raise ValueError('逐目标CV指标汇总与各折指标不一致。')
        winner = min(range(len(candidates)), key=lambda i: (candidates[i]['score_mean'], i))
        if (type(selected[target]) is not int or selected[target] != winner
                or record.get('selected_index') != winner or record.get('spec') != asdict(config.candidates[winner])
                or record.get('family') != config.candidates[winner].family):
            raise ValueError('逐目标胜出模型未按开发集验证结果选取。')
        winner_spec, winner_folds = config.candidates[winner], candidates[winner]['folds']
        budget = None
        if winner_spec.family == 'ann':
            budget = max(1, int(np.median([fold['best_epoch'] for fold in winner_folds])))
        elif winner_spec.family == 'xgboost':
            budget = np.maximum(1, np.median(np.array([fold['best_rounds'] for fold in winner_folds]), axis=0).astype(int)).tolist()
        if record.get('refit_budget') != budget:
            raise ValueError('最终重拟合预算不是由开发集各折早停结果确定的。')
    return config


def _save_v3(path, trained, optimization):
    validate_splits(trained.data)
    trained.validate()
    _validate_independent_metadata(trained.data, trained.search_metadata, trained.fit_indices)
    metadata = {'kind': 'independent_targets', 'data_config': asdict(trained.data.config),
                'fingerprint': trained.data.fingerprint, 'splits': trained.data.splits, 'report': trained.data.report,
                'fit_indices': trained.fit_indices, 'target_order': list(trained.data.config.targets),
                'metrics': trained.metrics, 'support_bounds': trained.support_bounds,
                'search_metadata': trained.search_metadata, 'models': []}
    payloads = {'data.csv': _csv(trained.data.frame)}
    for i, target in enumerate(trained.data.config.targets):
        model = trained.target_models[target]
        model.spec.validate(len(trained.data.config.inputs), 1)
        _validate_final_history(model)
        local_metadata = {key: model.search_metadata[key] for key in
                          ('seed', 'best_rounds', 'booster_history', 'selected_index', 'refit_budget') if key in model.search_metadata}
        metadata['models'].append({'target': target, 'spec': asdict(model.spec), 'fit_indices': model.fit_indices,
                                   'x_scale': model.x_scale.to_dict(), 'y_scale': model.y_scale.to_dict(),
                                   'history': model.history, 'best_epoch': model.best_epoch, 'fit_metadata': local_metadata})
        for name, blob in _encode_model_artifacts(model).items():
            payloads[f'target_{i:04d}_{name}'] = blob
    payloads['metadata.json'] = _json(metadata)
    if optimization is not None:
        optimization.config.validate(trained)
        payloads['candidates.csv'] = _csv(optimization.candidates)
        payloads['optimization.json'] = _json({'config': asdict(optimization.config), 'evaluations': optimization.evaluations})
    _write_archive(path, payloads, 3)


def _load_v3(path):
    from .modeling import CompositeSurrogate, FittedSurrogate, ModelSpec, target_data, metric_report
    import re
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = {e.filename for e in entries}
        fixed = {'manifest.json', 'data.csv', 'metadata.json', 'candidates.csv', 'optimization.json'}
        if (len(entries) != len(names) or len(entries) > 10000
                or not {'manifest.json', 'data.csv', 'metadata.json'} <= names
                or any(n not in fixed and not re.fullmatch(r'target_[0-9]{4}_(weights\.pt|forest\.npz|booster_0\.json)', n) for n in names)):
            raise ValueError('版本3项目成员名称不合法（拒绝路径、重复或额外模型）。')
        if sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES or any(e.flag_bits & 1 for e in entries):
            raise ValueError('版本3项目超大或加密。')
        blobs = {n: archive.read(n) for n in names}
    _check_archive_storage_budget(blobs)
    manifest = _read_json(blobs.pop('manifest.json'))
    hashes = manifest.get('sha256', {})
    if (manifest.get('schema_version') != 3 or set(hashes) != set(blobs)
            or any(hashlib.sha256(blob).hexdigest() != hashes[name] for name, blob in blobs.items())):
        raise ValueError('版本3项目完整性校验失败。')
    meta = _read_json(blobs['metadata.json'])
    required = {'kind', 'data_config', 'fingerprint', 'splits', 'report', 'fit_indices', 'target_order',
                'metrics', 'support_bounds', 'search_metadata', 'models'}
    if not isinstance(meta, dict) or set(meta) != required or meta['kind'] != 'independent_targets':
        raise ValueError('版本3项目元数据结构无效。')
    dc = DataConfig(**meta['data_config'])
    dc.validate()
    if meta['target_order'] != dc.targets or not isinstance(meta['models'], list) or len(meta['models']) != len(dc.targets):
        raise ValueError('组合项目目标顺序/数量不一致。')
    raw = pd.read_csv(BytesIO(blobs['data.csv']), float_precision='round_trip', dtype={dc.group: str} if dc.group else None)
    data = prepare_data(raw, dc)
    if data.fingerprint != meta['fingerprint']:
        raise ValueError('数据指纹不匹配。')
    _validate_split_types(meta['splits'])
    data.splits, data.report = meta['splits'], meta['report']
    validate_splits(data)
    _validate_independent_metadata(data, meta['search_metadata'], meta['fit_indices'])
    models, consumed = {}, {'metadata.json', 'data.csv', 'candidates.csv', 'optimization.json'}
    for i, target in enumerate(dc.targets):
        item = meta['models'][i]
        required_model = {'target', 'spec', 'fit_indices', 'x_scale', 'y_scale', 'history', 'best_epoch', 'fit_metadata'}
        if (not isinstance(item, dict) or set(item) != required_model or item['target'] != target
                or item['fit_indices'] != meta['fit_indices']):
            raise ValueError('组合子模型目标/公共拟合索引不一致。')
        record = meta['search_metadata']['target_results'][target]
        spec = ModelSpec.from_dict(item['spec'])
        spec.validate(len(dc.inputs), 1)
        if item['spec'] != record['spec'] or not isinstance(item['fit_metadata'], dict):
            raise ValueError('组合子模型规格与逐目标验证胜者不一致。')
        fit_meta = item['fit_metadata']
        if (fit_meta.get('selected_index') != record['selected_index']
                or fit_meta.get('refit_budget') != record['refit_budget']
                or fit_meta.get('seed') != meta['search_metadata']['seed']):
            raise ValueError('组合子模型最终重拟合记录不一致。')
        xs, ys = Scale.from_dict(item['x_scale'], len(dc.inputs)), Scale.from_dict(item['y_scale'], 1)
        view = target_data(data, target)
        for stored, expected in ((xs, Scale.fit(data.X[meta['fit_indices']])),
                                  (ys, Scale.fit(view.Y[meta['fit_indices']]))):
            if (not np.allclose(stored.mean, expected.mean, rtol=1e-12, atol=1e-12)
                    or not np.allclose(stored.scale, expected.scale, rtol=1e-12, atol=1e-12)):
                raise ValueError('组合子模型缩放器不是由公共拟合训练子集计算的。')
        prefix = f'target_{i:04d}_'
        artifacts = {name[len(prefix):]: blob for name, blob in blobs.items() if name.startswith(prefix)}
        backend = _decode_model_artifacts(artifacts, spec, len(dc.inputs), 1)
        consumed.update(prefix + name for name in artifacts)
        model = FittedSurrogate(view, spec, backend, xs, ys, list(item['fit_indices']), item['history'],
                                best_epoch=item['best_epoch'], search_metadata={
                                    **fit_meta, 'target_mode': 'independent', 'plan': meta['search_metadata']['plan'],
                                    'config': meta['search_metadata']['config'], 'candidates': record['candidates']})
        _validate_final_history(model)
        if ((model.family == 'ann' and model.best_epoch != record['refit_budget'])
                or (model.family == 'xgboost' and model.search_metadata['best_rounds'] != record['refit_budget'])):
            raise ValueError('组合子模型实际重拟合预算不一致。')
        models[target] = model
    if set(blobs) - consumed:
        raise ValueError('组合项目包含未引用的额外模型文件。')
    trained = CompositeSurrogate(data, models, meta['fit_indices'], search_metadata=meta['search_metadata'])
    if meta['support_bounds'] != {k: list(v) for k, v in trained.support_bounds.items()}:
        raise ValueError('组合模型支持范围与公共拟合子集不一致。')
    for split, idx in (('train', trained.fit_indices), ('test', data.splits['test'])):
        trained.metrics[split] = metric_report(data.Y[idx], trained.predict(data.X[idx]), dc.targets)
        for target in dc.targets:
            models[target].metrics[split] = {target: trained.metrics[split][target]}
    return trained, _load_optimization(blobs, trained)


def load_project(path):
    path = Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('项目超过128MB。')
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('manifest.json')
        if info.file_size > MAX_MANIFEST_BYTES:
            raise ValueError('项目清单过大。')
        version = _read_json(archive.read(info)).get('schema_version')
    if version == 1:
        return _load_v1(path)
    if version == 2:
        return _load_v2(path)
    if version == 3:
        return _load_v3(path)
    raise ValueError('不支持该项目版本。')
