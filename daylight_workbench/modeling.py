"""Bounded serial model comparison with locked test data and group-safe CV.

No test values are inspected until a candidate is selected and refit. Every
preprocessor is fit on the rows actually passed to that model fit. Native JSON,
tensor-only weights and validated numeric trees are the persistence contracts.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
import math
import threading
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .core import (PreparedData, Scale, TrainConfig, TrainedModel, _leakage_components,
                   check_cancel, make_model, validate_splits)

ACTIVATIONS = ('relu', 'tanh', 'sigmoid', 'gelu', 'leaky_relu')


@dataclass
class LayerSpec:
    width: int = 64
    activation: str = 'relu'

    def validate(self):
        if type(self.width) is not int or not 1 <= self.width <= 2048 or self.activation not in ACTIVATIONS:
            raise ValueError('ANN隐藏层宽度范围1-2048；激活函数必须来自预设列表。')


@dataclass
class ModelSpec:
    family: str = 'ann'
    params: dict = field(default_factory=dict)
    layers: list[LayerSpec] = field(default_factory=lambda: [LayerSpec(), LayerSpec()])

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, dict) or set(value) - {'family', 'params', 'layers'}
                or not isinstance(value.get('params', {}), dict) or not isinstance(value.get('layers', []), list)):
            raise ValueError('模型规格结构无效。')
        value = dict(value)
        value['layers'] = [LayerSpec(**layer) for layer in value.get('layers', [])]
        return cls(**value)

    def validate(self, n_inputs=1, n_outputs=1):
        if self.family not in ('ann', 'rf', 'xgboost') or not isinstance(self.params, dict) or not isinstance(self.layers, list):
            raise ValueError('模型必须为ANN、随机森林或XGBoost。')
        allowed = {'ann': {'epochs', 'patience', 'learning_rate', 'batch_size'},
                   'rf': {'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'max_leaf_nodes'},
                   'xgboost': {'n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_lambda', 'patience'}}[self.family]
        if set(self.params) - allowed:
            raise ValueError(f'不支持的{self.family}参数：{set(self.params) - allowed}')
        integer_params = {'ann': {'epochs', 'patience', 'batch_size'},
                          'rf': {'n_estimators', 'max_depth', 'min_samples_leaf', 'max_leaf_nodes'},
                          'xgboost': {'n_estimators', 'max_depth', 'patience'}}[self.family]
        for key, value in self.params.items():
            if type(value) not in (int, float) or (key in integer_params and type(value) is not int):
                raise ValueError('模型参数必须为正确类型的数值，不能用布尔值或文本替代。')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('模型参数必须是有限值。')
        if self.family == 'ann':
            if not self.layers:
                raise ValueError('ANN至少需要一层隐藏层。')
            for layer in self.layers:
                if not isinstance(layer, LayerSpec):
                    raise ValueError('ANN层规格无效。')
                layer.validate()
            self.train_config().validate(n_inputs, n_outputs)
            if self.params.get('epochs', 300) > 3000:
                raise ValueError('网格搜索ANN单次最多3000轮。')
        elif self.family == 'rf':
            p = self.rf_params()
            if any(type(p[k]) is not int for k in ('n_estimators', 'max_depth', 'min_samples_leaf', 'max_leaf_nodes')):
                raise ValueError('随机森林的树数、深度、叶节点和最小叶样本数必须是整数。')
            if not 1 <= p['n_estimators'] <= 500 or not 1 <= p['max_depth'] <= 30 or not 2 <= p['max_leaf_nodes'] <= 1024 or not 1 <= p['min_samples_leaf'] <= 10000 or not 0 < p['max_features'] <= 1:
                raise ValueError('随机森林参数超出P0.2安全预算。')
        else:
            p = self.xgb_params()
            if type(p['n_estimators']) is not int or type(p['max_depth']) is not int or not 1 <= p['n_estimators'] <= 1500 or not 1 <= p['max_depth'] <= 12:
                raise ValueError('XGBoost迭代数1-1500，树深度1-12。')
            if not 0 < p['learning_rate'] <= 1 or not 0 < p['subsample'] <= 1 or not 0 < p['colsample_bytree'] <= 1 or not 0 <= p['reg_lambda'] <= 10000 or not 1 <= p['patience'] <= 1000:
                raise ValueError('XGBoost学习率/采样/正则/早停参数无效。')
        if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in self.params.values()):
            raise ValueError('模型参数必须是有限值。')

    def train_config(self, seed=42):
        return TrainConfig(hidden=[l.width for l in self.layers], epochs=int(self.params.get('epochs', 300)),
                           patience=int(self.params.get('patience', 30)), learning_rate=float(self.params.get('learning_rate', .001)),
                           batch_size=int(self.params.get('batch_size', 64)), seed=seed,
                           activations=[l.activation for l in self.layers])

    def rf_params(self):
        return {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 1, 'max_features': 1., 'max_leaf_nodes': 256, **self.params}

    def xgb_params(self):
        return {'n_estimators': 150, 'max_depth': 4, 'learning_rate': .05, 'subsample': 1., 'colsample_bytree': 1., 'reg_lambda': 1., 'patience': 20, **self.params}


@dataclass
class SearchConfig:
    candidates: list[ModelSpec]
    mode: str = 'cv'
    folds: int = 5
    max_fits: int = 200
    seed: int = 42
    cpu_threads: int = 2
    target_mode: str = 'independent'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or not isinstance(value.get('candidates'), list):
            raise ValueError('搜索配置结构无效。')
        value = dict(value)
        value['candidates'] = [ModelSpec.from_dict(v) for v in value['candidates']]
        # Projects written before independent targets existed meant shared
        # selection. Never reinterpret their provenance as a different method.
        value.setdefault('target_mode', 'shared')
        return cls(**value)

    def validate(self, data):
        if (self.target_mode not in ('independent', 'shared') or self.mode not in ('cv', 'fixed')
                or any(type(v) is not int for v in (self.folds, self.max_fits, self.cpu_threads, self.seed))
                or not 0 <= self.seed <= 2**32 - 1 or not 2 <= self.folds <= 10
                or not 1 <= self.max_fits <= 2000 or not 1 <= self.cpu_threads <= 4):
            raise ValueError('模式/折数/拟合预算/CPU线程数无效。')
        if not isinstance(self.candidates, list) or not 1 <= len(self.candidates) <= 100:
            raise ValueError('一次搜索须有1-100个明确候选，不运行无界AutoML。')
        for spec in self.candidates:
            if not isinstance(spec, ModelSpec):
                raise ValueError('候选必须是明确模型规格。')
            spec.validate(len(data.config.inputs), 1 if self.target_mode == 'independent' else len(data.config.targets))


def _balanced_folds(indices, components, n_folds, seed):
    groups = list(np.unique(components[indices]))
    if len(groups) < n_folds:
        raise ValueError(f'开发集独立组仅{len(groups)}个，不能执行{n_folds}折；请增加数据或切换固定划分。')
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    grouped = [indices[components[indices] == group].tolist() for group in groups]
    grouped.sort(key=len, reverse=True)
    buckets = [[] for _ in range(n_folds)]
    for rows in grouped:
        buckets[min(range(n_folds), key=lambda k: len(buckets[k]))].extend(rows)
    folds = []
    for validation in buckets:
        train = sorted(set(indices.tolist()) - set(validation))
        if len(validation) < 2 or len(train) < 4:
            raise ValueError('某折样本过少，无法评估。')
        folds.append({'train': train, 'validation': sorted(validation)})
    return folds


def inner_split(indices, components, seed):
    indices = np.asarray(indices, dtype=int)
    unique = np.unique(components[indices])
    if len(unique) < 2:
        raise ValueError('ANN/XGBoost内部早停至少需要2个独立训练组；不能使用外层验证集替代。')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chosen, count = [], 0
    target = max(2, int(round(len(indices) * .15)))
    for group in unique[:-1]:
        chosen.append(group)
        count += int(np.sum(components[indices] == group))
        if count >= target:
            break
    val = indices[np.isin(components[indices], chosen)]
    train = indices[~np.isin(components[indices], chosen)]
    if min(len(train), len(val)) < 2:
        raise ValueError('内部早停划分不足2行，请增加独立组。')
    return train.tolist(), val.tolist()


def make_search_plan(data: PreparedData, config: SearchConfig) -> dict:
    validate_splits(data)
    config.validate(data)
    # One immutable locked test partition reused across every candidate/model.
    test = list(data.splits['test'])
    development = np.array(sorted(data.splits['train'] + data.splits['validation']), dtype=int)
    components = _leakage_components(data.frame, data.config)
    if config.mode == 'cv':
        folds = _balanced_folds(development, components, config.folds, config.seed)
    else:
        folds = [{'train': list(data.splits['train']), 'validation': list(data.splits['validation'])}]
    for k, fold in enumerate(folds):
        # Same inner split manifest is shared by ANN and boosting candidates.
        it, iv = inner_split(fold['train'], components, config.seed + k + 100)
        fold['inner_train'], fold['inner_validation'] = it, iv
    outputs = len(data.config.targets)
    independent = config.target_mode == 'independent'
    per_candidate = [(outputs if independent or s.family == 'xgboost' else 1) * len(folds) for s in config.candidates]
    refit_budget = outputs if independent else max(outputs if s.family == 'xgboost' else 1 for s in config.candidates)
    total = sum(per_candidate) + refit_budget
    if total > config.max_fits:
        raise ValueError(f'预算需要最多{total}次实际拟合，超过上限{config.max_fits}。请减少候选/折数或明确提高上限。')
    return {'schema_version': 2, 'seed': config.seed, 'mode': config.mode, 'target_mode': config.target_mode, 'test': test,
            'development': development.tolist(), 'components': components.tolist(), 'folds': folds,
            'candidate_count': len(config.candidates), 'fold_count': len(folds),
            'target_order': list(data.config.targets), 'target_count': outputs,
            'candidate_fold_jobs': len(config.candidates) * len(folds) * (outputs if independent else 1), 'fit_budget': total,
            'per_target_fit_budget': len(config.candidates) * len(folds) + 1 if independent else None,
            'refit_fit_budget': refit_budget,
            'per_candidate_fits': per_candidate, 'test_fraction_actual': len(test) / len(data.frame),
            'selection_score': ('independent per-target mean outer-fold RMSE / outer-training target std (constant std -> 1)' if independent
                                else 'equal-weight mean per-target RMSE / outer-training target std (constant std -> 1)'),
            'notice': '测试集锁定且只在选优/重拟合之后评估；不参与候选排名。'}


def metric_report(actual, predicted, targets):
    return {target: {'MAE': float(mean_absolute_error(actual[:, j], predicted[:, j])),
                     'MSE': float(mean_squared_error(actual[:, j], predicted[:, j])),
                     'RMSE': float(np.sqrt(mean_squared_error(actual[:, j], predicted[:, j]))),
                     'R2': float(r2_score(actual[:, j], predicted[:, j], force_finite=False)) if np.ptp(actual[:, j]) > 0 else None}
            for j, target in enumerate(targets)}


class NumericForest:
    """Immutable bounded tree arrays; inference parity-tested against sklearn RF."""
    def __init__(self, trees, n_inputs, n_outputs):
        self.trees, self.n_inputs, self.n_outputs = trees, n_inputs, n_outputs
        self.validate()

    @classmethod
    def from_sklearn(cls, forest, n_inputs, n_outputs):
        trees = []
        for estimator in forest.estimators_:
            tree = estimator.tree_
            trees.append({'left': tree.children_left.copy(), 'right': tree.children_right.copy(),
                          'feature': tree.feature.copy(), 'threshold': tree.threshold.copy(),
                          'value': tree.value[:, :, 0].copy()})
        return cls(trees, n_inputs, n_outputs)

    def validate(self):
        if (type(self.n_inputs) is not int or self.n_inputs < 1 or type(self.n_outputs) is not int or self.n_outputs < 1
                or not 1 <= len(self.trees) <= 500 or sum(len(t['left']) for t in self.trees) > 1_100_000):
            raise ValueError('森林树数/节点数超出安全限制。')
        for t in self.trees:
            if set(t) != {'left', 'right', 'feature', 'threshold', 'value'}:
                raise ValueError('树结构字段无效。')
            n = len(t['left'])
            if not n or any(np.asarray(t[k]).shape != (n,) for k in ('left', 'right', 'feature', 'threshold')) or t['value'].shape != (n, self.n_outputs):
                raise ValueError('树数组形状无效。')
            if not all(np.isfinite(v).all() for v in t.values()):
                raise ValueError('树包含非有限数值。')
            for k in ('left', 'right', 'feature'):
                if t[k].dtype.kind not in 'iu':
                    raise ValueError('树索引必须是整数。')
            internal = t['left'] != -1
            if not np.array_equal(internal, t['right'] != -1):
                raise ValueError('树左右子节点定义不一致。')
            ids = np.arange(n)[internal]
            if np.any(t['left'][internal] <= ids) or np.any(t['right'][internal] <= ids) or np.any(t['left'][internal] >= n) or np.any(t['right'][internal] >= n) or np.any(t['feature'][internal] < 0) or np.any(t['feature'][internal] >= self.n_inputs):
                raise ValueError('树路径存在环或越界。')
            children = np.concatenate((t['left'][internal], t['right'][internal]))
            if sorted(children.tolist()) != list(range(1, n)):
                raise ValueError('树包含重复子节点或不连通节点。')

    def predict(self, X):
        # sklearn converts feature data to float32 before tree threshold comparisons.
        X = np.asarray(X, np.float32)
        output = np.zeros((len(X), self.n_outputs), dtype=np.float64)
        for tree in self.trees:
            node = np.zeros(len(X), dtype=np.int64)
            while True:
                active = tree['left'][node] != -1
                if not active.any():
                    break
                rows = np.flatnonzero(active)
                old = node[rows]
                go_left = X[rows, tree['feature'][old]] <= tree['threshold'][old]
                node[rows] = np.where(go_left, tree['left'][old], tree['right'][old])
            output += tree['value'][node]
        return output / len(self.trees)


@dataclass
class FittedSurrogate:
    data: PreparedData
    spec: ModelSpec
    backend: object
    x_scale: Scale
    y_scale: Scale
    fit_indices: list[int]
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    best_epoch: int = 0
    search_metadata: dict = field(default_factory=dict)

    @property
    def family(self):
        return self.spec.family

    @property
    def config(self):
        # Compatibility with the early ANN-only UI/export contract.
        return self.spec.train_config(self.search_metadata.get('seed', 42))

    @property
    def support_bounds(self):
        X = self.data.X[self.fit_indices]
        return {name: (float(X[:, j].min()), float(X[:, j].max())) for j, name in enumerate(self.data.config.inputs)}

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.data.config.inputs) or not np.isfinite(X).all():
            raise ValueError('预测输入维度或数值无效。')
        scaled = self.x_scale.transform(X)
        if self.family == 'ann':
            import torch
            self.backend.eval()
            with torch.no_grad():
                y = self.backend(torch.as_tensor(scaled, dtype=torch.float32)).cpu().numpy()
        elif self.family == 'rf':
            y = self.backend.predict(scaled)
        else:
            import xgboost as xgb
            dm = xgb.DMatrix(scaled)
            y = np.column_stack([booster.predict(dm) for booster in self.backend])
        y = np.asarray(y)
        if y.shape != (len(X), len(self.data.config.targets)):
            raise ValueError('模型预测输出维度与目标定义不一致。')
        result = self.y_scale.inverse(y)
        if not np.isfinite(result).all():
            raise ValueError('模型预测出现非有限值。')
        return result


def target_data(data: PreparedData, target: str) -> PreparedData:
    """A scalar label view, retaining exactly the common sample identities.

    Do not prepare/split again: even duplicate rows with conflicting labels
    must retain the original multi-target project's leakage components.
    """
    if target not in data.config.targets:
        raise ValueError('独立模型的目标不属于项目目标。')
    config = replace(data.config, targets=[target],
                     directions={target: data.config.directions[target]} if data.config.directions else {})
    return PreparedData(data.frame, config, data.splits, data.fingerprint, data.report)


@dataclass
class CompositeSurrogate:
    """One independently selected scalar model per target, in declared order."""
    data: PreparedData
    target_models: dict[str, FittedSurrogate]
    fit_indices: list[int]
    metrics: dict = field(default_factory=dict)
    search_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    @property
    def family(self):
        return 'composite'

    @property
    def support_bounds(self):
        self.validate()
        X = self.data.X[self.fit_indices]
        return {name: (float(X[:, j].min()), float(X[:, j].max())) for j, name in enumerate(self.data.config.inputs)}

    def validate(self):
        targets = self.data.config.targets
        if not isinstance(self.target_models, dict) or set(self.target_models) != set(targets):
            raise ValueError('组合模型必须完整且唯一覆盖项目目标。')
        if (not isinstance(self.fit_indices, list) or not self.fit_indices
                or any(type(i) is not int for i in self.fit_indices)
                or len(set(self.fit_indices)) != len(self.fit_indices)
                or min(self.fit_indices) < 0 or max(self.fit_indices) >= len(self.data.frame)):
            raise ValueError('组合模型公共拟合索引无效。')
        for target in targets:
            model = self.target_models[target]
            if (not isinstance(model, FittedSurrogate) or model.data.config.targets != [target]
                    or model.data.config.inputs != self.data.config.inputs
                    or model.data.frame is not self.data.frame
                    or model.data.splits != self.data.splits or model.data.fingerprint != self.data.fingerprint
                    or model.fit_indices != self.fit_indices):
                raise ValueError('组合子模型的目标、输入顺序或公共拟合样本不一致。')

    def predict(self, X):
        self.validate()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.data.config.inputs) or not np.isfinite(X).all():
            raise ValueError('组合模型预测输入维度或数值无效。')
        # A dict's insertion order is deliberately NOT the output contract.
        columns = [self.target_models[target].predict(X) for target in self.data.config.targets]
        result = np.column_stack([column[:, 0] for column in columns])
        if result.shape != (len(X), len(self.data.config.targets)) or not np.isfinite(result).all():
            raise ValueError('组合模型预测输出维度或数值无效。')
        return result


@dataclass
class ModelSearchResult:
    selected: FittedSurrogate | CompositeSurrogate
    config: SearchConfig
    plan: dict
    candidates: list[dict]
    selected_index: int | dict[str, int]
    target_results: dict[str, dict] = field(default_factory=dict)


def _fit_ann(data, spec, train, validation, seed, cancel, progress, snapshot, context, fixed_epochs=None):
    import torch
    cfg = spec.train_config(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    # Materialize only rows used by this fit. In particular, locked test
    # labels must not be converted to float32 or inspected before selection.
    train_frame = data.frame.iloc[train]
    train_x = train_frame[data.config.inputs].to_numpy(dtype=np.float64)
    train_y = train_frame[data.config.targets].to_numpy(dtype=np.float64)
    xs, ys = Scale.fit(train_x), Scale.fit(train_y)
    X = torch.tensor(xs.transform(train_x), dtype=torch.float32)
    Y = torch.tensor(ys.transform(train_y), dtype=torch.float32)
    X_val = Y_val = None
    if validation:
        validation_frame = data.frame.iloc[validation]
        X_val = torch.tensor(xs.transform(validation_frame[data.config.inputs].to_numpy(dtype=np.float64)), dtype=torch.float32)
        Y_val = torch.tensor(ys.transform(validation_frame[data.config.targets].to_numpy(dtype=np.float64)), dtype=torch.float32)
    if not all(torch.isfinite(values).all() for values in (X, Y, X_val, Y_val) if values is not None):
        raise ValueError('缩放后数值超出float32范围。')
    model = make_model(X.shape[1], Y.shape[1], cfg)
    adam = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    best, best_epoch, weights, stale, history = float('inf'), 0, None, 0, []
    epochs = fixed_epochs if fixed_epochs is not None else cfg.epochs
    last_emit = 0.
    for epoch in range(1, epochs + 1):
        check_cancel(cancel)
        model.train()
        order, total = rng.permutation(len(train)), 0.
        for start in range(0, len(order), cfg.batch_size):
            check_cancel(cancel)
            batch = order[start:start + cfg.batch_size]
            adam.zero_grad()
            loss = loss_fn(model(X[batch]), Y[batch])
            if not torch.isfinite(loss):
                raise ValueError('ANN训练出现非有限损失。')
            loss.backward()
            adam.step()
            total += float(loss.item()) * len(batch)
        val_loss = None
        if validation:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(X_val), Y_val).item())
            if not math.isfinite(val_loss):
                raise ValueError('ANN验证损失非有限。')
            if val_loss < best - 1e-8:
                best, best_epoch, stale = val_loss, epoch, 0
                weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append({'epoch': epoch, 'train_loss': total / len(train), 'validation_loss': val_loss})
        stop = bool(validation and stale >= cfg.patience)
        if snapshot and (time.monotonic() - last_emit > .12 or epoch == epochs or stop):
            snapshot({'kind': 'loss', **context, 'family': 'ann', 'epoch': epoch, 'best_epoch': best_epoch,
                      'early_stopped': stop, 'refit': fixed_epochs is not None, 'history': list(history)})
            last_emit = time.monotonic()
        if progress and (epoch == 1 or epoch % 10 == 0 or stop):
            progress(f"{context.get('label','ANN')}：轮次{epoch}/{epochs}，最佳{best_epoch}" + ('（早停）' if stop else ''))
        if stop:
            break
    check_cancel(cancel)
    if weights is not None:
        model.load_state_dict(weights)
    return FittedSurrogate(data, spec, model, xs, ys, list(train), history=history, best_epoch=best_epoch)


def _fit_rf(data, spec, train, seed, threads, cancel, progress, snapshot, context):
    frame = data.frame.iloc[train]
    train_x, train_y = frame[data.config.inputs].to_numpy(float), frame[data.config.targets].to_numpy(float)
    xs, ys = Scale.fit(train_x), Scale.fit(train_y)
    X, Y = xs.transform(train_x), ys.transform(train_y)
    p = spec.rf_params()
    count = p.pop('n_estimators')
    forest = RandomForestRegressor(**p, n_estimators=1, random_state=seed, n_jobs=threads, warm_start=True)
    history = []
    for k in range(1, count + 1):
        check_cancel(cancel)
        forest.n_estimators = k
        forest.fit(X, Y[:, 0] if Y.shape[1] == 1 else Y)
        if snapshot and (k == 1 or k % 5 == 0 or k == count):
            snapshot({'kind': 'fit', **context, 'family': 'rf', 'tree': k, 'trees': count,
                      'notice': '真实已拟合树数；随机森林不显示伪造损失曲线。'})
        if progress and (k == 1 or k % 10 == 0 or k == count):
            progress(f"{context.get('label','RF')}：已拟合{k}/{count}棵树（单棵树结束后可取消）")
    check_cancel(cancel)
    backend = NumericForest.from_sklearn(forest, X.shape[1], Y.shape[1])
    # A failed persistence representation is never silently accepted.
    np.testing.assert_allclose(backend.predict(X[:128]), np.asarray(forest.predict(X[:128])).reshape(-1, Y.shape[1]), rtol=1e-12, atol=1e-12)
    return FittedSurrogate(data, spec, backend, xs, ys, list(train))


def _fit_xgb(data, spec, train, validation, seed, threads, cancel, progress, snapshot, context, fixed_rounds=None):
    import xgboost as xgb
    frame = data.frame.iloc[train]
    train_x, train_y = frame[data.config.inputs].to_numpy(float), frame[data.config.targets].to_numpy(float)
    xs, ys = Scale.fit(train_x), Scale.fit(train_y)
    X, Y = xs.transform(train_x), ys.transform(train_y)
    X_val = Y_val = None
    if validation:
        val_frame = data.frame.iloc[validation]
        X_val = xs.transform(val_frame[data.config.inputs].to_numpy(float))
        Y_val = ys.transform(val_frame[data.config.targets].to_numpy(float))
    p = spec.xgb_params()
    rounds, patience = p.pop('n_estimators'), p.pop('patience')
    p = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
         'nthread': threads, 'seed': seed, **p}
    boosters, best_rounds, all_history = [], [], []
    for j, target in enumerate(data.config.targets):
        check_cancel(cancel)
        dm = xgb.DMatrix(X, label=Y[:, j])
        evals = [(dm, 'train')]
        if validation:
            evals.append((xgb.DMatrix(X_val, label=Y_val[:, j]), 'validation'))
        records = {}
        local_history = []
        class Monitor(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                check_cancel(cancel)
                local_history.append({'epoch': epoch + 1, 'train_loss': float(evals_log['train']['rmse'][-1]),
                                      'validation_loss': float(evals_log['validation']['rmse'][-1]) if validation else None})
                if snapshot and (epoch == 0 or (epoch + 1) % 5 == 0):
                    snapshot({'kind': 'loss', **context, 'family': 'xgboost', 'target': target,
                              'epoch': epoch + 1, 'metric': 'scaled RMSE', 'refit': fixed_rounds is not None,
                              'history': list(local_history)})
                return False
        count = fixed_rounds[j] if fixed_rounds is not None else rounds
        booster = xgb.train(p, dm, num_boost_round=count, evals=evals, evals_result=records,
                            early_stopping_rounds=patience if validation else None,
                            callbacks=[Monitor()], verbose_eval=False)
        best = int(booster.best_iteration) + 1 if validation else count
        if validation:
            booster = booster[:best]
        boosters.append(booster)
        best_rounds.append(best)
        all_history.append({'target': target, 'history': local_history})
        if snapshot:
            snapshot({'kind': 'loss', **context, 'family': 'xgboost', 'target': target,
                      'epoch': len(local_history), 'best_epoch': best, 'metric': 'scaled RMSE',
                      'refit': fixed_rounds is not None, 'early_stopped': bool(validation and len(local_history) < count),
                      'history': list(local_history)})
        if progress:
            progress(f"{context.get('label','XGBoost')}：目标{target}，真实迭代{len(local_history)}，保留{best}轮")
    result = FittedSurrogate(data, spec, boosters, xs, ys, list(train), best_epoch=max(best_rounds))
    result.search_metadata['best_rounds'] = best_rounds
    result.search_metadata['booster_history'] = all_history
    return result


def _fit(data, spec, outer_train, inner_train, inner_val, seed, threads, cancel, progress, snapshot, context, fixed_budget=None):
    if spec.family == 'ann':
        return _fit_ann(data, spec, outer_train if fixed_budget is not None else inner_train,
                        [] if fixed_budget is not None else inner_val, seed, cancel, progress, snapshot, context,
                        fixed_epochs=fixed_budget)
    if spec.family == 'rf':
        return _fit_rf(data, spec, outer_train, seed, threads, cancel, progress, snapshot, context)
    return _fit_xgb(data, spec, outer_train if fixed_budget is not None else inner_train,
                    [] if fixed_budget is not None else inner_val, seed, threads, cancel, progress, snapshot,
                    context, fixed_rounds=fixed_budget)


def _rows(data, indices, columns):
    return data.frame.iloc[indices][columns].to_numpy(dtype=np.float64)


def _search_candidates(data, config, plan, cancel, progress, snapshot, target_context, job_counter):
    """Evaluate only development folds. No locked-test labels are materialized."""
    results, epoch_counts = [], []
    for ci, spec in enumerate(config.candidates):
        folds, budgets = [], []
        for fi, fold in enumerate(plan['folds']):
            check_cancel(cancel)
            job_counter[0] += 1
            prefix = (target_context.get('target', '') + ' · ') if target_context else ''
            context = {**target_context, 'candidate': ci + 1, 'candidates': len(config.candidates),
                       'fold': fi + 1, 'folds': len(plan['folds']), 'fit_job': job_counter[0],
                       'fit_jobs': plan['candidate_fold_jobs'],
                       'label': f'{prefix}{spec.family} 候选{ci+1} 折{fi+1}'}
            if snapshot:
                snapshot({'kind': 'search', **context, 'stage': 'fit', 'fit_budget': plan['fit_budget']})
            fit = _fit(data, spec, fold['train'], fold['inner_train'], fold['inner_validation'],
                       config.seed + fi, config.cpu_threads, cancel, progress, snapshot, context)
            check_cancel(cancel)
            actual = _rows(data, fold['validation'], data.config.targets)
            predicted = fit.predict(_rows(data, fold['validation'], data.config.inputs))
            metrics = metric_report(actual, predicted, data.config.targets)
            std = _rows(data, fold['train'], data.config.targets).std(axis=0)
            denom = np.where(std < 1e-12, 1., std)
            per_target = np.sqrt(np.mean((actual - predicted)**2, axis=0)) / denom
            score = float(per_target.mean())
            if not math.isfinite(score):
                raise ValueError('候选验证分数非有限，无法科学选优。')
            folds.append({'fold': fi + 1, 'metrics': metrics, 'score': score,
                          'normalization_std': denom.tolist(), 'constant_training_targets': (std < 1e-12).tolist(),
                          'fit_indices': fit.fit_indices, 'best_epoch': fit.best_epoch,
                          'best_rounds': fit.search_metadata.get('best_rounds')})
            budgets.append(fit.search_metadata.get('best_rounds') if spec.family == 'xgboost' else fit.best_epoch)
            if snapshot:
                snapshot({'kind': 'search', **context, 'stage': 'scored', 'score': score, 'metrics': metrics})
        scores = [f['score'] for f in folds]
        spread = {}
        for target in data.config.targets:
            spread[target] = {}
            for key in ('MAE', 'MSE', 'RMSE', 'R2'):
                values = [f['metrics'][target][key] for f in folds if f['metrics'][target][key] is not None]
                spread[target][key] = {'mean': float(np.mean(values)) if values else None,
                                       'std': float(np.std(values)) if values else None}
        results.append({'candidate': ci + 1, 'family': spec.family, 'spec': asdict(spec),
                        **({'target': target_context['target']} if target_context else {}),
                        'score_mean': float(np.mean(scores)), 'score_std': float(np.std(scores)),
                        'folds': folds, 'metric_spread': spread})
        epoch_counts.append(budgets)
    check_cancel(cancel)
    chosen = min(range(len(results)), key=lambda i: (results[i]['score_mean'], i))
    spec = config.candidates[chosen]
    budget = None
    if spec.family == 'ann':
        budget = max(1, int(np.median(epoch_counts[chosen])))
    elif spec.family == 'xgboost':
        budget = np.maximum(1, np.median(np.array(epoch_counts[chosen]), axis=0).astype(int)).tolist()
    return results, chosen, budget


def _refit_winner(data, config, plan, results, chosen, budget, cancel, progress, snapshot, target_context):
    check_cancel(cancel)
    spec = config.candidates[chosen]
    prefix = (target_context.get('target', '') + ' · ') if target_context else ''
    if progress:
        progress(f'{prefix}验证选优完成：候选{chosen+1} {spec.family}；全部开发集重拟合，未查看测试集。')
    context = {**target_context, 'candidate': chosen + 1, 'family': spec.family,
               'label': prefix + '最终开发集重拟合', 'stage': 'refit'}
    if snapshot:
        snapshot({'kind': 'search', **context})
    selected = _fit(data, spec, plan['development'], [], [], config.seed, config.cpu_threads, cancel,
                    progress, snapshot, context, fixed_budget=budget if spec.family != 'rf' else 1)
    selected.search_metadata.update({'schema_version': 2, 'seed': config.seed, 'target_mode': config.target_mode,
                                     'config': asdict(config), 'plan': plan, 'candidates': results,
                                     'selected_index': chosen, 'refit_budget': budget,
                                     'selection_score': plan['selection_score']})
    check_cancel(cancel)
    return selected


def run_model_search(data: PreparedData, config: SearchConfig, cancel=None, progress=None, snapshot=None) -> ModelSearchResult:
    import torch
    plan = make_search_plan(data, config)
    torch.set_num_threads(config.cpu_threads)
    check_cancel(cancel)
    jobs = [0]
    targets = data.config.targets
    if config.target_mode == 'shared':
        results, chosen, budget = _search_candidates(data, config, plan, cancel, progress, snapshot, {}, jobs)
        selected = _refit_winner(data, config, plan, results, chosen, budget, cancel, progress, snapshot, {})
        per_target = {}
    else:
        winners, per_target, results = {}, {}, []
        # Selection is target-specific, but every target sees the exact same
        # fold and inner early-stop indices. Different winners are permitted.
        selections = {}
        for j, target in enumerate(targets):
            view = target_data(data, target)
            context = {'target': target, 'target_index': j + 1, 'targets': len(targets),
                       'target_mode': 'independent'}
            local, chosen, budget = _search_candidates(view, config, plan, cancel, progress, snapshot, context, jobs)
            selections[target] = (view, local, chosen, budget, context)
            per_target[target] = {'target': target, 'selected_index': chosen, 'family': config.candidates[chosen].family,
                                  'spec': asdict(config.candidates[chosen]), 'candidates': local, 'refit_budget': budget}
            results.extend(local)
            if snapshot:
                snapshot({'kind': 'search', **context, 'stage': 'target_selected',
                          'selected_index': chosen, 'family': config.candidates[chosen].family,
                          'score': local[chosen]['score_mean']})
        # Refit all winners before the FIRST access to any locked-test labels.
        for target in targets:
            view, local, chosen, budget, context = selections[target]
            winners[target] = _refit_winner(view, config, plan, local, chosen, budget, cancel, progress, snapshot, context)
        chosen = {target: per_target[target]['selected_index'] for target in targets}
        if len(targets) == 1:
            # A scalar target is already an independent model. Keep the public
            # v2 scalar artifact/API compatible for existing projects/tests.
            selected = winners[targets[0]]
            chosen = chosen[targets[0]]
            selected.search_metadata['target_results'] = per_target
        else:
            selected = CompositeSurrogate(data, winners, list(plan['development']), search_metadata={
                'schema_version': 3, 'seed': config.seed, 'target_mode': 'independent',
                'target_order': list(targets), 'config': asdict(config), 'plan': plan,
                'target_results': per_target, 'selected_index': chosen,
                'selection_score': plan['selection_score']})
    check_cancel(cancel)
    # First locked-test label access in the search. Never select a family,
    # hyperparameter, architecture, epoch or boosting round from these values.
    for split, indices in (('train', plan['development']), ('test', plan['test'])):
        actual = _rows(data, indices, targets)
        predicted = selected.predict(_rows(data, indices, data.config.inputs))
        selected.metrics[split] = metric_report(actual, predicted, targets)
        if isinstance(selected, CompositeSurrogate):
            for target in targets:
                selected.target_models[target].metrics[split] = {target: selected.metrics[split][target]}
    check_cancel(cancel)
    if snapshot:
        snapshot({'kind': 'search', 'stage': 'complete', 'target_mode': config.target_mode,
                  'selected_index': chosen, 'test_metrics': selected.metrics['test']})
    return ModelSearchResult(selected, config, plan, results, chosen, per_target)
