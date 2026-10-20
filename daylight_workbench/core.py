"""GUI-independent data, training, and optimization pipeline. No network access."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
import hashlib
import csv
import json
import math
import random
import threading
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Cancelled(RuntimeError):
    pass


def check_cancel(cancel: threading.Event | None):
    if cancel is not None and cancel.is_set():
        raise Cancelled('操作已取消；未完成的结果不替换当前项目。')


@dataclass
class DataConfig:
    inputs: list[str]
    targets: list[str]
    group: str | None = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    drop_duplicates: bool = False
    directions: dict[str, str | None] = field(default_factory=dict)

    def validate(self):
        if not self.inputs or not self.targets:
            raise ValueError('请选择至少一个自变量和一个目标。')
        cols = self.inputs + self.targets
        if len(cols) != len(set(cols)) or (self.group and self.group in cols):
            raise ValueError('自变量、目标和分组列不能重复或重叠。')
        if any(c in {'candidate_id', 'generation', 'pareto'} or c.startswith('predicted::') for c in cols):
            raise ValueError('自变量列名与候选元数据/预测列名冲突，请重命名。')
        if self.directions and (set(self.directions) != set(self.targets) or any(d not in ('min', 'max', None) for d in self.directions.values())):
            raise ValueError('目标方向定义与目标列不一致。')
        if not all(isinstance(c, str) and c for c in cols):
            raise ValueError('列名必须是非空文本。')
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        if not all(math.isfinite(r) and 0 < r < 1 for r in ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-8):
            raise ValueError('训练/验证/测试比例必须均大于0，且合计为1。')
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError('随机种子必须在 0 至 2^32-1 范围内。')


@dataclass
class PreparedData:
    frame: pd.DataFrame
    config: DataConfig
    splits: dict[str, list[int]]
    fingerprint: str
    report: dict

    @property
    def X(self):
        return self.frame[self.config.inputs].to_numpy(dtype=np.float64)

    @property
    def Y(self):
        return self.frame[self.config.targets].to_numpy(dtype=np.float64)


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('P0 单个数据文件限制为128MB。')
    if path.suffix.lower() == '.csv':
        encoding = 'utf-8-sig'
        try:
            with path.open(encoding=encoding, newline='') as handle:
                header = next(csv.reader(handle), [])
        except UnicodeDecodeError:
            encoding = 'gb18030'
            with path.open(encoding=encoding, newline='') as handle:
                header = next(csv.reader(handle), [])
        if not header or any(not str(c).strip() for c in header) or len(set(header)) != len(header):
            raise ValueError('源文件列名为空或重复；请先修正，避免自动改名造成歧义。')
        df = pd.read_csv(path, encoding=encoding, float_precision='round_trip')
    elif path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            header = next(workbook.worksheets[0].iter_rows(values_only=True), [])
            header = [str(c) if c is not None else '' for c in header]
        finally:
            workbook.close()
        if not header or any(not c.strip() for c in header) or len(set(header)) != len(header):
            raise ValueError('源文件列名为空或重复；请先修正。')
        df = pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError('只支持 .csv 和 .xlsx（首个工作表）。')
    df.columns = [str(c) for c in df.columns]
    if not len(df) or not df.columns.is_unique:
        raise ValueError('数据为空或包含重复列名。')
    return df


def frame_fingerprint(frame: pd.DataFrame) -> str:
    # JSON float round-trip is stable across project CSV reloads (17-digit export).
    blob = frame.to_json(orient='split', index=False, double_precision=15, force_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _leakage_components(frame: pd.DataFrame, config: DataConfig) -> np.ndarray:
    """Union group IDs and identical input vectors, including conflicting targets."""
    parent = list(range(len(frame)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen_x, seen_group = {}, {}
    for i, row in enumerate(frame[config.inputs].itertuples(index=False, name=None)):
        if row in seen_x:
            union(i, seen_x[row])
        else:
            seen_x[row] = i
    if config.group:
        for i, group in enumerate(frame[config.group].astype(str)):
            if group in seen_group:
                union(i, seen_group[group])
            else:
                seen_group[group] = i
    return np.array([find(i) for i in range(len(frame))])


def _split_components(components: np.ndarray, ratios: list[float], seed: int) -> dict[str, list[int]]:
    unique = np.unique(components)
    if len(unique) < 3:
        raise ValueError('去重/分组后至少需要3个独立组，才能无泄漏划分训练、验证和测试集。')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    # Largest components first, random tie breaks. Greedy normalized deficit balances rows.
    groups = [np.flatnonzero(components == g).tolist() for g in unique]
    groups.sort(key=len, reverse=True)
    desired = np.asarray(ratios) * len(components)
    result = [[], [], []]
    for pos, group in enumerate(groups):
        empty = [i for i, rows in enumerate(result) if not rows]
        remaining = len(groups) - pos
        if remaining == len(empty):
            k = empty[0]
        else:
            deficit = desired - np.array([len(rows) for rows in result])
            k = int(np.argmax(deficit))
        result[k].extend(group)
    if any(len(rows) < 2 for rows in result):
        raise ValueError('每个划分至少需要2行。请增加数据或调整分组/比例。')
    return {name: sorted(rows) for name, rows in zip(('train', 'validation', 'test'), result)}


def prepare_data(raw: pd.DataFrame, config: DataConfig) -> PreparedData:
    config.validate()
    columns = config.inputs + config.targets + ([config.group] if config.group else [])
    if not raw.columns.is_unique or not set(columns).issubset(raw.columns):
        raise ValueError('所选列不存在或源数据包含重复列名。')
    frame = raw[columns].copy()
    for col in config.inputs + config.targets:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').astype(np.float64)
    numeric = frame[config.inputs + config.targets].to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(numeric)
    if invalid.any():
        rows, cols = np.where(invalid)
        examples = [f'第{int(r)+2}行 {columns[int(c)]}' for r, c in zip(rows[:8], cols[:8])]
        raise ValueError('数值列存在缺失、非数值或无穷值，请先修复：' + '；'.join(examples))
    if config.group:
        if frame[config.group].isna().any():
            raise ValueError('分组列不可有空值。')
        frame[config.group] = frame[config.group].astype(str)
    exact_cols = config.inputs + config.targets + ([config.group] if config.group else [])
    duplicate_count = int(frame.duplicated(subset=exact_cols).sum())
    if config.drop_duplicates:
        frame = frame.drop_duplicates(subset=exact_cols)
    frame = frame.reset_index(drop=True)
    if len(frame) < 12:
        raise ValueError('有效数据至少需要12行；科研使用通常需要更多样本。')
    components = _leakage_components(frame, config)
    splits = _split_components(components, [config.train_ratio, config.val_ratio, config.test_ratio], config.seed)
    report = {
        'source_rows': len(raw), 'retained_rows': len(frame), 'exact_duplicates': duplicate_count,
        'duplicates_removed': duplicate_count if config.drop_duplicates else 0,
        'independent_components': int(len(np.unique(components))),
        'split_counts': {name: len(rows) for name, rows in splits.items()},
        'repeated_input_rows': int(frame.duplicated(subset=config.inputs, keep=False).sum()),
        'conflicting_input_rows': int(frame.groupby(config.inputs, sort=False)[config.targets].transform('nunique').gt(1).any(axis=1).sum()),
        'warning': '相同自变量（即使目标不同）及同组样本始终位于同一划分；比例可能因组大小偏离。',
    }
    return PreparedData(frame, config, splits, frame_fingerprint(frame), report)


def validate_splits(data: PreparedData):
    flat = sum([data.splits.get(k, []) for k in ('train', 'validation', 'test')], [])
    if len(flat) != len(data.frame) or sorted(flat) != list(range(len(data.frame))):
        raise ValueError('划分索引必须无重叠且完整覆盖数据。')
    if any(len(data.splits[k]) < 2 for k in ('train', 'validation', 'test')):
        raise ValueError('每个划分至少2行。')
    comps = _leakage_components(data.frame, data.config)
    seen = set()
    for name in ('train', 'validation', 'test'):
        current = set(comps[data.splits[name]])
        if seen & current:
            raise ValueError('项目划分存在组或重复自变量泄漏。')
        seen |= current


@dataclass
class TrainConfig:
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    epochs: int = 500
    patience: int = 35
    learning_rate: float = 0.001
    batch_size: int = 64
    seed: int = 42
    activations: list[str] = field(default_factory=list)

    def validate(self, n_inputs=1, n_outputs=1):
        if not 1 <= len(self.hidden) <= 8 or any(type(n) is not int or not 1 <= n <= 2048 for n in self.hidden):
            raise ValueError('隐藏层需要1-8层，每层1-2048个神经元。')
        if self.activations and (len(self.activations) != len(self.hidden) or any(a not in ('relu','tanh','sigmoid','gelu','leaky_relu') for a in self.activations)):
            raise ValueError('每层激活函数必须与隐藏层逐项对应。')
        sizes = [n_inputs] + self.hidden + [n_outputs]
        if sum((a + 1) * b for a, b in zip(sizes, sizes[1:])) > 5_000_000:
            raise ValueError('P0 模型参数上限为500万。')
        if not 1 <= self.epochs <= 100000 or not 1 <= self.patience <= 10000 or not 1 <= self.batch_size <= 100000:
            raise ValueError('训练轮数、早停耐心值或批量大小超出范围。')
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('学习率必须大于0且不超过1。')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32-1:
            raise ValueError('随机种子超出范围。')


@dataclass
class Scale:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, a):
        mean, scale = a.mean(axis=0), a.std(axis=0)
        return cls(mean, np.where(scale < 1e-12, 1.0, scale))

    def transform(self, a):
        return (a - self.mean) / self.scale

    def inverse(self, a):
        return a * self.scale + self.mean

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, d, size):
        mean, scale = np.asarray(d['mean'], float), np.asarray(d['scale'], float)
        if mean.shape != (size,) or scale.shape != (size,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError('缩放器格式无效。')
        return cls(mean, scale)


def make_model(n_inputs, n_outputs, config):
    import torch
    config.validate(n_inputs, n_outputs)
    sizes = [n_inputs] + config.hidden + [n_outputs]
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            activation = config.activations[i] if config.activations else 'relu'
            layers.append({'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid, 'gelu': torch.nn.GELU, 'leaky_relu': torch.nn.LeakyReLU}[activation]())
    return torch.nn.Sequential(*layers)


@dataclass
class TrainedModel:
    data: PreparedData
    config: TrainConfig
    model: object
    x_scale: Scale
    y_scale: Scale
    history: list[dict]
    metrics: dict
    best_epoch: int

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.data.config.inputs) or not np.isfinite(X).all():
            raise ValueError('预测输入维度或数值无效。')
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.as_tensor(self.x_scale.transform(X), dtype=torch.float32)).cpu().numpy()
        result = self.y_scale.inverse(out)
        if not np.isfinite(result).all():
            raise ValueError('模型预测出现非有限值。')
        return result


def train_model(data: PreparedData, config: TrainConfig, cancel=None, progress: Callable | None = None, snapshot=None) -> TrainedModel:
    import torch
    config.validate(len(data.config.inputs), len(data.config.targets))
    validate_splits(data)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)
    tr, va = data.splits['train'], data.splits['validation']
    # The held-out test labels are evaluated only after training; even a finite
    # label outside float32 range must not affect fitting or early stopping.
    train_frame, validation_frame = data.frame.iloc[tr], data.frame.iloc[va]
    train_x = train_frame[data.config.inputs].to_numpy(dtype=np.float64)
    train_y = train_frame[data.config.targets].to_numpy(dtype=np.float64)
    xs, ys = Scale.fit(train_x), Scale.fit(train_y)
    X = torch.tensor(xs.transform(train_x), dtype=torch.float32)
    Y = torch.tensor(ys.transform(train_y), dtype=torch.float32)
    X_val = torch.tensor(xs.transform(validation_frame[data.config.inputs].to_numpy(dtype=np.float64)), dtype=torch.float32)
    Y_val = torch.tensor(ys.transform(validation_frame[data.config.targets].to_numpy(dtype=np.float64)), dtype=torch.float32)
    if not all(torch.isfinite(values).all() for values in (X, Y, X_val, Y_val)):
        raise ValueError('缩放后数值超出float32范围，请检查数据量级。')
    model = make_model(X.shape[1], Y.shape[1], config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(config.seed)
    best, best_epoch, best_weights, stale, history = float('inf'), 0, None, 0, []
    for epoch in range(1, config.epochs + 1):
        check_cancel(cancel)
        model.train()
        order = rng.permutation(len(tr))
        train_sum = 0.0
        for start in range(0, len(order), config.batch_size):
            check_cancel(cancel)
            batch = order[start:start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X[batch]), Y[batch])
            if not torch.isfinite(loss):
                raise ValueError('训练损失非有限值，请降低学习率并检查数据。')
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val), Y_val).item())
        if not math.isfinite(val_loss):
            raise ValueError('验证损失非有限值。')
        history.append({'epoch': epoch, 'train_loss': train_sum / len(tr), 'validation_loss': val_loss})
        if val_loss < best - 1e-8:
            best, best_epoch, stale = val_loss, epoch, 0
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if snapshot and (epoch == 1 or epoch % 5 == 0 or stale >= config.patience or epoch == config.epochs):
            snapshot({'kind':'loss', 'family':'ann', 'epoch':epoch, 'best_epoch':best_epoch, 'early_stopped':stale >= config.patience, 'history':list(history), 'label':'快速固定划分ANN'})
        if progress and (epoch == 1 or epoch % 5 == 0):
            progress(f'训练 {epoch}/{config.epochs}，验证损失 {val_loss:.5g}，最佳轮次 {best_epoch}')
        if stale >= config.patience:
            break
    check_cancel(cancel)
    model.load_state_dict(best_weights)
    trained = TrainedModel(data, config, model, xs, ys, history, {}, best_epoch)
    for name, indices in data.splits.items():
        actual, predicted = data.Y[indices], trained.predict(data.X[indices])
        trained.metrics[name] = {
            target: {'R2': float(r2_score(actual[:, j], predicted[:, j], force_finite=False)) if np.ptp(actual[:, j]) > 0 else None,
                     'MSE': float(mean_squared_error(actual[:, j], predicted[:, j])),
                     'MAE': float(mean_absolute_error(actual[:, j], predicted[:, j])),
                     'RMSE': float(math.sqrt(mean_squared_error(actual[:, j], predicted[:, j])))}
            for j, target in enumerate(data.config.targets)
        }
    return trained


@dataclass
class Variable:
    name: str
    lower: float
    upper: float
    kind: str = 'continuous'
    fixed: bool = False


@dataclass
class OptimizeConfig:
    variables: list[Variable]
    directions: dict[str, str]
    population: int = 80
    generations: int = 60
    seed: int = 42
    allow_extrapolation: bool = False

    def validate(self, trained):
        expected = trained.data.config.inputs
        if any(c in {'candidate_id', 'generation', 'pareto'} or c.startswith('predicted::') for c in expected) or any(prediction_column(t) in expected for t in trained.data.config.targets):
            raise ValueError('优化输入列名与保留结果列名冲突。')
        if [v.name for v in self.variables] != expected:
            raise ValueError('优化变量必须与模型自变量的名称和顺序完全一致。')
        if set(self.directions) != set(trained.data.config.targets) or any(d not in ('min', 'max') for d in self.directions.values()):
            raise ValueError('每个目标必须明确选择最小化或最大化。')
        if not 4 <= self.population <= 2000 or not 1 <= self.generations <= 10000:
            raise ValueError('种群范围4-2000，迭代代数范围1-10000。')
        if self.population * self.generations > 20000:
            raise ValueError('P0 单次评估预算上限20000。')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError('随机种子超出范围。')
        fit_indices = getattr(trained, 'fit_indices', trained.data.splits['train'])
        X = trained.data.X[fit_indices]
        for j, v in enumerate(self.variables):
            if v.kind not in ('continuous', 'integer') or not np.isfinite([v.lower, v.upper]).all() or v.lower > v.upper:
                raise ValueError(f'{v.name} 的类型或上下界无效。')
            if v.fixed and v.lower != v.upper:
                raise ValueError(f'{v.name} 固定输入需要上下界相等。')
            if v.kind == 'integer' and (v.lower != round(v.lower) or v.upper != round(v.upper)):
                raise ValueError(f'{v.name} 整数变量的上下界必须是整数。')
            if not self.allow_extrapolation and (v.lower < X[:, j].min() - 1e-10 or v.upper > X[:, j].max() + 1e-10):
                raise ValueError(f'{v.name} 超出训练集观测范围；默认禁止外推。')


@dataclass
class OptimizationResult:
    config: OptimizeConfig
    candidates: pd.DataFrame
    evaluations: int


def prediction_column(name: str) -> str:
    return f'predicted::{name}'


def optimize_model(trained: TrainedModel, config: OptimizeConfig, cancel=None, progress=None, snapshot=None) -> OptimizationResult:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.core.repair import Repair
    config.validate(trained)
    targets = trained.data.config.targets
    signs = np.array([1 if config.directions[t] == 'min' else -1 for t in targets])
    active = [j for j, v in enumerate(config.variables) if v.lower < v.upper and not v.fixed]
    records_x, records_y, records_g = [], [], []
    generation = [1]
    last_snapshot = [0.]

    def evaluate(decisions):
        check_cancel(cancel)
        full = np.tile([v.lower for v in config.variables], (len(decisions), 1)).astype(float)
        if active:
            full[:, active] = decisions
        for j, v in enumerate(config.variables):
            if v.kind == 'integer':
                full[:, j] = np.clip(np.rint(full[:, j]), v.lower, v.upper)
        y = trained.predict(full)
        records_x.append(full.copy())
        records_y.append(y.copy())
        records_g.extend([generation[0]] * len(full))
        return y * signs

    if active:
        class IntegerRepair(Repair):
            def _do(self, problem, X, **kwargs):
                # Repair the actual genotype before evaluation and duplicate elimination.
                for k, j in enumerate(active):
                    if config.variables[j].kind == 'integer':
                        X[:, k] = np.clip(np.rint(X[:, k]), config.variables[j].lower, config.variables[j].upper)
                return X
        class SurrogateProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(active), n_obj=len(targets), xl=[config.variables[j].lower for j in active], xu=[config.variables[j].upper for j in active])
            def _evaluate(self, x, out, *args, **kwargs):
                out['F'] = evaluate(x)
        class Monitor(Callback):
            def notify(self, algorithm):
                check_cancel(cancel)
                generation[0] = int(algorithm.n_gen) + 1
                if snapshot and (time.monotonic() - last_snapshot[0] > .15 or algorithm.n_gen == config.generations):
                    last_snapshot[0] = time.monotonic()
                    active_x = algorithm.pop.get('X')
                    full_x = np.tile([v.lower for v in config.variables], (len(active_x), 1)).astype(float)
                    full_x[:, active] = active_x
                    snapshot({'kind': 'optimization', 'generation': int(algorithm.n_gen), 'generations': config.generations, 'evaluations': int(algorithm.evaluator.n_eval), 'X': full_x.tolist(), 'Y': (algorithm.pop.get('F') * signs).tolist(), 'directions': config.directions, 'inputs': trained.data.config.inputs, 'targets': targets, 'pareto_Y': (algorithm.opt.get('F') * signs).tolist()})
                if progress:
                    progress(f'NSGA-II {algorithm.n_gen}/{config.generations}代，已评估 {algorithm.evaluator.n_eval} 个候选')
        minimize(SurrogateProblem(), NSGA2(pop_size=config.population, repair=IntegerRepair()), ('n_gen', config.generations), seed=config.seed, callback=Monitor(), verbose=False)
    else:
        evaluate(np.zeros((1, 0)))
    check_cancel(cancel)
    xx, yy = np.vstack(records_x), np.vstack(records_y)
    df = pd.DataFrame(xx, columns=trained.data.config.inputs)
    for j, target in enumerate(targets):
        if prediction_column(target) in df.columns:
            raise ValueError('输入列名与预测输出列名冲突，请重命名。')
        df[prediction_column(target)] = yy[:, j]
    df['generation'] = records_g
    evaluations = len(df)
    df = df.drop_duplicates(subset=trained.data.config.inputs).reset_index(drop=True)
    objectives = df[[prediction_column(t) for t in targets]].to_numpy() * signs
    # Bounded-memory incremental archive: no dense N×N dominance matrix.
    front = []
    for i, point in enumerate(objectives):
        if i % 32 == 0:
            check_cancel(cancel)
        if front:
            current = objectives[front]
            if np.any(np.all(current <= point, axis=1) & np.any(current < point, axis=1)):
                continue
            dominated = np.all(point <= current, axis=1) & np.any(point < current, axis=1)
            front = [idx for idx, remove in zip(front, dominated) if not remove]
        front.append(i)
    check_cancel(cancel)
    df['pareto'] = False
    df.loc[front, 'pareto'] = True
    df.insert(0, 'candidate_id', np.arange(1, len(df) + 1))
    return OptimizationResult(config, df, evaluations)


def filter_candidates(frame: pd.DataFrame, ranges: dict[str, tuple[float | None, float | None]], pareto_only=False) -> pd.DataFrame:
    result = frame[frame['pareto']].copy() if pareto_only else frame.copy()
    for col, (low, high) in ranges.items():
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'不能筛选列：{col}')
        if (low is not None and not math.isfinite(low)) or (high is not None and not math.isfinite(high)):
            raise ValueError('筛选上下限必须是有限数值。')
        if low is not None and high is not None and low > high:
            raise ValueError('筛选下限不能大于上限。')
        if low is not None:
            result = result[result[col] >= low]
        if high is not None:
            result = result[result[col] <= high]
    return result


def export_candidates(path: str | Path, frame: pd.DataFrame, trained: TrainedModel, optimization: OptimizationResult):
    path = Path(path)
    # Neutralize spreadsheet formulas in headers as well as cells, without altering internal data.
    safe = frame.copy()
    safe.columns = [('\'' + str(c)) if str(c).startswith(('=', '+', '-', '@', '\t', '\r')) else str(c) for c in safe.columns]
    safe.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.17g')
    composite = hasattr(trained, 'target_models')
    provenance = {'schema_version': 3 if composite else 1, 'app': 'Daylight Workbench 0.3.0',
                  'notice': '代理模型预测候选，不是经 Radiance / EnergyPlus 验证的物理仿真结果。',
                  'dataset_sha256': trained.data.fingerprint, 'data_config': asdict(trained.data.config),
                  'test_metrics': trained.metrics['test'], 'optimization_config': asdict(optimization.config),
                  'exported_rows': len(frame), 'evaluations': optimization.evaluations,
                  'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if composite:
        provenance['model_family'] = 'independent_targets'
        provenance['target_order'] = list(trained.data.config.targets)
        provenance['target_models'] = {target: {'family': trained.target_models[target].family,
                                              'spec': asdict(trained.target_models[target].spec),
                                              'best_epoch': trained.target_models[target].best_epoch,
                                              'refit_budget': trained.target_models[target].search_metadata.get('refit_budget')}
                                       for target in trained.data.config.targets}
        provenance['actual_fit_indices'] = trained.fit_indices
        provenance['actual_training_support'] = trained.support_bounds
        provenance['search'] = trained.search_metadata
    else:
        provenance['training_config'] = asdict(trained.config)
        provenance['best_epoch'] = trained.best_epoch
    if hasattr(trained, 'spec'):
        provenance['model_spec'] = asdict(trained.spec)
        provenance['model_family'] = trained.family
        provenance['actual_fit_indices'] = trained.fit_indices
        provenance['actual_training_support'] = trained.support_bounds
        provenance['search'] = trained.search_metadata
    path.with_suffix(path.suffix + '.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
