import hashlib
import io
import json
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest

from daylight_workbench.core import (Cancelled, DataConfig, TrainConfig, Variable, OptimizeConfig,
                                    prepare_data, train_model, optimize_model, read_dataset,
                                    filter_candidates, export_candidates)
from daylight_workbench.demo import make_demo
from daylight_workbench.project import load_project, save_project


@pytest.fixture(scope='module')
def trained():
    frame = make_demo(90)
    columns = list(frame.columns)
    data = prepare_data(frame, DataConfig(columns[:10], columns[10:]))
    return train_model(data, TrainConfig([8], epochs=3, patience=2))


def test_dataset_io_and_missing(trained, tmp_path):
    frame = trained.data.frame
    for ext in ['csv', 'xlsx']:
        path = tmp_path / ('data.' + ext)
        if ext == 'csv':
            frame.to_csv(path, index=False)
        else:
            frame.to_excel(path, index=False)
        loaded = read_dataset(path)
        np.testing.assert_allclose(loaded, frame, atol=1e-12)
    invalid = frame.copy()
    invalid.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='数值'):
        prepare_data(invalid, trained.data.config)


def test_exact_duplicate_keep_or_explicit_drop(trained):
    frame = pd.concat([trained.data.frame, trained.data.frame.iloc[:3]], ignore_index=True)
    config = DataConfig(trained.data.config.inputs, trained.data.config.targets)
    kept = prepare_data(frame, config)
    assert len(kept.frame) == len(frame)
    assert kept.report['exact_duplicates'] == 3
    assert kept.report['conflicting_input_rows'] == 0
    config.drop_duplicates = True
    dropped = prepare_data(frame, config)
    assert len(dropped.frame) == len(frame) - 3


def test_cancellation_before_training_and_optimization(trained):
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        train_model(trained.data, TrainConfig([4], epochs=5), event)
    x = trained.data.X[trained.data.splits['train']]
    config = OptimizeConfig([Variable(c, float(x[:, j].min()), float(x[:, j].max())) for j, c in enumerate(trained.data.config.inputs)],
                            {c: 'min' for c in trained.data.config.targets}, population=4, generations=1)
    with pytest.raises(Cancelled):
        optimize_model(trained, config, event)


def test_bounds_direction_and_budget_guards(trained):
    x = trained.data.X[trained.data.splits['train']]
    variables = [Variable(c, float(x[:, j].min()), float(x[:, j].max())) for j, c in enumerate(trained.data.config.inputs)]
    config = OptimizeConfig(variables, {c: 'min' for c in trained.data.config.targets})
    config.validate(trained)
    variables[0].lower -= 1
    with pytest.raises(ValueError, match='外推'):
        config.validate(trained)
    variables[0].lower += 1
    config.generations = 300
    with pytest.raises(ValueError, match='20000'):
        config.validate(trained)


def test_project_integrity_and_member_path_guard(trained, tmp_path):
    path = tmp_path / 'valid.dwb'
    save_project(path, trained)
    original = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(original)) as z:
        blobs = {n: z.read(n) for n in z.namelist()}
    for bad, change in [('path', {'../../escape': b'bad'}), ('tamper', {'data.csv': b'bad'})]:
        target = tmp_path / (bad + '.dwb')
        with zipfile.ZipFile(target, 'w') as z:
            for name, value in {**blobs, **change}.items():
                z.writestr(name, value)
        with pytest.raises(ValueError):
            load_project(target)
    assert not (tmp_path.parent / 'escape').exists()


def test_filter_export_and_formula_header_escape(trained, tmp_path):
    from daylight_workbench.core import OptimizationResult
    frame = pd.DataFrame({'candidate_id': [1, 2, 3], 'x': [1., 2., 3.], 'predicted::y': [5., 6., 7.], 'pareto': [True, False, True]})
    filtered = filter_candidates(frame, {'x': (2, None)}, pareto_only=True)
    assert filtered.candidate_id.tolist() == [3]
    with pytest.raises(ValueError):
        filter_candidates(frame, {'x': (3, 2)})
    result = OptimizationResult(OptimizeConfig([], {}), frame, 3)
    frame['=FORMULA'] = 0
    path = tmp_path / 'export.csv'
    export_candidates(path, frame, trained, result)
    assert "'=FORMULA" in path.read_text(encoding='utf-8-sig').splitlines()[0]
    provenance = json.loads(path.with_suffix('.csv.provenance.json').read_text(encoding='utf-8'))
    assert provenance['csv_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance['dataset_sha256'] == trained.data.fingerprint
