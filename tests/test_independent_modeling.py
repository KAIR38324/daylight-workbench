"""Scientific and adversarial regressions for independent target models.

Synthetic fixtures only. Existing shared-mode tests keep their exact old
assertions; these tests independently prove the new default contract.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import threading
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pandas as pd
import pytest

from daylight_workbench import modeling
from daylight_workbench.core import (Cancelled, DataConfig, OptimizeConfig, Scale,
    TrainConfig, Variable, _leakage_components, export_candidates, optimize_model,
    prepare_data, train_model)
from daylight_workbench.modeling import (CompositeSurrogate, LayerSpec, ModelSpec,
    SearchConfig, make_search_plan, run_model_search)
from daylight_workbench.project import load_project, save_project


def tiny_spec(family):
    if family == 'ann':
        return ModelSpec('ann', {'epochs': 3, 'patience': 1, 'batch_size': 16}, [LayerSpec(4, 'tanh')])
    if family == 'rf':
        return ModelSpec('rf', {'n_estimators': 3, 'max_depth': 3}, [])
    return ModelSpec('xgboost', {'n_estimators': 3, 'max_depth': 2, 'patience': 1}, [])


@pytest.fixture
def grouped_data():
    rows = []
    for group in range(36):
        for repeat in range(2):
            x, z = group * .1 + repeat * .023, np.sin(group) + repeat * .04
            # Bind two distinct named groups via duplicate inputs, even with
            # conflicting labels. Never split these components independently.
            if group == 1 and repeat == 0:
                x, z = 0., 0.
            rows.append({'x': x, 'z': z, 'a': 1.4*x + z, 'b': np.cos(x) + repeat*.2,
                         'group': f'g{group}'})
    return prepare_data(pd.DataFrame(rows), DataConfig(['x', 'z'], ['b', 'a'], group='group', seed=8))


@pytest.fixture(scope='module')
def mixed_search():
    rng = np.random.default_rng(5)
    x, z = rng.uniform(-3, 3, 160), rng.uniform(-1, 1, 160)
    raw = pd.DataFrame({'x': x, 'z': z, 'linear': 2*x + .1*z,
                        'step': (x > 0).astype(float)*10, 'wave': np.sin(3*x) + .2*z})
    data = prepare_data(raw, DataConfig(['x', 'z'], ['linear', 'step', 'wave'],
                        directions={'linear': 'max', 'step': 'min', 'wave': 'min'}))
    candidates = [ModelSpec('ann', {'epochs': 160, 'patience': 25, 'learning_rate': .02,
                                  'batch_size': 64}, [LayerSpec(8, 'leaky_relu')]),
                  ModelSpec('rf', {'n_estimators': 10, 'max_depth': 4}, []),
                  ModelSpec('xgboost', {'n_estimators': 40, 'max_depth': 2,
                                       'learning_rate': .2, 'patience': 10}, [])]
    return run_model_search(data, SearchConfig(candidates, folds=2, max_fits=21, cpu_threads=1))


def rewrite_archive(source, destination, change_meta=None, change_blobs=None):
    with zipfile.ZipFile(source) as archive:
        blobs = {name: archive.read(name) for name in archive.namelist()}
    if change_meta:
        metadata = json.loads(blobs['metadata.json'])
        change_meta(metadata)
        blobs['metadata.json'] = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode()
    if change_blobs:
        change_blobs(blobs)
    manifest = json.loads(blobs['manifest.json'])
    manifest['sha256'] = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items() if name != 'manifest.json'}
    blobs['manifest.json'] = json.dumps(manifest).encode()
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in blobs.items():
            archive.writestr(name, blob)


def test_default_budget_counts_all_targets_and_shared_mode_is_retained(grouped_data):
    specs = [tiny_spec(name) for name in ('ann', 'rf', 'xgboost')]
    cfg = SearchConfig(specs, folds=3, max_fits=20, cpu_threads=1)
    assert cfg.target_mode == 'independent'
    plan = make_search_plan(grouped_data, cfg)
    assert plan['fit_budget'] == 20  # 2 targets * (3 candidates * 3 folds + 1 refit)
    assert plan['candidate_fold_jobs'] == 18
    assert plan['per_candidate_fits'] == [6, 6, 6]
    assert plan['per_target_fit_budget'] == 10
    assert plan['refit_fit_budget'] == 2
    shared = make_search_plan(grouped_data, SearchConfig(specs, folds=3, max_fits=14,
                                                       cpu_threads=1, target_mode='shared'))
    assert shared['fit_budget'] == 14
    assert shared['per_candidate_fits'] == [3, 3, 6]
    assert shared['folds'] == plan['folds'] and shared['test'] == plan['test']
    with pytest.raises(ValueError, match='20次实际拟合'):
        make_search_plan(grouped_data, SearchConfig(specs, folds=3, max_fits=19))


def test_legacy_configuration_has_shared_meaning_but_new_config_is_independent():
    old = asdict(SearchConfig([tiny_spec('rf')], target_mode='shared'))
    del old['target_mode']
    assert SearchConfig.from_dict(old).target_mode == 'shared'
    new = SearchConfig([tiny_spec('rf')])
    assert SearchConfig.from_dict(asdict(new)).target_mode == 'independent'


def test_actual_mixed_family_winners_order_and_independent_oracle(mixed_search):
    result, model = mixed_search, mixed_search.selected
    assert isinstance(model, CompositeSurrogate)
    assert len({local.family for local in model.target_models.values()}) >= 2
    assert set(result.selected_index) == set(model.data.config.targets)
    X = model.data.X[:12]
    expected = np.column_stack([model.target_models[t].predict(X)[:, 0] for t in model.data.config.targets])
    np.testing.assert_array_equal(model.predict(X), expected)
    # Dict insertion order is adversarial, never a scientific output contract.
    model.target_models = dict(reversed(list(model.target_models.items())))
    np.testing.assert_array_equal(model.predict(X), expected)
    for target in model.data.config.targets:
        record = result.target_results[target]
        oracle = min(range(len(record['candidates'])), key=lambda i: (record['candidates'][i]['score_mean'], i))
        assert record['selected_index'] == oracle == result.selected_index[target]
        assert model.target_models[target].spec == result.config.candidates[oracle]
        for candidate in record['candidates']:
            values = []
            for fold in candidate['folds']:
                assert set(fold['metrics']) == {target}
                value = fold['metrics'][target]['RMSE'] / fold['normalization_std'][0]
                assert fold['score'] == pytest.approx(value, rel=1e-12)
                values.append(value)
            assert candidate['score_mean'] == pytest.approx(np.mean(values), rel=1e-12)
            assert candidate['score_std'] == pytest.approx(np.std(values), rel=1e-12)


def test_all_fits_are_scalar_common_fold_safe_and_preprocessing_is_training_only(grouped_data, monkeypatch):
    fitted, heldout_reads = [], []
    original_fit, original_rows = modeling._fit, modeling._rows
    targets = grouped_data.config.targets
    test = set(grouped_data.splits['test'])
    def audited_fit(data, spec, outer, inner, validation, seed, threads, cancel, progress, snapshot, context, fixed_budget=None):
        assert len(data.config.targets) == 1
        assert not test.intersection(outer + inner + validation)
        model = original_fit(data, spec, outer, inner, validation, seed, threads, cancel,
                             progress, snapshot, context, fixed_budget)
        idx = outer if spec.family == 'rf' or fixed_budget is not None else inner
        assert model.fit_indices == idx
        for scale, values in ((model.x_scale, data.frame.iloc[idx][data.config.inputs].to_numpy(float)),
                              (model.y_scale, data.frame.iloc[idx][data.config.targets].to_numpy(float))):
            expected = Scale.fit(values)
            np.testing.assert_array_equal(scale.mean, expected.mean)
            np.testing.assert_array_equal(scale.scale, expected.scale)
        fitted.append((data.config.targets[0], spec.family, context.copy(), idx))
        return model
    def audited_rows(data, idx, columns):
        if test.intersection(idx) and set(columns).intersection(targets):
            # No target's test labels may be read until ALL winners have refit.
            assert len([entry for entry in fitted if entry[2].get('stage') == 'refit']) == len(targets)
            heldout_reads.append(list(idx))
        return original_rows(data, idx, columns)
    monkeypatch.setattr(modeling, '_fit', audited_fit)
    monkeypatch.setattr(modeling, '_rows', audited_rows)
    cfg = SearchConfig([tiny_spec(name) for name in ('ann', 'rf', 'xgboost')],
                       folds=2, max_fits=14, cpu_threads=1)
    result = run_model_search(grouped_data, cfg)
    components = _leakage_components(grouped_data.frame, grouped_data.config)
    assert components[0] == components[2]
    assert heldout_reads == [grouped_data.splits['test']]
    for fold in result.plan['folds']:
        for left, right in (('train', 'validation'), ('inner_train', 'inner_validation')):
            assert set(components[fold[left]]).isdisjoint(components[fold[right]])
    for target in targets:
        for candidate in result.target_results[target]['candidates']:
            for fi, fold in enumerate(candidate['folds']):
                expected = result.plan['folds'][fi]
                assert fold['fit_indices'] == expected['train' if candidate['family'] == 'rf' else 'inner_train']
    assert result.selected.fit_indices == result.plan['development']


@pytest.mark.parametrize('family', ['ann', 'rf', 'xgboost'])
@pytest.mark.parametrize('mode', ['cv', 'fixed'])
def test_extreme_locked_labels_never_change_any_target_selection(grouped_data, family, mode):
    cfg = SearchConfig([tiny_spec(family)], mode=mode, folds=2, max_fits=6, cpu_threads=1)
    baseline = run_model_search(grouped_data, cfg)
    changed = deepcopy(grouped_data)
    changed.frame.loc[changed.splits['test'], changed.config.targets] = 1e40
    mutated = run_model_search(changed, cfg)
    assert mutated.selected_index == baseline.selected_index
    assert mutated.candidates == baseline.candidates
    for target in changed.config.targets:
        assert mutated.selected.target_models[target].history == baseline.selected.target_models[target].history
    np.testing.assert_array_equal(mutated.selected.predict(grouped_data.X), baseline.selected.predict(grouped_data.X))
    assert all(metrics['MSE'] > 1e79 for metrics in mutated.selected.metrics['test'].values())


@pytest.mark.parametrize('stage', ['target_selected', 'second_refit'])
def test_cancellation_never_returns_partial_composite_or_reads_test(grouped_data, monkeypatch, stage):
    cancel, seen = threading.Event(), []
    original_rows = modeling._rows
    def rows(data, indices, columns):
        assert not (set(indices).intersection(data.splits['test']) and set(columns).intersection(data.config.targets))
        return original_rows(data, indices, columns)
    monkeypatch.setattr(modeling, '_rows', rows)
    def snapshot(event):
        seen.append(event)
        if (stage == 'target_selected' and event.get('stage') == 'target_selected') or (
                stage == 'second_refit' and event.get('stage') == 'refit' and event.get('target_index') == 2
                and event.get('kind') == 'fit' and event.get('tree') == 1):
            cancel.set()
    with pytest.raises(Cancelled):
        run_model_search(grouped_data, SearchConfig([tiny_spec('rf')], folds=2, max_fits=6, cpu_threads=1),
                         cancel=cancel, snapshot=snapshot)
    assert cancel.is_set()
    assert not any(event.get('stage') == 'complete' for event in seen)


def test_mixed_composite_safe_roundtrip_optimizer_and_complete_provenance(mixed_search, tmp_path):
    model = mixed_search.selected
    cfg = OptimizeConfig([Variable(name, *model.support_bounds[name]) for name in model.data.config.inputs],
                         model.data.config.directions, population=6, generations=2)
    optimized = optimize_model(model, cfg)
    path = tmp_path / 'mixed.dwb'
    save_project(path, model, optimized)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['schema_version'] == 3
        assert all('/' not in name and '..' not in name for name in archive.namelist())
        assert all(not name.endswith(('.pkl', '.pickle')) for name in archive.namelist())
    loaded, restored = load_project(path)
    assert isinstance(loaded, CompositeSurrogate)
    np.testing.assert_array_equal(loaded.predict(model.data.X), model.predict(model.data.X))
    assert loaded.search_metadata == model.search_metadata
    assert restored.candidates.candidate_id.tolist() == optimized.candidates.candidate_id.tolist()
    np.testing.assert_allclose(restored.candidates.drop(columns='pareto').to_numpy(float),
                               optimized.candidates.drop(columns='pareto').to_numpy(float), rtol=0, atol=0)
    assert loaded.metrics == model.metrics
    exported = tmp_path / 'candidates.csv'
    export_candidates(exported, restored.candidates, loaded, restored)
    provenance = json.loads(exported.with_suffix('.csv.provenance.json').read_text())
    assert provenance['model_family'] == 'independent_targets'
    assert provenance['target_order'] == model.data.config.targets
    assert list(provenance['target_models']) == model.data.config.targets
    assert provenance['search']['selected_index'] == mixed_search.selected_index
    # Bounds use COMMON final fitting support, not a stale first-target split.
    assert loaded.support_bounds == model.support_bounds
    with pytest.raises(ValueError, match='超出训练集观测范围'):
        OptimizeConfig([Variable('x', -1e6, 1e6), Variable('z', *model.support_bounds['z'])],
                        model.data.config.directions).validate(loaded)


@pytest.mark.parametrize('family', ['ann', 'rf', 'xgboost'])
def test_each_scalar_backend_composite_archive_roundtrips(grouped_data, family, tmp_path):
    model = run_model_search(grouped_data, SearchConfig([tiny_spec(family)], folds=2,
                             max_fits=6, cpu_threads=1)).selected
    path = tmp_path / (family + '.dwb')
    save_project(path, model)
    loaded, _ = load_project(path)
    np.testing.assert_array_equal(loaded.predict(grouped_data.X), model.predict(grouped_data.X))
    assert all(local.family == family for local in loaded.target_models.values())


@pytest.mark.parametrize('mutation', ['target_order', 'model_order', 'missing_target', 'shared_fit',
    'test_fold', 'scaler', 'selected_index', 'selected_spec', 'score', 'history', 'extra_path', 'extra_model'])
def test_composite_rejects_rehashed_semantic_tampering(mixed_search, tmp_path, mutation):
    source, bad = tmp_path/'original.dwb', tmp_path/'bad.dwb'
    save_project(source, mixed_search.selected)
    def change(meta):
        if mutation == 'target_order': meta['target_order'].reverse()
        elif mutation == 'model_order': meta['models'].reverse()
        elif mutation == 'missing_target': meta['models'].pop()
        elif mutation == 'shared_fit': meta['models'][1]['fit_indices'] = meta['splits']['train']
        elif mutation == 'test_fold': meta['search_metadata']['plan']['folds'][0]['train'][0] = meta['splits']['test'][0]
        elif mutation == 'scaler': meta['models'][0]['x_scale']['mean'][0] += 100.
        elif mutation == 'selected_index': meta['search_metadata']['selected_index']['linear'] = 99
        elif mutation == 'selected_spec': meta['models'][0]['spec']['params']['learning_rate'] = .9
        elif mutation == 'score': meta['search_metadata']['target_results']['wave']['candidates'][0]['score_mean'] = -1.
        elif mutation == 'history': meta['models'][0]['best_epoch'] = -1
    def blobs(items):
        if mutation == 'extra_path': items['../escape'] = b'not extracted'
        elif mutation == 'extra_model': items['target_9999_weights.pt'] = b'not loaded'
    rewrite_archive(source, bad, change, blobs)
    with pytest.raises((ValueError, KeyError)):
        load_project(bad)
    assert not (tmp_path.parent / 'escape').exists()


def test_old_v1_and_v2_projects_keep_shared_predictions_and_config(grouped_data, tmp_path):
    v1 = train_model(grouped_data, TrainConfig([3], epochs=2, patience=1, batch_size=16))
    path1 = tmp_path/'old-v1.dwb'
    save_project(path1, v1)
    loaded1, _ = load_project(path1)
    np.testing.assert_array_equal(loaded1.predict(grouped_data.X), v1.predict(grouped_data.X))
    shared = run_model_search(grouped_data, SearchConfig([tiny_spec('rf')], folds=2,
                              max_fits=3, cpu_threads=1, target_mode='shared')).selected
    fresh, old = tmp_path/'shared.dwb', tmp_path/'old-v2.dwb'
    save_project(fresh, shared)
    def legacy(meta):
        metadata = meta['search_metadata']
        metadata['config'].pop('target_mode')
        metadata.pop('target_mode')
        for key in ('target_mode', 'target_order', 'target_count', 'per_target_fit_budget', 'refit_fit_budget'):
            metadata['plan'].pop(key, None)
        metadata['plan']['schema_version'] = 1
    rewrite_archive(fresh, old, legacy)
    loaded2, _ = load_project(old)
    assert not isinstance(loaded2, CompositeSurrogate)
    assert loaded2.search_metadata['config']['target_mode'] == 'shared'
    assert loaded2.search_metadata['target_mode'] == 'shared'
    np.testing.assert_array_equal(loaded2.predict(grouped_data.X), shared.predict(grouped_data.X))
    rerun = run_model_search(grouped_data, SearchConfig.from_dict(loaded2.search_metadata['config']))
    np.testing.assert_array_equal(rerun.selected.predict(grouped_data.X), shared.predict(grouped_data.X))


@pytest.mark.parametrize('family,param,value', [('ann', 'epochs', True), ('ann', 'learning_rate', '0.01'),
    ('rf', 'n_estimators', 3.5), ('rf', 'max_features', False), ('xgboost', 'patience', 1.5),
    ('xgboost', 'learning_rate', float('nan'))])
def test_model_specs_reject_ambiguous_or_nonfinite_parameter_types(family, param, value):
    spec = tiny_spec(family)
    spec.params[param] = value
    with pytest.raises(ValueError):
        spec.validate(2, 1)


@pytest.mark.parametrize('field,value', [('folds', True), ('seed', -1), ('cpu_threads', 1.5),
                                         ('target_mode', 'automatic'), ('max_fits', False)])
def test_search_configuration_rejects_ambiguous_types(grouped_data, field, value):
    config = SearchConfig([tiny_spec('rf')], folds=2, max_fits=6)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        make_search_plan(grouped_data, config)


def test_json_reader_rejects_duplicate_keys_and_exponent_overflow():
    from daylight_workbench.project import _read_json
    for blob in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
        with pytest.raises(ValueError):
            _read_json(blob)


def test_nested_torch_storage_is_bounded_before_loading(monkeypatch):
    from io import BytesIO
    from daylight_workbench import project
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('archive/data/0', bytes(5000))
    monkeypatch.setattr(project, 'MAX_ARCHIVE_BYTES', 4096)
    with pytest.raises(ValueError, match='权重归档'):
        project._validate_tensor_archive(stream.getvalue())


def test_native_xgb_cyclic_json_rejected_before_native_loading(mixed_search, tmp_path, monkeypatch):
    import xgboost as xgb
    source, bad = tmp_path/'original.dwb', tmp_path/'cycle.dwb'
    save_project(source, mixed_search.selected)
    def blobs(items):
        name = next(name for name in items if name.endswith('booster_0.json'))
        native = json.loads(items[name])
        tree = native['learner']['gradient_booster']['model']['trees'][0]
        tree['left_children'][0] = 0
        tree['right_children'][0] = 0
        items[name] = json.dumps(native).encode()
    rewrite_archive(source, bad, change_blobs=blobs)
    loaded = []
    monkeypatch.setattr(xgb.Booster, 'load_model', lambda *args, **kwargs: loaded.append(True))
    with pytest.raises(ValueError, match='树路径'):
        load_project(bad)
    assert loaded == []


def test_refit_budget_is_rederived_from_outer_folds(mixed_search, tmp_path):
    source, bad = tmp_path/'original.dwb', tmp_path/'budget.dwb'
    save_project(source, mixed_search.selected)
    def change(meta):
        record = meta['search_metadata']['target_results']['linear']
        record['refit_budget'] = record['refit_budget'] + 1
        meta['models'][0]['fit_metadata']['refit_budget'] = record['refit_budget']
    rewrite_archive(source, bad, change_meta=change)
    with pytest.raises(ValueError, match='各折早停'):
        load_project(bad)


def test_composite_aggregate_nested_storage_budget_precedes_all_native_loaders(grouped_data, tmp_path, monkeypatch):
    from io import BytesIO
    from daylight_workbench import project
    model = run_model_search(grouped_data, SearchConfig([tiny_spec('rf')], folds=2,
                             max_fits=6, cpu_threads=1)).selected
    source, bad = tmp_path/'original.dwb', tmp_path/'aggregate.dwb'
    save_project(source, model)
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as nested:
        nested.writestr('array.npy', bytes(30000))
    compact = stream.getvalue()
    assert len(compact) < 30000
    def change(blobs):
        for name in list(blobs):
            if name.endswith('.npz'):
                blobs[name] = compact
    rewrite_archive(source, bad, change_blobs=change)
    with zipfile.ZipFile(bad) as archive:
        stored = sum(info.file_size for info in archive.infolist())
    # Either compressed forest individually fits, but the aggregate does not.
    limit = stored + 30000 - len(compact) + 1
    monkeypatch.setattr(project, 'MAX_ARCHIVE_BYTES', limit)
    decoded = []
    monkeypatch.setattr(project, '_decode_model_artifacts', lambda *args: decoded.append(True))
    with pytest.raises(ValueError, match='展开总量'):
        load_project(bad)
    assert decoded == []
    with zipfile.ZipFile(bad) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist() if name != 'manifest.json'}
    with pytest.raises(ValueError, match='展开总量'):
        project._write_archive(tmp_path/'never-written.dwb', payloads, 3)
    assert not (tmp_path/'never-written.dwb').exists()


def test_writer_rejects_oversized_manifest_without_replacing_existing_file(tmp_path):
    from daylight_workbench.project import _write_archive
    path = tmp_path/'preserved.dwb'
    path.write_bytes(b'previous project must survive rejected save')
    # Representative many-target names. Size is checked before numeric parsing
    # or creating an atomic temporary output, so no unreadable file is written.
    payloads = {f'target_{i:04d}_weights.pt': b'' for i in range(1000)}
    with pytest.raises(ValueError, match='清单超过64KB'):
        _write_archive(path, payloads, 3)
    assert path.read_bytes() == b'previous project must survive rejected save'
