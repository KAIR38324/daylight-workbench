"""Bounded synthetic v0.3 runtime check: independent targets and local decisions.

This module is also run by the packaged EXE self-test. It never sends a request
to DeepSeek or reads user data.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np


def main(gui=False):
    from .core import DataConfig, OptimizeConfig, Variable, optimize_model, prepare_data
    from .demo import make_demo
    from .modeling import CompositeSurrogate, ModelSpec, SearchConfig, run_model_search
    from .project import load_project, save_project

    raw = make_demo(90)
    names = list(raw)
    config = DataConfig(names[:10], names[10:],
                        directions=dict(zip(names[10:], ('max', 'min', 'min'))))
    data = prepare_data(raw, config)
    search = SearchConfig([ModelSpec('rf', {'n_estimators': 3, 'max_depth': 3}, [])],
                          folds=2, max_fits=9, cpu_threads=1)
    assert search.target_mode == 'independent'
    result = run_model_search(data, search)
    model = result.selected
    assert isinstance(model, CompositeSurrogate)
    assert result.plan['fit_budget'] == 9
    assert result.plan['candidate_fold_jobs'] == 6
    assert list(result.target_results) == config.targets
    assert set(result.selected_index) == set(config.targets)
    assert all(model.target_models[target].fit_indices == result.plan['development']
               for target in config.targets)
    predicted = model.predict(data.X[:5])
    assert predicted.shape == (5, 3)
    np.testing.assert_allclose(predicted, np.column_stack([
        model.target_models[target].predict(data.X[:5])[:, 0]
        for target in config.targets]), rtol=0, atol=0)
    assert all({'MAE', 'MSE', 'RMSE', 'R2'} <= set(model.metrics['test'][target])
               for target in config.targets)

    variables = [Variable(name, *model.support_bounds[name],
                          'integer' if name == 'floor_level' else 'continuous')
                 for name in config.inputs]
    optimized = optimize_model(model, OptimizeConfig(variables, config.directions,
                                                      population=4, generations=2))
    assert len(optimized.candidates) > 0
    with tempfile.TemporaryDirectory(prefix='daylight-v3-selftest-') as directory:
        path = Path(directory) / 'synthetic-independent.dwb'
        save_project(path, model, optimized)
        loaded, restored = load_project(path)
        assert isinstance(loaded, CompositeSurrogate)
        np.testing.assert_allclose(loaded.predict(data.X[:5]), predicted, rtol=1e-10, atol=1e-10)
        assert restored is not None
        assert restored.candidates.candidate_id.tolist() == optimized.candidates.candidate_id.tolist()

    if gui:
        from PySide6.QtWidgets import QApplication
        from .ui import MainWindow
        from .preference_engine import default_preference_plan

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.set_raw(raw, 'SYNTHETIC V0.3 SELF-TEST', config, config.directions)
        window.trained, window.prepared, window.optimization = loaded, loaded.data, restored
        window.show_training()
        window.populate_bounds(restored.config)
        window.show_results()
        page = window.decision_page
        assert window.tabs.count() == 5 and len(page._frame) == len(restored.candidates)
        assert page.ranking is None
        page.confirm_mapping()
        assert page._mapping_confirmed
        plan = default_preference_plan(page._objectives, scope='all')
        plan['limit'] = 2
        page._load_plan_editor(plan)  # A reviewed local proposal; no provider call.
        page.apply_reviewed_plan()
        assert page.ranking is not None and len(page.ranking.frame) == 2
        assert set(page.ranking.frame.candidate_id) <= set(restored.candidates.candidate_id)
        assert page.ranking.frame.candidate_id.tolist() == page.selected_ids()
        assert len(window.results_model.frame) > 2  # Full comparison stays intact.
        assert window.last_filter_plan['kind'] == 'preference_v1'
        app.processEvents()
        window.close()

    print('V0.3 SYNTHETIC SELF-TEST PASSED: independent composite, v3 project, '
          'NSGA-II' + (', fifth-page reviewed local preferences' if gui else ''))


if __name__ == '__main__':
    main(gui=True)
