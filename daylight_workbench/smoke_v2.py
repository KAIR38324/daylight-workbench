"""Small synthetic v0.2 packaged-runtime self-test. Never uses network or real data."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def main(gui=False):
    from .core import DataConfig, OptimizeConfig, Variable, optimize_model, prediction_column
    from .demo import make_demo
    from .external_results import prepare_external_results
    from .llm_filter import apply_plan, validate_plan
    from .modeling import LayerSpec, ModelSpec, SearchConfig, run_model_search
    from .project import load_project, save_project
    from .core import prepare_data

    raw = make_demo(120)
    columns = list(raw)
    config = DataConfig(columns[:10], columns[10:], directions=dict(zip(columns[10:], ('max', 'min', 'min'))))
    data = prepare_data(raw, config)
    search = SearchConfig([
        ModelSpec('ann', {'epochs': 2, 'patience': 1, 'learning_rate': .01, 'batch_size': 64}, [LayerSpec(4)]),
        ModelSpec('rf', {'n_estimators': 3, 'max_depth': 3}, []),
        ModelSpec('xgboost', {'n_estimators': 3, 'max_depth': 2, 'learning_rate': .1, 'patience': 1}, []),
    ], folds=3, max_fits=18, cpu_threads=1, target_mode='shared')
    result = run_model_search(data, search)
    assert [c['family'] for c in result.candidates] == ['ann', 'rf', 'xgboost']
    model = result.selected
    assert model.fit_indices == result.plan['development']
    assert all({'MAE', 'MSE', 'RMSE', 'R2'} <= set(m) for m in model.metrics['test'].values())
    bounds = model.support_bounds
    variables = [Variable(c, *bounds[c], 'integer' if c == 'floor_level' else 'continuous') for c in config.inputs]
    optimized = optimize_model(model, OptimizeConfig(variables, config.directions, population=4, generations=2))
    assert len(optimized.candidates) > 0

    with tempfile.TemporaryDirectory(prefix='daylight-v2-selftest-') as directory:
        path = Path(directory) / 'synthetic-v2.dwb'
        save_project(path, model, optimized)
        loaded, restored = load_project(path)
        np.testing.assert_allclose(loaded.predict(data.X[:6]), model.predict(data.X[:6]), rtol=1e-10, atol=1e-10)
        assert restored is not None and len(restored.candidates) == len(optimized.candidates)

    objective = prediction_column(config.targets[0])
    frame = optimized.candidates
    plan = validate_plan({'filters': [{'column': objective, 'operator': '>=',
                                      'value': float(frame[objective].min())}],
                          'sort': [], 'limit': 2, 'needs_clarification': False,
                          'explanation': 'Synthetic local self-test.'}, frame)
    filtered = apply_plan(frame, plan)
    assert filtered.candidate_id.tolist() == frame.candidate_id.tolist()[:2]
    external = prepare_external_results(pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 2., 1.]}),
                                        [], ['a', 'b'], {'a': 'min', 'b': 'min'}, 'synthetic-self-test')
    assert external.inputs == [] and external.targets == ['a', 'b']

    if gui:
        from PySide6.QtWidgets import QApplication
        from .interactive_plot import InteractiveParetoWidget
        from .ui import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.set_raw(raw, 'SYNTHETIC SELF-TEST', config, config.directions)
        window.trained, window.prepared, window.optimization = loaded, loaded.data, restored
        window.show_training()
        window.populate_bounds(restored.config)
        window.show_results()
        assert window.metrics_model.rowCount() > 0 and window.results_model.rowCount() > 0
        assert isinstance(window.interactive_plot, InteractiveParetoWidget)
        assert (window.interactive_plot._asset_dir / 'plotly.min.js').is_file()
        app.processEvents()
        window.close()
    print('V0.2 SYNTHETIC SELF-TEST PASSED: ANN/RF/XGBoost CV, NSGA-II, project, local plan, external objectives' +
          (', Qt and bundled offline Plotly asset' if gui else ''))


if __name__ == '__main__':
    main(gui=True)
