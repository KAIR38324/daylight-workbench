"""One small real ANN->NSGA->save/reopen->export test. No network calls."""
from pathlib import Path
import argparse
import os
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args(argv)
    if args.gui:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'daylight-mpl-cache'))
    import numpy as np
    from daylight_workbench.core import DataConfig, TrainConfig, Variable, OptimizeConfig, prepare_data, train_model, optimize_model, export_candidates
    from daylight_workbench.demo import make_demo
    from daylight_workbench.project import save_project, load_project
    raw = make_demo(120)
    columns = list(raw.columns)
    dc = DataConfig(columns[:10], columns[10:], directions=dict(zip(columns[10:], ['max', 'min', 'min'])))
    data = prepare_data(raw, dc)
    model = train_model(data, TrainConfig([12, 12], epochs=12, patience=5))
    x = data.X[data.splits['train']]
    variables = [Variable(c, float(x[:, j].min()), float(x[:, j].max()), 'integer' if c == 'floor_level' else 'continuous') for j, c in enumerate(dc.inputs)]
    result = optimize_model(model, OptimizeConfig(variables, dc.directions, population=8, generations=2))
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'smoke.dwb'
        save_project(path, model, result)
        loaded, optimized = load_project(path)
        np.testing.assert_array_equal(model.predict(data.X), loaded.predict(data.X))
        np.testing.assert_allclose(optimized.candidates.drop(columns='pareto').to_numpy(float), result.candidates.drop(columns='pareto').to_numpy(float), rtol=0, atol=0)
        assert optimized.candidates.pareto.tolist() == result.candidates.pareto.tolist()
        export_candidates(Path(temp) / 'candidates.csv', result.candidates, model, result)
        assert (Path(temp) / 'candidates.csv.provenance.json').exists()
    if args.gui:
        from PySide6.QtWidgets import QApplication
        from daylight_workbench.ui import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.load_demo()
        window.trained, window.prepared = model, data
        window.show_training()
        window.populate_bounds()
        window.optimization = result
        window.show_results()
        window.results_table.selectRow(0)
        app.processEvents()
        assert window.metrics_model.rowCount() == 9
        assert window.results_model.rowCount() > 0
        window.close()
    print('SMOKE PASSED: ANN, integer NSGA-II, project, CSV provenance' + (', offscreen Qt views' if args.gui else ''))


if __name__ == '__main__':
    main()
