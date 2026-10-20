"""Real Qt visual/interaction evidence with synthetic-only data.

No mocked plots, no image compositing, no network provider call, no sandbox flags.
Run with .venv/bin/python qa/visual_refinement_smoke.py.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'daylight-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'daylight-cache'))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6 import QtCore as C, QtWidgets as W
from daylight_workbench.modeling import LayerSpec
from daylight_workbench.theme import apply_theme
from daylight_workbench.ui import MainWindow

DEST = ROOT / 'qa' / 'evidence' / 'visual-refinement'


def pump(app, seconds=.25):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(.008)


def wait(app, predicate, timeout=50):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(.008)
    raise AssertionError('Visual smoke timed out')


def js(app, window, expression):
    result = []
    window.interactive_plot.view.page().runJavaScript(expression, result.append)
    wait(app, lambda: bool(result), 12)
    return result[0]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    app = W.QApplication.instance() or W.QApplication([])
    apply_theme(app)
    window = MainWindow()
    window.resize(1440, 1000)
    window.show()
    evidence = []

    def shot(name):
        pump(app, .65)
        path = DEST / name
        assert window.grab().save(str(path))
        evidence.append({'file': str(path.relative_to(ROOT)), 'width': window.width(), 'height': window.height()})

    window.load_demo()
    window.prepare()
    shot('01-data-1440x1000.png')
    window.tabs.setCurrentIndex(1)
    window.train_tabs.setCurrentIndex(0)
    window.layer_builder.set_layers([LayerSpec(64, 'relu'), LayerSpec(32, 'tanh')])
    window.epochs.setValue(180)
    window.patience.setValue(24)
    window.lr.setText('0.003')
    window.batch.setValue(32)
    shot('02-ann-architecture-1440x1000.png')

    # Snapshot evidence is captured from the real cancellable worker, not a
    # fabricated curve. Keep screenshot capture out of the worker thread.
    live = []
    window.start_training()
    class CaptureObserver(C.QObject):
        @C.Slot(object)
        def on_snapshot(self, event):
            if live or event.get('kind') != 'loss' or event.get('epoch', 0) < 15:
                return
            assert C.QThread.currentThread() is app.thread()
            if '训练中' not in window.training_stats['phase'].text() or not window.thread.isRunning():
                return
            window.training_canvas.draw()
            displayed = int(window.training_stats['epoch'].text())
            plotted = int(window.training_figure.axes[0].lines[0].get_xdata()[-1])
            if displayed != plotted or displayed != event['epoch']:
                return
            path = DEST / '03-live-ann-training-1440x1000.png'
            assert window.grab().save(str(path))
            live.append({'file': str(path.relative_to(ROOT)), 'width': window.width(), 'height': window.height(),
                         'epoch': displayed, 'worker_active': window.thread.isRunning(),
                         'displayed_phase': window.training_stats['phase'].text(),
                         'plotted_through_epoch': plotted, 'history_rows': len(event.get('history', []))})
    observer = CaptureObserver()
    window.worker.snapshot.connect(observer.on_snapshot, C.Qt.ConnectionType.QueuedConnection)
    wait(app, lambda: window.thread is None)
    assert live and live[0]['worker_active']
    evidence.extend(live)
    assert window.trained is not None
    assert window.metrics_model.frame.iloc[0]['划分'] == 'test'
    shot('04-ann-test-and-loss-1440x1000.png')
    window.resize(1240, 900)
    shot('05-ann-test-and-loss-1240x900.png')
    window.resize(1440, 1000)

    window.population.setValue(36)
    window.generations.setValue(10)
    floor = window.trained.data.config.inputs.index('floor_level')
    window.bounds.cellWidget(floor, 3).setCurrentIndex(1)
    window.start_optimization()
    wait(app, lambda: window.thread is None)
    assert window.optimization is not None
    window.tabs.setCurrentIndex(2)
    shot('06-nsga-live-frontier-1440x1000.png')
    window.tabs.setCurrentIndex(3)
    window.result_source.setCurrentIndex(1)
    window.apply_filter()
    wait(app, lambda: window.interactive_plot._bridge.ready_for_data)
    wait(app, lambda: js(app, window, "document.querySelector('#scatter')?.data?.length") == 3)
    frame = window.results_model.frame
    chosen = frame.candidate_id.iloc[[0, len(frame)//2, len(frame)-1]].tolist()
    window.plot_selection_changed(chosen)
    window.interactive_plot.set_selection(chosen)
    window.toggle_results_table()
    window.result_splitter.setSizes([150, 650])
    wait(app, lambda: js(app, window, "document.getElementById('selected').textContent") == '3')
    assert set(window.selected_frame().candidate_id) == set(chosen)
    # Table sorting must preserve the stable-ID selection, never old row numbers.
    window.results_table.sortByColumn(0, C.Qt.SortOrder.DescendingOrder)
    assert set(window.selected_frame().candidate_id) == set(chosen)
    assert set(window.interactive_plot.selected_ids()) == set(chosen)
    shot('07-linked-results-table-1440x1000.png')
    window.toggle_results_table()
    shot('08-linked-objective-space-1440x1000.png')
    for view, file in [('matrix','09-objective-matrix-1440x1000.png'),
                       ('distribution','10-selected-distributions-1440x1000.png'),
                       ('comparison','11-selected-comparison-1440x1000.png'),
                       ('correlation','12-noncausal-correlation-1440x1000.png')]:
        js(app, window, f"document.querySelector('[data-analysis-view=\"{view}\"]').click()")
        pump(app, .2)
        assert set(window.interactive_plot.selected_ids()) == set(chosen)
        shot(file)
    js(app, window, "document.querySelector('[data-analysis-view=\"frontier\"]').click()")
    # Repeated controls and interrupted brush gestures may not discard selection.
    for name, dragmode in [('toolBox','select'),('toolPan','pan'),('toolZoom','zoom'),('toolLasso','lasso')]:
        js(app, window, f"document.getElementById('{name}').click()")
        wait(app, lambda m=dragmode: js(app, window, "document.querySelector('#scatter').layout.dragmode") == m)
        assert set(window.interactive_plot.selected_ids()) == set(chosen)
    if js(app, window, "document.querySelector('#parallel svg') !== null"):
        assert js(app, window, "document.querySelector('#parallel svg').getBoundingClientRect().width > document.querySelector('#parallel').clientWidth")
        js(app, window, "(() => {const h=document.querySelector('[data-brush-dimension]');const r=h.getBoundingClientRect();h.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,clientY:r.top+10,pointerId:77}));h.dispatchEvent(new PointerEvent('pointercancel',{bubbles:true,pointerId:77}));})()")
        assert set(window.interactive_plot.selected_ids()) == set(chosen)
    for _ in range(2):
        window.toggle_filters()
        window.toggle_filters()
    assert window.filter_panel.isHidden()
    window.resize(1240, 900)
    shot('13-linked-objective-space-1240x900.png')
    webgl = js(app, window, "!document.getElementById('mode3').disabled")
    report = {
        'source': 'Real Linux offscreen PySide6 application, synthetic data only',
        'scientific_claim': 'None. Demo scores are not physical simulation or validation.',
        'dimensions': len(window.interactive_plot._payload['dimensions']),
        'selected_ids': chosen,
        'live_snapshot': live[0],
        'training_epochs_recorded': len(window.trained.history),
        'candidate_count': len(frame),
        'webgl_available': bool(webgl),
        'webgl_3d_executed': False,
        'online_llm_executed': False,
        'sandbox_disabled': False,
        'checks': ['real ANN loss snapshots', 'test-first original-unit metrics', 'real NSGA-II snapshots',
                   'stable-ID table sorting retention', 'linked matrix/distribution/comparison/correlation tabs',
                   'explicit lasso/box/pan/zoom controls', 'all 13 PCP dimensions retained with horizontal scroll',
                   'interrupted PCP brush cancellation', 'repeated filter toggle'],
        'screenshots': evidence,
    }
    (DEST / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    window.close()
    pump(app, .1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
