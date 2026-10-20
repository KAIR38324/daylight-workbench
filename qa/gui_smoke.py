"""Bounded offscreen Qt integration smoke; run from project root with python."""
import os
import tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(tempfile.gettempdir()) / "daylight-mpl"))

from pathlib import Path
import json
import sys
import time

from PySide6 import QtCore, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVIDENCE = ROOT / "qa" / "evidence"
EVIDENCE.mkdir(parents=True, exist_ok=True)

from daylight_workbench.ui import MainWindow


def wait_idle(window, app, timeout=45):
    ticks = [0]
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.__setitem__(0, ticks[0] + 1))
    timer.start(10)
    deadline = time.monotonic() + timeout
    while window.thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    app.processEvents()
    timer.stop()
    assert window.thread is None, "background task did not finish before timeout"
    return ticks[0]


def screenshot(window, name):
    window.repaint()
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage()
    assert not image.isNull()
    path = EVIDENCE / name
    assert image.save(str(path))
    return str(path.relative_to(ROOT))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    shots = [screenshot(window, "main-data.png")]

    window.load_demo()
    prepared = window.prepare()
    assert len(prepared.frame) == 420
    window.hidden.setText("4")
    window.epochs.setValue(4)
    window.patience.setValue(3)
    window.batch.setValue(128)
    window.start_training()
    train_ticks = wait_idle(window, app)
    assert window.trained is not None
    assert window.trained.best_epoch >= 1
    assert len(window.metrics_model.frame) == 9
    shots.append(screenshot(window, "main-training.png"))

    window.population.setValue(4)
    window.generations.setValue(2)
    window.opt_seed.setValue(5)
    # The floor count is an integer variable; other demo variables remain continuous.
    floor_row = window.trained.data.config.inputs.index("floor_level")
    window.bounds.cellWidget(floor_row, 3).setCurrentIndex(
        window.bounds.cellWidget(floor_row, 3).findData("integer"))
    window.start_optimization()
    opt_ticks = wait_idle(window, app)
    assert window.optimization is not None
    assert len(window.optimization.candidates) > 0
    assert window.optimization.candidates["pareto"].any()
    assert (window.optimization.candidates["floor_level"] % 1 == 0).all()

    # Exercise the UI filter using a real bound and verify every surviving row.
    window.result_source.setCurrentIndex(1)
    x_col = window.trained.data.config.inputs[0]
    threshold = float(window.optimization.candidates[x_col].median())
    window.filters.cellWidget(0, 1).setText(str(threshold))
    window.apply_filter()
    filtered = window.results_model.frame
    assert len(filtered) > 0
    assert (filtered[x_col] >= threshold - 1e-12).all()
    shots.append(screenshot(window, "main-results.png"))

    # Drive the actual save/open UI methods while replacing only the native
    # chooser return value, keeping the round trip local and deterministic.
    project = EVIDENCE / "gui-roundtrip.dwb"
    save_picker = QtWidgets.QFileDialog.getSaveFileName
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *args, **kwargs: (str(project), "Daylight project (*.dwb)"))
    try:
        window.save_project()
    finally:
        QtWidgets.QFileDialog.getSaveFileName = save_picker
    assert project.is_file()

    open_picker = QtWidgets.QFileDialog.getOpenFileName
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *args, **kwargs: (str(project), "Daylight project (*.dwb)"))
    try:
        window.open_project()
    finally:
        QtWidgets.QFileDialog.getOpenFileName = open_picker
    load_ticks = wait_idle(window, app)
    assert window.trained is not None and window.optimization is not None
    assert len(window.results_model.frame) > 0
    shots.append(screenshot(window, "main-reloaded.png"))

    # Cancel a bounded but longer NSGA-II run shortly after launch. Previously
    # valid model survives. Editing optimization settings explicitly invalidates
    # the old candidate set; cancellation must not bring those stale results back.
    old_model, old_result = window.trained, window.optimization
    window.population.setValue(80)
    window.generations.setValue(60)
    assert window.optimization is None, "edited optimization config must invalidate old candidates"
    window.start_optimization()
    QtCore.QTimer.singleShot(20, window.cancel_task)
    cancel_ticks = wait_idle(window, app)
    assert window.trained is old_model
    assert window.optimization is None, "cancelled revised config must not adopt partial or stale candidates"

    report = {
        "mode": f"Qt offscreen on {sys.platform}; not manual target-machine verification",
        "optimization_config_invalidation": "passed",
        "rows": len(prepared.frame),
        "training_epochs": len(old_model.history),
        "optimization_candidates": len(old_result.candidates),
        "filtered_candidates": len(filtered),
        "responsive_timer_ticks": {"training": train_ticks, "optimization": opt_ticks,
                                    "project_reload": load_ticks, "cancelled_optimization": cancel_ticks},
        "cancel_status": window.status.text(),
        "screenshots": shots,
        "project": str(project.relative_to(ROOT)),
    }
    (EVIDENCE / "gui-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    window.close()
    app.processEvents()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
