def main():
    import sys, os
    # Windowed PyInstaller apps have no console streams. Libraries may still
    # write diagnostics; provide valid sinks rather than None.
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))
    if '--self-test' in sys.argv:
        import json, tempfile, traceback
        from pathlib import Path
        report = Path(tempfile.gettempdir()) / 'daylight-workbench-selftest.json'
        try:
            from .smoke import main as smoke_main
            from .smoke_v2 import main as smoke_v2_main
            from .smoke_v3 import main as smoke_v3_main
            smoke_main(['--gui'])
            smoke_v2_main(gui=True)
            smoke_v3_main(gui=True)
            report.write_text(json.dumps({'status': 'passed', 'scope': 'synthetic v0.1/v0.2 shared ANN/RF/XGBoost and v0.3 independent composite/CV/NSGA/v3-project/fifth-page local preference/offscreen-Qt/offline-plot-assets; no live API'}, indent=2), encoding='utf-8')
        except Exception:
            report.write_text(json.dumps({'status': 'failed', 'error': traceback.format_exc()}, indent=2), encoding='utf-8')
            raise SystemExit(1)
        return
    from PySide6.QtWidgets import QApplication
    from .branding import APP_NAME, app_icon
    from .ui import MainWindow
    from .theme import apply_theme
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    apply_theme(app)
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
