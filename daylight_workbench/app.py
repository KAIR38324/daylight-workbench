def main():
    import sys
    if '--self-test' in sys.argv:
        from .smoke import main as smoke_main
        import json, tempfile, traceback
        from pathlib import Path
        report = Path(tempfile.gettempdir()) / 'daylight-workbench-selftest.json'
        try:
            smoke_main(['--gui'])
            report.write_text(json.dumps({'status': 'passed', 'scope': 'synthetic ANN/NSGA/project/export/offscreen Qt'}, indent=2), encoding='utf-8')
        except Exception:
            report.write_text(json.dumps({'status': 'failed', 'error': traceback.format_exc()}, indent=2), encoding='utf-8')
            raise SystemExit(1)
        return
    from PySide6.QtWidgets import QApplication
    from .ui import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('Daylight Workbench')
    window = MainWindow()
    window.show()
    raise SystemExit(app.exec())
