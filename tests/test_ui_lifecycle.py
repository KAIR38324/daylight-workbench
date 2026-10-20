import os
import time
import threading
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread

from daylight_workbench.core import check_cancel
from daylight_workbench.ui import MainWindow


def pump_until(app, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert predicate(), 'Qt worker did not finish within timeout'


def test_worker_single_task_cancel_and_close():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.load_demo()
    received = []
    def task(cancel, progress):
        for _ in range(300):
            check_cancel(cancel)
            time.sleep(.01)
        return 42
    window.start_worker(task, received.append)
    first_thread = window.thread
    window.start_worker(lambda c, p: 99, received.append)
    assert window.thread is first_thread
    assert not window.tabs.isEnabled()
    window.cancel_task()
    pump_until(app, lambda: window.thread is None)
    assert received == []
    assert window.tabs.isEnabled()
    window.start_worker(task, received.append)
    window.close()
    assert window._close_pending
    pump_until(app, lambda: window.thread is None)
    assert received == []
    window.close()


def test_python_completion_callback_is_marshaled_to_gui_thread():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    receiver_threads = []
    def completion(result):
        receiver_threads.append(QThread.currentThread())
        window.status.setText(str(result))
    window.start_worker(lambda cancel, progress: 123, completion)
    pump_until(app, lambda: window.thread is None)
    assert receiver_threads == [app.thread()]
    assert window.status.text() == '123'
    window.close()
