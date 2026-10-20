"""User-facing name and local application icon."""
from pathlib import Path

from PySide6.QtGui import QIcon


APP_NAME = '建筑采光多目标优化平台'
APP_SUBTITLE = 'Daylight Multi-objective Optimization Workbench'
ICON_PATH = Path(__file__).resolve().parent / 'assets' / 'app-icon.png'


def app_icon() -> QIcon:
    return QIcon(str(ICON_PATH))
