"""A restrained light theme for the native Daylight Workbench controls.

The stylesheet is deliberately limited to presentation. It does not change any
widget state, layout, model data, or application behavior.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


_STYLESHEET = """
QWidget {
    color: #242b33;
    background-color: #ffffff;
    font-size: 10pt;
    selection-background-color: #dceafb;
    selection-color: #1e2935;
}

QMainWindow, QDialog, QTabWidget::pane, QScrollArea, QAbstractScrollArea {
    background: #ffffff;
}

QToolTip {
    color: #202832;
    background: #ffffff;
    border: 1px solid #cfd6de;
    padding: 4px 6px;
}

QFrame {
    border: 1px solid #e0e5ea;
    border-radius: 4px;
    background: #ffffff;
}

/* QLabel inherits QFrame. Keep ordinary text labels borderless. */
QLabel {
    border: none;
    border-radius: 0;
    background: transparent;
}

QPushButton {
    color: #26313c;
    background: #ffffff;
    border: 1px solid #cbd3dc;
    border-radius: 4px;
    padding: 6px 10px;
}
QPushButton:hover {
    background: #f7f9fb;
    border-color: #9eabb9;
}
QPushButton:pressed {
    background: #edf3f9;
    border-color: #8098af;
}
QPushButton:focus {
    border: 2px solid #5b8fbd;
}
QPushButton:default {
    color: #ffffff;
    background: #315f87;
    border-color: #315f87;
}
QPushButton:default:hover {
    background: #284f72;
    border-color: #284f72;
}
QPushButton:disabled {
    color: #98a1aa;
    background: #f6f7f8;
    border-color: #e1e5e9;
}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    color: #202832;
    background: #ffffff;
    border: 1px solid #cbd3dc;
    border-radius: 3px;
    padding: 5px 7px;
    selection-background-color: #dceafb;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color: #9eabb9;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4f86b5;
}
QLineEdit:read-only, QLineEdit:disabled, QPlainTextEdit:disabled,
QTextEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {
    color: #7e8893;
    background: #f6f7f8;
}
QLineEdit::placeholder {
    color: #8c96a1;
}

QComboBox QAbstractItemView {
    color: #202832;
    background: #ffffff;
    border: 1px solid #cbd3dc;
    selection-background-color: #e7f0f9;
    selection-color: #1e2935;
    outline: 0;
    padding: 3px;
}

QCheckBox, QRadioButton {
    spacing: 6px;
    background: transparent;
}
QCheckBox:disabled, QRadioButton:disabled {
    color: #929ba5;
}

QTabWidget::pane {
    border: 1px solid #d7dde4;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    color: #58636f;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 12px 6px 12px;
    margin-right: 2px;
}
QTabBar::tab:hover {
    color: #304a62;
    background: #f6f8fa;
}
QTabBar::tab:selected {
    color: #244f73;
    border-bottom-color: #3975a5;
    font-weight: 600;
}
QTabBar::tab:disabled {
    color: #a0a8b1;
}

QTableView, QTableWidget, QTreeView, QListView {
    color: #202832;
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #d3dae2;
    gridline-color: #e4e8ed;
    selection-background-color: #d9eafb;
    selection-color: #1d2b39;
}
QTableView::item, QTableWidget::item, QTreeView::item, QListView::item {
    padding: 4px 6px;
    border-bottom: 1px solid #edf0f3;
}
QTableView::item:selected, QTableWidget::item:selected,
QTreeView::item:selected, QListView::item:selected {
    background: #d9eafb;
    color: #1d2b39;
}
QHeaderView::section {
    color: #34404c;
    background: #f4f6f8;
    border: none;
    border-right: 1px solid #e1e6eb;
    border-bottom: 1px solid #cfd6de;
    padding: 6px 7px;
    font-weight: 600;
}
QHeaderView::section:horizontal:hover, QHeaderView::section:vertical:hover {
    background: #edf1f5;
}

QScrollBar:vertical {
    background: #f7f8fa;
    width: 12px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #c4cdd6;
    min-height: 26px;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #aab6c2; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal {
    background: #f7f8fa;
    height: 12px;
    margin: 0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #c4cdd6;
    min-width: 26px;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: #aab6c2; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

QMenu {
    color: #202832;
    background: #ffffff;
    border: 1px solid #d3dae2;
    padding: 4px;
}
QMenu::item { padding: 5px 22px 5px 8px; }
QMenu::item:selected { color: #1e2935; background: #eaf2fa; }
"""


class _WorkbenchStyle(QtWidgets.QProxyStyle):
    """Keep Fusion's native controls while making spin arrows clearly legible."""

    def drawPrimitive(self, element, option, painter, widget=None):
        up = QtWidgets.QStyle.PrimitiveElement.PE_IndicatorSpinUp
        down = QtWidgets.QStyle.PrimitiveElement.PE_IndicatorSpinDown
        if element in (up, down):
            rect = option.rect
            center_x, center_y = rect.center().x(), rect.center().y()
            half_width = min(4, max(2, (rect.width() - 4) // 2))
            half_height = min(3, max(2, (rect.height() - 4) // 2))
            if element == up:
                points = [
                    QtCore.QPoint(center_x - half_width, center_y + half_height),
                    QtCore.QPoint(center_x + half_width, center_y + half_height),
                    QtCore.QPoint(center_x, center_y - half_height),
                ]
            else:
                points = [
                    QtCore.QPoint(center_x - half_width, center_y - half_height),
                    QtCore.QPoint(center_x + half_width, center_y - half_height),
                    QtCore.QPoint(center_x, center_y + half_height),
                ]
            enabled = bool(option.state & QtWidgets.QStyle.StateFlag.State_Enabled)
            painter.save()
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#596775" if enabled else "#aab3bc"))
            painter.drawPolygon(QtGui.QPolygon(points))
            painter.restore()
            return
        super().drawPrimitive(element, option, painter, widget)


def _font_family(app: QtWidgets.QApplication) -> str:
    """Prefer Segoe UI on Windows, with sensible installed-font fallbacks."""
    try:
        installed = {name.casefold(): name for name in QtGui.QFontDatabase.families()}
    except Exception:
        installed = {}
    for preferred in ("Segoe UI", "Noto Sans CJK SC", "Noto Sans", "Arial", "Liberation Sans"):
        if preferred.casefold() in installed:
            return installed[preferred.casefold()]
    return app.font().family() or "Sans Serif"


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Apply the shared clean-white theme to a QApplication before showing UI."""
    if app is None:
        raise TypeError("apply_theme requires a QApplication instance")

    # Fusion provides consistent control geometry, while Qt still falls back to
    # platform fonts (including CJK glyphs) when the preferred family is absent.
    style = _WorkbenchStyle("Fusion")
    app.setStyle(style)
    # Hold a Python reference as well as Qt's ownership reference.
    app._daylight_workbench_theme_style = style
    font = QtGui.QFont(_font_family(app))
    font.setPointSizeF(10.0)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setStyleSheet(_STYLESHEET)

