"""A restrained light theme for the native Daylight Workbench controls.

The stylesheet is deliberately limited to presentation. It does not change any
widget state, layout, model data, or application behavior.
"""
from __future__ import annotations

from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets


_STYLESHEET = """
QWidget {
    color: #273544;
    background-color: #ffffff;
    font-size: 10pt;
    selection-background-color: #fff0d8;
    selection-color: #273544;
}

QWidget#workbenchRoot { background: #f5f7f9; }
QFrame#appHeader { border: none; border-bottom: 1px solid #dfe5eb; border-radius: 0; }
QLabel#appTitle { font-size: 17pt; font-weight: 600; color: #1e3449; }
QLabel#appSubtitle, QLabel[role="muted"] { color: #6d7b8b; font-size: 9pt; }
QLabel[role="sectionTitle"] { font-size: 12pt; font-weight: 600; color: #233b50; }
QLabel[role="eyebrow"] { color: #738292; font-size: 8pt; font-weight: 600; }
QLabel[role="notice"] { color: #687482; font-size: 9pt; padding: 2px 0; }
QLabel[role="badge"] { color: #3e596c; background: #edf3f7; padding: 4px 9px; border-radius: 4px; font-size: 9pt; }
QLabel[role="syntheticBadge"] { color: #865c20; background: #fff3df; padding: 4px 9px; border-radius: 4px; font-size: 9pt; }
QFrame[role="card"] { border: 1px solid #dfe5eb; border-radius: 6px; background: #ffffff; }
QFrame[role="softCard"] { border: 1px solid #e3e9ee; border-radius: 5px; background: #fafbfd; }
QFrame#statusBar { border: none; border-top: 1px solid #e0e6ec; border-radius: 0; background: #f5f7f9; }
QLabel[role="statValue"] { color: #233b50; font-size: 13pt; font-weight: 600; }
QLabel[role="statLabel"] { color: #758291; font-size: 8pt; }
QPushButton[role="primary"] { background: #315f87; color: #ffffff; border-color: #315f87; font-weight: 600; }
QPushButton[role="primary"]:hover { background: #284f72; border-color: #284f72; }
QPushButton[role="primary"]:disabled { background: #eaf0f5; color: #93a2b0; border-color: #e0e7ee; }
QPushButton[role="quiet"] { color: #5c6c7c; background: transparent; border-color: transparent; }
QPushButton[role="quiet"]:hover { background: #eef3f7; border-color: #d5dfe8; }
QToolButton { color: #334c61; background: #ffffff; border: 1px solid #cbd5df; border-radius: 4px; padding: 6px 10px; }
QToolButton:hover { background: #f4f7fa; border-color: #9eafbd; }
QToolButton::menu-button { border-left: 1px solid #dfe5eb; width: 20px; }
QGroupBox { border: 1px solid #e0e6ec; border-radius: 6px; margin-top: 13px; padding: 15px 12px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #3b5268; }
QProgressBar { border: none; border-radius: 3px; background: #e7edf3; min-height: 5px; max-height: 5px; }
QProgressBar::chunk { background: #4d7c9f; border-radius: 3px; }
QSplitter::handle { background: #edf1f5; }
QSplitter::handle:horizontal { width: 7px; }
QSplitter::handle:vertical { height: 7px; }
QGraphicsView { border: none; background: #fafbfd; }

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
    border: none;
    border-radius: 0;
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
    border: 1px solid #5b8fbd;
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
QComboBox { padding-right: 24px; }
QComboBox::drop-down { border: none; width: 23px; }
QComboBox::down-arrow { image: url(__ASSETS__/chevron.svg); width: 11px; height: 11px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 20px; height: 17px; border: none; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; height: 17px; border: none; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(__ASSETS__/chevron-up.svg); width: 10px; height: 10px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(__ASSETS__/chevron.svg); width: 10px; height: 10px; }

QCheckBox, QRadioButton {
    spacing: 6px;
    background: transparent;
}
QCheckBox:disabled, QRadioButton:disabled {
    color: #929ba5;
}
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #bfccd8; border-radius: 3px; background: #ffffff; }
QCheckBox::indicator:hover { border-color: #6b91ad; background: #f6f9fc; }
QCheckBox::indicator:checked { background: #4a7595; border-color: #4a7595; image: url(__ASSETS__/check.svg); }
QCheckBox::indicator:disabled { background: #f3f6f8; border-color: #dfe6ec; }

QTabWidget::pane {
    border: 1px solid #dfe5eb;
    background: #ffffff;
    top: -1px;
}
QTabBar::tab {
    color: #58636f;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 15px 8px 15px;
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
QTabWidget#workflowTabs > QTabBar::tab { padding: 12px 19px 11px; font-size: 10pt; }
QTabWidget#workflowTabs > QTabBar::tab:selected { color: #244f73; background: #ffffff; }
QTabWidget#trainingTabs::pane, QTabWidget#resultTabs::pane { border: none; border-top: 1px solid #dfe5eb; }

QTableView, QTableWidget, QTreeView, QListView {
    color: #202832;
    background: #ffffff;
    alternate-background-color: #fafbfd;
    border: 1px solid #e0e6ec;
    gridline-color: #edf1f5;
    selection-background-color: #fff0d8;
    selection-color: #1d2b39;
}
QTableView::item, QTableWidget::item, QTreeView::item, QListView::item {
    padding: 4px 6px;
    border-bottom: 1px solid #edf0f3;
}
QTableView::item:selected, QTableWidget::item:selected,
QTreeView::item:selected, QListView::item:selected {
    background: #fff0d8;
    color: #1d2b39;
}
QHeaderView::section {
    color: #34404c;
    background: #f4f6f8;
    border: none;
    border-right: none;
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
    assets = (Path(__file__).resolve().parent / 'plot_assets').as_posix()
    app.setStyleSheet(_STYLESHEET.replace('__ASSETS__', assets))

