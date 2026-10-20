"""Offline, ID-linked Plotly analysis view for multi-objective candidates.

Only bundled ``plot_assets`` files are loaded. The WebEngine profile denies remote
requests; the JavaScript bridge exchanges JSON strings rather than executable code.
"""
from __future__ import annotations

import html
import json
import math
import shutil
import tempfile
from importlib import resources
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

ASSET_DIR = Path(__file__).resolve().parent / 'plot_assets'


def _plotly_js_resource():
    try:
        source = resources.files('plotly').joinpath('package_data', 'plotly.min.js')
        if source.is_file():
            return source
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    raise FileNotFoundError('Plotly JS is not bundled and the installed plotly package is unavailable')


def prepare_plotly_asset() -> Path:
    """Copy JS from the installed Plotly wheel for a distributable offline build."""
    target = ASSET_DIR / 'plotly.min.js'
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    with resources.as_file(_plotly_js_resource()) as source:
        shutil.copyfile(source, target)
    return target


@lru_cache(maxsize=1)
def _runtime_asset_dir():
    """Return bundled assets, or private temporary local copies for source runs."""
    if (ASSET_DIR / 'plotly.min.js').is_file():
        return ASSET_DIR, None
    temporary = tempfile.TemporaryDirectory(prefix='daylight-plot-')
    destination = Path(temporary.name)
    for name in ('index.html', 'style.css', 'plot.js'):
        shutil.copyfile(ASSET_DIR / name, destination / name)
    with resources.as_file(_plotly_js_resource()) as source:
        shutil.copyfile(source, destination / 'plotly.min.js')
    return destination, temporary


def _is_missing(value):
    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False
    except (ValueError, TypeError):
        return False


def _json_value(value):
    """Return a finite JSON scalar, retaining ID type when possible."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if _is_missing(value):
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _key(value):
    return json.dumps(_json_value(value), ensure_ascii=True, allow_nan=False,
                      separators=(',', ':'))


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _front(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'y', 'pareto', 'front'}
    try:
        return bool(value) if not _is_missing(value) else False
    except (TypeError, ValueError):
        return False


def prepare_plot_payload(frame: pd.DataFrame, objective_columns: Iterable[str],
                         input_columns: Iterable[str] = (), id_column: str = 'candidate_id',
                         pareto_column: str = 'pareto') -> dict:
    """Validate/serialize all rows without sampling or modifying the caller's frame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('frame must be a pandas DataFrame')
    objectives = list(objective_columns)
    inputs = list(input_columns)
    if len(objectives) < 2:
        raise ValueError('At least two objective columns are required')
    if len(set(objectives)) != len(objectives) or len(set(inputs)) != len(inputs):
        raise ValueError('Column lists cannot contain duplicates')
    if set(objectives) & set(inputs):
        raise ValueError('Input and objective columns must be distinct')
    required = [id_column, *objectives, *inputs]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if len(set(required)) != len(required):
        raise ValueError('ID, input, and objective columns must be distinct')

    ids = [_json_value(v) for v in frame[id_column].tolist()]
    if any(v is None for v in ids):
        raise ValueError('Candidate IDs must not be missing')
    keys = [_key(v) for v in ids]
    if len(set(keys)) != len(keys):
        raise ValueError('Candidate IDs must be unique (including their JSON types)')
    for name in objectives:
        nonmissing = frame[name].dropna()
        if any(_number(v) is None for v in nonmissing):
            raise ValueError(f'Objective column must be numeric: {name}')

    # Parallel coordinates supports categorical design variables via visible tick labels.
    dimensions = []
    for name in [*objectives, *inputs]:
        values = frame[name].tolist()
        if name in objectives or all(_number(v) is not None or _is_missing(v) for v in values):
            dimensions.append({'name': name, 'kind': 'numeric'})
        else:
            categories = list(dict.fromkeys(str(v) for v in values if not _is_missing(v)))
            dimensions.append({'name': name, 'kind': 'category',
                               'categories': categories})
    has_front = pareto_column in frame.columns
    rows = []
    for i, (_, row) in enumerate(frame.iterrows()):
        values = {}
        for dim in dimensions:
            name = dim['name']
            value = row[name]
            if dim['kind'] == 'numeric':
                values[name] = _number(value)
            else:
                values[name] = None if _is_missing(value) else dim['categories'].index(str(value))
        rows.append({'id': ids[i], 'id_key': keys[i], 'display_id': str(ids[i]), 'hover_id': html.escape(str(ids[i])),
                     'pareto': _front(row[pareto_column]) if has_front else None,
                     'values': values})
    numeric_inputs = [d['name'] for d in dimensions if d['name'] in inputs and d['kind'] == 'numeric']
    excluded_inputs = [d['name'] for d in dimensions if d['name'] in inputs and d['kind'] != 'numeric']
    correlations = []
    for input_name in numeric_inputs:
        cells = []
        for objective_name in objectives:
            pairs = [(r['values'][input_name], r['values'][objective_name]) for r in rows]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            coefficient = None
            if len(pairs) >= 2:
                xy = np.asarray(pairs, dtype=float)
                if np.ptp(xy[:, 0]) > 0 and np.ptp(xy[:, 1]) > 0:
                    value = float(np.corrcoef(xy[:, 0], xy[:, 1])[0, 1])
                    if math.isfinite(value):
                        coefficient = max(-1.0, min(1.0, value))
            cells.append({'objective': objective_name, 'r': coefficient, 'n': len(pairs)})
        correlations.append({'input': input_name, 'cells': cells})
    return {'objectives': objectives, 'inputs': inputs, 'dimensions': dimensions,
            'has_pareto': has_front, 'rows': rows, 'count': len(rows),
            'correlations': correlations, 'excluded_inputs': excluded_inputs}


class _LocalOnlyInterceptor(QWebEngineUrlRequestInterceptor):
    def __init__(self, asset_dir=ASSET_DIR, parent=None):
        super().__init__(parent)
        self.asset_dir = Path(asset_dir).resolve()

    def interceptRequest(self, info):
        url = info.requestUrl()
        scheme = url.scheme().lower()
        if scheme == 'file':
            path = Path(url.toLocalFile()).resolve()
            allowed = path == self.asset_dir or self.asset_dir in path.parents
        else:
            allowed = scheme in {'qrc', 'data', 'about', 'blob'}
        if not allowed:
            info.block(True)


_shared_interceptor = None


def _profile_for_assets(asset_dir):
    global _shared_interceptor
    profile = QWebEngineProfile.defaultProfile()
    profile.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
    profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.NoPersistentCookies)
    if _shared_interceptor is None:
        _shared_interceptor = _LocalOnlyInterceptor(asset_dir)
        profile.setUrlRequestInterceptor(_shared_interceptor)
    return profile


class _LocalPage(QWebEnginePage):
    def __init__(self, profile, parent, asset_dir):
        super().__init__(profile, parent)
        self.asset_dir = Path(asset_dir).resolve()

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if not is_main_frame:
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)
        if url.scheme().lower() == 'file':
            path = Path(url.toLocalFile()).resolve()
            return path == self.asset_dir / 'index.html'
        return url.scheme().lower() == 'about'


class _PlotBridge(QObject):
    payloadChanged = Signal(str)
    selectionChanged = Signal(str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.ready_for_data = False

    @Slot()
    def ready(self):
        self.ready_for_data = True
        self.owner._send_payload()
        self.owner._send_selection()

    @Slot(str)
    def reportSelection(self, raw):
        self.owner._receive_selection(raw)


class InteractiveParetoWidget(QWidget):
    """A local-only linked Pareto scatter / parallel-coordinates QWidget.

    ``selectionChanged`` sends actual candidate IDs, not current row indices.
    The frame is never sampled; the caller's table/export remains authoritative.
    """
    selectionChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._asset_dir, self._temporary_assets = _runtime_asset_dir()
        self._payload = {'objectives': [], 'inputs': [], 'dimensions': [],
                         'has_pareto': False, 'rows': [], 'count': 0, 'version': 0}
        self._ids_by_key = {}
        self._selected_keys = []
        self._version = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        layout.addWidget(self.view)
        self._profile = _profile_for_assets(self._asset_dir)
        page = _LocalPage(self._profile, self.view, self._asset_dir)
        page.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        page.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        self.view.setPage(page)
        self._bridge = _PlotBridge(self)
        self._channel = QWebChannel(page)
        self._channel.registerObject('plotBridge', self._bridge)
        page.setWebChannel(self._channel)
        self.view.loadStarted.connect(self._on_load_started)
        self.view.load(QUrl.fromLocalFile(str(self._asset_dir / 'index.html')))

    def _on_load_started(self):
        self._bridge.ready_for_data = False

    def set_data(self, frame: pd.DataFrame, objective_columns: Iterable[str],
                 input_columns: Iterable[str] = (), id_column: str = 'candidate_id',
                 pareto_column: str = 'pareto') -> None:
        payload = prepare_plot_payload(frame, objective_columns, input_columns,
                                       id_column, pareto_column)
        self._version += 1
        payload['version'] = self._version
        self._payload = payload
        ids_by_key = {r['id_key']: r['id'] for r in payload['rows']}
        # A refresh with identical IDs retains selection; a changed dataset does not.
        if set(ids_by_key) != set(self._ids_by_key) or payload['objectives'] != getattr(self, '_old_objectives', None):
            was_selected = bool(self._selected_keys)
            self._selected_keys = []
            if was_selected:
                self.selectionChanged.emit([])
        self._old_objectives = payload['objectives'][:]
        self._ids_by_key = ids_by_key
        self._send_payload()
        self._send_selection()

    def set_selection(self, ids: Iterable) -> None:
        seen = set()
        keys = []
        for value in ids:
            key = _key(value)
            if key in self._ids_by_key and key not in seen:
                keys.append(key)
                seen.add(key)
        if keys != self._selected_keys:
            self._selected_keys = keys
            self._send_selection()

    def selected_ids(self) -> list:
        return [self._ids_by_key[k] for k in self._selected_keys]

    def _send_payload(self):
        if self._bridge.ready_for_data:
            self._bridge.payloadChanged.emit(json.dumps(self._payload, ensure_ascii=True,
                                                         allow_nan=False, separators=(',', ':')))

    def _send_selection(self):
        if self._bridge.ready_for_data:
            self._bridge.selectionChanged.emit(json.dumps(
                {'version': self._version, 'keys': self._selected_keys},
                ensure_ascii=True, allow_nan=False, separators=(',', ':')))

    def _receive_selection(self, raw: str):
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get('version') != self._version:
                return
            requested = data.get('keys')
            if not isinstance(requested, list) or len(requested) > len(self._ids_by_key):
                return
            keys = []
            seen = set()
            for value in requested:
                if not isinstance(value, str) or value not in self._ids_by_key:
                    return
                key = value
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        except (ValueError, TypeError, OverflowError):
            return
        if keys != self._selected_keys:
            self._selected_keys = keys
            self.selectionChanged.emit(self.selected_ids())
            self._send_selection()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare local Plotly JS for an offline build')
    parser.add_argument('--prepare-assets', action='store_true')
    args = parser.parse_args()
    if args.prepare_assets:
        print(prepare_plotly_asset())
    else:
        parser.print_help()
