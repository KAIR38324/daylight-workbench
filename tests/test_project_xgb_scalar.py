"""A v2 archive may contain only scalar-output regression boosters per target."""
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from daylight_workbench.core import DataConfig, prepare_data
from daylight_workbench.modeling import ModelSpec, SearchConfig, run_model_search
from daylight_workbench.project import load_project, save_project


def test_project_rejects_valid_xgboost_multi_output_booster(tmp_path):
    x = np.linspace(-2, 2, 60)
    raw = pd.DataFrame({'x1': x, 'x2': np.sin(x * 2), 'y': x + .2 * np.sin(x * 2)})
    data = prepare_data(raw, DataConfig(['x1', 'x2'], ['y']))
    search = SearchConfig([ModelSpec('xgboost', {'n_estimators': 2, 'max_depth': 2,
                                               'learning_rate': .1, 'patience': 1}, [])],
                          folds=2, max_fits=3, cpu_threads=1)
    selected = run_model_search(data, search).selected
    original = tmp_path / 'scalar.dwb'
    save_project(original, selected)
    loaded, _ = load_project(original)
    assert loaded.predict(data.X[:3]).shape == (3, 1)

    # Same feature dimension and a fully valid native XGBoost JSON model,
    # but it predicts two targets while the signed metadata declares one.
    train = selected.fit_indices
    xgb_data = xgb.DMatrix(data.X[train], label=np.column_stack([data.Y[train, 0], -data.Y[train, 0]]))
    multi = xgb.train({'objective': 'reg:squarederror', 'nthread': 1, 'max_depth': 2},
                      xgb_data, num_boost_round=2)
    assert multi.num_features() == 2
    assert multi.predict(xgb.DMatrix(data.X[:3])).shape == (3, 2)
    with zipfile.ZipFile(original) as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads['booster_0.json'] = bytes(multi.save_raw(raw_format='json'))
    manifest = json.loads(payloads['manifest.json'])
    manifest['sha256']['booster_0.json'] = hashlib.sha256(payloads['booster_0.json']).hexdigest()
    payloads['manifest.json'] = json.dumps(manifest, ensure_ascii=False).encode('utf-8')
    swapped = tmp_path / 'swapped.dwb'
    with zipfile.ZipFile(swapped, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, blob in payloads.items():
            archive.writestr(name, blob)
    with pytest.raises(ValueError, match='标量回归输出'):
        load_project(swapped)
