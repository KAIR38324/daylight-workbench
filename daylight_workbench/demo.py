"""Entirely synthetic classroom-shaped data. Not a physical simulation."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo(n=420, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 10))
    frame = pd.DataFrame({
        'window_ratio': .15 + .65*x[:, 0], 'glass_transmittance': .25 + .60*x[:, 1],
        'room_depth_m': 5 + 7*x[:, 2], 'room_width_m': 5 + 6*x[:, 3],
        'room_height_m': 2.7 + 1.8*x[:, 4], 'overhang_m': 1.8*x[:, 5],
        'wall_reflectance': .3 + .55*x[:, 6], 'ceiling_reflectance': .6 + .3*x[:, 7],
        'orientation_deg': -90 + 180*x[:, 8], 'floor_level': rng.integers(1, 7, n),
    })
    noise = rng.normal(0, .4, (n, 3))
    frame['synthetic_daylight_score'] = 20 + 60*x[:, 0]*x[:, 1] - 15*x[:, 2] + 8*x[:, 6] - 8*x[:, 5] + noise[:, 0]
    frame['synthetic_glare_score'] = 5 + 25*x[:, 0]*x[:, 1] + 8*x[:, 8] - 12*x[:, 5] + noise[:, 1]
    frame['synthetic_energy_score'] = 30 + 20*x[:, 0] + 12*x[:, 2] - 10*x[:, 5] + 6*x[:, 8] + noise[:, 2]
    return frame


def write_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_demo().to_csv(path, index=False, float_format='%.17g')


if __name__ == '__main__':
    import sys
    write_demo(sys.argv[1] if len(sys.argv) > 1 else 'demo/synthetic_classroom.csv')
