"""Compare four perturbed 1492 declination maps using the original isogonic.py plotting code."""
from pathlib import Path
import tempfile

import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from pymagglobal import Model, field
from pymagglobal.utils import get_grid

from declination_uncertainty import (
    parse_cals3k_4b, write_model_file,
)

# If a different model is used you'd have to modify the parser
MODEL_PATH = 'CALS3k.4b'
YEAR = 1492
RNG_SEED = 42

model_path = Path(__file__).resolve().parent / MODEL_PATH

knots, mu, sigma, degree, nspl, start, end = parse_cals3k_4b(model_path)

rng = np.random.default_rng(RNG_SEED)

grid = get_grid(2000, t=YEAR)
grid[1] -= 180

proj = ccrs.Mercator()

fig, axes = plt.subplots(2, 2, figsize=(20, 12),
                            subplot_kw={'projection': proj})

pts = proj.transform_points(ccrs.Geodetic(), grid[1], 90 - grid[0])
plt_lon, plt_lat = pts[:, 0], pts[:, 1]
levels = np.arange(-180, 181, 10)

with tempfile.TemporaryDirectory() as tmp:
    for number, ax in enumerate(axes.flat, 1):
        path = str(Path(tmp) / f'perturbed_{number}.dat')

        coefficients = mu + sigma * rng.standard_normal((1, mu.shape[1]))

        write_model_file(path, knots, coefficients, degree, nspl, start, end)
        model = Model(f'perturbed_{number}', fname=path)

        declination, inclination, intensity = field(grid, model, field_type='dif')
        ax.tricontourf(plt_lon, plt_lat, declination,
                        levels=levels, cmap='RdBu_r', extend='both')
        ax.tricontour(plt_lon, plt_lat, declination,
                        levels=levels, colors='k', linewidths=0.5, alpha=0.5)
        ax.tricontour(plt_lon, plt_lat, declination,
                        levels=[0], colors='gold', linewidths=2.0)
        ax.set_global()
        ax.coastlines()
        ax.set_title(f'Model {number}')

fig.suptitle(f'CALS3k.4b, {YEAR} — four perturbed models')
fig.tight_layout()
fig.savefig('isogonic_perturbed.png', dpi=150., bbox_inches='tight')

plt.show()

