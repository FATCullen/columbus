import numpy as np

from cartopy import crs as ccrs

from matplotlib import pyplot as plt

from pymagglobal import Model, field
from pymagglobal.utils import get_grid

model = Model('CALS3k.4b', fname='CALS3k.4b')
year = 1492

grid = get_grid(2000, t=year)

grid[1] -= 180

declination, inclination, intensity = field(grid, model, field_type='dif')

proj = ccrs.Mercator()

fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': proj})

pts = proj.transform_points(
    ccrs.Geodetic(),
    grid[1],
    90 - grid[0],
)
plt_lon, plt_lat = pts[:, 0], pts[:, 1]

levels = np.arange(-180, 181, 10)

cf = ax.tricontourf(plt_lon, plt_lat, declination,
                    levels=levels, cmap='RdBu_r', extend='both')

cs = ax.tricontour(plt_lon, plt_lat, declination,
                   levels=levels, colors='k', linewidths=0.5, alpha=0.5)

ax.tricontour(plt_lon, plt_lat, declination,
              levels=[0], colors='gold', linewidths=2.0)

ax.set_global()
ax.coastlines()

plt.show()