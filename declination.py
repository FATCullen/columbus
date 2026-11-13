import math

import numpy as np

from cartopy import crs as ccrs

from matplotlib import pyplot as plt

from geographiclib.geodesic import Geodesic

from pymagglobal import Model, field, local_curve
from pymagglobal.utils import get_grid

geod = Geodesic.WGS84

model = Model('CALS3k.4b', fname='CALS3k.4b')
year = 1492

# Canary islands, between Gomera and Tenerife
start_lat, start_lon = 28.168932, -16.968030

# [Month, Day, Leg, Bearing, Distance (leagues)]
course = [[9,8,1,270,9],
[9,9,1,270,18],
[9,9,2,281.5,36],
[9,10,1,270,60],
[9,11,1,270,24],
[9,11,2,270,20],
[9,12,1,270,33],
[9,13,1,270,33],
[9,14,1,270,20],
[9,15,1,270,32],
[9,16,1,270,39],
[9,17,1,270,50],
[9,18,1,270,55],
[9,19,1,270,25],
[9,20,1,286.88,9],
[9,21,1,270,16],
[9,22,1,292.5,36],
[9,23,1,315,32],
[9,24,1,270,14.5],
[9,25,1,270,4.5],
[9,25,2,225,17],
[9,26,1,270,8],
[9,26,2,225,6],
[9,26,3,270,17],
[9,27,1,270,24],
[9,28,1,270,14],
[9,29,1,270,24],
[9,30,1,270,14],
[10,1,1,270,25],
[10,2,1,270,39],
[10,3,1,270,47],
[10,4,1,270,63],
[10,5,1,270,57],
[10,6,1,270,40],
[10,7,1,270,23],
[10,7,2,247.5,5],
[10,8,1,247.5,11.5],
[10,9,1,225,5],
[10,9,2,281.25,4],
[10,9,3,247.5,22.5],
[10,10,1,247.5,59],
[10,11,1,247.5,27],
[10,11,2,270,22.5]]

def leagues_to_m(leagues):
    # Returns distance to nearest meter
    return round(leagues * 1609.34 * 3)


def plot_course(course, model, year):
    # Set start position
    lats = [start_lat]
    lons = [start_lon]

    # Loop through each leg
    for leg in course:
        # Pull leg data
        month, day, leg_num, bearing, distance = leg

        # Starts at last position
        starting_loc = (lats[-1], lons[-1])
        # Correct bearing for declination
        d, i, f = local_curve([year], starting_loc, model)
        corrected_bearing = bearing + d[0] -4

        # Calculate new position
        result = geod.Direct(starting_loc[0], starting_loc[1], corrected_bearing, leagues_to_m(distance))

        # Append new position to list
        lats.append(result['lat2'])
        lons.append(result['lon2'])
    
    print(f"Final position: {lats[-1]:.6f}, {lons[-1]:.6f}")

    # Configure plot
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([-100, 10, 0, 50])
    plt.plot(lons, lats, marker='o')

    plt.savefig('declination.png', dpi=150, bbox_inches='tight')

    plt.show()

plot_course(course, model, year)