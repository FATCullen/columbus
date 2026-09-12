import math
import copy
import tempfile
import os

import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from geographiclib.geodesic import Geodesic
from pymagglobal import Model, local_curve

geod = Geodesic.WGS84

# If a different model is used you'd have to modify the parser
MODEL_PATH = 'CALS3k.4b'
YEAR = 1492
# Monte Carlo parameters
N_TRIALS = 500
RNG_SEED = 42

# In between Gomera and Tenerife, Canary Islands
start_lat, start_lon = 28.168932, -16.968030

# [Month, Day, Leg, Bearing, Distance (leagues, converted from geometric units)]
course = [
    [9,8,1,270,9],[9,9,1,270,18],[9,9,2,281.5,36],[9,10,1,270,60],
    [9,11,1,270,24],[9,11,2,270,20],[9,12,1,270,33],[9,13,1,270,33],
    [9,14,1,270,20],[9,15,1,270,32],[9,16,1,270,39],[9,17,1,270,50],
    [9,18,1,270,55],[9,19,1,270,25],[9,20,1,286.88,9],[9,21,1,270,16],
    [9,22,1,292.5,36],[9,23,1,315,32],[9,24,1,270,14.5],[9,25,1,270,4.5],
    [9,25,2,225,17],[9,26,1,270,8],[9,26,2,225,6],[9,26,3,270,17],
    [9,27,1,270,24],[9,28,1,270,14],[9,29,1,270,24],[9,30,1,270,14],
    [10,1,1,270,25],[10,2,1,270,39],[10,3,1,270,47],[10,4,1,270,63],
    [10,5,1,270,57],[10,6,1,270,40],[10,7,1,270,23],[10,7,2,247.5,5],
    [10,8,1,247.5,11.5],[10,9,1,225,5],[10,9,2,281.25,4],
    [10,9,3,247.5,22.5],[10,10,1,247.5,59],[10,11,1,247.5,27],
    [10,11,2,270,22.5],
]

def leagues_to_m(leagues):
    # Returns distance to nearest meter
    return round(leagues * 1609.34 * 3)

def parse_cals3k_4b(path):
    """
    Custom parser, specific to CALS3k.4b formatting:
        knot_years : 1-D array of spline knot epochs
        mu         : 2-D array [n_knots, n_coeffs] of mean Gauss coefficients
        sigma      : 2-D array [n_knots, n_coeffs] of 1-sigma uncertainties
    """
    with open(path, 'r') as f:
        line1 = f.readline().strip().split()
        line2 = f.readline().strip().split()

        t_start = float(line1[0])
        t_end = float(line1[1])

        n_degree = int(line2[0])
        n_splines = int(line2[2])

        n_coeffs = (n_degree + 1) ** 2 - 1

        knot_years = np.array(list(map(float, line2[3:])))

        f.readline()  # skip blank line

        mu = np.array(list(map(float, f.readline().split())))
        mu = mu.reshape(n_splines, n_coeffs)

        sigma = np.array(list(map(float, f.readline().split())))
        sigma = sigma.reshape(n_splines, n_coeffs)

    return knot_years, mu, sigma, n_degree, n_splines, t_start, t_end


def write_model_file(path, knot_years, coeffs, n_degree, n_splines, t_start, t_end):
    """
    Write a pymagglobal-compatible model file with the given coefficient block
    """
    n_coeffs = coeffs.shape[1]
    with open(path, 'w') as f:
        # Line 1
        f.write(f"  {t_start:.1f}  {t_end:.1f}\n")
        # Line 2
        f.write(f"  {n_degree}   0   {n_splines}\n")
        # Knot years
        for i, yr in enumerate(knot_years):
            f.write(f"  {yr:.10f}")
        f.write("\n")
        # Mean coefficients
        for i, val in enumerate(coeffs.flat):
            f.write(f"  {val:.15e}")
        f.write("\n")

def simulate_voyage(model, year=YEAR):
    """
    Simulate the voyage using the given model, returning lists of latitudes and longitudes.
    """
    lats = [start_lat]
    lons = [start_lon]
    for leg in course:
        _, _, _, bearing, distance = leg
        loc = (lats[-1], lons[-1])
        d, i, f = local_curve([year], loc, model)
        corrected = bearing + d[0]
        result = geod.Direct(loc[0], loc[1], corrected, leagues_to_m(distance))
        lats.append(result['lat2'])
        lons.append(result['lon2'])
    return lats, lons

def run_monte_carlo(model_path, n_trials=N_TRIALS, seed=RNG_SEED):
    """
    Monte Carlo sim
    Perturbs model coefficients based on uncertainties for each trial
    """
    rng = np.random.default_rng(seed)

    knot_years, mu, sigma, n_degree, n_splines, t_start, t_end = parse_cals3k_4b(model_path)

    # Baseline (mean model)
    baseline_model = Model('CALS3k.4b', fname=model_path)
    base_lats, base_lons = simulate_voyage(baseline_model)
    base_final = (base_lats[-1], base_lons[-1])
    print(f"Baseline final position: {base_final[0]:.4f}°N, {base_final[1]:.4f}°E\n")

    # To store trial results
    final_lats = []
    final_lons = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, 'perturbed_model.dat')

        for trial in range(n_trials):
            if (trial + 1) % 50 == 0:
                print(f"Trial {trial + 1}/{n_trials} ...")

            # Array of normal standard normally distributed perturbations in shape of coefficients
            perturbation = rng.standard_normal(mu.shape)
            # Perturb coefficients based on uncertainties and normal random draw
            coeffs_perturbed = mu + sigma * perturbation

            # Write temporary model file with perturbed coefficients
            write_model_file(tmp_path, knot_years, coeffs_perturbed,
                             n_degree, n_splines, t_start, t_end)

            try:
                # Run voyage simulation with perturbed model
                trial_model = Model('perturbed', fname=tmp_path)
                lats, lons = simulate_voyage(trial_model)
                final_lats.append(lats[-1])
                final_lons.append(lons[-1])
            except Exception as e:
                print(f"Warning: trial {trial+1} failed ({e}), skipping")

    final_lats = np.array(final_lats)
    final_lons = np.array(final_lons)

    return base_lats, base_lons, base_final, final_lats, final_lons

def print_statistics(base_final, final_lats, final_lons):
    """
    Prints statistics on the Monte Carlo results
    """
    mean_lat = np.mean(final_lats)
    mean_lon = np.mean(final_lons)
    best_fit = np.polyfit(final_lons, final_lats, 1)

    # Distance from baseline to MC mean
    dist_result = geod.Inverse(base_final[0], base_final[1], mean_lat, mean_lon)
    mean_offset_km = dist_result['s12'] / 1000

    # 1-sigma and 2-sigma spread in km (approximate, using lat std as proxy)
    dists_km = []
    for la, lo in zip(final_lats, final_lons):
        r = geod.Inverse(mean_lat, mean_lon, la, lo)
        dists_km.append(r['s12'] / 1000)
    dists_km = np.array(dists_km)

    print("\n" + "="*55)
    print("Monte Carlo Results")
    print("="*55)
    print(f"  Trials completed.  : {len(final_lats)}")
    print(f"  Baseline final     : {base_final[0]:.4f}°N, {base_final[1]:.4f}°E")
    print(f"  MC mean final      : {mean_lat:.4f}°N, {mean_lon:.4f}°E")
    print(f"  Offset (base->mean): {mean_offset_km:.1f} km")
    print(f"  Best fit slope     : {best_fit[0]:.4f}")
    print(f"  Radial 1std spread : {np.percentile(dists_km, 68.3):.1f} km")
    print(f"  Radial 2std spread : {np.percentile(dists_km, 95.4):.1f} km")
    print(f"  Radial 3std spread : {np.percentile(dists_km, 99.7):.1f} km")
    print("="*55)

def plot_results(base_lats, base_lons, base_final, final_lats, final_lons):
    """
    Plots the Monte Carlo results
    """
    mean_lat = np.mean(final_lats)
    mean_lon = np.mean(final_lons)
    best_fit = np.polyfit(final_lons, final_lats, 1, cov=True)
    std_dist_km = np.std([geod.Inverse(mean_lat, mean_lon, la, lo)['s12'] / 1000 for la, lo in zip(final_lats, final_lons)])

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.set_extent([-80, -60, 10, 30])
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    # MC scatter of final positions
    ax.scatter(final_lons, final_lats,
               s=4, alpha=0.3, color='steelblue', zorder=3,
               label=f'MC endpoints (n={len(final_lats)})',
               transform=ccrs.PlateCarree())

    # MC mean final position
    ax.plot(mean_lon, mean_lat, 'b^', markersize=10, zorder=6,
            label=f'MC mean: ({mean_lat:.2f}°N, {mean_lon:.2f}°E)',
            transform=ccrs.PlateCarree())

    # Baseline voyage track
    ax.plot(base_lons, base_lats,
            'r-o', markersize=3, linewidth=1.5, zorder=4,
            label='Baseline track (mean model)',
            transform=ccrs.PlateCarree())

    # Baseline final position
    ax.plot(base_final[1], base_final[0], 'r*', markersize=14, zorder=7,
            label=f'Baseline: ({base_final[0]:.2f}°N, {base_final[1]:.2f}°E)',
            transform=ccrs.PlateCarree())

    # Best fit line, with marks at std distances
    best_fit = np.polyfit(final_lons, final_lats, 1, cov=True)
        std_dist_km = np.std([geod.Inverse(mean_lat, mean_lon, la, lo)['s12'] / 1000 for la, lo in zip(final_lats, final_lons)])
    ax.plot(final_lons, best_fit[0][0] * final_lons + best_fit[0][1],
            'k--', linewidth=1.5, zorder=5,
            label=f'Best fit line (slope={best_fit[0][0]:.4f})',
            transform=ccrs.PlateCarree())
    ax.

    ax.legend(loc='lower left', fontsize=9)
    ax.set_title(
        f"Columbus 1492 voyage\n"
        f"(n={len(final_lats)} trials, year={YEAR})",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig('columbus_monte_carlo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to columbus_monte_carlo.png")

if __name__ == '__main__':
    base_lats, base_lons, base_final, final_lats, final_lons = run_monte_carlo(
        MODEL_PATH, n_trials=N_TRIALS
    )
    print_statistics(base_final, final_lats, final_lons)
    plot_results(base_lats, base_lons, base_final, final_lats, final_lons)