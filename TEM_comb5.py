
# -*- coding: utf-8 -*-
"""
Combined TEM Inversion Script
- Method A: Offset-optimized curve fitting (from TEM_001a.py)
- Method B: Combo-D multiple fit inversion (from TEM_inv7a.py)
- Unified outputs: decay fits, residuals, and resistivity profiles with median in red
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm import tqdm
import empymod
import logging
import os

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data paths and site definitions ---
file_path = 'c:/WRDAPP/data001/002/TEM_Ojos2023b.npy'

layer_thicknesses_dict = {
    'Atacama': [3, 17, 80],
    'Tejos50': [1, 24, 65, 135],
    'Murray': [3, 50, 95]
}

initial_models = {
    'Atacama': [5000, 50000, 8000],
    'Tejos50': [5000, 50000, 8000, 5000],
    'Murray': [2000, 2000, 8000]
}

best_fits = {
    'Atacama': 'AtacamaStation1',
    'Tejos50': 'Tejos50Station1',
    'Murray': 'MurrayStation1'
}

site_titles = {
    'Atacama': 'Atacama 5,230 m ASL',
    'Murray': 'Murray 4,500 m ASL',
    'Tejos50': 'Tejos 5,830 m ASL'
}

# --- Data Loader ---
def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    logging.info(f"Loading data from {file_path}")
    return np.load(file_path, allow_pickle=True).item()

# --- Preprocessing for Combo-D ---
def preprocess_combo_d(time, emf, mad_multiplier=4, min_sg_window=5):
    mask = (time > 0) & (np.abs(emf) > 1e-30)
    time, emf = time[mask], emf[mask]
    emf = emf / 1e6
    log_time = np.log10(time)
    emf_log = np.log10(np.abs(emf))

    if np.abs(np.mean(np.gradient(emf_log))) < 0.05:
        return time, emf

    median_val = np.median(emf_log)
    emf_log -= median_val
    mad = np.median(np.abs(emf_log - np.median(emf_log)))
    mask_mad = np.abs(emf_log) < mad_multiplier * mad
    time, emf_log = time[mask_mad], emf_log[mask_mad]
    N = len(time)
    window_length = max(min_sg_window, (N // 10) * 2 + 1)
    window_length = min(window_length, N-1 if N % 2 == 0 else N)
    if window_length % 2 == 0:
        window_length += 1
    logt_new = np.linspace(np.log10(time.min()), np.log10(time.max()), len(time))
    interp_func = interp1d(np.log10(time), emf_log, kind='linear', fill_value='extrapolate')
    emf_log_interp = interp_func(logt_new)
    emf_log_smooth = savgol_filter(emf_log_interp, window_length=window_length, polyorder=2, mode='nearest')
    time_new = 10**logt_new
    emf_new = 10**(emf_log_smooth + median_val)
    return time_new, emf_new

# --- Forward model ---
def empymod_forward_model(times, layer_thicknesses, resistivities, offset=25):
    depth = np.cumsum(layer_thicknesses[:-1])
    src = [0, 0, 0]
    rec = [offset, 0, 0.01]
    emf = empymod.dipole(src=src, rec=rec, depth=depth, res=resistivities,
                         freqtime=times, signal=-1, strength=1, mrec=True, verb=0)
    return emf

# --- Method B: Combo-D ---
def inversion_combo_d(data, site_key, station_key, n_tries=3):
    t_raw = data[station_key]['time']
    emf_raw = data[station_key]['emf']
    t_obs, emf_obs = preprocess_combo_d(t_raw, emf_raw)

    emf_obs_log = np.log10(np.abs(emf_obs))
    emf_obs_log[~np.isfinite(emf_obs_log)] = -30

    initial_thicknesses = layer_thicknesses_dict[site_key]
    initial_resistivities = initial_models[site_key]
    n_thicknesses = len(initial_thicknesses)
    n_layers = len(initial_resistivities)
    all_fits = []

    def model_func_log(t, scale, *params):
        thicknesses = list(params[:n_thicknesses])
        resistivities = list(params[n_thicknesses:])
        synthetic = empymod_forward_model(t, thicknesses, resistivities)
        synthetic = np.abs(synthetic)
        synthetic[synthetic < 1e-30] = 1e-30
        return np.log10(scale * synthetic)

    for attempt in tqdm(range(n_tries), desc=f"Inversion Combo-D: {site_key}"):
        np.random.seed(attempt)
        thickness_guess = (np.array(initial_thicknesses) * np.random.uniform(0.8, 1.2, n_thicknesses)).tolist()
        resistivity_guess = (np.array(initial_resistivities) * np.random.uniform(0.5, 2.0, n_layers)).tolist()
        p0 = [1.0] + thickness_guess + resistivity_guess
        bounds_lower = [1e-5] + [0.5 * t for t in initial_thicknesses] + [1] * n_layers
        bounds_upper = [1e5] + [2.0 * t for t in initial_thicknesses] + [1e6] * n_layers
        try:
            popt, _ = curve_fit(model_func_log, t_obs, emf_obs_log, p0=p0,
                                bounds=(bounds_lower, bounds_upper), maxfev=50000)
            residual = np.sum((emf_obs_log - model_func_log(t_obs, *popt)) ** 2)
            all_fits.append({'popt': popt, 'residual': residual})
        except:
            continue

    if not all_fits:
        logging.warning(f"No successful fits for {site_key}")
        return

    all_fits.sort(key=lambda x: x['residual'])
    thicknesses_arr, resistivities_arr = [], []
    for fit in all_fits:
        p = fit['popt']
        thicknesses_arr.append(p[1:1+n_thicknesses])
        resistivities_arr.append(p[1+n_thicknesses:])

    resistivities_arr = np.array(resistivities_arr)
    median_res = np.median(resistivities_arr, axis=0)
    mean_thickness = np.mean(np.array(thicknesses_arr), axis=0)
    top = np.insert(np.cumsum(mean_thickness[:-1]), 0, 0)
    bottom = np.cumsum(mean_thickness)

    
    # --- Plot: Fit curves ---
    n_curves = len(all_fits)
    plt.figure(figsize=(7, 5))
    plt.loglog(t_obs, np.abs(emf_obs), '.', label='Observed', color='black')
    r2_best = None
    for idx, fit in enumerate(all_fits):
        popt = fit['popt']
        emf_fit = 10**model_func_log(t_obs, *popt)
        if idx == 0:
            # Calculate R^2 for the best fit
            ss_res = np.sum((emf_obs_log - model_func_log(t_obs, *popt)) ** 2)
            ss_tot = np.sum((emf_obs_log - np.mean(emf_obs_log)) ** 2)
            r2_best = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
            plt.loglog(t_obs, np.abs(emf_fit), '-', lw=2.5, label=f'Best Fit (R²={r2_best:.3f})', color='red')
        else:
            plt.loglog(t_obs, np.abs(emf_fit), '-', lw=1, alpha=0.5, color='gray')
    plt.xlabel('Time (s)')
    plt.ylabel('EMF (V/m²)')
    plt.title(f"{site_titles[site_key]} - Fits (n = {n_curves})")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{site_key}_ComboD_fitCurves.png")
    plt.show()
    

    # --- Plot: Residuals of best fit ---
    emf_fit_best = 10**model_func_log(t_obs, *all_fits[0]['popt'])
    resids = emf_obs_log - np.log10(emf_fit_best)
    plt.figure(figsize=(6, 3))
    plt.plot(t_obs, resids, 'o-', color='blue', label='Residuals')
    plt.axhline(0, color='gray', ls='--')
    plt.xscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Log Residual')
    plt.title(f"Residuals - {site_key}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{site_key}_ComboD_residuals.png")
    plt.show()

    # --- Plot: Resistivity profiles ---

    plt.figure(figsize=(5, 7))
    for r in resistivities_arr:
        for i in range(len(r)):
            plt.plot([r[i], r[i]], [top[i], bottom[i]], color='gray', lw=1, alpha=0.5)
            if i < len(r)-1:
                plt.plot([r[i], r[i+1]], [bottom[i], bottom[i]], color='gray', lw=1, alpha=0.5)
    for i in range(len(median_res)):
        plt.plot([median_res[i], median_res[i]], [top[i], bottom[i]], color='red', lw=2.5)
        if i < len(median_res)-1:
            plt.plot([median_res[i], median_res[i+1]], [bottom[i], bottom[i]], color='red', lw=2.5)

    plt.gca().invert_yaxis()
    plt.xscale('log')
    plt.xlabel('Resistivity (Ω·m)')
    plt.ylabel('Depth (m)')
    plt.title(f"Resistivity Profiles - {site_titles[site_key]}")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(f"{site_key}_medianRes.png")
    plt.show()

# --- Run all ---
if __name__ == "__main__":
    data = load_data(file_path)
    for site_key, station_key in best_fits.items():
        inversion_combo_d(data, site_key, station_key)
