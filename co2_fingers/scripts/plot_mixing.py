"""
scripts/plot_mixing.py
======================
Standalone script for late-stage mixing analysis.
Run independently — NOT called by run_analysis.py.

Usage (Colab / Jupyter):
    %run scripts/plot_mixing.py
or import and call directly:
    from scripts.plot_mixing import run_sherwood, run_horizontal_slicing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import co2_fingers as cf
from co2_fingers.config import load_config
from co2_fingers.mixing import sherwood_number, horizontal_slice_profile


# ============================================================
#  CONFIG — edit for each experiment
# ============================================================
CONFIG_PATH = '/content/drive/My Drive/CO2_diff/configs/C3R5.yaml'
CSV_PATH    = '/content/drive/My Drive/CO2_diff/time_csvs/c3r5_time.csv'
IMAGE_DIR   = '/content/drive/My Drive/CO2_diff/Experiment images/C3/C3R5'

H_M          = 0.27     # aquifer height in metres
D            = 2e-9     # CO2 diffusivity in water (m²/s)
T_BOTTOM_H   = 23.3     # h — when first finger hits bottom; used as x-ref line
N_SLICES     = 50       # depth bins for horizontal slicing
SMOOTH_WIN   = 10       # frames to smooth c̄(t) before differentiating
INVERT       = False    # True if dye darkens with CO2 concentration
# ============================================================


def _load_inputs(config_path, csv_path, image_dir):
    """Load config, timestamps, and matched sorted image paths."""
    cfg = load_config(config_path)
    crop = {
        'y_top':   cfg.crop.y_top,
        'y_bot':   cfg.crop.y_bot,
        'x_left':  cfg.crop.x_left,
        'x_right': cfg.crop.x_right,
    }

    df_t = pd.read_csv(csv_path, sep=None, engine='python')
    df_t.columns = df_t.columns.str.strip()
    fname_col = 'Filename'
    time_col  = 'TimeSinceStart(min)'
    fname_to_h = dict(zip(df_t[fname_col], df_t[time_col] / 60.0))

    all_paths = sorted(Path(image_dir).glob('*.JPG'))
    matched   = [(p, fname_to_h[p.name]) for p in all_paths if p.name in fname_to_h]
    if not matched:
        raise ValueError("No images matched timestamps — check CSV column names.")

    image_paths = [m[0] for m in matched]
    times_h     = np.array([m[1] for m in matched])
    times_sec   = times_h * 3600.0
    exp_name    = cfg.name
    return image_paths, times_h, times_sec, crop, exp_name


def run_sherwood():
    """Compute and plot the Sherwood number time series."""
    image_paths, times_h, times_sec, crop, exp_name = _load_inputs(
        CONFIG_PATH, CSV_PATH, IMAGE_DIR
    )

    print(f"Computing Sherwood number for {exp_name}  "
          f"({len(image_paths)} frames)...")
    sh = sherwood_number(
        image_paths, times_sec, crop,
        H_m=H_M, D=D,
        smooth_window=SMOOTH_WIN,
        invert=INVERT,
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(times_h, sh['Sh'], color='#7eb8f7', lw=1.5, label='Sh(t)')
    ax.axhline(1, color='gray', ls='--', lw=1, label='Sh = 1 (pure diffusion)')
    if T_BOTTOM_H is not None:
        ax.axvline(T_BOTTOM_H, color='#cc2222', lw=1.8,
                   label=f'First finger hits bottom  ({T_BOTTOM_H} h)')
    t_peak_h = sh['t_Sh_peak'] / 3600
    ax.axvline(t_peak_h, color='#7af5a0', lw=1.5, ls='--',
               label=f'Sh peak = {sh["Sh_peak"]:.2f}  at {t_peak_h:.2f} h')
    ax.set(xlabel='Time (h)', ylabel='Sh(t)',
           title=f'{exp_name} — Sherwood Number')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"\nSh peak = {sh['Sh_peak']:.2f}  at  {t_peak_h:.2f} h  ← B→C transition")
    return sh


def run_horizontal_slicing(t_start_h=None):
    """
    Compute and plot horizontal slicing diagnostics.

    Parameters
    ----------
    t_start_h : float or None
        If given, restrict analysis to frames after this time (hours).
        Defaults to T_BOTTOM_H.
    """
    if t_start_h is None:
        t_start_h = T_BOTTOM_H

    image_paths, times_h, times_sec, crop, exp_name = _load_inputs(
        CONFIG_PATH, CSV_PATH, IMAGE_DIR
    )

    if t_start_h is not None:
        mask        = times_h >= t_start_h
        image_paths = [p for p, m in zip(image_paths, mask) if m]
        times_h     = times_h[mask]
        times_sec   = times_sec[mask]
        print(f"Restricting to {len(image_paths)} frames after {t_start_h} h...")

    print(f"Computing horizontal slices for {exp_name}  "
          f"({len(image_paths)} frames, {N_SLICES} depth bins)...")
    hs = horizontal_slice_profile(
        image_paths, times_sec, crop,
        n_slices=N_SLICES,
        invert=INVERT,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{exp_name} — Late-stage mixing', fontweight='bold')

    # Hovmöller
    axes[0].imshow(
        hs['C_z_t'], aspect='auto', origin='upper', cmap='Blues',
        extent=[times_h[0], times_h[-1], H_M, 0],
    )
    axes[0].set(xlabel='Time (h)', ylabel='Depth (norm.)',
                title='C̄(z,t) — Hovmöller')

    # Mixing fraction
    axes[1].plot(times_h, hs['mixing_fraction'], color='#7af5a0', lw=2)
    axes[1].axhline(0.95, color='#f55a5a', ls='--', lw=1, label='95% mixed')
    if hs['t_95pct'] is not None:
        t95h = hs['t_95pct'] / 3600
        axes[1].axvline(t95h, color='#f55a5a', lw=1.5, ls=':',
                        label=f'95% at {t95h:.2f} h')
    axes[1].set(xlabel='Time (h)', ylabel='Mixing fraction',
                title='Mixing progress', ylim=(0, 1.05))
    axes[1].legend(fontsize=8)

    # Spatial variance
    axes[2].plot(times_h, hs['sigma2'], color='#f7c97e', lw=2)
    axes[2].set(xlabel='Time (h)', ylabel='σ²(t)',
                title='Spatial variance → 0 = fully mixed')

    plt.tight_layout()
    plt.show()

    if hs['t_95pct'] is not None:
        print(f"95% mixed at t = {hs['t_95pct']/3600:.2f} h")
    else:
        print("95% mixing threshold not reached in this window.")
    return hs


if __name__ == "__main__":
    sh = run_sherwood()
    hs = run_horizontal_slicing()
