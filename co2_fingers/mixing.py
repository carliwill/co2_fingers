"""
co2_fingers.mixing
==================
Late-stage mixing analysis — Sherwood number and horizontal slicing.

These functions are independent of the main fingering pipeline and operate
directly on raw image sequences.  They are designed to be called *after*
fingers have hit the bottom of the rig (period C in Hassanzadeh et al. 2007).

Functions
---------
- diffusion_flux          : Theoretical pure-diffusion dc/dt from Carslaw & Jaeger series
- sherwood_number         : Sh(t) = measured flux / diffusion flux
- horizontal_slice_profile: C̄(z, t) from horizontally-averaged pixel intensity
- mixing_fraction         : Per-frame uniformity metric (0 = unmixed, 1 = fully mixed)
- spatial_variance        : σ²(t) across the concentration field
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Diffusion reference
# ---------------------------------------------------------------------------

def diffusion_flux(
    t_sec: np.ndarray,
    H_m: float,
    D: float = 2e-9,
    n_terms: int = 50,
) -> np.ndarray:
    """
    Theoretical pure-diffusion dissolution flux dc/dt from the Carslaw &
    Jaeger series solution for a finite slab.

    The dimensionless flux is:

        d c̄ / dt_D = (4/π) Σ exp[ -((2n+1)/2 · π)² · t_D ]

    converted back to real time via the chain rule dc/dt = (dc/dt_D) · (D/H²).

    Parameters
    ----------
    t_sec : np.ndarray
        Elapsed time in seconds.
    H_m : float
        Aquifer (slab) height in metres.
    D : float
        CO₂ diffusivity in water (m²/s). Default 2e-9.
    n_terms : int
        Number of series terms. 50 is more than sufficient for t_D > 1e-4.

    Returns
    -------
    np.ndarray
        Theoretical flux at each time step (same shape as *t_sec*).
    """
    t_D    = D * np.asarray(t_sec, dtype=float) / H_m**2
    flux   = np.zeros_like(t_D)
    for n in range(n_terms):
        flux += np.exp(-((2*n + 1) / 2 * np.pi)**2 * t_D)
    flux *= (4 / np.pi) * (D / H_m**2)
    return flux


# ---------------------------------------------------------------------------
# Concentration field from images
# ---------------------------------------------------------------------------

def _load_concentration_field(
    image_path: str,
    crop: dict,
    invert: bool = False,
) -> np.ndarray:
    """
    Load one image and return a normalized [0, 1] greyscale concentration field.

    Parameters
    ----------
    image_path : str
    crop : dict
        Must contain keys y_top, y_bot, x_left, x_right.
    invert : bool
        If True, return 1 - intensity (for dyes that darken with concentration).

    Returns
    -------
    np.ndarray  shape (H_crop, W_crop), float32 in [0, 1]
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    c = img[crop['y_top']:crop['y_bot'],
            crop['x_left']:crop['x_right']].astype(np.float32) / 255.0
    return 1.0 - c if invert else c


# ---------------------------------------------------------------------------
# Sherwood number
# ---------------------------------------------------------------------------

def sherwood_number(
    image_paths: list,
    times_sec: np.ndarray,
    crop: dict,
    H_m: float,
    D: float = 2e-9,
    smooth_window: int = 10,
    invert: bool = False,
    n_terms: int = 50,
) -> dict:
    """
    Compute the Sherwood number Sh(t) from an image sequence.

    Sh(t) = [d c̄/dt]_measured / [d c̄/dt]_diffusion

    where c̄(t) is the domain-mean normalized pixel intensity.

    Parameters
    ----------
    image_paths : list of str or Path
        Sorted image paths, one per frame.
    times_sec : np.ndarray
        Elapsed time in seconds, same length as image_paths.
    crop : dict
        Crop parameters: y_top, y_bot, x_left, x_right.
    H_m : float
        Aquifer height in metres.
    D : float
        CO₂ diffusivity (m²/s). Default 2e-9.
    smooth_window : int
        Uniform filter window (frames) applied to c̄(t) before differentiating.
        Larger values reduce noise but may smooth real transitions. Default 10.
    invert : bool
        Set True if your dye darkens with CO₂ concentration. Default False.
    n_terms : int
        Series terms for the diffusion reference. Default 50.

    Returns
    -------
    dict with keys:
        - ``times_sec``   : np.ndarray — input times
        - ``c_bar``       : np.ndarray — raw domain-mean concentration
        - ``c_smooth``    : np.ndarray — smoothed c̄(t)
        - ``dc_dt_meas``  : np.ndarray — measured flux (numerical derivative)
        - ``dc_dt_diff``  : np.ndarray — theoretical diffusion flux
        - ``Sh``          : np.ndarray — Sherwood number time series
        - ``t_Sh_peak``   : float      — time (s) of Sh peak (B→C transition)
        - ``Sh_peak``     : float      — peak Sh value
    """
    times_sec = np.asarray(times_sec, dtype=float)
    c_bar = np.zeros(len(image_paths))

    for i, path in enumerate(image_paths):
        try:
            c = _load_concentration_field(str(path), crop, invert=invert)
            c_bar[i] = c.mean()
        except Exception as e:
            print(f"  SKIP {Path(path).name}: {e}")
            c_bar[i] = np.nan

    c_smooth   = uniform_filter1d(np.nan_to_num(c_bar), size=smooth_window)
    dc_dt_meas = np.gradient(c_smooth, times_sec)
    dc_dt_diff = diffusion_flux(times_sec, H_m, D, n_terms)
    dc_dt_diff = np.clip(dc_dt_diff, 1e-15, None)

    Sh = dc_dt_meas / dc_dt_diff

    valid = np.isfinite(Sh)
    i_peak    = int(np.nanargmax(Sh[valid])) if valid.any() else 0
    t_Sh_peak = float(times_sec[valid][i_peak])
    Sh_peak   = float(Sh[valid][i_peak])

    return {
        "times_sec":  times_sec,
        "c_bar":      c_bar,
        "c_smooth":   c_smooth,
        "dc_dt_meas": dc_dt_meas,
        "dc_dt_diff": dc_dt_diff,
        "Sh":         Sh,
        "t_Sh_peak":  t_Sh_peak,
        "Sh_peak":    Sh_peak,
    }


# ---------------------------------------------------------------------------
# Horizontal slicing
# ---------------------------------------------------------------------------

def horizontal_slice_profile(
    image_paths: list,
    times_sec: np.ndarray,
    crop: dict,
    n_slices: int = 50,
    invert: bool = False,
) -> dict:
    """
    Compute the horizontally-averaged concentration profile C̄(z, t).

    At each depth bin z, averages pixel intensity across the full width of
    the crop.  Returns a 2-D array (n_slices × n_frames) suitable for a
    Hovmöller diagram.

    Parameters
    ----------
    image_paths : list
        Sorted image paths.
    times_sec : np.ndarray
        Elapsed time per frame in seconds.
    crop : dict
        y_top, y_bot, x_left, x_right.
    n_slices : int
        Number of horizontal depth bins. Default 50.
    invert : bool
        Invert intensity → concentration. Default False.

    Returns
    -------
    dict with keys:
        - ``C_z_t``          : np.ndarray shape (n_slices, n_frames)
        - ``times_sec``      : np.ndarray
        - ``depths_m``       : np.ndarray — depth of each slice mid-point (0 = top)
                               in normalised units [0, 1]
        - ``mixing_fraction``: np.ndarray — per-frame mixing metric (0–1)
        - ``sigma2``         : np.ndarray — spatial variance per frame
        - ``t_95pct``        : float or None — time (s) when mixing_fraction ≥ 0.95
    """
    times_sec = np.asarray(times_sec, dtype=float)
    n_frames  = len(image_paths)
    C_z_t     = np.full((n_slices, n_frames), np.nan)

    for j, path in enumerate(image_paths):
        try:
            c = _load_concentration_field(str(path), crop, invert=invert)
            bands = np.array_split(c, n_slices, axis=0)
            C_z_t[:, j] = np.array([b.mean() for b in bands])
        except Exception as e:
            print(f"  SKIP {Path(path).name}: {e}")

    depths_m = np.linspace(0, 1, n_slices)

    # mixing fraction: 1 - normalised range of vertical profile
    C_min = np.nanmin(C_z_t, axis=0)
    C_max = np.nanmax(C_z_t, axis=0)
    max_range = np.nanmax(C_max - C_min)
    mixing_fraction = 1.0 - (C_max - C_min) / (max_range + 1e-12)

    sigma2 = np.nanvar(C_z_t, axis=0)

    idx_95 = np.where(mixing_fraction >= 0.95)[0]
    t_95pct = float(times_sec[idx_95[0]]) if len(idx_95) else None

    return {
        "C_z_t":           C_z_t,
        "times_sec":       times_sec,
        "depths_norm":     depths_m,
        "mixing_fraction": mixing_fraction,
        "sigma2":          sigma2,
        "t_95pct":         t_95pct,
    }


# ---------------------------------------------------------------------------
# High-level callable functions (config-driven)
# ---------------------------------------------------------------------------

def _load_inputs(config_path, csv_path, image_dir):
    """Load config, timestamps, and matched sorted image paths."""
    import pandas as pd
    from .config import load_config

    cfg = load_config(config_path)
    crop = {
        'y_top':   cfg.crop.y_top,
        'y_bot':   cfg.crop.y_bot,
        'x_left':  cfg.crop.x_left,
        'x_right': cfg.crop.x_right,
    }

    df_t = pd.read_csv(csv_path, sep=None, engine='python')
    df_t.columns = df_t.columns.str.strip()
    fname_to_h = dict(zip(df_t['Filename'], df_t['TimeSinceStart(min)'] / 60.0))

    all_paths = sorted(Path(image_dir).glob('*.JPG'))
    matched   = [(p, fname_to_h[p.name]) for p in all_paths if p.name in fname_to_h]
    if not matched:
        raise ValueError("No images matched timestamps — check CSV Filename column.")

    image_paths = [m[0] for m in matched]
    times_h     = np.array([m[1] for m in matched])
    times_sec   = times_h * 3600.0
    return image_paths, times_h, times_sec, crop, cfg.name


def run_sherwood(
    config_path: str,
    csv_path: str,
    image_dir: str,
    H_m: float = 0.27,
    D: float = 2e-9,
    t_bottom_h: float | None = None,
    smooth_window: int = 10,
    invert: bool = False,
) -> dict:
    """
    Load images, compute and plot the Sherwood number time series.

    Parameters
    ----------
    config_path : str
        Path to experiment YAML config.
    csv_path : str
        Path to timestamp CSV (columns: Filename, TimeSinceStart(min)).
    image_dir : str
        Directory of JPG images.
    H_m : float
        Aquifer height in metres (default 0.27).
    D : float
        CO₂ diffusivity in water m²/s (default 2e-9).
    t_bottom_h : float or None
        Hours when first finger hits bottom — drawn as reference line.
    smooth_window : int
        Smoothing window in frames before differentiating (default 10).
    invert : bool
        Invert pixel intensity if dye darkens with concentration.

    Returns
    -------
    dict
        Full output from :func:`sherwood_number`.
    """
    import matplotlib.pyplot as plt

    image_paths, times_h, times_sec, crop, exp_name = _load_inputs(
        config_path, csv_path, image_dir
    )
    print(f"Computing Sherwood number for {exp_name} ({len(image_paths)} frames)...")

    sh = sherwood_number(
        image_paths, times_sec, crop,
        H_m=H_m, D=D,
        smooth_window=smooth_window,
        invert=invert,
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(times_h, sh['Sh'], color='#7eb8f7', lw=1.5, label='Sh(t)')
    ax.axhline(1, color='gray', ls='--', lw=1, label='Sh = 1 (pure diffusion)')
    if t_bottom_h is not None:
        ax.axvline(t_bottom_h, color='#cc2222', lw=1.8,
                   label=f'First finger hits bottom  ({t_bottom_h} h)')
    t_peak_h = sh['t_Sh_peak'] / 3600
    ax.axvline(t_peak_h, color='#7af5a0', lw=1.5, ls='--',
               label=f'Sh peak = {sh["Sh_peak"]:.2f}  at {t_peak_h:.2f} h')
    ax.set(xlabel='Time (h)', ylabel='Sh(t)', title=f'{exp_name} — Sherwood Number')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"Sh peak = {sh['Sh_peak']:.2f}  at  {t_peak_h:.2f} h  ← B→C transition")
    return sh


def run_horizontal_slicing(
    config_path: str,
    csv_path: str,
    image_dir: str,
    H_m: float = 0.27,
    t_start_h: float | None = None,
    n_slices: int = 50,
    invert: bool = False,
) -> dict:
    """
    Compute and plot horizontal slicing diagnostics (Hovmöller, mixing
    fraction, spatial variance).

    Parameters
    ----------
    config_path : str
        Path to experiment YAML config.
    csv_path : str
        Path to timestamp CSV.
    image_dir : str
        Directory of JPG images.
    H_m : float
        Aquifer height in metres (default 0.27).
    t_start_h : float or None
        Restrict to frames at or after this time (hours). Use your
        T_BOTTOM_H value to focus on post-fingering mixing.
    n_slices : int
        Number of horizontal depth bins (default 50).
    invert : bool
        Invert pixel intensity if dye darkens with concentration.

    Returns
    -------
    dict
        Full output from :func:`horizontal_slice_profile`.
    """
    import matplotlib.pyplot as plt

    image_paths, times_h, times_sec, crop, exp_name = _load_inputs(
        config_path, csv_path, image_dir
    )

    if t_start_h is not None:
        mask        = times_h >= t_start_h
        image_paths = [p for p, m in zip(image_paths, mask) if m]
        times_h     = times_h[mask]
        times_sec   = times_sec[mask]
        print(f"Restricting to {len(image_paths)} frames after {t_start_h} h...")

    print(f"Computing horizontal slices for {exp_name} "
          f"({len(image_paths)} frames, {n_slices} depth bins)...")

    hs = horizontal_slice_profile(
        image_paths, times_sec, crop,
        n_slices=n_slices,
        invert=invert,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'{exp_name} — Late-stage mixing', fontweight='bold')

    axes[0].imshow(
        hs['C_z_t'], aspect='auto', origin='upper', cmap='Blues',
        extent=[times_h[0], times_h[-1], H_m, 0],
    )
    axes[0].set(xlabel='Time (h)', ylabel='Depth (norm.)', title='C̄(z,t) — Hovmöller')

    axes[1].plot(times_h, hs['mixing_fraction'], color='#7af5a0', lw=2)
    axes[1].axhline(0.95, color='#f55a5a', ls='--', lw=1, label='95% mixed')
    if hs['t_95pct'] is not None:
        t95h = hs['t_95pct'] / 3600
        axes[1].axvline(t95h, color='#f55a5a', lw=1.5, ls=':',
                        label=f'95% at {t95h:.2f} h')
    axes[1].set(xlabel='Time (h)', ylabel='Mixing fraction',
                title='Mixing progress', ylim=(0, 1.05))
    axes[1].legend(fontsize=8)

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
