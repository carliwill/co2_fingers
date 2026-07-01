"""
co2_fingers.fingers
===================
Finger detection, roughness statistics, width measurement, and
tip-depth extraction.

All functions replicate the original notebook logic exactly.
"""

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ---------------------------------------------------------------------------
# Interface statistics
# ---------------------------------------------------------------------------

def roughness_sigma(smooth_interface: np.ndarray, y_bar: np.ndarray) -> float:
    """
    Compute interface roughness σ_h = std(smooth_interface − y_bar).

    Parameters
    ----------
    smooth_interface : np.ndarray
        Smoothed y(x) interface values.
    y_bar : np.ndarray
        Mean / baseline front at same x positions.

    Returns
    -------
    float
        Standard deviation of the residual (pixels).
    """
    return float(np.nanstd(smooth_interface - y_bar))


def mean_front(
    y: np.ndarray,
    window_frac: float = 0.6,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Estimate the smooth mean front y_bar(x) using a Savitzky–Golay filter.

    Parameters
    ----------
    y : np.ndarray
        1-D interface signal.
    window_frac : float
        Fraction of signal length used as the Savitzky–Golay window
        (default 0.6).  Window is forced to be odd.
    polyorder : int
        Polynomial order for the Savitzky–Golay filter (default 2).

    Returns
    -------
    np.ndarray
        Smoothed baseline array, same shape as *y*.
    """
    n = len(y)
    w = max(3, int(window_frac * n))
    if w % 2 == 0:
        w += 1
    return savgol_filter(y, window_length=w, polyorder=polyorder)


# ---------------------------------------------------------------------------
# Finger detection
# ---------------------------------------------------------------------------

def detect_fingers(
    smooth_interface: np.ndarray,
    residual: np.ndarray,
    x_vals: np.ndarray,
    distance: int = 30,
    prominence_override: float | None = None,
    roughness_k: float = 2.5,
    baseline_y_bar: np.ndarray | None = None,
) -> np.ndarray:
    """
    Detect finger indices along the smoothed interface.

    Uses ``scipy.signal.find_peaks`` on ``smooth_interface`` (y increases
    downward, so fingers are *peaks*).  Prominence is set to
    ``roughness_k x std(residual)`` unless overridden.

    Optionally filters out fingers that do not protrude below the static
    baseline ``baseline_y_bar`` (mirrors the notebook logic for C2R5).

    Parameters
    ----------
    smooth_interface : np.ndarray
        Smoothed y(x) interface values (output of
        :func:`~co2_fingers.interface.gauss_outline`).
    residual : np.ndarray
        Residual = ``smooth_interface - y_bar`` for the *baseline* frame.
    x_vals : np.ndarray
        Corresponding x-coordinates (pixels).
    distance : int
        Minimum horizontal distance between detected fingers in pixels
        (default 30).
    prominence_override : float or None
        If supplied, use this fixed prominence instead of computing it from
        *residual* (default None).
    roughness_k : float
        Multiplier applied to ``std(residual)`` to set prominence when
        *prominence_override* is None (default 2.5).
    baseline_y_bar : np.ndarray or None
        If supplied, fingers whose tip y-value does not exceed the baseline
        y_bar at that x are filtered out.

    Returns
    -------
    np.ndarray
        Integer array of indices into *smooth_interface* / *x_vals* where
        fingers were detected.
    """
    
    prom = prominence_override if prominence_override is not None \
            else 0

    valley_idx, _ = find_peaks(smooth_interface, distance=distance, prominence=prom)

    if baseline_y_bar is not None:
        y_bar_at_fingers = np.interp(x_vals[valley_idx], x_vals, baseline_y_bar)
        valley_idx = valley_idx[smooth_interface[valley_idx] > y_bar_at_fingers]

    return valley_idx


# ---------------------------------------------------------------------------
# Finger widths
# ---------------------------------------------------------------------------

def measure_finger_widths(
    smooth_interface: np.ndarray,
    x_vals: np.ndarray,
    finger_idx: np.ndarray,
    px_per_metre: float,
    min_width_cm: float,
    max_width_cm: float,
) -> dict:
    """
    Measure finger widths from shoulder to shoulder, filtering by width range.

    A "shoulder" is the local minimum of ``smooth_interface`` (i.e. local
    maximum of ``-smooth_interface``) nearest to each finger tip.  Width is
    the horizontal distance between the left and right shoulder.

    Fingers whose shoulder-to-shoulder width falls outside
    [*min_width_cm*, *max_width_cm*] are excluded from the accepted count
    and statistics, but are retained in ``rejected_fingers`` for inspection.

    Parameters
    ----------
    smooth_interface : np.ndarray
        Smoothed interface y-values.
    x_vals : np.ndarray
        Corresponding x-coordinates (pixels).
    finger_idx : np.ndarray
        Indices of detected fingers (output of :func:`detect_fingers`).
    px_per_metre : float
        Pixel-to-metre conversion factor
        (``frame_width_px / frame_width_m``).
    min_width_cm : float
        Minimum accepted finger width in centimetres (default 0.3).
    max_width_cm : float
        Maximum accepted finger width in centimetres (default 5.0).

    Returns
    -------
    dict with keys:

    Accepted fingers only
        - ``widths_px``       : list of widths in pixels.
        - ``widths_cm``       : list of widths in centimetres.
        - ``left_shoulders``  : list of left-shoulder indices.
        - ``right_shoulders`` : list of right-shoulder indices.
        - ``n_fingers``       : number of accepted fingers.
        - ``mean_width_cm``   : mean width of accepted fingers (cm).
        - ``std_width_cm``    : std of accepted finger widths (cm).
        - ``min_width_cm``    : minimum accepted width (cm).
        - ``max_width_cm``    : maximum accepted width (cm).

    Rejected fingers
        - ``rejected_fingers`` : list of dicts, each with keys
          ``finger_idx`` (int), ``x_px`` (float), ``width_cm`` (float),
          ``reason`` ("too_narrow" | "too_wide").
        - ``n_rejected``       : number of rejected fingers.
    """
    from scipy.signal import find_peaks as _find_peaks
    shoulder_idx, _ = _find_peaks(-smooth_interface)

    # --- measure all fingers first ---
    all_results = []
    for fi in finger_idx:
        left_candidates  = shoulder_idx[shoulder_idx < fi]
        right_candidates = shoulder_idx[shoulder_idx > fi]
        left_sh  = left_candidates[-1]  if len(left_candidates)  > 0 else 0
        right_sh = right_candidates[0]  if len(right_candidates) > 0 else len(smooth_interface) - 1
        width_px = int(x_vals[right_sh] - x_vals[left_sh])
        width_cm = width_px / px_per_metre * 100.0
        all_results.append({
            "finger_idx": int(fi),
            "left_sh":    left_sh,
            "right_sh":   right_sh,
            "width_px":   width_px,
            "width_cm":   width_cm,
        })

    # --- split into accepted / rejected ---
    widths_px, widths_cm_acc    = [], []
    left_sh_list, right_sh_list = [], []
    rejected_fingers             = []

    for r in all_results:
        w = r["width_cm"]
        if w < min_width_cm:
            rejected_fingers.append({
                "finger_idx": r["finger_idx"],
                "x_px":       float(x_vals[r["finger_idx"]]),
                "width_cm":   w,
                "reason":     "too_narrow",
            })
        elif w > max_width_cm:
            rejected_fingers.append({
                "finger_idx": r["finger_idx"],
                "x_px":       float(x_vals[r["finger_idx"]]),
                "width_cm":   w,
                "reason":     "too_wide",
            })
        else:
            widths_px.append(r["width_px"])
            widths_cm_acc.append(w)
            left_sh_list.append(r["left_sh"])
            right_sh_list.append(r["right_sh"])

    arr = np.array(widths_cm_acc) if widths_cm_acc else np.array([np.nan])

    return {
        # accepted
        "widths_px":        widths_px,
        "widths_cm":        widths_cm_acc,
        "left_shoulders":   left_sh_list,
        "right_shoulders":  right_sh_list,
        "n_fingers":        len(widths_px),
        "mean_width_cm":    float(np.nanmean(arr)),
        "std_width_cm":     float(np.nanstd(arr)),
        "min_width_cm":     float(np.nanmin(arr)),
        "max_width_cm":     float(np.nanmax(arr)),
        # rejected
        "rejected_fingers": rejected_fingers,
        "n_rejected":       len(rejected_fingers),
    }


# ---------------------------------------------------------------------------
# Tip depth
# ---------------------------------------------------------------------------

def tip_from_valleys(
    v_x: np.ndarray,
    v_y: np.ndarray,
    flow_direction: str = "down",
) -> float:
    """
    Return the deepest finger-tip position.

    Parameters
    ----------
    v_x, v_y : np.ndarray
        x and y coordinates of detected finger tips.
    flow_direction : str
        ``'down'`` (use max y) or ``'right'`` (use max x).

    Returns
    -------
    float
        Deepest tip coordinate in pixels; 0.0 if no fingers present.
    """
    if len(v_x) == 0:
        return 0.0
    return float(np.max(v_y if flow_direction == "down" else v_x))
