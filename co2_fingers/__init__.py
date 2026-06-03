"""
co2_fingers
===========
A Python package for analyzing CO₂ convective fingering in FluidFlower experiments.

Modules
-------
- io         : Time extraction from CSV files and image loading
- preprocessing : Image binarization, cropping, mask cleaning
- interface  : Contour extraction and Gaussian interface smoothing
- fingers    : Finger detection, counting, and width measurement
- baseline   : Static baseline computation and roughness-band analysis
- regimes    : TimeRegimeDetector class (onset, linear, merging, convective)
- plotting   : All visualization functions
- heatmap    : Heatmap overlay for multiple user-selected frames

Function dictionary is available via co2_fingers.FUNCTION_REGISTRY.
"""

from .io import load_timestamps, load_image, load_images
from .preprocessing import preprocess, crop_image
from .interface import raw_outline, gauss_outline
from .baseline import compute_static_baseline
from .fingers import detect_fingers, measure_finger_widths, roughness_sigma, mean_front, tip_from_valleys
from .regimes import TimeRegimeDetector, REGIME_COLORS, REGIME_LABELS
from .plotting import (
    plot_baseline_frame,
    plot_finger_check,
    plot_finger_widths_per_frame,
    plot_finger_count,
    plot_merging_metric,
    plot_loglog_tip,
    plot_heatmap_overlay,
    plot_time_regimes,
)
from .heatmap import build_heatmap_overlay
from .mixing import diffusion_flux, sherwood_number, horizontal_slice_profile
from ._registry import FUNCTION_REGISTRY

__version__ = "0.1.0"
__all__ = [
    "load_timestamps", "load_image", "load_images",
    "preprocess", "crop_image",
    "raw_outline", "gauss_outline",
    "compute_static_baseline",
    "detect_fingers", "measure_finger_widths", "roughness_sigma", "mean_front", "tip_from_valleys",
    "TimeRegimeDetector", "REGIME_COLORS", "REGIME_LABELS",
    "plot_baseline_frame", "plot_finger_check", "plot_finger_widths_per_frame",
    "plot_finger_count", "plot_merging_metric", "plot_loglog_tip",
    "plot_heatmap_overlay", "plot_time_regimes",
    "build_heatmap_overlay",
    "diffusion_flux", "sherwood_number", "horizontal_slice_profile",
    "FUNCTION_REGISTRY",
]
