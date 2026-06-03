"""
scripts/plot_mixing.py
======================
Config wrapper for late-stage mixing analysis.
Edit the CONFIG block, then run this script OR call cf.run_sherwood /
cf.run_horizontal_slicing directly from a notebook.

NOT called by run_analysis.py.
"""

import co2_fingers as cf

# ── CONFIG — edit for each experiment ───────────────────────
CONFIG_PATH = '/content/drive/My Drive/CO2_diff/configs/C3R5.yaml'
CSV_PATH    = '/content/drive/My Drive/CO2_diff/time_csvs/c3r5_time.csv'
IMAGE_DIR   = '/content/drive/My Drive/CO2_diff/Experiment images/C3/C3R5'

H_M        = 0.27    # aquifer height in metres
D          = 2e-9    # CO2 diffusivity in water (m²/s)
T_BOTTOM_H = 23.3    # h — when first finger hits bottom
N_SLICES   = 50      # depth bins for horizontal slicing
INVERT     = False   # True if dye darkens with CO2 concentration
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sh = cf.run_sherwood(
        CONFIG_PATH, CSV_PATH, IMAGE_DIR,
        H_m=H_M, D=D, t_bottom_h=T_BOTTOM_H,
    )
    hs = cf.run_horizontal_slicing(
        CONFIG_PATH, CSV_PATH, IMAGE_DIR,
        H_m=H_M, t_start_h=T_BOTTOM_H, n_slices=N_SLICES,
    )
