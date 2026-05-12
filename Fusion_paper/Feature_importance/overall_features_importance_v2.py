#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import IPython as IP
IP.get_ipython().run_line_magic('reset', '-sf')

# %% imports & style
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.close('all')

# %% data & common settings
file_path = "TDA_PCA_template.xlsx"                 # experimental file
sim_file_path = "Abnormalities_Features-V5-2.xlsx" # simulation file

excel_header_row = 0
EPS = 1e-12

FEATURE_COLS = [
    " entropy (H0)", " entropy (H1)",
    "Nop (H0)", "Nop (H1)",
    "Bot (H0)", "Bot (H1)",
    "Was (H0)", "Was (H1)",
    "Land (H0)", "Land (H1)",
    "Image (H0)", "Image (H1)",
    "Betti (H0)", "Betti (H1)",
    "heat (H0)", "heat (H1)"
]

# %% manual controls



COMMON_YMAX = 0.35
COMMON_YTICKS = np.arange(0, 0.351, 0.05)

LABEL_FS = 9.5     # axis-label font size
XTICK_FS = 9.5      # feature-name font size
YTICK_FS = 9.5      # y-axis tick font size
BAR_WIDTH = 0.75

def h_color(name: str) -> str:
    n = name.upper()
    return "tab:blue" if "H0" in n else ("tab:orange" if "H1" in n else "gray")

def clean_label(name: str) -> str:
    # keep each label as one vertical line
    return name.strip()

def load_and_prepare(path, feature_cols, header_row=0, eps=1e-12):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, header=header_row)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls or .csv")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"File '{path}' missing columns: {missing}")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # fill NaN with column median
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    ri, ci = np.where(np.isnan(X))
    X[ri, ci] = col_med[ci]

    # remove constant columns
    std = X.std(axis=0, ddof=0)
    keep = std >= eps
    Xk = X[:, keep]
    feat_names_kept = [c for c, k in zip(feature_cols, keep) if k]

    if Xk.shape[1] == 0:
        raise ValueError(f"All features are constant in '{path}'; PCA undefined.")

    # z-score standardization
    mu = Xk.mean(axis=0)
    sg = Xk.std(axis=0, ddof=0)
    sg[sg < eps] = 1.0
    Xz = (Xk - mu) / sg

    # PCA and PC1 loadings
    pca = PCA(n_components=2, random_state=0).fit(Xz)
    loadings = pca.components_.T
    pc1 = loadings[:, 0]

    # sort by absolute importance
    order = np.argsort(-np.abs(pc1))
    feat_sorted = np.array(feat_names_kept)[order]
    pc1_sorted = pc1[order]
    mag = np.abs(pc1_sorted)

    labels = [clean_label(f) for f in feat_sorted]
    colors = [h_color(n) for n in feat_sorted]
    xpos = np.arange(len(mag))

    return xpos, mag, labels, colors

# %% prepare both datasets
xpos_sim, mag_sim, labels_sim, colors_sim = load_and_prepare(
    sim_file_path, FEATURE_COLS, excel_header_row, EPS
)

xpos_exp, mag_exp, labels_exp, colors_exp = load_and_prepare(
    file_path, FEATURE_COLS, excel_header_row, EPS
)

# %% draw the two-panel feature-importance figure: 1 x 2
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2))

# --- left panel: (a) simulation
ax = axes[0]
ax.bar(xpos_sim, mag_sim, color=colors_sim, width=BAR_WIDTH)
ax.set_ylabel(r"PC1 loading (importance)", fontsize=LABEL_FS)
ax.set_xticks(xpos_sim)
ax.set_xticklabels(labels_sim, rotation=90, ha="center", va="top", fontsize=XTICK_FS)
ax.set_yticks(COMMON_YTICKS)
ax.set_ylim(0, COMMON_YMAX)
ax.set_xlabel('features\n(a)', fontsize=LABEL_FS)
ax.tick_params(axis='y', labelsize=YTICK_FS)
ax.tick_params(axis='x', pad=1)

# --- right panel: (b) experimental
ax = axes[1]
ax.bar(xpos_exp, mag_exp, color=colors_exp, width=BAR_WIDTH)
ax.set_ylabel(r"PC1 loading (importance)", fontsize=LABEL_FS)
ax.set_xticks(xpos_exp)
ax.set_xticklabels(labels_exp, rotation=90, ha="center", va="top", fontsize=XTICK_FS)
ax.set_yticks(COMMON_YTICKS)
ax.set_ylim(0, COMMON_YMAX)
ax.set_xlabel('features\n(b)', fontsize=LABEL_FS)
ax.tick_params(axis='y', labelsize=YTICK_FS)
ax.tick_params(axis='x', pad=1)

plt.tight_layout(pad=0.25, w_pad=1.0)
plt.savefig('overall_features_importance_abnormalities_1x2.png', dpi=300, bbox_inches="tight")
plt.show()