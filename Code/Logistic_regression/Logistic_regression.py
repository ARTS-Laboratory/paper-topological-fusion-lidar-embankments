
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score


# data importing

file_path = "TDA_PCA_template.xlsx"
t_label = 28.0  # humidity threshold (%) for wet/dry labels

# avoid MiKTeX font error
use_tex = False

#  figure layout controls 
figsize        = (6.5, 6.5)   # size per 4×2 figure
nrows, ncols   = 4, 2
wspace, hspace = 0.55, 0.90
title_fs       = 11
axis_fs        = 11
suptitle_y     = 0.98

# one shared colorbar per row (to the right of the row)
row_cbar_fraction = 0.035
row_cbar_pad      = 0.04

#  subplot style 
grid_on     = True
grid_ls     = "--"
grid_lw     = 0.5
grid_alpha  = 0.35

point_s_dry = 35
point_s_wet = 40
point_lw    = 1.0

curve_std_span = 3.0
curve_lw       = 1.6
vline_lw       = 1.0
jitter_width   = 0.06

show_dates        = False
date_label_offset = (3, 3)


#  global plotting style

plt.rcParams.update({'text.usetex': bool(use_tex)})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})


#  canonical headers 

canon_features = [
    "persistence entropy (H0)","persistence entropy (H1)",
    "number of points (H0)","number of points (H1)",
    "bottleneck (H0)","bottleneck (H1)",
    "wasserstein (H0)","wasserstein (H1)",
    "landscape (H0)","landscape (H1)",
    "persistence image (H0)","persistence image (H1)",
    "Betti (H0)","Betti (H1)",
    "heat (H0)","heat (H1)"
]
required = ["date", "humidity_percent"] + canon_features

def normalize_col_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\(", " (", s)  # exactly one space before '('
    return s

def keyize(s: str) -> str:
    return normalize_col_name(s).lower()


#  load (first sheet only) + normalize headers

if file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path, sheet_name=0)
elif file_path.lower().endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    raise ValueError("Use .xlsx/.xls or .csv")

col_key_to_actual = {keyize(c): c for c in df.columns}
rename_map, missing = {}, []
for canon in required:
    k = keyize(canon)
    actual = col_key_to_actual.get(k)
    if actual is None:
        missing.append(canon)
    else:
        rename_map[actual] = normalize_col_name(canon)
if missing:
    print("Columns present:", list(df.columns))
    raise ValueError(f"Required columns not found after normalization: {missing}")
df = df.rename(columns=rename_map)


# coerce numerics, labels, color scale

df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
for c in canon_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=canon_features, how="all").copy()
if df.shape[0] < 2:
    raise ValueError("Need at least 2 rows with some feature values.")

dates    = df["date"].astype(str).values
humidity = df["humidity_percent"].values
if np.isnan(humidity).all():
    raise ValueError("humidity_percent is all NaN.")

# binary labels from threshold
y_all = (humidity >= t_label).astype(int)
print(f"Label threshold = {t_label:.1f}% → dry: {(y_all==0).sum()}, wet: {(y_all==1).sum()}")

# shared color scale (actual humidity)
norm = plt.Normalize(vmin=np.nanmin(humidity), vmax=np.nanmax(humidity))
cmap = plt.cm.viridis
scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)


# helper that plots one 4×2 figure with a colorbar per row

def plot_grid_for(features_subset, fig_title_suffix):
    loo = LeaveOneOut()

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes)

    # ensure x-axis bottom on every subplot
    for ax in axes.ravel():
        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True)

    for i, feat in enumerate(features_subset):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        x_raw = df[feat].values.astype(float)
        mask  = ~np.isnan(x_raw) & ~np.isnan(humidity)
        x     = x_raw[mask]
        y     = y_all[mask]
        hum   = humidity[mask]
        dts   = df.loc[mask, "date"].astype(str).values
        n_pts = len(x)

        if grid_on:
            ax.set_axisbelow(True)
            ax.grid(True, ls=grid_ls, lw=grid_lw, alpha=grid_alpha)

        # Degenerate cases
        if n_pts < 2 or np.nanstd(x) == 0:
            ax.text(0.5, 0.5, "constant/NaN\nfeature", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"(n={n_pts})", fontsize=title_fs)                  # <-- TOP: info only
            ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xlabel(feat, fontsize=axis_fs)                            # <-- BOTTOM: feature name
            ax.set_ylabel(r"P(wet)", fontsize=axis_fs)
            continue

        if len(np.unique(y)) < 2:
            ax.text(0.5, 0.5, "only one class\nunder this threshold", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"(n={n_pts})", fontsize=title_fs)                  # <-- TOP: info only
            ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xlabel(feat, fontsize=axis_fs)                            # <-- BOTTOM: feature name
            ax.set_ylabel(r"P(wet)", fontsize=axis_fs)
            continue

        # Pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                                      max_iter=1000, class_weight="balanced", random_state=0))
        ])

        # LOO accuracy
        try:
            loo_acc = cross_val_score(pipe, x.reshape(-1,1), y, cv=loo, scoring="accuracy").mean()
        except Exception:
            loo_acc = np.nan

        # Fit & curve
        pipe.fit(x.reshape(-1,1), y)
        mu = x.mean()
        sigma = x.std() if x.std() > 0 else 1.0
        xmin = mu - curve_std_span * sigma
        xmax = mu + curve_std_span * sigma
        grid = np.linspace(xmin, xmax, 300).reshape(-1, 1)
        proba = pipe.predict_proba(grid)[:, 1]

        # p=0.5 crossing (original units)
        beta1 = pipe.named_steps["lr"].coef_[0, 0]
        beta0 = pipe.named_steps["lr"].intercept_[0]
        x_star = np.nan
        if beta1 != 0:
            z_star = -beta0 / beta1
            x_star = mu + sigma * z_star

        # points colored by humidity
        rng = np.random.default_rng(0)
        jitter = (rng.random(n_pts) - 0.5) * jitter_width
        y_obs = y + jitter
        colors = cmap(norm(hum))
        dry   = (y == 0)
        wet   = ~dry
        ax.scatter(x[dry], y_obs[dry], marker="o", s=point_s_dry,
           c=colors[dry], edgecolor='k', linewidths=point_lw, label="dry")
        ax.scatter(x[wet], y_obs[wet], marker="o", s=point_s_wet,
           c=colors[wet], edgecolor='k', linewidths=point_lw, label="wet")

        if show_dates:
            for xi, yi, dt in zip(x, y_obs, dts):
                ax.annotate(str(dt), xy=(xi, yi), xytext=date_label_offset,
                            textcoords="offset points", fontsize=8)

        ax.plot(grid.ravel(), proba, lw=curve_lw, color="k")
        if np.isfinite(x_star):
            ax.axvline(x_star, color="k", ls=":", lw=vline_lw)

        # --- TOP: only info; BOTTOM: feature name
        ax.set_title(f"(n={n_pts}, LOO acc={loo_acc:.2f})", fontsize=title_fs)
        ax.set_xlabel(feat, fontsize=axis_fs)
        ax.set_ylabel(r"P(wet)", fontsize=axis_fs)
        ax.set_ylim(-0.1, 1.1); ax.set_yticks([0, 0.5, 1.0])

        # Legend once
        if r == 0 and c == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=8)

    # spacing + title
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08,
                        wspace=wspace, hspace=hspace)
    fig.suptitle(rf"1D Logistic Regression per feature  (wet if humidity $\geq$ {t_label:.1f}\%)",
                 y=suptitle_y)

    # one colorbar per row
    for r in range(nrows):
        row_axes = [axes[r, 0], axes[r, 1]]
        cbar = fig.colorbar(scalar_mappable, ax=row_axes,
                            fraction=row_cbar_fraction, pad=row_cbar_pad)
        cbar.set_label(r"Soil moisture (\%)", fontsize=axis_fs)
        
       # These must match the colorbar normalization range
        cbar.set_ticks([17, 22.6, 28.1, 33.7, 39.2, 44.8])
        cbar.set_ticklabels(["17.0", "22.6","28.1","33.7", "39.2", "44.8"])  # Not set_yticklabels (for the new API)
        cbar.ax.tick_params(labelsize=axis_fs) 
        
    plt.show()


# split into two figures (8 + 8) and show

first8  = canon_features[:8]
second8 = canon_features[8:]

plot_grid_for(first8,  "A")
plot_grid_for(second8, "B")
