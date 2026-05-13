import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# paths with basic config
SIM_XLS   = "Abnormalities_Features-V5-2.xlsx"
EXP_XLS   = "TDA_PCA_template.xlsx"
USE_LAST_N = 16                                  # for simulation table (None = all)

T_LABEL = 28.0                                   # wet/dry threshold (experimental)
RANDOM_SEED = 42

FIGSIZE   = (6.5, 2.5)        # two small panels
LABEL_FS  = 9
TICK_FS   = 9
EDGE_LW   = 0.7
SCATTER_S = 30

# colorbar ranges for simulation
VMIN, VMAX = -5.0, 5.0
CB_TICKS_SIM = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# NEW: simulation-only heatmap colormap (distinct from experimental viridis)
# negative -> red, zero -> white, positive -> orange
SIM_CMAP = LinearSegmentedColormap.from_list(
    "index_heat",
    ["#c62828", "#ffffff", "#f39c12"],  # red -> white -> orange
    N=256
)
SIM_NORM = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)

# style and formatting
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
#plt.rcParams['savefig.dpi'] = 300   # high-res export only

def label_contour_upright(ax, contour, txt, color, fs=8):
    try:
        path = contour.collections[0].get_paths()[0].vertices
        i = path.shape[0] // 2
        x0, y0 = path[i]; x1, y1 = path[min(i+1, path.shape[0]-1)]
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if ang > 90:  ang -= 180
        if ang < -90: ang += 180
        ax.text(x0, y0, txt, color=color, fontsize=fs, rotation=ang,
                ha="center", va="center", rotation_mode="anchor",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
    except Exception:
        pass

def even_ticks(ax, n=5):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xticks(np.linspace(x0, x1, n))
    ax.set_yticks(np.linspace(y0, y1, n))

# simulation analysis
def load_sim_table(path):
    for header in (1, 0):
        df = pd.read_excel(path, header=header)
        df.columns = [str(c).strip() for c in df.columns]
        idx_cols = [c for c in df.columns if "index" in c.lower()]
        if idx_cols:
            index_col = idx_cols[0]
            numeric_cols = [c for c in df.columns
                            if c != index_col and pd.api.types.is_numeric_dtype(df[c])]
            df = (df[[index_col] + numeric_cols]
                    .replace([np.inf, -np.inf], np.nan).dropna())
            return df, index_col, numeric_cols
    raise ValueError("Could not find a column with name containing 'index' in simulation file.")

def build_sim_panel(ax):
    df, index_col, numeric_cols = load_sim_table(SIM_XLS)
    if USE_LAST_N is not None:
        df = df.tail(USE_LAST_N).copy()

    idx_vals = df[index_col].astype(float).to_numpy()
    y = (idx_vals > 0).astype(int)     # 0=cavity, 1=hump
    X = df[numeric_cols].to_numpy()

    Xs = StandardScaler().fit_transform(X)
    ncomp = min(Xs.shape[0], Xs.shape[1])
    pca  = PCA(n_components=ncomp, random_state=RANDOM_SEED).fit(Xs)
    Zall = pca.transform(Xs)
    Z    = Zall[:, :2]
    pc1v, pc2v = 100*pca.explained_variance_ratio_[0], 100*pca.explained_variance_ratio_[1]

    if np.unique(y).size >= 2:
        lr = LogisticRegression(penalty="l2", solver="liblinear", C=1.0,
                                class_weight="balanced", max_iter=1000,
                                random_state=RANDOM_SEED).fit(Z, y)
        pad = 0.6
        x_min, x_max = Z[:,0].min() - pad, Z[:,0].max() + pad
        y_min, y_max = Z[:,1].min() - pad, Z[:,1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                             np.linspace(y_min, y_max, 400))
        grid  = np.c_[xx.ravel(), yy.ravel()]
        proba = lr.predict_proba(grid)[:, 1].reshape(xx.shape)

        ax.contourf(xx, yy, proba, levels=[0.0, 0.5, 1.0],
                    colors=["#dbe8fb", "#f3e7d8"], alpha=0.35)
        c025 = ax.contour(xx, yy, proba, levels=[0.25], linestyles="--",
                          linewidths=1.6, colors="black")
        c075 = ax.contour(xx, yy, proba, levels=[0.75], linestyles="--",
                          linewidths=1.6, colors="black")
        c050 = ax.contour(xx, yy, proba, levels=[0.50], linestyles="--",
                          linewidths=2.0, colors="red")
        label_contour_upright(ax, c025, "0.25", "black", fs=8)
        label_contour_upright(ax, c050, "0.5",  "red",   fs=8)
        label_contour_upright(ax, c075, "0.75", "black", fs=8)

    # CHANGED: use distinct "heat" colormap + centered normalization (white at 0)
    sc = ax.scatter(
        Z[:,0], Z[:,1],
        c=idx_vals,
        cmap="BuPu",
        norm=SIM_NORM,
        s=SCATTER_S,
        edgecolor="black", linewidth=EDGE_LW, zorder=3
    )

    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("index size", fontsize=LABEL_FS)
    cb.set_ticks(CB_TICKS_SIM)
    cb.ax.set_yticklabels([f"{t:g}" for t in CB_TICKS_SIM])
    cb.ax.tick_params(labelsize=TICK_FS)

    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="gray", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(rf"PC1 ({pc1v:.1f}\%\ var)", fontsize=LABEL_FS)
    ax.set_ylabel(rf"PC2 ({pc2v:.1f}\%\ var)", fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    even_ticks(ax, 5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
    ax.text(0.5, -0.28, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=LABEL_FS)

# experimental analysis
def build_exp_panel(ax):
    FEATURE_COLS = [
        "entropy (H0)","entropy (H1)",
        "Nop (H0)","Nop (H1)",
        "Bot (H0)","Bot (H1)",
        "Was (H0)","Was (H1)",
        "Land (H0)","Land (H1)",
        "Image (H0)","Image (H1)",
        "Betti (H0)","Betti (H1)",
        "heat (H0)","heat (H1)"
    ]
    REQUIRED = ["date","humidity_percent"] + FEATURE_COLS

    df = pd.read_excel(EXP_XLS, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Experimental file is missing columns: {missing}")

    dates    = df["date"].astype(str).to_numpy()
    humidity = pd.to_numeric(df["humidity_percent"], errors="coerce").to_numpy()
    Xraw     = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()

    col_med      = np.nanmedian(Xraw, axis=0)
    col_med      = np.where(np.isnan(col_med), 0.0, col_med)
    ri, ci       = np.where(np.isnan(Xraw)); Xraw[ri, ci] = col_med[ci]
    mu, sg       = Xraw.mean(axis=0), Xraw.std(axis=0, ddof=0)
    sg[sg < 1e-12] = 1.0
    Xz           = (Xraw - mu) / sg

    pca  = PCA(n_components=2, random_state=0)
    Z    = pca.fit_transform(Xz)
    evr  = pca.explained_variance_ratio_
    Zm   = Z.mean(axis=0)

    y_lab = (humidity >= T_LABEL).astype(int)

    if np.unique(y_lab).size >= 2:
        lr = LogisticRegression(penalty="l2", solver="lbfgs",
                                C=1.0, max_iter=2000,
                                class_weight="balanced", random_state=0).fit(Z, y_lab)
        pad = 0.6
        x_min, x_max = Z[:,0].min() - pad, Z[:,0].max() + pad
        y_min, y_max = Z[:,1].min() - pad, Z[:,1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        proba = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1].reshape(xx.shape)

        ax.contourf(xx, yy, proba, levels=[0.0, 0.5, 1.0],
                    colors=["#f3e7d8", "#dbe8fb"], alpha=0.35)
        c025 = ax.contour(xx, yy, proba, levels=[0.25], linestyles="--",
                          linewidths=1.2, colors="black")
        c075 = ax.contour(xx, yy, proba, levels=[0.75], linestyles="--",
                          linewidths=1.2, colors="black")
        c050 = ax.contour(xx, yy, proba, levels=[0.5],  linestyles="--",
                          linewidths=1.6, colors="red")
        label_contour_upright(ax, c025, "0.25", "black", fs=8)
        label_contour_upright(ax, c050, "0.5",  "red",   fs=8)
        label_contour_upright(ax, c075, "0.75", "black", fs=8)

    sc = ax.scatter(Z[:,0], Z[:,1],
                    c=humidity, cmap="viridis", s=SCATTER_S,
                    edgecolors="k", linewidths=EDGE_LW, zorder=3)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"soil moisture (\%)", fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    cb.set_ticks(np.linspace(np.nanmin(humidity), np.nanmax(humidity), 6))
    cb.formatter = plt.FuncFormatter(lambda x, pos: f"{x:.1f}")
    cb.update_ticks()

    offset_map = {'2021-06': (5, -10), '2022-10': (15, -4)}
    def label_offset(x, y, d_pts=7):
        ox = d_pts if x >= Zm[0] else -d_pts
        oy = d_pts if y >= Zm[1] else -d_pts
        return ox, oy
    for (x, yv), dlabel in zip(Z, dates):
        ox, oy = offset_map.get(dlabel, label_offset(x, yv))
        ax.annotate(dlabel, xy=(x, yv), xytext=(ox, oy),
                    textcoords='offset points',
                    ha='left' if ox >= 0 else 'right',
                    va='bottom' if oy >= 0 else 'top',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85, lw=0),
                    fontsize=8, zorder=4)

    ax.axhline(0, lw=0.8, color="gray", alpha=0.7)
    ax.axvline(0, lw=0.8, color="gray", alpha=0.7)
    ax.set_xlabel(rf"PC1 ({evr[0]*100:.1f}\%\ var)", fontsize=LABEL_FS)
    ax.set_ylabel(rf"PC2 ({evr[1]*100:.1f}\%\ var)", fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
    even_ticks(ax, 5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.2f}"))
    ax.text(0.5, -0.28, "(b)", transform=ax.transAxes, ha="center", va="top", fontsize=LABEL_FS)

# plot generating
fig, (axL, axR) = plt.subplots(1, 2, figsize=FIGSIZE)
build_sim_panel(axL)
build_exp_panel(axR)

for ax in (axL, axR):
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle="--", lw=0.5, alpha=0.35, zorder=0)

plt.tight_layout(w_pad=2.0)
fig.savefig("PC12_LR_sim_vs_exp_subplots.pdf", bbox_inches="tight")
plt.show()

