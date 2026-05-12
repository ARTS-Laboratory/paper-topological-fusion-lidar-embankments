import laspy
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude

plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'font.size': 8})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})


# =========================
# manual controls
# =========================
FIG_W = 6.5
FIG_H = 2.6

LABEL_FS = 9
TICK_FS = 9
LEGEND_FS = 7
TITLE_FS = 9

WSPACE = 0.28   # space between left/right panels

X_TICKS = [0, 1000, 2000, 3000, 4000, 5000]
X_LIM = (0, 5000)
Y_LIM = (-0.05, 1.05)
Y_TICKS = np.linspace(0, 1.0, 6)


# =========================
# Class definition for TDA
# =========================
class tda:
    def __init__(self, homo_dim=1, fts='all') -> None:
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=-1,
            max_edge_length=1e9
        )
        self.fts = fts
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        self.metrics = ["bottleneck", "wasserstein", "landscape", "persistence_image", "betti", "heat"]
        self.diag = None

    def random_sampling_consensus(self, pcd, m=100, K=10):
        features_list = []
        for i in range(K):
            print(f"Iteration {i+1}/{K}:")
            if pcd.shape[0] > m:
                idx = np.random.choice(pcd.shape[0], size=m, replace=False)
                subset = pcd[idx, :]
            else:
                subset = pcd

            diag = self.persistence.fit_transform([subset])
            feat_entropy = self.persistence_entropy.fit_transform(diag)
            feat_num = self.NumOfPoint.fit_transform(diag)

            amps = []
            for metric in self.metrics:
                AMP = Amplitude(metric=metric)
                amp = AMP.fit_transform(diag)
                amps.append(amp)

            feat_amp = np.hstack(amps) if amps else np.array([])
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            features_list.append(iteration_features)

        features_array = np.vstack(features_list)
        median_features = np.median(features_array, axis=0)
        return features_array

    def forward(self, pcd_list):
        self.diag = self.persistence.fit_transform(pcd_list)
        features_entropy = self.persistence_entropy.fit_transform(self.diag)
        features_num = self.NumOfPoint.fit_transform(self.diag)

        amps = []
        for metric in self.metrics:
            AMP = Amplitude(metric=metric)
            amp = AMP.fit_transform(self.diag)
            amps.append(amp)

        features_amp = np.hstack(amps) if amps else np.array([])
        all_features = np.hstack((features_entropy, features_num, features_amp))
        return all_features


# =========================
# Load the LAS file
# =========================
file_path = "surface_with_smooth_circular_cavity_20.las"
print("Opening LAS file:", file_path)

with laspy.open(file_path) as f:
    las = f.read()

x = las.x
y = las.y
z = las.z
point_cloud = np.vstack((x, y, z)).T
print("Original point cloud shape:", point_cloud.shape)

my_tda = tda(homo_dim=1, fts='all')

# sample sizes
m_values = [50, 200, 500, 1000, 1500, 2500, 3500, 4500]
K = 10
median_features_list = []

for m in m_values:
    print(f"\nProcessing sample size m = {m}")
    iteration_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)
    median_features = np.median(iteration_features, axis=0)
    median_features_list.append(median_features)

median_features_array = np.vstack(median_features_list)

# normalize columnwise
den = (median_features_array.max(axis=0) - median_features_array.min(axis=0))
den[den == 0] = 1.0
normalized_features = (median_features_array - median_features_array.min(axis=0)) / den

# flip columns if decreasing from first to last
for i in range(normalized_features.shape[1]):
    if normalized_features[0, i] > normalized_features[-1, i]:
        normalized_features[:, i] = 1 - normalized_features[:, i]

# feature labels
feature_labels = [
    "entropy H0", "entropy H1",
    "numPoints H0", "numPoints H1",
    "bottleneck H0", "bottleneck H1",
    "wasserstein H0", "wasserstein H1",
    "landscape H0", "landscape H1",
    "image H0", "image H1",
    "Betti H0", "Betti H1",
    "heat H0", "heat H1"
]

print("\nAvailable features:")
for idx, feature in enumerate(feature_labels):
    print(f"{idx + 1}: {feature}")

user_input = input("\nEnter the feature numbers you want to plot (comma-separated, or type 'all' to plot all): ")

if user_input.lower() == 'all':
    selected_features = feature_labels
else:
    selected_indices = [int(v.strip()) - 1 for v in user_input.split(',')]
    selected_features = [feature_labels[i] for i in selected_indices if 0 <= i < len(feature_labels)]

print("\nSelected features to plot:", selected_features)
print("X-axis ticks:", X_TICKS)


# =========================
# split features into H0 and H1
# =========================
selected_h0 = []
selected_h1 = []

for i, label in enumerate(feature_labels):
    if label in selected_features:
        if label.endswith("H0"):
            selected_h0.append((i, label))
        elif label.endswith("H1"):
            selected_h1.append((i, label))


# =========================
# plot: 1 row x 2 columns
# =========================
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

# ---- left panel: H0
ax = axes[0]
for i, label in selected_h0:
    ax.plot(m_values, normalized_features[:, i], marker='o', markersize=3, label=label)

ax.set_xlabel("sample size (m)\n(a)", fontsize=LABEL_FS)
ax.set_ylabel("normalized median feature value", fontsize=LABEL_FS)
ax.set_xlim(X_LIM)
ax.set_ylim(Y_LIM)
ax.set_xticks(X_TICKS)
ax.set_yticks(Y_TICKS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.grid(True, color='0.85', linewidth=0.8)

leg = ax.legend(loc='lower right', fontsize=LEGEND_FS, frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(1.0)

# ---- right panel: H1
ax = axes[1]
for i, label in selected_h1:
    ax.plot(m_values, normalized_features[:, i], marker='o', markersize=3, label=label)

ax.set_xlabel("sample size (m)\n(b)", fontsize=LABEL_FS)
ax.set_ylabel("normalized median feature value", fontsize=LABEL_FS)
ax.set_xlim(X_LIM)
ax.set_ylim(Y_LIM)
ax.set_xticks(X_TICKS)
ax.set_yticks(Y_TICKS)
ax.tick_params(axis='both', labelsize=TICK_FS)
ax.grid(True, color='0.85', linewidth=0.8)

leg = ax.legend(loc='lower right', fontsize=LEGEND_FS, frameon=True)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(1.0)

plt.subplots_adjust(left=0.08, right=0.985, bottom=0.23, top=0.95, wspace=WSPACE)
plt.savefig("topological_features_lidar_1x2.png", dpi=300, bbox_inches="tight")
plt.show()
