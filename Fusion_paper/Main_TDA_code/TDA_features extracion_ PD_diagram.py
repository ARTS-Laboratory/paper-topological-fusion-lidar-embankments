from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
import numpy as np
import laspy
import csv
import os
import matplotlib.pyplot as plt

class tda:
    def __init__(self, homo_dim=1, fts='all') -> None:
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)

        # FIX 1: keep a REAL max_edge_length, only change n_jobs
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=1,          # safer memory than -1
            max_edge_length=1e9
        )

        self.fts = fts
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        self.metrics = ["bottleneck", "wasserstein", "landscape",
                        "persistence_image", "betti", "heat"]
        self.diag = None

    def random_sampling_consensus(self, pcd, m=1000, K=10):
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
            from gtda.diagrams import Amplitude
            for metric in self.metrics:
                AMP = Amplitude(metric=metric)
                amp = AMP.fit_transform(diag)
                amps.append(amp)
            feat_amp = np.hstack(amps) if amps else np.array([])
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            print(f"  Computed features: {iteration_features}")
            features_list.append(iteration_features)

        features_array = np.vstack(features_list)
        if os.path.exists("iteration_features.csv"):
            os.remove("iteration_features.csv")
        np.savetxt("iteration_features.csv", features_array, delimiter=",",
                   header="All iteration feature vectors", comments="")
        median_features = np.median(features_array, axis=0)
        print("Median consensus features:", median_features)
        if os.path.exists("median_consensus_features.csv"):
            os.remove("median_consensus_features.csv")
        np.savetxt("median_consensus_features.csv",
                   median_features.reshape(1, -1), delimiter=",",
                   header="Median consensus feature vector", comments="")
        return median_features

    def forward(self, pcd_list):
        print("Computing persistence diagrams on point cloud(s)...")
        self.diag = self.persistence.fit_transform(pcd_list)
        print("Persistence diagrams computed.")

    def save_h0_h1_to_csv(self, diagram_index=0, filename_prefix="point_cloud"):
        diag = self.diag[diagram_index]
        H0 = diag[diag[:, 2] == 0]
        H1 = diag[diag[:, 2] == 1]
        np.savetxt(f"{filename_prefix}_H0.csv", H0, delimiter=",",
                   header="birth,death,dimension", comments="")
        np.savetxt(f"{filename_prefix}_H1.csv", H1, delimiter=",",
                   header="birth,death,dimension", comments="")
        print(f"Saved H0 to {filename_prefix}_H0.csv")
        print(f"Saved H1 to {filename_prefix}_H1.csv")
        return H0, H1

    def plot_pd_h0_h1(self, H0, H1, title_prefix="point_cloud"):
        plt.figure(figsize=(6, 6))

        all_pts = np.vstack([H0[:, :2], H1[:, :2]]) if H0.size and H1.size \
                  else (H0[:, :2] if H0.size else H1[:, :2])
        bmin = np.min(all_pts[:, 0])
        bmax = np.max(all_pts[:, 1])
        plt.plot([bmin, bmax], [bmin, bmax], "k--", linewidth=1)

        if H0.size > 0:
            plt.scatter(H0[:, 0], H0[:, 1], s=10, c="tab:blue", label=r"$H_0$")
        if H1.size > 0:
            plt.scatter(H1[:, 0], H1[:, 1], s=10, c="tab:orange", label=r"$H_1$")

        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title(f"Persistence Diagram (H0, H1): {title_prefix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def __call__(self, pcd_list):
        return self.forward(pcd_list)


def process_single_las_file():
    las_path = r"C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"

    my_tda = tda(homo_dim=1, fts='all')

    print(f"\nNow processing file: {las_path}")
    with laspy.open(las_path) as f:
        las = f.read()
    print("LAS file read successfully.")

    x = las.x
    y = las.y
    z = las.z
    point_cloud = np.vstack((x, y, z)).T
    print("Point cloud shape:", point_cloud.shape)

    # 1) consensus features on subsamples
    m = 1000
    K = 10
    median_features = my_tda.random_sampling_consensus(point_cloud, m=m, K=K)

    # 2) PD for H0/H1 on one subsample of size m (avoid full cloud)
    if point_cloud.shape[0] > m:
        idx = np.random.choice(point_cloud.shape[0], size=m, replace=False)
        subset = point_cloud[idx, :]
    else:
        subset = point_cloud

    my_tda([subset])
    prefix = os.path.splitext(os.path.basename(las_path))[0]

    H0, H1 = my_tda.save_h0_h1_to_csv(diagram_index=0, filename_prefix=prefix)
    my_tda.plot_pd_h0_h1(H0, H1, title_prefix=prefix)

    # optional: save median features
    header_list = [
        "Persistence Entropy (H0)", "Persistence Entropy (H1)",
        "Number of Points (H0)", "Number of Points (H1)",
        "Bottleneck Amplitude (H0)", "Bottleneck Amplitude (H1)",
        "Wasserstein Amplitude (H0)", "Wasserstein Amplitude (H1)",
        "Landscape Amplitude (H0)", "Landscape Amplitude (H1)",
        "Persistence Image Amplitude (H0)", "Persistence Image Amplitude (H1)",
        "Betti Amplitude (H0)", "Betti Amplitude (H1)",
        "Heat Amplitude (H0)", "Heat Amplitude (H1)",
    ]
    out_csv = f"{prefix}_median_consensus_features_single.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_list)
        writer.writerow(median_features)
    print(f"Median consensus features saved to {out_csv}")


if __name__ == "__main__":
    process_single_las_file()


