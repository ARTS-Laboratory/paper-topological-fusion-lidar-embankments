from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, NumberOfPoints, Amplitude
import numpy as np
import laspy
import csv
import os
import traceback
import matplotlib.pyplot as plt


class tda:
    def __init__(self, homo_dim=1, fts='all') -> None:
        self.homology_dimensions = list(range(homo_dim + 1))
        print("Homology dimensions:", self.homology_dimensions)

        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=self.homology_dimensions,
            n_jobs=1,
            max_edge_length=1e9
        )

        self.fts = fts
        self.persistence_entropy = PersistenceEntropy()
        self.NumOfPoint = NumberOfPoints()
        self.metrics = [
            "bottleneck", "wasserstein", "landscape",
            "persistence_image", "betti", "heat"
        ]
        self.diag = None

    def random_sampling_consensus(self, pcd, m=20000, K=10, out_prefix=None):
        features_list = []

        for i in range(K):
            print(f"Iteration {i + 1}/{K}")
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
                amp_obj = Amplitude(metric=metric)
                amp = amp_obj.fit_transform(diag)
                amps.append(amp)

            feat_amp = np.hstack(amps) if amps else np.array([])
            iteration_features = np.hstack((feat_entropy, feat_num, feat_amp))
            features_list.append(iteration_features)

        features_array = np.vstack(features_list)
        median_features = np.median(features_array, axis=0)

        if out_prefix is not None:
            iter_csv = f"{out_prefix}_iteration_features.csv"
            median_csv = f"{out_prefix}_median_consensus_features.csv"

            np.savetxt(
                iter_csv,
                features_array,
                delimiter=",",
                header="All iteration feature vectors",
                comments=""
            )
            np.savetxt(
                median_csv,
                median_features.reshape(1, -1),
                delimiter=",",
                header="Median consensus feature vector",
                comments=""
            )

            print(f"Saved: {iter_csv}")
            print(f"Saved: {median_csv}")

        return median_features, features_array

    def forward(self, pcd_list):
        print("Computing persistence diagrams...")
        self.diag = self.persistence.fit_transform(pcd_list)
        print("Persistence diagrams computed.")

    def save_h0_h1_to_csv(self, diagram_index=0, filename_prefix="point_cloud"):
        diag = self.diag[diagram_index]
        H0 = diag[diag[:, 2] == 0]
        H1 = diag[diag[:, 2] == 1]

        h0_csv = f"{filename_prefix}_H0.csv"
        h1_csv = f"{filename_prefix}_H1.csv"

        np.savetxt(
            h0_csv,
            H0,
            delimiter=",",
            header="birth,death,dimension",
            comments=""
        )
        np.savetxt(
            h1_csv,
            H1,
            delimiter=",",
            header="birth,death,dimension",
            comments=""
        )

        print(f"Saved: {h0_csv}")
        print(f"Saved: {h1_csv}")
        return H0, H1

    def plot_pd_h0_h1(self, H0, H1, title_prefix="point_cloud", show=True):
        plt.figure(figsize=(6, 6))

        if H0.size == 0 and H1.size == 0:
            plt.title(f"Persistence Diagram (empty): {title_prefix}")
            plt.xlabel("Birth")
            plt.ylabel("Death")
            plt.grid(True)
            plt.tight_layout()
            if show:
                plt.show()
            return

        if H0.size and H1.size:
            all_pts = np.vstack([H0[:, :2], H1[:, :2]])
        elif H0.size:
            all_pts = H0[:, :2]
        else:
            all_pts = H1[:, :2]

        bmin = np.min(all_pts[:, 0])
        bmax = np.max(all_pts[:, 1])

        plt.plot([bmin, bmax], [bmin, bmax], "k--", linewidth=1)

        if H0.size > 0:
            plt.scatter(H0[:, 0], H0[:, 1], s=12, c="tab:blue", label=r"$H_0$")
        if H1.size > 0:
            plt.scatter(H1[:, 0], H1[:, 1], s=12, c="tab:orange", label=r"$H_1$")

        plt.xlabel("Birth")
        plt.ylabel("Death")
        plt.title(f"Persistence Diagram: {title_prefix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if show:
            plt.show()

    def __call__(self, pcd_list):
        return self.forward(pcd_list)


def feature_header_list(homo_dim=1):
    if homo_dim != 1:
        raise ValueError("This helper currently assumes homo_dim=1.")

    return [
        "Persistence Entropy (H0)", "Persistence Entropy (H1)",
        "Number of Points (H0)", "Number of Points (H1)",
        "Bottleneck Amplitude (H0)", "Bottleneck Amplitude (H1)",
        "Wasserstein Amplitude (H0)", "Wasserstein Amplitude (H1)",
        "Landscape Amplitude (H0)", "Landscape Amplitude (H1)",
        "Persistence Image Amplitude (H0)", "Persistence Image Amplitude (H1)",
        "Betti Amplitude (H0)", "Betti Amplitude (H1)",
        "Heat Amplitude (H0)", "Heat Amplitude (H1)",
    ]


def save_features_csv(output_csv, features, homo_dim=1):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_header_list(homo_dim=homo_dim))
        writer.writerow(features)
    print(f"Saved: {output_csv}")


def save_all_segments_feature_summary(output_csv, all_rows, homo_dim=1):
    header = [
        "segment_id", "axis", "range_start", "range_end", "n_points"
    ] + feature_header_list(homo_dim=homo_dim)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    print(f"Saved: {output_csv}")


def load_las_as_point_cloud(las_path):
    with laspy.open(las_path) as f:
        las = f.read()

    point_cloud = np.vstack((las.x, las.y, las.z)).T
    return point_cloud


def plot_point_cloud_3d(
    point_cloud,
    title,
    sample_for_plot=None,
    color_values=None,
    cmap="viridis",
    point_size=1
):
    pcd = point_cloud

    if sample_for_plot is not None and point_cloud.shape[0] > sample_for_plot:
        idx = np.random.choice(point_cloud.shape[0], size=sample_for_plot, replace=False)
        pcd = point_cloud[idx, :]
        if color_values is not None:
            color_values = color_values[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if color_values is None:
        sc = ax.scatter(
            pcd[:, 0], pcd[:, 1], pcd[:, 2],
            c=pcd[:, 2], cmap=cmap, s=point_size
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Elevation (Z)")
    else:
        sc = ax.scatter(
            pcd[:, 0], pcd[:, 1], pcd[:, 2],
            c=color_values, cmap=cmap, s=point_size
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="Segment")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def select_representative_reduced_cloud(pcd, m=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    if pcd.shape[0] > m:
        idx = rng.choice(pcd.shape[0], size=m, replace=False)
        reduced_cloud = pcd[idx, :]
    else:
        reduced_cloud = pcd.copy()
    return reduced_cloud


def segment_point_cloud_with_overlap(point_cloud, n_segments=4, overlap_ratio=0.2, axis="x"):
    """
    Segmentation logic:
    1. Choose one axis ("x" or "y")
    2. Take min and max of that axis on the reduced dataset
    3. Divide the axis extent into n_segments equal base intervals
    4. Expand each interval by overlap_ratio * base_width
    5. Use those overlapping intervals as segment windows
    """
    if axis not in ["x", "y"]:
        raise ValueError("axis must be 'x' or 'y'")

    axis_idx = 0 if axis == "x" else 1
    coord = point_cloud[:, axis_idx]

    cmin = np.min(coord)
    cmax = np.max(coord)
    total_length = cmax - cmin

    if total_length == 0:
        raise ValueError(f"All points have the same {axis}-coordinate; cannot segment.")

    base_width = total_length / n_segments
    overlap_width = overlap_ratio * base_width

    segments = []
    summary_rows = []

    for i in range(n_segments):
        base_start = cmin + i * base_width
        base_end = cmin + (i + 1) * base_width

        seg_start = max(cmin, base_start - overlap_width / 2.0)
        seg_end = min(cmax, base_end + overlap_width / 2.0)

        if i == 0:
            seg_start = cmin
        if i == n_segments - 1:
            seg_end = cmax

        mask = (coord >= seg_start) & (coord <= seg_end)
        seg_points = point_cloud[mask]
        seg_id = f"segment_{i + 1:02d}"

        segments.append({
            "segment_id": seg_id,
            "segment_index": i + 1,
            "axis": axis,
            "range_start": seg_start,
            "range_end": seg_end,
            "points": seg_points
        })

        summary_rows.append([
            seg_id,
            axis,
            seg_start,
            seg_end,
            overlap_ratio,
            seg_points.shape[0]
        ])

    return segments, summary_rows


def save_segment_summary(summary_rows, output_csv):
    header = [
        "segment_id",
        "axis",
        "range_start",
        "range_end",
        "overlap_ratio",
        "n_points"
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(summary_rows)
    print(f"Saved: {output_csv}")


def assign_segment_colors(point_cloud, segments, axis="x"):
    axis_idx = 0 if axis == "x" else 1
    coord = point_cloud[:, axis_idx]
    color_values = np.full(point_cloud.shape[0], -1, dtype=float)

    for seg in segments:
        idx = seg["segment_index"]
        seg_start = seg["range_start"]
        seg_end = seg["range_end"]
        mask = (coord >= seg_start) & (coord <= seg_end) & (color_values < 0)
        color_values[mask] = idx

    color_values[color_values < 0] = 0
    return color_values


def plot_segmented_cloud_3d(
    point_cloud,
    segments,
    axis,
    title="Segmented Cloud",
    sample_for_plot=None,
    point_size=10
):
    color_values = assign_segment_colors(point_cloud, segments, axis=axis)

    plot_point_cloud_3d(
        point_cloud=point_cloud,
        title=title,
        sample_for_plot=sample_for_plot,
        color_values=color_values,
        cmap="tab20",
        point_size=point_size
    )


def plot_segment_boundaries_2d(point_cloud, segments, axis="x", title="Segment boundaries on slope"):
    """
    2D helper plot to make segment intervals easier to read.
    It plots the segmentation axis against elevation Z.
    """
    if axis == "x":
        axis_idx = 0
        axis_label = "X"
    elif axis == "y":
        axis_idx = 1
        axis_label = "Y"
    else:
        raise ValueError("axis must be 'x' or 'y'")

    coord = point_cloud[:, axis_idx]
    z = point_cloud[:, 2]

    plt.figure(figsize=(11, 6))
    plt.scatter(coord, z, s=2, alpha=0.5)

    for seg in segments:
        x0 = seg["range_start"]
        x1 = seg["range_end"]
        seg_name = seg["segment_id"]
        xm = 0.5 * (x0 + x1)

        plt.axvline(x0, linestyle="--", linewidth=1)
        plt.axvline(x1, linestyle="--", linewidth=1)
        plt.text(xm, np.max(z), seg_name, ha="center", va="bottom", fontsize=9)

    plt.xlabel(axis_label)
    plt.ylabel("Z")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def process_single_las_file():
    # =========================
    # USER SETTINGS
    # =========================
    las_path = r"C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/Data/2021-06/laz/2021-06.laz"

    homo_dim = 1

    # reduction settings
    m = 1000                      # increase this for more points per TDA sample
    K = 10                        # increase this for more stable consensus features
    reduced_random_state = 42

    # segmentation settings
    n_segments = 5                # user input 1: number of segments
    overlap_ratio = 0.20          # user input 2: overlap ratio (0.20 = 20%)
    segment_axis = "x"            # choose "x" or "y"

    # plotting settings
    plot_sample_full = 30000
    plot_sample_reduced = None

    # csv output folder
    input_dir = os.path.dirname(las_path)
    prefix = os.path.splitext(os.path.basename(las_path))[0]
    output_dir = os.path.join(input_dir, f"{prefix}_tda_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input file: {las_path}")
    print(f"CSV output folder: {output_dir}")

    my_tda = tda(homo_dim=homo_dim, fts='all')

    # =========================
    # 1) LOAD RAW POINT CLOUD
    # =========================
    point_cloud = load_las_as_point_cloud(las_path)

    print("LAS/LAZ file read successfully.")
    print("Original point cloud shape:", point_cloud.shape)
    print("Initial number of points in main slope:", point_cloud.shape[0])

    # =========================
    # 2) SHOW ORIGINAL RAW SLOPE
    # =========================
    plot_point_cloud_3d(
        point_cloud=point_cloud,
        title=f"Original Slope - {prefix}",
        sample_for_plot=plot_sample_full,
        point_size=1
    )

    # =========================
    # 3) BUILD REDUCED DATASET
    # =========================
    reduced_cloud = select_representative_reduced_cloud(
        point_cloud,
        m=m,
        random_state=reduced_random_state
    )

    print("Reduced cloud shape:", reduced_cloud.shape)
    print("Number of points in reduced slope:", reduced_cloud.shape[0])

    reduced_cloud_csv = os.path.join(output_dir, f"{prefix}_whole_reduced_cloud.csv")
    np.savetxt(
        reduced_cloud_csv,
        reduced_cloud,
        delimiter=",",
        header="x,y,z",
        comments=""
    )
    print(f"Saved: {reduced_cloud_csv}")

    # =========================
    # 4) SHOW REDUCED SLOPE
    # =========================
    plot_point_cloud_3d(
        point_cloud=reduced_cloud,
        title=f"Reduced Slope - {prefix}",
        sample_for_plot=plot_sample_reduced,
        point_size=12
    )

    # =========================
    # 5) SEGMENT REDUCED DATASET
    # =========================
    segments, segment_summary_rows = segment_point_cloud_with_overlap(
        reduced_cloud,
        n_segments=n_segments,
        overlap_ratio=overlap_ratio,
        axis=segment_axis
    )

    segment_summary_csv = os.path.join(output_dir, f"{prefix}_segment_summary.csv")
    save_segment_summary(segment_summary_rows, segment_summary_csv)

    # =========================
    # 6) SHOW SEGMENT LOCATIONS ON MAIN SLOPE
    # =========================
    plot_segmented_cloud_3d(
        point_cloud=point_cloud,
        segments=segments,
        axis=segment_axis,
        title=f"Segment locations on MAIN slope ({segment_axis}-axis) - {prefix}",
        sample_for_plot=plot_sample_full,
        point_size=2
    )

    plot_segment_boundaries_2d(
        point_cloud=point_cloud,
        segments=segments,
        axis=segment_axis,
        title=f"Segment boundaries on MAIN slope ({segment_axis}-axis) - {prefix}"
    )

    # =========================
    # 7) SHOW SEGMENT LOCATIONS ON REDUCED SLOPE
    # =========================
    plot_segmented_cloud_3d(
        point_cloud=reduced_cloud,
        segments=segments,
        axis=segment_axis,
        title=f"Segment locations on REDUCED slope ({segment_axis}-axis) - {prefix}",
        sample_for_plot=plot_sample_reduced,
        point_size=20
    )

    plot_segment_boundaries_2d(
        point_cloud=reduced_cloud,
        segments=segments,
        axis=segment_axis,
        title=f"Segment boundaries on REDUCED slope ({segment_axis}-axis) - {prefix}"
    )

    # =========================
    # 8) TDA FOR WHOLE REDUCED DATASET
    # =========================
    whole_reduced_prefix = os.path.join(output_dir, f"{prefix}_whole_reduced")

    whole_reduced_features, _ = my_tda.random_sampling_consensus(
        reduced_cloud,
        m=min(m, reduced_cloud.shape[0]),
        K=K,
        out_prefix=whole_reduced_prefix
    )

    whole_reduced_features_csv = os.path.join(output_dir, f"{prefix}_whole_reduced_features.csv")
    save_features_csv(
        whole_reduced_features_csv,
        whole_reduced_features,
        homo_dim=homo_dim
    )

    my_tda([reduced_cloud])

    H0_whole, H1_whole = my_tda.save_h0_h1_to_csv(
        diagram_index=0,
        filename_prefix=whole_reduced_prefix
    )

    my_tda.plot_pd_h0_h1(
        H0_whole,
        H1_whole,
        title_prefix=f"{prefix} - Whole Reduced Data",
        show=True
    )

    # =========================
    # 9) TDA FOR EACH SEGMENT
    # =========================
    all_segment_feature_rows = []

    for seg in segments:
        seg_id = seg["segment_id"]
        seg_points = seg["points"]
        seg_start = seg["range_start"]
        seg_end = seg["range_end"]

        print("\n" + "=" * 70)
        print(f"Processing {seg_id}")
        print(f"Range: {seg_start:.6f} to {seg_end:.6f}")
        print(f"Points in segment: {seg_points.shape[0]}")
        print("=" * 70)

        if seg_points.shape[0] < 2:
            print(f"Skipping {seg_id}: not enough points.")
            continue

        seg_prefix = os.path.join(output_dir, f"{prefix}_{seg_id}")

        seg_features, _ = my_tda.random_sampling_consensus(
            seg_points,
            m=min(m, seg_points.shape[0]),
            K=K,
            out_prefix=seg_prefix
        )

        seg_features_csv = os.path.join(output_dir, f"{prefix}_{seg_id}_features.csv")
        save_features_csv(seg_features_csv, seg_features, homo_dim=homo_dim)

        my_tda([seg_points])

        H0_seg, H1_seg = my_tda.save_h0_h1_to_csv(
            diagram_index=0,
            filename_prefix=seg_prefix
        )

        my_tda.plot_pd_h0_h1(
            H0_seg,
            H1_seg,
            title_prefix=f"{prefix} - {seg_id}",
            show=True
        )

        row = [
            seg_id,
            segment_axis,
            seg_start,
            seg_end,
            seg_points.shape[0]
        ] + list(seg_features)

        all_segment_feature_rows.append(row)

    # =========================
    # 10) COMBINED SEGMENT SUMMARY CSV
    # =========================
    all_segments_csv = os.path.join(output_dir, f"{prefix}_all_segments_features_summary.csv")
    save_all_segments_feature_summary(
        all_segments_csv,
        all_segment_feature_rows,
        homo_dim=homo_dim
    )

    print("\nCompleted successfully.")
    print(f"CSV files are in:\n{output_dir}")


if __name__ == "__main__":
    try:
        process_single_las_file()
    except Exception as e:
        print("\nERROR OCCURRED:")
        print(str(e))
        print("\nFULL TRACEBACK:")
        traceback.print_exc()