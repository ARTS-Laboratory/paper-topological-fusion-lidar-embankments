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

    def random_sampling_consensus(self, pcd, m=100000, K=10, out_prefix=None):
        features_list = []
        last_subset = None

        for i in range(K):
            print(f"Iteration {i + 1}/{K}")
            if pcd.shape[0] > m:
                idx = np.random.choice(pcd.shape[0], size=m, replace=False)
                subset = pcd[idx, :]
            else:
                subset = pcd

            last_subset = subset.copy()

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

        return median_features, features_array, last_subset

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
        "segment_id", "x_range_start", "x_range_end",
        "y_range_start", "y_range_end", "n_points"
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


def segment_point_cloud_with_overlap_2d(
    point_cloud,
    n_segments_x=4,
    n_segments_y=4,
    overlap_ratio_x=0.2,
    overlap_ratio_y=0.2
):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    x_total = xmax - xmin
    y_total = ymax - ymin

    if x_total == 0 or y_total == 0:
        raise ValueError("Point cloud extent in x or y is zero; cannot do 2D segmentation.")

    x_base_width = x_total / n_segments_x
    y_base_width = y_total / n_segments_y

    x_overlap_width = overlap_ratio_x * x_base_width
    y_overlap_width = overlap_ratio_y * y_base_width

    segments = []
    summary_rows = []

    for ix in range(n_segments_x):
        x_base_start = xmin + ix * x_base_width
        x_base_end = xmin + (ix + 1) * x_base_width

        x_start = max(xmin, x_base_start - x_overlap_width / 2.0)
        x_end = min(xmax, x_base_end + x_overlap_width / 2.0)

        if ix == 0:
            x_start = xmin
        if ix == n_segments_x - 1:
            x_end = xmax

        for iy in range(n_segments_y):
            y_base_start = ymin + iy * y_base_width
            y_base_end = ymin + (iy + 1) * y_base_width

            y_start = max(ymin, y_base_start - y_overlap_width / 2.0)
            y_end = min(ymax, y_base_end + y_overlap_width / 2.0)

            if iy == 0:
                y_start = ymin
            if iy == n_segments_y - 1:
                y_end = ymax

            mask = (
                (x >= x_start) & (x <= x_end) &
                (y >= y_start) & (y <= y_end)
            )
            seg_points = point_cloud[mask]

            seg_id = f"segment_x{ix + 1:02d}_y{iy + 1:02d}"

            segments.append({
                "segment_id": seg_id,
                "segment_index_x": ix + 1,
                "segment_index_y": iy + 1,
                "x_range_start": x_start,
                "x_range_end": x_end,
                "y_range_start": y_start,
                "y_range_end": y_end,
                "points": seg_points
            })

            summary_rows.append([
                seg_id,
                x_start,
                x_end,
                y_start,
                y_end,
                overlap_ratio_x,
                overlap_ratio_y,
                seg_points.shape[0]
            ])

    return segments, summary_rows


def save_segment_summary(summary_rows, output_csv):
    header = [
        "segment_id",
        "x_range_start",
        "x_range_end",
        "y_range_start",
        "y_range_end",
        "overlap_ratio_x",
        "overlap_ratio_y",
        "n_points"
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(summary_rows)
    print(f"Saved: {output_csv}")


def assign_segment_colors_2d(point_cloud, segments):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    color_values = np.full(point_cloud.shape[0], -1, dtype=float)

    for i, seg in enumerate(segments, start=1):
        x0 = seg["x_range_start"]
        x1 = seg["x_range_end"]
        y0 = seg["y_range_start"]
        y1 = seg["y_range_end"]

        mask = (
            (x >= x0) & (x <= x1) &
            (y >= y0) & (y <= y1) &
            (color_values < 0)
        )
        color_values[mask] = i

    color_values[color_values < 0] = 0
    return color_values


def plot_segmented_cloud_3d(
    point_cloud,
    segments,
    title="Segmented Cloud",
    sample_for_plot=None,
    point_size=10
):
    color_values = assign_segment_colors_2d(point_cloud, segments)

    plot_point_cloud_3d(
        point_cloud=point_cloud,
        title=title,
        sample_for_plot=sample_for_plot,
        color_values=color_values,
        cmap="tab20",
        point_size=point_size
    )


def plot_segment_boundaries_2d(point_cloud, segments, title="2D segment boundaries on slope"):
    plt.figure(figsize=(10, 8))
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=2, alpha=0.5)

    for seg in segments:
        x0 = seg["x_range_start"]
        x1 = seg["x_range_end"]
        y0 = seg["y_range_start"]
        y1 = seg["y_range_end"]
        seg_name = seg["segment_id"]

        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        plt.plot(xs, ys, linewidth=1)

        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        plt.text(xm, ym, seg_name, ha="center", va="center", fontsize=8)

    plt.xlabel("X")
    plt.ylabel("Y")
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
    m = 1000
    K = 10

    # segmentation settings
    n_segments_x = 5
    n_segments_y = 4
    overlap_ratio_x = 0.20
    overlap_ratio_y = 0.20

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
    # 3) BUILD REDUCED DATASET FROM THE LAST SAMPLE OF YOUR K RUNS
    # =========================
    whole_reduced_prefix = os.path.join(output_dir, f"{prefix}_whole_reduced")

    whole_reduced_features, _, reduced_cloud = my_tda.random_sampling_consensus(
        point_cloud,
        m=m,
        K=K,
        out_prefix=whole_reduced_prefix
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

    whole_reduced_features_csv = os.path.join(output_dir, f"{prefix}_whole_reduced_features.csv")
    save_features_csv(
        whole_reduced_features_csv,
        whole_reduced_features,
        homo_dim=homo_dim
    )

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
    # 5) SEGMENT REDUCED DATASET IN 2D (X AND Y)
    # =========================
    segments, segment_summary_rows = segment_point_cloud_with_overlap_2d(
        reduced_cloud,
        n_segments_x=n_segments_x,
        n_segments_y=n_segments_y,
        overlap_ratio_x=overlap_ratio_x,
        overlap_ratio_y=overlap_ratio_y
    )

    segment_summary_csv = os.path.join(output_dir, f"{prefix}_segment_summary.csv")
    save_segment_summary(segment_summary_rows, segment_summary_csv)

    # =========================
    # 6) SHOW SEGMENT LOCATIONS ON MAIN SLOPE
    # =========================
    plot_segmented_cloud_3d(
        point_cloud=point_cloud,
        segments=segments,
        title=f"Segment locations on MAIN slope (2D x-y segmentation) - {prefix}",
        sample_for_plot=plot_sample_full,
        point_size=2
    )

    plot_segment_boundaries_2d(
        point_cloud=point_cloud,
        segments=segments,
        title=f"2D segment boundaries on MAIN slope - {prefix}"
    )

    # =========================
    # 7) SHOW SEGMENT LOCATIONS ON REDUCED SLOPE
    # =========================
    plot_segmented_cloud_3d(
        point_cloud=reduced_cloud,
        segments=segments,
        title=f"Segment locations on REDUCED slope (2D x-y segmentation) - {prefix}",
        sample_for_plot=plot_sample_reduced,
        point_size=20
    )

    plot_segment_boundaries_2d(
        point_cloud=reduced_cloud,
        segments=segments,
        title=f"2D segment boundaries on REDUCED slope - {prefix}"
    )

    # =========================
    # 8) TDA FOR WHOLE REDUCED DATASET
    # =========================
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
        x_start = seg["x_range_start"]
        x_end = seg["x_range_end"]
        y_start = seg["y_range_start"]
        y_end = seg["y_range_end"]

        print("\n" + "=" * 70)
        print(f"Processing {seg_id}")
        print(f"X range: {x_start:.6f} to {x_end:.6f}")
        print(f"Y range: {y_start:.6f} to {y_end:.6f}")
        print(f"Points in segment: {seg_points.shape[0]}")
        print("=" * 70)

        if seg_points.shape[0] < 2:
            print(f"Skipping {seg_id}: not enough points.")
            continue

        seg_prefix = os.path.join(output_dir, f"{prefix}_{seg_id}")

        seg_features, _, _ = my_tda.random_sampling_consensus(
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
            x_start,
            x_end,
            y_start,
            y_end,
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

