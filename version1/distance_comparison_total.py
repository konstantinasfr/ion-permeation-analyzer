import MDAnalysis as mda
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def convert_to_pdb_numbering(residue_id, channel_type):
    """
    Converts a residue ID to a PDB-style numbering.
    """
    if channel_type == "G4":
        residues_per_chain = 325
        offset = 49
    elif channel_type == "G2" or channel_type == "G2_FD":
        residues_per_chain = 328
        offset = 54
    elif channel_type == "G12":
        residues_per_chain = 325
        offset = 53

    amino_acid_names = {152:"E",
                       184:"N",
                       141:"E",
                       173:"D",
                       }
    if channel_type == "G2_FD":
            amino_acid_names = {152:"E",
                       184:"N",
                       141:"E",
                       173:"D",
                       }
            
    if residue_id != "SF":
        residue_id = int(residue_id)
        chain_number = int(residue_id)//residues_per_chain
        if channel_type == "G2" or channel_type == "G2_FD":
            chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        elif channel_type == "G12":
            chain_dict = {0:"D", 1:"C", 2:"B", 3:"A"}
        pdb_number = residue_id-residues_per_chain*chain_number+offset
        if channel_type == "G12" and residue_id<=325:
            pdb_number = residue_id+42
        if channel_type == "G2_FD" and pdb_number==184 and chain_number==0:
            return "D184.A"
        else:
            return f"{amino_acid_names[pdb_number]}{pdb_number}.{chain_dict[chain_number]}", chain_dict[chain_number]
    else:
        return "SF"

# --- Residue dictionary for each channel type ---
RESIDUE_DICT = {
    "G2": {
        "glu_residues": [98, 426, 754, 1082],
        "sf_residues": [100, 428, 756, 1084],
        "hbc_residues": [138, 466, 794, 1122]
    },
    "G2_FD": {
        "glu_residues": [98, 426, 754, 1082],
        "sf_residues": [100, 428, 756, 1084],
        "hbc_residues": [138, 466, 794, 1122]
    },
    "G12": {
        "glu_residues": [99, 424, 749, 1074],
        "sf_residues": [101, 426, 751, 1076],
        "hbc_residues": [139, 464, 789, 1114]
    }
}

def compute_all_distances(topology_paths, trajectory_paths, channel_types, output_dir):
    """
    Compute distances for all systems and save combined data.

    Calculates z-offset, COM distances, min atom distances, and radial distances
    for all specified residues in all systems.

    Parameters:
        topology_paths (list): Paths to topology files (.prmtop)
        trajectory_paths (list): Paths to trajectory files (.nc)
        channel_types (list): List of channel types ("G2", "G12", "G2_FD")
        output_dir (str): Directory where output CSV will be saved

    Returns:
        DataFrame: Combined distances for all systems
    """
    all_records = []

    for system_idx, (top, traj, ch_type) in enumerate(zip(topology_paths, trajectory_paths, channel_types), start=1):
        print(f"📂 Processing system {system_idx} ({ch_type})...")
        group = 1 if ch_type == "G2" else (2 if ch_type == "G12" else 3)

        # Load trajectory
        u = mda.Universe(top, traj)

        # Get residues from RESIDUE_DICT
        sf_residues = RESIDUE_DICT[ch_type]["sf_residues"]
        hbc_residues = RESIDUE_DICT[ch_type]["hbc_residues"]
        glu_residues = RESIDUE_DICT[ch_type]["glu_residues"]

        # Select groups
        sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
        hbc_group = u.select_atoms("resid " + " ".join(map(str, hbc_residues)))
        residue_groups = {
            resid: u.select_atoms(f"resid {resid} and not name N CA C O HA H")
            for resid in glu_residues
        }

        # Iterate over trajectory
        for ts in tqdm(u.trajectory, desc=f"System {system_idx}", unit="frame"):
            sf_com = sf_group.center_of_mass()
            hbc_com = hbc_group.center_of_mass()

            # Define pore axis: SF → HBC
            axis_vector = sf_com - hbc_com
            axis_vector /= np.linalg.norm(axis_vector)

            for resid, atoms in residue_groups.items():
                res_com = atoms.center_of_mass()

                # z-offset relative to SF
                z_offset = sf_com[2] - res_com[2]

                # Distance metrics
                com_distance = np.linalg.norm(res_com - sf_com)
                min_distance = np.min(np.linalg.norm(atoms.positions - sf_com, axis=1))

                # Radial distance relative to pore axis
                v = res_com - hbc_com
                proj_length = np.dot(v, axis_vector)
                proj_point = hbc_com + proj_length * axis_vector
                radial_distance = np.linalg.norm(res_com - proj_point)

                # Residue labeling
                pdb_label, pdb_chain = convert_to_pdb_numbering(resid, ch_type)

                # Store record
                all_records.append({
                    "system": system_idx,
                    "channel_type": ch_type,
                    "group": group,
                    "frame": ts.frame,
                    "resid": resid,
                    "pdb_label": pdb_label,
                    "pdb_chain": pdb_chain,
                    "z_offset_from_sf": z_offset,
                    "com_to_sf_com_distance": com_distance,
                    "min_atom_to_sf_com_distance": min_distance,
                    "radial_distance": radial_distance
                })

    # Save combined data
    df = pd.DataFrame(all_records)
    combined_file = os.path.join(output_dir, "combined_distances.csv")
    df.to_csv(combined_file, index=False)
    print(f"✅ Saved combined distances: {combined_file}")
    return df


def compute_threshold_median_of_means(df, metric, start_fraction=0.05):
    """
    Computes threshold as median of per-residue means and determines direction.
    """
    mean_per_residue = df.groupby(["system", "resid"])[metric].mean().reset_index()
    threshold = mean_per_residue[metric].median()

    # Detect direction based on starting frames
    start_frames = df.groupby("system")["frame"].transform(
        lambda x: x <= (x.max() * start_fraction)
    )
    start_df = df[start_frames]
    above_count = (start_df[metric] > threshold).sum()
    below_count = (start_df[metric] <= threshold).sum()

    if above_count > below_count:
        direction = "drop"
        print(f"✅ Threshold for {metric}: {threshold:.2f} (majority start ABOVE → detecting DROPS)")
    else:
        direction = "rise"
        print(f"✅ Threshold for {metric}: {threshold:.2f} (majority start BELOW → detecting RISES)")

    return threshold, direction, mean_per_residue

def compute_residue_summaries(df, metric, threshold, direction="drop", output_dir="distance_comparison_results"):
    """
    Computes per-residue summary statistics for one metric and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    summaries = []

    for (system, resid), group in df.groupby(["system", "resid"]):
        values = group[metric].values

        mean_val = values.mean()
        variance_val = values.var()
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val

        # Fraction above/below threshold
        if direction == "rise":
            fraction = (values > threshold).sum() / len(values) * 100
        else:
            fraction = (values < threshold).sum() / len(values) * 100

        # Find first crossing frame
        event_label = "rise_frame" if direction == "rise" else "drop_frame"
        event_group = group[group[metric] > threshold] if direction == "rise" else group[group[metric] < threshold]
        event_frame = event_group["frame"].min() if not event_group.empty else "Never"

        # Threshold crossings
        crossings = ((values[:-1] <= threshold) & (values[1:] > threshold)).sum() if direction == "rise" \
                    else ((values[:-1] >= threshold) & (values[1:] < threshold)).sum()

        summaries.append({
            "system": system,
            "resid": resid,
            "pdb_label": group["pdb_label"].iloc[0],
            "pdb_chain": group["pdb_chain"].iloc[0],
            "channel_type": group["channel_type"].iloc[0],
            f"{metric}_mean": mean_val,
            f"{metric}_variance": variance_val,
            f"{metric}_min": min_val,
            f"{metric}_max": max_val,
            f"{metric}_range": range_val,
            f"fraction_{'above' if direction=='rise' else 'below'}_threshold": fraction,
            event_label: event_frame if not pd.isna(event_frame) else "Never",
            "threshold_crossings": crossings
        })

    summary_df = pd.DataFrame(summaries)
    summary_file = os.path.join(output_dir, f"{metric}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved summary: {summary_file}")
    return summary_df

def plot_residue_summaries_grouped(summary_df, threshold, output_dir, metric, direction="drop"):
    """
    Create scatter plots for all stats of a metric, grouping chains on X-axis
    and coloring by system. Legend includes system number and channel type.
    """
    import matplotlib.pyplot as plt
    import os
    import matplotlib.cm as cm

    base_folder = os.path.join(output_dir, "plots_grouped", metric)
    os.makedirs(base_folder, exist_ok=True)

    stats = ["mean", "variance", "min", "max", "range"]
    chains = ["A", "B", "C", "D"]

    # Define a colormap for systems (10 unique colors)
    system_ids = sorted(summary_df["system"].unique())
    system_channel_types = {
        row["system"]: row["channel_type"] for _, row in summary_df[["system", "channel_type"]].drop_duplicates().iterrows()
    }
    cmap = cm.get_cmap("tab10", len(system_ids))
    system_colors = {system: cmap(i) for i, system in enumerate(system_ids)}

    for stat in stats:
        plt.figure(figsize=(12, 6))

        # X-axis prep
        x_positions = []
        y_values = []
        color_values = []
        x_labels = []

        for idx, chain in enumerate(chains):
            df_chain = summary_df[summary_df["pdb_chain"] == chain]

            if df_chain.empty:
                continue

            # Assign X position for all dots in this chain
            x_pos = [idx] * len(df_chain)
            x_positions.extend(x_pos)
            y_values.extend(df_chain[f"{metric}_{stat}"])

            # Assign colors by system
            color_values.extend([system_colors[sys] for sys in df_chain["system"]])
            x_labels.append(f"Chain {chain}")

        # Scatter plot without black edges
        plt.scatter(x_positions, y_values,
                    color=color_values,
                    alpha=0.8, s=80)

        # Add threshold line (only for mean)
        if stat == "mean":
            plt.axhline(threshold, color="black", linestyle="--", linewidth=1.5,
                        label=f"Threshold = {threshold:.2f} Å")

        plt.title(f"{metric.replace('_', ' ').title()} – {stat.title()} (Grouped by Chain)", fontsize=16)
        plt.xticks(range(len(chains)), x_labels, fontsize=12)
        plt.ylabel(f"{metric.replace('_', ' ').title()} {stat.title()}", fontsize=14)
        plt.xlabel("Chains", fontsize=14)
        plt.yticks(fontsize=12)

        # System color legend (with channel types)
        handles = [plt.Line2D([0], [0], marker='o', color=system_colors[sys],
                              label=f"System {sys} ({system_channel_types[sys]})", markersize=8, linestyle='None')
                   for sys in system_ids]

        # Add threshold line to legend only if plotted
        if stat == "mean":
            handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                                      label=f"Threshold = {threshold:.2f} Å"))

        plt.legend(handles=handles, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        plot_file = os.path.join(base_folder, f"{metric}_{stat}_grouped_scatter.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"✅ Saved: {plot_file}")

    print(f"📊 All plots for {metric} saved in: {base_folder}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

def cluster_residues(summary_file, output_dir, metric_name):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    # Read the summary file
    df = pd.read_csv(summary_file)

    # Features for clustering
    features = [f"{metric_name}_mean", f"{metric_name}_variance",
                f"{metric_name}_min", f"{metric_name}_max", f"{metric_name}_range"]

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- KMeans Clustering ---
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = kmeans_labels

    # --- DBSCAN Clustering ---
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df["dbscan_cluster"] = dbscan_labels  # Includes noise (-1)
    df["is_dbscan_outlier"] = (dbscan_labels == -1)

    # Save results
    cluster_csv = os.path.join(output_dir, f"{metric_name}_clustering.csv")
    df.to_csv(cluster_csv, index=False)
    print(f"✅ Saved clustering results: {cluster_csv}")

    # --- PCA for KMeans visualization ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    for cluster in np.unique(kmeans_labels):
        cluster_data = df[df["kmeans_cluster"] == cluster]
        plt.scatter(cluster_data["PC1"], cluster_data["PC2"],
                    label=f"KMeans Cluster {cluster}", s=60)

    # Overlay DBSCAN outliers with X
    outliers = df[df["is_dbscan_outlier"]]
    plt.scatter(outliers["PC1"], outliers["PC2"],
                color="black", marker="x", s=100, label="DBSCAN Outlier")

    # Annotate DBSCAN outliers and systems 9 & 10
    to_annotate = df[df["is_dbscan_outlier"] | df["system"].isin([9, 10])]
    for _, row in to_annotate.iterrows():
        plt.text(row["PC1"], row["PC2"],
                 f"{row['pdb_label']} ({row['system']}, {row['channel_type']})",
                 fontsize=8, ha='right', va='bottom')

    plt.title(f"KMeans Clustering (PCA) for {metric_name.replace('_', ' ').title()}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    kmeans_plot_file = os.path.join(output_dir, f"{metric_name}_kmeans_pca.png")
    plt.savefig(kmeans_plot_file)
    plt.close()
    print(f"📊 Saved KMeans PCA plot: {kmeans_plot_file}")

    # --- t-SNE for DBSCAN visualization ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
    tsne_coords = tsne.fit_transform(X_scaled)
    df["TSNE1"] = tsne_coords[:, 0]
    df["TSNE2"] = tsne_coords[:, 1]

    plt.figure(figsize=(10, 6))
    for cluster in np.unique(dbscan_labels):
        if cluster == -1:
            continue  # Skip noise for now
        cluster_data = df[df["dbscan_cluster"] == cluster]
        plt.scatter(cluster_data["TSNE1"], cluster_data["TSNE2"],
                    label=f"DBSCAN Cluster {cluster}", s=60, marker='o')

    # Plot DBSCAN noise points as red dots
    noise = df[df["dbscan_cluster"] == -1]
    plt.scatter(noise["TSNE1"], noise["TSNE2"],
                color="red", marker="o", s=80, label="DBSCAN Noise")

    # Annotate DBSCAN noise points and systems 9 & 10
    to_annotate_tsne = df[df["is_dbscan_outlier"] | df["system"].isin([9, 10])]
    for _, row in to_annotate_tsne.iterrows():
        plt.text(row["TSNE1"], row["TSNE2"],
                 f"{row['pdb_label']} ({row['system']}, {row['channel_type']})",
                 fontsize=8, ha='right', va='bottom')

    plt.title(f"DBSCAN Clustering (t-SNE) for {metric_name.replace('_', ' ').title()}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()

    dbscan_plot_file = os.path.join(output_dir, f"{metric_name}_dbscan_tsne.png")
    plt.savefig(dbscan_plot_file)
    plt.close()
    print(f"📊 Saved DBSCAN t-SNE plot: {dbscan_plot_file}")




def run_clustering_on_all_summaries(summary_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    summary_files = [f for f in os.listdir(summary_folder) if f.endswith("_summary.csv")]

    for file in summary_files:
        metric_name = file.replace("_summary.csv", "")
        print(f"\n🔍 Processing metric: {metric_name}")
        summary_path = os.path.join(summary_folder, file)
        metric_output = os.path.join(output_folder, metric_name)
        os.makedirs(metric_output, exist_ok=True)

        cluster_residues(summary_path, metric_output, metric_name)



def run_full_pipeline(topology_paths, trajectory_paths, channel_types, output_dir):

    if not os.path.exists(f"{output_dir}/plots_grouped"):
        df = compute_all_distances(topology_paths, trajectory_paths, channel_types, output_dir)
        metrics = ["z_offset_from_sf", "com_to_sf_com_distance", "min_atom_to_sf_com_distance", "radial_distance"]

        for metric in metrics:
            threshold, direction, _ = compute_threshold_median_of_means(df, metric)
            summary_df = compute_residue_summaries(df, metric, threshold, direction, output_dir)
            plot_residue_summaries_grouped(
                    summary_df=summary_df,              # The DataFrame with all your summary metrics
                    threshold=threshold,                # The threshold value you computed
                    output_dir=output_dir,      # Folder where plots will be saved
                    metric=metric,             # The metric you want to plot
                    direction=direction                    # Either "drop" or "rise" (same as your threshold logic)
                )

    run_clustering_on_all_summaries(
        summary_folder=output_dir,
        output_folder=os.path.join(output_dir, "clustering_results")
    )




base = "/home/data/Konstantina/"
# Example usage
topology_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN1/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN2/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN3/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN4/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN5/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN6/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN7/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN8/com_4fs.prmtop",
    f"{base}/GIRK12_WT/RUN2/com_4fs.prmtop",
    f"{base}/GIRK2/GIRK2_FD_RUN2/com_4fs.prmtop",
]
trajectory_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN1/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN2/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN3/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN4/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN5/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN6/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN7/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN8/protein.nc",
    f"{base}/GIRK12_WT/RUN2/protein.nc",
    f"{base}/GIRK2/GIRK2_FD_RUN2/protein.nc",
]
channel_types = ["G2", "G2", "G2", "G2","G2", "G2", "G2", "G2","G12", "G2_FD"]





# topology_paths = [
#     f"{base}/GIRK2/G2_4KFM_RUN1/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN2/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN3/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN4/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN5/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN6/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN7/com_4fs.prmtop",
#     # f"{base}/GIRK2/G2_4KFM_RUN8/com_4fs.prmtop",
#     f"{base}/GIRK12_WT/RUN2/com_4fs.prmtop",
#     # f"{base}/GIRK2/GIRK2_FD_RUN2/com_4fs.prmtop",
# ]
# trajectory_paths = [
#     f"{base}/GIRK2/G2_4KFM_RUN1/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN2/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN3/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN4/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN5/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN6/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN7/protein.nc",
#     # f"{base}/GIRK2/G2_4KFM_RUN8/protein.nc",
#     f"{base}/GIRK12_WT/RUN2/protein.nc",
#     # f"{base}/GIRK2/GIRK2_FD_RUN2/protein.nc",
# ]
# channel_types = ["G2","G12"]



output_dir = "./distance_comparison_total_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
run_full_pipeline(topology_paths, trajectory_paths, channel_types, output_dir)