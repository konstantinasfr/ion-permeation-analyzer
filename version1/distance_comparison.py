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
        "sf_residues": [100, 428, 756, 1084]
    },
    "G2_FD": {
        "glu_residues": [98, 426, 754, 1082],
        "sf_residues": [100, 428, 756, 1084]
    },
    "G12": {
        "glu_residues": [99, 424, 749, 1074],
        "sf_residues": [101, 426, 751, 1076]
    }
}

# --- Step 1: Compute z-offsets for all systems ---
def compute_all_z_offsets(topology_paths, trajectory_paths, channel_types, output_dir):
    all_records = []
    for system_idx, (top, traj, ch_type) in enumerate(zip(topology_paths, trajectory_paths, channel_types), start=1):
        print(f"📂 Processing system {system_idx} ({ch_type})...")
        group = 1 if ch_type == "G2" else (2 if ch_type == "G12" else 3)
        u = mda.Universe(top, traj)

        sf_residues = RESIDUE_DICT[ch_type]["sf_residues"]
        glu_residues = RESIDUE_DICT[ch_type]["glu_residues"]

        sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
        residue_groups = {resid: u.select_atoms(f"resid {resid} and not name N CA C O HA H")
                          for resid in glu_residues}

        for ts in tqdm(u.trajectory, desc=f"System {system_idx}", unit="frame"):
            sf_com = sf_group.center_of_mass()
            for resid, atoms in residue_groups.items():
                res_com = atoms.center_of_mass()
                z_offset = sf_com[2] - res_com[2]
                all_records.append({
                    "system": system_idx,
                    "channel_type": ch_type,
                    "group": group,
                    "frame": ts.frame,
                    "resid": resid,
                    "pdb_label": convert_to_pdb_numbering(resid, ch_type)[0],
                    "pdb_chain": convert_to_pdb_numbering(resid, ch_type)[1],
                    "z_offset_from_sf": z_offset
                })

    df = pd.DataFrame(all_records)
    df.to_csv(f"{output_dir}/combined_z_offsets.csv", index=False)
    print("✅ Saved combined z-offsets: combined_z_offsets.csv")
    return df

# --- Step 2: Auto-detect threshold ---
def compute_threshold_median_of_means(df, start_fraction=0.05):
    """
    Computes threshold as median of per-residue means and determines direction automatically.
    
    Parameters:
        df (DataFrame): Combined z-offset data
        start_fraction (float): Fraction of frames at start of simulation to use for direction detection
    
    Returns:
        threshold (float): Detected threshold
        direction (str): 'drop' or 'rise'
        mean_per_residue (DataFrame): Mean z-offset per residue
    """
    # Compute median of per-residue means
    mean_per_residue = df.groupby(["system", "resid"])["z_offset_from_sf"].mean().reset_index()
    threshold = mean_per_residue["z_offset_from_sf"].median()

    # Detect direction based on starting frames
    start_frames = df.groupby("system")["frame"].transform(
        lambda x: x <= (x.max() * start_fraction)
    )
    start_df = df[start_frames]
    above_count = (start_df["z_offset_from_sf"] > threshold).sum()
    below_count = (start_df["z_offset_from_sf"] <= threshold).sum()

    if above_count > below_count:
        direction = "drop"
        print(f"✅ Threshold: {threshold:.2f} Å (majority start ABOVE → detecting DROPS)")
    else:
        direction = "rise"
        print(f"✅ Threshold: {threshold:.2f} Å (majority start BELOW → detecting RISES)")

    return threshold, direction, mean_per_residue


# --- Step 3: Compute residue summaries ---
def compute_residue_summaries(df, threshold, direction="drop", output_dir="distance_comparison_results"):
    os.makedirs(output_dir, exist_ok=True)
    summaries = []

    for (system, resid), group in df.groupby(["system", "resid"]):
        z_offsets = group["z_offset_from_sf"].values

        # Basic statistics
        mean_z = z_offsets.mean()
        variance_z = z_offsets.var()
        min_z = z_offsets.min()
        max_z = z_offsets.max()
        range_z = max_z - min_z

        # Fraction above or below threshold
        if direction == "rise":
            fraction = (z_offsets > threshold).sum() / len(z_offsets) * 100
        else:  # "drop"
            fraction = (z_offsets < threshold).sum() / len(z_offsets) * 100

        # Find first frame depending on direction
        if direction == "rise":
            event_group = group[group["z_offset_from_sf"] > threshold]
            event_label = "rise_frame"
        else:  # "drop"
            event_group = group[group["z_offset_from_sf"] < threshold]
            event_label = "drop_frame"

        event_frame = event_group["frame"].min() if not event_group.empty else "Never"

        # Count threshold crossings (upwards or downwards)
        if direction == "rise":
            crossings = ((z_offsets[:-1] <= threshold) & (z_offsets[1:] > threshold)).sum()
        else:  # "drop"
            crossings = ((z_offsets[:-1] >= threshold) & (z_offsets[1:] < threshold)).sum()

        summaries.append({
            "system": system,
            "resid": resid,
            "pdb_label": group["pdb_label"].iloc[0],
            "pdb_chain": group["pdb_chain"].iloc[0],
            "mean_z_offset": mean_z,
            "variance_z_offset": variance_z,
            "min_z_offset": min_z,
            "max_z_offset": max_z,
            "range_z_offset": range_z,
            f"fraction_{'above' if direction=='rise' else 'below'}_threshold": fraction,
            event_label: event_frame if not pd.isna(event_frame) else "Never",
            "threshold_crossings": crossings
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(f"{output_dir}/residue_summary_features.csv", index=False)
    print(f"✅ Residue summaries saved: {output_dir}/residue_summary_features.csv")
    return summary_df


# --- Step 4: Visualization ---
import matplotlib.pyplot as plt
import os

def plot_residue_summaries(summary_df, threshold, output_dir, direction="drop"):
    """
    Create 6 plots per chain, showing residue metrics across all systems.
    
    Parameters:
        summary_df (DataFrame): Output of compute_residue_summaries
        threshold (float): Auto-detected threshold
        output_dir (str): Base directory to save plots
        direction (str): "drop" or "rise"
    """
    plots_base = f"{output_dir}/plots"
    os.makedirs(plots_base, exist_ok=True)

    # Metrics to plot
    metrics = [
        "mean_z_offset",
        "variance_z_offset",
        "min_z_offset",
        "max_z_offset",
        "range_z_offset",
        f"fraction_{'above' if direction=='rise' else 'below'}_threshold"
    ]

    # Group data by chain
    for chain, df_chain in summary_df.groupby("pdb_chain"):
        chain_dir = f"{plots_base}/Chain_{chain}"
        os.makedirs(chain_dir, exist_ok=True)
        print(f"📂 Creating plots for Chain {chain}...")

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for system in sorted(df_chain["system"].unique()):
                df_sys = df_chain[df_chain["system"] == system]
                x_labels = df_sys["pdb_label"]
                y_values = df_sys[metric]
                plt.plot(x_labels, y_values, marker="o", label=f"System {system}")

            plt.axhline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.2f} Å")
            plt.title(f"{metric.replace('_', ' ').title()} (Chain {chain})", fontsize=16)
            plt.xlabel("Residue", fontsize=14)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
            plt.xticks(rotation=45, fontsize=10)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=10)
            plt.tight_layout()
            filename = f"{chain_dir}/{metric}.png"
            plt.savefig(filename)
            plt.close()
            print(f"✅ Saved: {filename}")

    print(f"📊 All plots saved in: {plots_base}")


# --- Main Pipeline ---
def run_full_pipeline(topology_paths, trajectory_paths, channel_types, output_dir):
    combined_df = compute_all_z_offsets(topology_paths, trajectory_paths, channel_types, output_dir)

    threshold, direction, mean_df = compute_threshold_median_of_means(combined_df)
    summary_df = compute_residue_summaries(combined_df, threshold, direction)

    plot_residue_summaries(summary_df, threshold, output_dir, direction)

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





topology_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN1/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN2/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN3/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN4/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN5/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN6/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN7/com_4fs.prmtop",
    # f"{base}/GIRK2/G2_4KFM_RUN8/com_4fs.prmtop",
    f"{base}/GIRK12_WT/RUN2/com_4fs.prmtop",
    # f"{base}/GIRK2/GIRK2_FD_RUN2/com_4fs.prmtop",
]
trajectory_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN1/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN2/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN3/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN4/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN5/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN6/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN7/protein.nc",
    # f"{base}/GIRK2/G2_4KFM_RUN8/protein.nc",
    f"{base}/GIRK12_WT/RUN2/protein.nc",
    # f"{base}/GIRK2/GIRK2_FD_RUN2/protein.nc",
]
channel_types = ["G2","G12"]



output_dir = "./distance_comparison_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
run_full_pipeline(topology_paths, trajectory_paths, channel_types, output_dir)
