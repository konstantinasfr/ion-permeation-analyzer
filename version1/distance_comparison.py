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
            return f"{amino_acid_names[pdb_number]}{pdb_number}.{chain_dict[chain_number]}"
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
def compute_all_z_offsets(topology_paths, trajectory_paths, channel_types):
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
                    "pdb_label": convert_to_pdb_numbering(resid, ch_type),
                    "z_offset_from_sf": z_offset
                })

    df = pd.DataFrame(all_records)
    df.to_csv("combined_z_offsets.csv", index=False)
    print("✅ Saved combined z-offsets: combined_z_offsets.csv")
    return df

# --- Step 2: Auto-detect threshold ---
def compute_threshold_median_of_means(df):
    mean_per_residue = df.groupby(["system", "resid"])["z_offset_from_sf"].mean().reset_index()
    threshold = mean_per_residue["z_offset_from_sf"].median()
    print(f"✅ Auto-detected threshold (median of means): {threshold:.2f} Å")
    return threshold, mean_per_residue

# --- Step 3: Compute residue summaries ---
def compute_residue_summaries(df, threshold):
    summaries = []
    for (system, resid), group in df.groupby(["system", "resid"]):
        z_offsets = group["z_offset_from_sf"].values
        mean_z = z_offsets.mean()
        variance_z = z_offsets.var()
        fraction_above = (z_offsets > threshold).sum() / len(z_offsets) * 100
        drop_frame = group[group["z_offset_from_sf"] < threshold]["frame"].min()
        summaries.append({
            "system": system,
            "resid": resid,
            "pdb_label": group["pdb_label"].iloc[0],
            "mean_z_offset": mean_z,
            "variance_z_offset": variance_z,
            "fraction_above_threshold": fraction_above,
            "drop_frame": drop_frame if not np.isnan(drop_frame) else "Never"
        })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("residue_summary_features.csv", index=False)
    print("✅ Residue summaries saved: residue_summary_features.csv")
    return summary_df

# --- Step 4: Visualization ---
def plot_residue_summaries(summary_df, threshold):
    os.makedirs("plots", exist_ok=True)
    for resid, df_res in summary_df.groupby("resid"):
        plt.figure(figsize=(8, 5))
        plt.bar(df_res["system"], df_res["mean_z_offset"],
                color=["blue" if ct=="G2" else "green" if ct=="G12" else "red" for ct in df_res["pdb_label"]])
        plt.axhline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.2f} Å")
        plt.title(f"Mean z-offset: {df_res['pdb_label'].iloc[0]}")
        plt.xlabel("System")
        plt.ylabel("Mean z-offset (Å)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{df_res['pdb_label'].iloc[0]}_mean_z_offset.png")
        plt.close()
    print("📊 Plots saved in 'plots/' folder.")

# --- Main Pipeline ---
def run_full_pipeline(topology_paths, trajectory_paths, channel_types):
    combined_df = compute_all_z_offsets(topology_paths, trajectory_paths, channel_types)
    threshold, _ = compute_threshold_median_of_means(combined_df)
    summary_df = compute_residue_summaries(combined_df, threshold)
    plot_residue_summaries(summary_df, threshold)

base = "/home/data/Konstantina/"
# Example usage
topology_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN_1/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_2/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_3/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_4/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_5/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_6/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_7/com_4fs.prmtop",
    f"{base}/GIRK2/G2_4KFM_RUN_8/com_4fs.prmtop",
    f"{base}/GIRK12_WT/RUN_2/com_4fs.prmtop",
    f"{base}/GIRK2/GIRK2_FD_RUN2/com_4fs.prmtop",
]
trajectory_paths = [
    f"{base}/GIRK2/G2_4KFM_RUN_1/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_2/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_3/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_4/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_5/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_6/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_7/protein.nc",
    f"{base}/GIRK2/G2_4KFM_RUN_8/protein.nc",
    f"{base}/GIRK12_WT/RUN_2/protein.nc",
    f"{base}/GIRK2/GIRK2_FD_RUN2/protein.nc",
]
channel_types = ["G2", "G2", "G2", "G2","G2", "G2", "G2", "G2","G12", "G2_FD"]

run_full_pipeline(topology_paths, trajectory_paths, channel_types)
