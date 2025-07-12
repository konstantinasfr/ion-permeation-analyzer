import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from scipy.stats import norm
from relaxation_frame_finder import analyze_system_relaxation

# Residue dictionary for each channel type
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

    amino_acid_names = {152: "E", 184: "N", 141: "E", 173: "D"}
    if channel_type == "G2_FD":
        amino_acid_names = {152: "E", 184: "N", 141: "E", 173: "D"}

    if residue_id != "SF":
        residue_id = int(residue_id)
        chain_number = int(residue_id) // residues_per_chain
        if channel_type == "G2" or channel_type == "G2_FD":
            chain_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
        elif channel_type == "G12":
            chain_dict = {0: "D", 1: "C", 2: "B", 3: "A"}
        pdb_number = residue_id - residues_per_chain * chain_number + offset
        if channel_type == "G12" and residue_id <= 325:
            pdb_number = residue_id + 42
        if channel_type == "G2_FD" and pdb_number == 184 and chain_number == 0:
            return "D184.A"
        else:
            return f"{amino_acid_names[pdb_number]}{pdb_number}.{chain_dict[chain_number]}", chain_dict[chain_number]
    else:
        return "SF"

def bootstrap_and_fit_gaussian(values, sample_size=50, n_bootstrap=1000):
    """
    Perform bootstrap sampling and fit a Gaussian to the distribution of sample means.
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=sample_size, replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)

    # Fit Gaussian: estimate mean and stddev
    mu, std = norm.fit(bootstrap_means)
    return bootstrap_means, mu, std


def analyze_simulation(top_path, traj_path, simulation_id, channel_type, output_dir, sidechain=True, relaxation=True,
                        sample_size=50, n_bootstrap=1000):
    """
    Analyze a single simulation, computing Gaussians for specified residues and metrics
    in a single trajectory pass.
    """
    u = mda.Universe(top_path, traj_path)

    # Create folder for this simulation
    if sidechain:
        output_dir_full = os.path.join(output_dir, "sidechain")
    else:
        output_dir_full = os.path.join(output_dir, "full_residue")

    # Decide subfolder based on relaxation flag
    mode = "relaxation" if relaxation else "no_relaxation"
    output_dir_full = os.path.join(output_dir_full, mode)

    # Set simulation folder
    sim_output_dir = os.path.join(output_dir_full, f"simulation_{simulation_id}")

    # Determine start frame
    start_frame = (
        analyze_system_relaxation(top_path, traj_path, sim_output_dir)
        if relaxation else 0
    )

    os.makedirs(sim_output_dir, exist_ok=True)

    sf_residues = RESIDUE_DICT[channel_type]["sf_residues"]
    hbc_residues = RESIDUE_DICT[channel_type]["hbc_residues"]
    glu_residues = RESIDUE_DICT[channel_type]["glu_residues"]

    # Select groups
    sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
    hbc_group = u.select_atoms("resid " + " ".join(map(str, hbc_residues)))
    if sidechain:
        residue_groups = {
            resid: u.select_atoms(f"resid {resid} and not name N CA C O HA H")
            for resid in glu_residues
        }
    else:
        residue_groups = {
            resid: u.select_atoms(f"resid {resid}")
            for resid in glu_residues
        }
    # Metrics dictionary: {metric: {resid: [values]}}
    metrics_data = {
        "z_offset_from_sf": {resid: [] for resid in glu_residues},
        "com_to_sf_com_distance": {resid: [] for resid in glu_residues},
        "min_atom_to_sf_com_distance": {resid: [] for resid in glu_residues},
        "radial_distance": {resid: [] for resid in glu_residues},
    }

    print(f"🔄 Parsing trajectory for Simulation {simulation_id}...")
    for ts in tqdm(u.trajectory[start_frame:], desc=f"Simulation {simulation_id}", unit="frame"):
        sf_com = sf_group.center_of_mass()
        hbc_com = hbc_group.center_of_mass()
        axis_vector = sf_com - hbc_com
        axis_vector /= np.linalg.norm(axis_vector)

        for resid, atoms in residue_groups.items():
            res_com = atoms.center_of_mass()

            # Calculate all metrics
            z_offset = sf_com[2] - res_com[2]
            com_distance = np.linalg.norm(res_com - sf_com)
            min_distance = np.min(np.linalg.norm(atoms.positions - sf_com, axis=1))
            v = res_com - hbc_com
            proj_length = np.dot(v, axis_vector)
            proj_point = hbc_com + proj_length * axis_vector
            radial_distance = np.linalg.norm(res_com - proj_point)

            # Store values
            metrics_data["z_offset_from_sf"][resid].append(z_offset)
            metrics_data["com_to_sf_com_distance"][resid].append(com_distance)
            metrics_data["min_atom_to_sf_com_distance"][resid].append(min_distance)
            metrics_data["radial_distance"][resid].append(radial_distance)

    # Create a dictionary to store all Gaussian parameters for all metrics
    all_gaussian_params_by_metric = {}

    # Process metrics and save results
    for metric, residues_data in metrics_data.items():
        plt.figure(figsize=(10, 6))
        
        # Initialize list to collect all Gaussian params for this metric
        if metric not in all_gaussian_params_by_metric:
            all_gaussian_params_by_metric[metric] = []

        for resid, values in residues_data.items():
            values = np.array(values)

            # Bootstrap and fit Gaussian
            bootstrap_means, mu, std = bootstrap_and_fit_gaussian(
                values, sample_size=sample_size, n_bootstrap=n_bootstrap
            )

            # Convert to PDB numbering
            pdb_label, pdb_chain = convert_to_pdb_numbering(resid, channel_type)

            # Save Gaussian parameters for this residue
            all_gaussian_params_by_metric[metric].append({
                "Simulation": simulation_id,
                "Metric": metric,
                "Residue": resid,
                "PDB_Label": pdb_label,
                "Gaussian_Mean": mu,
                "Gaussian_StdDev": std
            })

            # Fixed color mapping for chains
            CHAIN_COLORS = {
                "A": "blue",
                "B": "orange",
                "C": "green",
                "D": "red"
            }

            # Plot Gaussian curve with fixed colors
            sns.kdeplot(
                bootstrap_means,
                label=f"{pdb_label} (μ={mu:.2f}, σ={std:.2f})",
                color=CHAIN_COLORS[pdb_chain]
            )

        # Save Gaussian plot for this simulation and metric
        plt.title(f"Simulation {simulation_id} – {metric}")
        plt.xlabel("Bootstrap Sample Means")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sim_output_dir, f"{metric}_gaussians.png"))
        plt.close()

        print(f"✅ Saved Gaussian plot for {metric} in Simulation {simulation_id}")

    # After processing all simulations, save combined CSVs per metric
    for metric, params_list in all_gaussian_params_by_metric.items():
        df_params = pd.DataFrame(params_list)
        os.makedirs(os.path.join(output_dir_full, "plots"), exist_ok=True)
        csv_path = os.path.join(output_dir_full, "plots", f"{metric}_gaussian_parameters.csv")
        # If file exists, append without header
        if os.path.exists(csv_path):
            df_params.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # If file doesn't exist, write with header
            df_params.to_csv(csv_path, index=False)

        print(f"✅ Saved combined Gaussian parameters for {metric} in {csv_path}")




def analyze_all_simulations(topology_paths, trajectory_paths, simulation_ids, channel_types,
                            output_dir, sidechain=True, relaxation=False, sample_size=50, n_bootstrap=1000):
    """
    Run Gaussian analysis for all simulations.
    """
    os.makedirs(output_dir, exist_ok=True)
    for top, traj, sim_id, ch_type in zip(topology_paths, trajectory_paths, simulation_ids, channel_types):
        analyze_simulation(top, traj, sim_id, ch_type, output_dir, sidechain=sidechain, relaxation=relaxation,
                           sample_size=sample_size, n_bootstrap=n_bootstrap)


# ======== USER SETTINGS ========
# Topologies and trajectories
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
simulation_ids = ["G2_RUN1", "G2_RUN2", "G2_RUN3", "G2_RUN4",
                 "G2_RUN5", "G2_RUN6", "G2_RUN7", "G2_RUN8",
                 "G12_RUN2", "G2_FD_RUN2"]



# base = "/home/data/Konstantina/"
# topology_paths = [
#     f"{base}/GIRK2/G2_4KFM_RUN1/com_4fs.prmtop",
#     f"{base}/GIRK12_WT/RUN2/com_4fs.prmtop"
# ]
# trajectory_paths = [
#     f"{base}/GIRK2/G2_4KFM_RUN1/protein.nc",
#     f"{base}/GIRK12_WT/RUN2/protein.nc"
# ]
# simulation_ids = ["G2_RUN1", "G12_RUN2"]
# channel_types = ["G2", "G12"]


# Output directory
output_dir = "./gaussian_analysis_results"
sidechain = False  # Set to True for sidechain analysis, False for full residue analysis
relaxation = False  
# Run analysis
analyze_all_simulations(topology_paths, trajectory_paths, simulation_ids,
                        channel_types, output_dir, sidechain=sidechain,relaxation=relaxation,
                        sample_size=50, n_bootstrap=5000)
