import MDAnalysis as mda
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from MDAnalysis.analysis.dihedrals import Dihedral
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def convert_to_pdb_numbering(residue_id, channel_type):
    if channel_type == "G4":
        residues_per_chain = 325
        offset = 49
    elif channel_type == "G2":
        residues_per_chain = 328
        offset = 54
    elif channel_type == "G12":
        residues_per_chain = 325
        offset = 53

    amino_acid_names = {152: "E", 184: "N", 141: "E", 173: "D", 148: "S", 137:"F"}

    if residue_id != "SF":
        residue_id = int(residue_id)
        chain_number = int(residue_id) // residues_per_chain
        if channel_type == "G2":
            chain_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
        elif channel_type == "G12":
            chain_dict = {0: "D", 1: "C", 2: "B", 3: "A"}
        pdb_number = residue_id - residues_per_chain * chain_number + offset
        if channel_type == "G12" and residue_id <= 325:
            pdb_number = residue_id + 42
        return f"{amino_acid_names.get(pdb_number, 'X')}{pdb_number}.{chain_dict[chain_number]}"
    else:
        return "SF"

def compute_residue_distances(u, sf_residues, hbc_residues, target_residues, channel_type="G4", output_dir="./residue_distances_output", residue_part="sidechain"):
    os.makedirs(output_dir, exist_ok=True)

    sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
    hbc_group = u.select_atoms("resid " + " ".join(map(str, hbc_residues)))

    if residue_part == "sidechain":
        residue_groups = {resid: u.select_atoms(f"resid {resid} and not name N CA C O HA H") for resid in target_residues}
    elif residue_part == "full":
        residue_groups = {resid: u.select_atoms(f"resid {resid}") for resid in target_residues}
    elif residue_part == "backbone":
        residue_groups = {resid: u.select_atoms(f"resid {resid} and name N CA C O HA H") for resid in target_residues}
    else:
        raise ValueError("Invalid residue_part. Choose from 'sidechain', 'full', or 'backbone'.")

    records = []

    tqmd_text = f"Calculating distances {residue_part} residue"
    for ts in tqdm(u.trajectory, desc=tqmd_text, unit="frame"):
        frame = ts.frame
        sf_com = sf_group.center_of_mass()
        hbc_com = hbc_group.center_of_mass()
        axis_vector = sf_com - hbc_com
        axis_vector /= np.linalg.norm(axis_vector)

        for resid in target_residues:
            group = residue_groups[resid]
            res_com = group.center_of_mass()
            z_offset_from_sf = sf_com[2] - res_com[2]  # Adjusted to match the original code logic


            com_distance = np.linalg.norm(res_com - sf_com)
            min_distance = np.min(np.linalg.norm(group.positions - sf_com, axis=1))

            v = res_com - hbc_com
            proj_length = np.dot(v, axis_vector)
            proj_point = hbc_com + proj_length * axis_vector
            radial_distance = np.linalg.norm(res_com - proj_point)

            records.append({
                "frame": frame,
                "resid": resid,
                "pdb_label": convert_to_pdb_numbering(resid, channel_type),
                "com_to_sf_com_distance": com_distance,
                "min_atom_to_sf_com_distance": min_distance,
                "radial_distance": radial_distance,
                "z_offset_from_sf": z_offset_from_sf
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "residue_distances_all_frames.csv"), index=False)

    os.makedirs(f"{output_dir}/csv", exist_ok=True)
    for resid in df["resid"].unique():
        df_res = df[df["resid"] == resid]
        df_res.to_csv(os.path.join(f"{output_dir}/csv", f"residue_{resid}_distances.csv"), index=False)

    return df

def generate_residue_distance_plots_with_ion_lines(csv_folder, ion_json_path, output_base="./plots", residue_part="sidechain"):
    os.makedirs(output_base, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    with open(ion_json_path, "r") as f:
        ion_events = json.load(f)
    start_frames = [e["start_frame"] for e in ion_events]
    all_exit_frames = [e["exit_frame"] for e in ion_events]
    exit_frames = [x for x in all_exit_frames if x != max(all_exit_frames)]

    for file in csv_files:
        filepath = os.path.join(csv_folder, file)
        df = pd.read_csv(filepath)

        if "pdb_label" not in df.columns:
            print(f"Skipping {file} (no pdb_label column).")
            continue

        residue_label = df["pdb_label"].iloc[0]
        suffix = f"_{residue_part}"
        residue_folder_name = residue_label.replace(".", "") + suffix
        plot_dir = os.path.join(output_base, f"plots/{residue_folder_name}")
        os.makedirs(plot_dir, exist_ok=True)

        def plot_with_lines(y, ylabel, title, filename, lines=None, color="blue", linecolor="black", linestyle="--", linelabel=None):
            plt.figure()
            plt.plot(df["frame"], df[y], color=color, label="Distance")
            if lines:
                for i, x in enumerate(lines):
                    if i == 0 and linelabel:
                        plt.axvline(x=x, linestyle=linestyle, color=linecolor, linewidth=1, label=linelabel)
                    else:
                        plt.axvline(x=x, linestyle=linestyle, color=linecolor, linewidth=1)
            plt.title(f"{residue_label} – {title}", fontsize=16)
            plt.xlabel("Frame", fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
                # Larger ticks
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if linelabel:
                plt.legend(loc="best", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()

        def plot_with_2lines(y, ylabel, title, filename,
                             lines1=None, label1=None, color1="green", style1="--",
                             lines2=None, label2=None, color2="red", style2="--",
                             color="blue"):
            plt.figure()
            plt.plot(df["frame"], df[y], color=color, label="Distance")
            if lines1:
                for i, x in enumerate(lines1):
                    if i == 0 and label1:
                        plt.axvline(x=x, linestyle=style1, color=color1, linewidth=1, label=label1)
                    else:
                        plt.axvline(x=x, linestyle=style1, color=color1, linewidth=1)
            if lines2:
                for i, x in enumerate(lines2):
                    if i == 0 and label2:
                        plt.axvline(x=x, linestyle=style2, color=color2, linewidth=1, label=label2)
                    else:
                        plt.axvline(x=x, linestyle=style2, color=color2, linewidth=1)
            plt.title(f"{residue_label} – {title}", fontsize=16)
            plt.xlabel("Frame" , fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
                # Larger ticks
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if label1 or label2:
                plt.legend(loc="best", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()

        # === BASIC PLOTS ===
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "1_com_to_sf_com_distance.png")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Closest Atom in Residue to SF COM", "2_min_atom_to_sf_com_distance.png")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance from Pore Axis", "3_radial_distance.png", color="red")

        plt.figure()
        plt.scatter(df["min_atom_to_sf_com_distance"], df["radial_distance"], color="purple", s=10)
        plt.title(f"{residue_label} – Radial vs. Closest Atom Distance", fontsize=16)
        plt.xlabel("Min Atom-to-SF COM Distance (Å)", fontsize=14)
        plt.ylabel("Radial Distance (Å)", fontsize=14)
            # Larger ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "4_min_vs_radial.png"))
        plt.close()

        plt.figure()
        plt.scatter(df["com_to_sf_com_distance"], df["radial_distance"], color="orange", s=10)
        plt.title(f"{residue_label} – Radial vs. COM Distance",fontsize=16)
        plt.xlabel("COM-to-SF COM Distance (Å)", fontsize=14)
        plt.ylabel("Radial Distance (Å)",fontsize=14)
            # Larger ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "5_com_vs_radial.png"))
        plt.close()

        # === LINE PLOTS ===
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "6_com_with_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Min Atom–SF Distance", "7_min_with_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance", "8_radial_with_start_lines.png", lines=start_frames, color="red", linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance", "8_radial.png", color="black")
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "9_com_with_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Min Atom–SF Distance", "10_min_with_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance", "11_radial_with_exit_lines.png", lines=exit_frames, color="red", linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "15_z_offset_basic.png", color="darkblue")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "16_z_offset_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion exits SF", color="darkblue")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "17_z_offset_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion exits GLU/ASN", color="darkblue")

        # === DOUBLE LINE PLOTS ===
        plot_with_2lines("com_to_sf_com_distance", "Distance (Å)", "COM to SF COM\n", "12_com_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="blue")
        plot_with_2lines("min_atom_to_sf_com_distance", "Distance (Å)", "Closest Atom to SF COM\n", "13_min_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="blue")
        plot_with_2lines("radial_distance", "Radial Distance (Å)", "Radial Distance from Pore Axis\n", "14_radial_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="black")
        plot_with_2lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM\n", "18_z_offset_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="darkblue")


def compute_residue_pair_distances(u, residue_list_1, residue_list_2, channel_type="G2", prefix="glu_asn", output_dir="./"):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    assert len(residue_list_1) == len(residue_list_2), "Both residue lists must have the same length"

    residue_pairs = list(zip(residue_list_1, residue_list_2))
    pair_labels = [f"{convert_to_pdb_numbering(r1,channel_type)}_{convert_to_pdb_numbering(r2,channel_type)}" for r1, r2 in residue_pairs]
    pair_data = {label: [] for label in pair_labels}

    for ts in tqdm(u.trajectory, desc="Computing residue pair distances", unit="frame"):
        for (r1, r2), label in zip(residue_pairs, pair_labels):
            sel1 = u.select_atoms(f"resid {r1} and not name N CA C O HA H")
            sel2 = u.select_atoms(f"resid {r2} and not name N CA C O HA H")
            if len(sel1) == 0 or len(sel2) == 0:
                continue
            com1 = sel1.center_of_mass()
            com2 = sel2.center_of_mass()
            dist = np.linalg.norm(com1 - com2)
            pair_data[label].append(dist)

    # Create a DataFrame
    frames = list(range(len(u.trajectory)))
    df = pd.DataFrame({"frame": frames})
    for label, distances in pair_data.items():
        df[label] = distances

    # Save CSV
    df.to_csv(os.path.join(output_dir, f"{prefix}_combined_pairwise_distances.csv"), index=False)

    # Plot all distances in one figure
    plt.figure(figsize=(10, 6))
    for label in pair_labels:
        if label in df.columns:
            plt.plot(df["frame"], df[label], label=f"{label}", linewidth=1.8)

    # Set a larger title based on prefix
    if prefix == "glu_asn":
        plt.title("Distance Between Residue GLU and ASN Sidechains", fontsize=16)
    elif prefix == "ser_glu":
        plt.title("Distance Between Residue GLU and SER Sidechains", fontsize=16)
    else:
        plt.title("Distance Between Residue Pairs (Sidechains)", fontsize=16)

    # Larger axis labels
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Distance (Å)", fontsize=14)

    # Larger ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Larger legend
    plt.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_all_pairs_distance_plot.png"))
    plt.close()

    # Smoothed plot version (rolling mean)
    window_size = 50  # Adjust this for more or less smoothing
    plt.figure(figsize=(10, 6))
    for label in pair_labels:
        if label in df.columns:
            smoothed = df[label].rolling(window=window_size, center=True).mean()
            plt.plot(df["frame"], smoothed, label=f"{label}", linewidth=2)

    if prefix == "glu_asn":
        plt.title("Smoothed Distance Between Residue GLU and ASN Sidechains", fontsize=16)
    elif prefix == "ser_glu":
        plt.title("Smoothed Distance Between Residue GLU and SER Sidechains", fontsize=16)
    else:
        plt.title("Smoothed Distance Between Residue Pairs (Sidechains)", fontsize=16)

    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Distance (Å)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_smoothed_pairs_distance_plot.png"))
    plt.close()

# def compute_chi1_angles(u, residue_ids, channel_type='G2', prefix="glu", output_dir="./chi1_output", window_size=50):

#     os.makedirs(output_dir, exist_ok=True)

#     # Complete chi1 atom mapping (N-CA-CB-X where X is the fourth atom)
#     chi1_atom_map = {
#         "GLU": "CG", "ASP": "CG", "ASN": "CG", "GLN": "CG",
#         "TYR": "CG", "PHE": "CG", "HIS": "CG", "TRP": "CG",
#         "CYS": "SG", "SER": "OG", "THR": "OG1", "MET": "SD",
#         "LYS": "CG", "ARG": "CG", "LEU": "CG", "ILE": "CG1",
#         "VAL": "CG1", "PRO": "CG"
#     }

#     all_raw = {}
#     all_smoothed = {}
#     max_frames = len(u.trajectory)

#     for resid in tqdm(residue_ids, desc=f"{prefix}: Computing χ1 angles"):
#         try:
#             res_atoms = u.select_atoms(f"resid {resid}")
#             if len(res_atoms) == 0:
#                 print(f"[Warning] Resid {resid} not found. Skipping.")
#                 continue
                
#             resname = res_atoms[0].resname.upper()
            
#             # Skip glycine (no chi1 angle)
#             if resname == "GLY":
#                 print(f"[Info] Resid {resid} is GLY (no chi1 angle). Skipping.")
#                 continue
            
#             # Get the fourth atom for chi1 angle
#             atom4 = chi1_atom_map.get(resname)
#             if atom4 is None:
#                 print(f"[Warning] Resid {resid} ({resname}) not in chi1_atom_map. Skipping.")
#                 continue

#             # Select atoms in the correct order for chi1: N-CA-CB-X
#             try:
#                 atom_N = u.select_atoms(f"resid {resid} and name N")
#                 atom_CA = u.select_atoms(f"resid {resid} and name CA")
#                 atom_CB = u.select_atoms(f"resid {resid} and name CB")
#                 atom_X = u.select_atoms(f"resid {resid} and name {atom4}")
                
#                 # Check if all atoms exist
#                 if len(atom_N) != 1 or len(atom_CA) != 1 or len(atom_CB) != 1 or len(atom_X) != 1:
#                     print(f"[Warning] Resid {resid} ({resname}) missing required atoms. Skipping.")
#                     continue
                
#                 # Create atomgroup in correct order for dihedral calculation
#                 atomgroup = atom_N + atom_CA + atom_CB + atom_X
                
#             except Exception as e:
#                 print(f"[Error] Resid {resid} atom selection failed: {e}")
#                 continue

#             # Calculate dihedral angle
#             dihedral = Dihedral([atomgroup])
#             dihedral.run()

#             # MDAnalysis Dihedral already returns angles in degrees
#             angles_deg = dihedral.results.angles[:, 0]

#             # Handle NaNs if any (e.g., interpolate)
#             if np.any(np.isnan(angles_deg)):
#                 mask = ~np.isnan(angles_deg)
#                 if np.sum(mask) > 0:  # Only interpolate if we have some valid values
#                     angles_deg[~mask] = np.interp(
#                         np.flatnonzero(~mask),
#                         np.flatnonzero(mask),
#                         angles_deg[mask]
#                     )
#                 else:
#                     print(f"[Warning] Resid {resid} has all NaN angles. Skipping.")
#                     continue

#             # Apply smoothing
#             if len(angles_deg) >= window_size:
#                 smooth_angles = np.convolve(angles_deg, np.ones(window_size) / window_size, mode='same')
#             else:
#                 smooth_angles = angles_deg.copy()
#                 print(f"[Info] Resid {resid}: trajectory shorter than window_size, no smoothing applied.")

#             label = f"{convert_to_pdb_numbering(resid, channel_type)}"
            
#             # Save individual residue data
#             np.savetxt(os.path.join(output_dir, f"{label}_chi1_angles.txt"), angles_deg)
#             all_raw[label] = angles_deg
#             all_smoothed[label] = smooth_angles

#         except Exception as e:
#             print(f"[Error] resid {resid}: {e}")
#             continue

#     if not all_raw:
#         print("[Warning] No valid chi1 angles calculated. Check your residue IDs and trajectory.")
#         return

#     # === Save combined CSV ===
#     df = pd.DataFrame({"frame": np.arange(max_frames)})
#     for label, angles in all_raw.items():
#         if len(angles) == max_frames:
#             df[label] = angles
#         else:
#             print(f"[Warning] {label} has {len(angles)} frames, expected {max_frames}")
    
#     df.to_csv(os.path.join(output_dir, f"{prefix}_chi1_all_residues_raw.csv"), index=False)

#     # === Combined Plot: Raw ===
#     plt.figure(figsize=(12, 8))
#     for label, data in all_raw.items():
#         plt.plot(range(len(data)), data, label=label, linewidth=1, alpha=0.8)
    
#     plt.title("χ1 Dihedral Angles (Raw)", fontsize=16)
#     plt.xlabel("Frame", fontsize=14)
#     plt.ylabel("Angle (degrees)", fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylim(-180, 180)
#     plt.grid(True, alpha=0.3)
    
#     # Handle legend for many residues
#     if len(all_raw) > 20:
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#     else:
#         plt.legend(fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{prefix}_chi1_all_residues_raw.png"), dpi=300, bbox_inches='tight')
#     plt.close()

#     # === Combined Plot: Smoothed ===
#     plt.figure(figsize=(12, 8))
#     for label, data in all_smoothed.items():
#         plt.plot(range(len(data)), data, label=label, linewidth=2, alpha=0.8)
    
#     plt.title("χ1 Dihedral Angles (Smoothed)", fontsize=16)
#     plt.xlabel("Frame", fontsize=14)
#     plt.ylabel("Angle (degrees)", fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.ylim(-180, 180)
#     plt.grid(True, alpha=0.3)
    
#     # Handle legend for many residues
#     if len(all_smoothed) > 20:
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#     else:
#         plt.legend(fontsize=12)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{prefix}_chi1_all_residues_smoothed.png"), dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"[Success] Calculated chi1 angles for {len(all_raw)} residues.")
#     print(f"[Success] Results saved to {output_dir}")
    
def compute_chi1_angles(u, residue_ids, channel_type='G2', prefix="glu", output_dir="./chi1_output", window_size=50):
    os.makedirs(output_dir, exist_ok=True)

    # Complete chi1 atom mapping (N-CA-CB-X where X is the fourth atom)
    chi1_atom_map = {
        "GLU": "CG", "ASP": "CG", "ASN": "CG", "GLN": "CG",
        "TYR": "CG", "PHE": "CG", "HIS": "CG", "TRP": "CG",
        "CYS": "SG", "SER": "OG", "THR": "OG1", "MET": "SD",
        "LYS": "CG", "ARG": "CG", "LEU": "CG", "ILE": "CG1",
        "VAL": "CG1", "PRO": "CG"
    }

    atomgroups = []
    labels = []

    for resid in tqdm(residue_ids, desc=f"{prefix}: Preparing AtomGroups"):
        try:
            res_atoms = u.select_atoms(f"resid {resid}")
            if len(res_atoms) == 0:
                continue

            resname = res_atoms[0].resname.upper()
            if resname == "GLY":
                continue

            atom4 = chi1_atom_map.get(resname)
            if atom4 is None:
                continue

            ag = u.select_atoms(f"resid {resid} and (name N or name CA or name CB or name {atom4})")
            if len(ag) == 4:
                atomgroups.append(ag)
                labels.append(f"{resname}_{convert_to_pdb_numbering(resid, channel_type)}")

        except Exception as e:
            print(f"[Error] resid {resid}: {e}")

    if not atomgroups:
        print("[Warning] No valid χ1 atomgroups found.")
        return

    # Compute all dihedrals in batch
    dih = Dihedral(atomgroups)
    dih.run()
    angles_matrix = dih.results.angles  # Already in degrees

    # Process output
    all_raw = {}
    all_smoothed = {}
    max_frames = len(u.trajectory)

    for i, label in enumerate(labels):
        angles_deg = angles_matrix[:, i]

        # Interpolate NaNs if needed
        if np.any(np.isnan(angles_deg)):
            mask = ~np.isnan(angles_deg)
            if np.any(mask):
                angles_deg[~mask] = np.interp(
                    np.flatnonzero(~mask),
                    np.flatnonzero(mask),
                    angles_deg[mask]
                )
            else:
                print(f"[Warning] {label} has only NaNs — skipping.")
                continue

        # Smoothing
        if len(angles_deg) >= window_size:
            smooth = np.convolve(angles_deg, np.ones(window_size) / window_size, mode='same')
        else:
            smooth = angles_deg.copy()

        np.savetxt(os.path.join(output_dir, f"{label}_chi1_angles.txt"), angles_deg)
        all_raw[label] = angles_deg
        all_smoothed[label] = smooth

    # === Save combined CSV ===
    df = pd.DataFrame({"frame": np.arange(max_frames)})
    for label, data in all_raw.items():
        df[label] = data
    df.to_csv(os.path.join(output_dir, f"{prefix}_chi1_all_residues_raw.csv"), index=False)

    # === Plot Raw ===
    plt.figure(figsize=(12, 8))
    for label, data in all_raw.items():
        plt.plot(df["frame"], data, label=label, linewidth=1, alpha=0.8)
    plt.title("χ1 Dihedral Angles (Raw)", fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Angle (degrees)", fontsize=14)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    if len(all_raw) > 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_chi1_all_residues_raw.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === Plot Smoothed ===
    plt.figure(figsize=(12, 8))
    for label, data in all_smoothed.items():
        plt.plot(df["frame"], data, label=label, linewidth=2, alpha=0.85)
    plt.title("χ1 Dihedral Angles (Smoothed)", fontsize=16)
    plt.xlabel("Frame", fontsize=14)
    plt.ylabel("Angle (degrees)", fontsize=14)
    plt.ylim(-180, 180)
    plt.grid(True, alpha=0.3)
    if len(all_smoothed) > 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    else:
        plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_chi1_all_residues_smoothed.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Success] Calculated χ1 angles for {len(all_raw)} residues.")
    print(f"[Success] Results saved in: {output_dir}")

# === MAIN EXECUTION ===
channel_type = "G12"
run_type = 2

# suffix = "_sidechain" if sidechain_only else "_full"
data_path = "/home/data/Konstantina/ion-permeation-analyzer-results/version1"

if channel_type == "G2":
    # topology_path = "/home/data/Konstantina/Rep0/com_4fs.prmtop"
    # trajectory_path = "/home/data/Konstantina/Rep0/protein.nc"
    # output_dir = f"./G2_geometry/"
    # ion_json_path = f"{data_path}/results_G2_5000_frames/ch2.json"

    topology_path = "/home/yongcheng/Konstantina/G2_4KFM_RUN2/com_4fs.prmtop"
    trajectory_path = "/home/yongcheng/Konstantina/G2_4KFM_RUN2/protein.nc"
    output_dir = f"./G2_CHL_geometry/"
    ion_json_path = f"{data_path}/results_G2_CHL_frames/ch2.json"

    glu_residues = [98, 426, 754, 1082]
    asn_residues = [130, 458, 786, 1114]
    sf_residues = [100, 428, 756, 1084]
    hbc_residues = [138, 466, 794, 1122]
    ser_residues = [94, 422, 750, 1078]

elif channel_type == "G12":
    topology_path = f"/home/data/Konstantina/GIRK12_WT/RUN{run_type}/com_4fs.prmtop"
    trajectory_path = f"/home/data/Konstantina/GIRK12_WT/RUN{run_type}/protein.nc"
    output_dir = f"./G12_RUN{run_type}_geometry/"
    if run_type == 2:
        ion_json_path = f"{data_path}/results_G12_duplicates/ch2.json"
    elif run_type == 1:
        ion_json_path = f"{data_path}/results_G12_RUN1/ch2.json"
    glu_residues = [99, 424, 749, 1074]
    asn_residues = [131, 456, 781, 1106]
    sf_residues = [101, 426, 751, 1076]
    hbc_residues = [139, 464, 789, 1114]
    ser_residues = [95, 420, 745, 1070]

target_residues = glu_residues + asn_residues

u = mda.Universe(topology_path, trajectory_path)

if not os.path.exists(f"{output_dir}/plots"):
    os.makedirs(output_dir)
    for part in ["sidechain", "full", "backbone"]:
        df = compute_residue_distances(u, sf_residues, hbc_residues, target_residues, channel_type=channel_type, output_dir=output_dir, residue_part=part)
        csv_folder = os.path.join(output_dir, "csv")
        generate_residue_distance_plots_with_ion_lines(csv_folder, ion_json_path, output_base=output_dir, residue_part=part)

if not os.path.exists(f"{output_dir}/glu_asn"):
    compute_residue_pair_distances(u, glu_residues, asn_residues, channel_type, prefix="glu_asn", output_dir=f"{output_dir}/glu_asn")
if not os.path.exists(f"{output_dir}/ser_glu"):
    compute_residue_pair_distances(u, glu_residues, ser_residues, channel_type, prefix="ser_glu", output_dir=f"{output_dir}/ser_glu")

# if not os.path.exists(f"{output_dir}/glu_chi1"):
compute_chi1_angles(u, glu_residues, channel_type, prefix="glu", output_dir=f"{output_dir}/glu_chi1", window_size=200)
# if not os.path.exists(f"{output_dir}/asn_chi1"):
compute_chi1_angles(u, asn_residues, channel_type, prefix="asn", output_dir=f"{output_dir}/asn_chi1", window_size=200)


print("All calculations and plots completed successfully.")