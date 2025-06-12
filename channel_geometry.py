import MDAnalysis as mda
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

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

    amino_acid_names = {152: "E", 184: "N", 141: "E", 173: "D"}

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
            plt.title(f"{residue_label} – {title}")
            plt.xlabel("Frame")
            plt.ylabel(ylabel)
            if linelabel:
                plt.legend(loc="best")
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
            plt.title(f"{residue_label} – {title}")
            plt.xlabel("Frame")
            plt.ylabel(ylabel)
            if label1 or label2:
                plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, filename))
            plt.close()

        # === BASIC PLOTS ===
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "1_com_to_sf_com_distance.png")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Closest Atom in Residue to SF COM", "2_min_atom_to_sf_com_distance.png")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance from Pore Axis", "3_radial_distance.png", color="red")

        plt.figure()
        plt.scatter(df["min_atom_to_sf_com_distance"], df["radial_distance"], color="purple", s=10)
        plt.title(f"{residue_label} – Radial vs. Closest Atom Distance")
        plt.xlabel("Min Atom-to-SF COM Distance (Å)")
        plt.ylabel("Radial Distance (Å)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "4_min_vs_radial.png"))
        plt.close()

        plt.figure()
        plt.scatter(df["com_to_sf_com_distance"], df["radial_distance"], color="orange", s=10)
        plt.title(f"{residue_label} – Radial vs. COM Distance")
        plt.xlabel("COM-to-SF COM Distance (Å)")
        plt.ylabel("Radial Distance (Å)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "5_com_vs_radial.png"))
        plt.close()

        # === LINE PLOTS ===
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "6_com_with_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Min Atom–SF Distance", "7_min_with_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance", "8_radial_with_start_lines.png", lines=start_frames, color="red", linecolor="green", linelabel="Ion leaves SF")
        plot_with_lines("com_to_sf_com_distance", "Distance (Å)", "Distance Between Residue COM and SF COM", "9_com_with_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("min_atom_to_sf_com_distance", "Distance (Å)", "Min Atom–SF Distance", "10_min_with_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("radial_distance", "Radial Distance (Å)", "Radial Distance", "11_radial_with_exit_lines.png", lines=exit_frames, color="red", linecolor="black", linelabel="Ion leaves GLU/ASN")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "15_z_offset_basic.png", color="darkblue")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "16_z_offset_start_lines.png", lines=start_frames, linecolor="green", linelabel="Ion exits SF", color="darkblue")
        plot_with_lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM", "17_z_offset_exit_lines.png", lines=exit_frames, linecolor="black", linelabel="Ion exits GLU/ASN", color="darkblue")

        # === DOUBLE LINE PLOTS ===
        plot_with_2lines("com_to_sf_com_distance", "Distance (Å)", "COM to SF COM\n(Entry & Exit Events)", "12_com_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="blue")
        plot_with_2lines("min_atom_to_sf_com_distance", "Distance (Å)", "Closest Atom to SF COM\n(Entry & Exit Events)", "13_min_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="blue")
        plot_with_2lines("radial_distance", "Radial Distance (Å)", "Radial Distance from Pore Axis\n(Entry & Exit Events)", "14_radial_with_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="black")
        plot_with_2lines("z_offset_from_sf", "Z Offset (Å)", "Z Difference from SF COM\n(Entry & Exit Events)", "18_z_offset_start_and_exit_lines.png", lines1=start_frames, label1="Ion exits SF", lines2=exit_frames, label2="Ion exits GLU/ASN", color="darkblue")
        
# === MAIN EXECUTION ===
channel_type = "G12"
run_type = 2

# suffix = "_sidechain" if sidechain_only else "_full"
data_path = "/home/data/Konstantina/ion-permeation-analyzer-results"

if channel_type == "G2":
    topology_path = "/home/data/Konstantina/Rep0/com_4fs.prmtop"
    trajectory_path = "/home/data/Konstantina/Rep0/protein.nc"
    output_dir = f"./G2_geometry/"
    ion_json_path = f"{data_path}/results_G2_5000_frames/ch2.json"
    glu_residues = [98, 426, 754, 1082]
    asn_residues = [130, 458, 786, 1114]
    sf_residues = [100, 428, 756, 1084]
    hbc_residues = [138, 466, 794, 1122]

elif channel_type == "G12":
    topology_path = f"../GIRK12_WT/RUN{run_type}/com_4fs.prmtop"
    trajectory_path = f"../GIRK12_WT/RUN{run_type}/protein.nc"
    output_dir = f"./G12_RUN{run_type}_geometry/"
    if run_type == 2:
        ion_json_path = f"{data_path}/results_G12_duplicates/ch2.json"
    elif run_type == 1:
        ion_json_path = f"{data_path}/results_G12_RUN1/ch2.json"
    glu_residues = [99, 424, 749, 1074]
    asn_residues = [131, 456, 781, 1106]
    sf_residues = [101, 426, 751, 1076]
    hbc_residues = [139, 464, 789, 1114]

target_residues = glu_residues + asn_residues

u = mda.Universe(topology_path, trajectory_path)

for part in ["sidechain", "full", "backbone"]:
    df = compute_residue_distances(u, sf_residues, hbc_residues, target_residues, channel_type=channel_type, output_dir=output_dir, residue_part=part)
    csv_folder = os.path.join(output_dir, "csv")
    generate_residue_distance_plots_with_ion_lines(csv_folder, ion_json_path, output_base=output_dir, residue_part=part)
print("All calculations and plots completed successfully.")