import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import json
from analysis.converter import convert_to_pdb_numbering

def compute_electric_field_at_point(point, atom_positions, atom_charges, k=332):
    point = np.asarray(point, dtype=float)
    total_field = np.zeros(3)
    for pos, q in zip(atom_positions, atom_charges):
        r_vec = point - pos
        r = np.linalg.norm(r_vec)
        if r < 1e-6:
            continue
        total_field += k * q * r_vec / (r ** 3)
    return total_field

def add_totals_to_result(result):
    def sum_fields(field_dict):
        return np.sum([np.array(entry["field"]) for entry in field_dict.values()], axis=0)

    glu_total = sum_fields(result["glu_residues"])
    asn_total = sum_fields(result["asn_residues"])
    pip2_total = sum_fields(result["pip2_residues"])
    ions_up = np.array(result["ionic_up_total_field"]["field"])
    ions_down = np.array(result["ionic_down_total_field"]["field"])

    result["total_glu"] = {
        "field": glu_total.tolist(),
        "magnitude": float(np.linalg.norm(glu_total))
    }
    result["total_asn"] = {
        "field": asn_total.tolist(),
        "magnitude": float(np.linalg.norm(asn_total))
    }

    result["total_no_ions"] = {
        "field": (glu_total + asn_total + pip2_total).tolist(),
        "magnitude": float(np.linalg.norm(glu_total + asn_total + pip2_total))
    }

    result["total"] = {
        "field": (glu_total + asn_total + pip2_total + ions_up + ions_down).tolist(),
        "magnitude": float(np.linalg.norm(glu_total + asn_total + pip2_total + ions_up + ions_down))
    }

    return result


def compute_frame_field(u, frame, point, glu_residues, asn_residues,
                        pip2_resids, exclude_backbone, headgroup_atoms,
                        ion_selection="resname K+", k=332):
    u.trajectory[frame]

    def collect_pa_ch(atom_group, exclude_backbone=False, allowed_names=None):
        pos, q = [], []
        for atom in atom_group:
            if exclude_backbone and atom.name in {"N", "CA", "C", "O", "HA", "H"}:
                continue
            if allowed_names and atom.name not in allowed_names:
                continue
            pos.append(atom.position)
            q.append(atom.charge)
        return pos, q

    result = {
        "glu_residues": {},
        "asn_residues": {},
        "pip2_residues": {},
        "ionic_up_fields": [],
        "ionic_down_fields": [],
        "ionic_up_total_field": None,
        "ionic_down_total_field": None
    }

    for resid_list, label in [(glu_residues, "glu_residues"), (asn_residues, "asn_residues")]:
        for resid in resid_list:
            atoms = u.select_atoms(f"resid {resid}")
            pos, q = collect_pa_ch(atoms, exclude_backbone)
            field = compute_electric_field_at_point(point, pos, q, k)
            result[label][int(resid)] = {
                "field": field.tolist(),
                "magnitude": float(np.linalg.norm(field))
            }

    pip2_fields = {}
    for resid in pip2_resids:
        atoms = u.select_atoms(f"resid {resid}")
        pos, q = collect_pa_ch(atoms, exclude_backbone, allowed_names=headgroup_atoms)
        field = compute_electric_field_at_point(point, pos, q, k)
        pip2_fields[int(resid)] = {
            "field": field.tolist(),
            "magnitude": float(np.linalg.norm(field))
        }
    result["pip2_residues"] = pip2_fields

    up_list, down_list = [], []
    up_total, down_total = np.zeros(3), np.zeros(3)

    for ion in u.select_atoms(ion_selection):
        field = compute_electric_field_at_point(point, [ion.position], [ion.charge], k)
        entry = {
            "ion_id": int(ion.index),
            "field": field.tolist(),
            "magnitude": float(np.linalg.norm(field)),
            "distance": float(np.linalg.norm(point - ion.position))
        }
        if abs(ion.position[2] - point[2]) <1:
            continue 

        if ion.position[2] > point[2]:
            up_list.append(entry)
            up_total += field
        else:
            down_list.append(entry)
            down_total += field

    result["ionic_up_fields"] = up_list
    result["ionic_down_fields"] = down_list
    result["ionic_up_total_field"] = {
        "field": up_total.tolist(), "magnitude": float(np.linalg.norm(up_total))
    }
    result["ionic_down_total_field"] = {
        "field": down_total.tolist(), "magnitude": float(np.linalg.norm(down_total))
    }

    return result

def detect_pip2_resids(u, pip2_resname):
    pip2_atoms = u.select_atoms(f"resname {pip2_resname}")
    pip2_resids = sorted(list(set(pip2_atoms.resids)))
    if len(pip2_resids) != 4:
        raise ValueError(f"Expected 4 PIP2 residues, but found {len(pip2_resids)}: {pip2_resids}")
    return pip2_resids

def run_field_analysis(u, sf_residues, glu_residues, asn_residues,
                       pip2_resname, headgroup_atoms, exclude_backbone,
                       output_path="sf_field_results.json",
                       point_strategy="sf_com", fixed_point=None,
                       ion_selection="resname K+ K"):
    
    pip2_resids = detect_pip2_resids(u, pip2_resname)
    field_by_frame = {}

    for ts in tqdm(u.trajectory, desc="Frames"):
        if point_strategy == "sf_com":
            sf_group = u.select_atoms("resid " + " ".join(map(str, sf_residues)))
            point = sf_group.center_of_mass()

        elif point_strategy == "sf_min_atoms":
            atom_indices = []
            for resid in sf_residues:
                residue_atoms = u.select_atoms(f"resid {resid}")
                if len(residue_atoms) == 0:
                    continue
                coords = residue_atoms.positions
                min_index = coords[:, 2].argmin()
                atom_indices.append(residue_atoms[min_index].index)
            sf_min_atoms = u.atoms[atom_indices]
            point = sf_min_atoms.center_of_mass()

        elif point_strategy == "fixed":
            if fixed_point is None:
                raise ValueError("fixed_point must be provided")
            point = np.asarray(fixed_point)

        else:
            raise ValueError("point_strategy must be one of 'sf_com', 'sf_min_atoms', or 'fixed'")

        frame_result = compute_frame_field(
            u, ts.frame, point, glu_residues, asn_residues,
            pip2_resids, exclude_backbone, headgroup_atoms, ion_selection, k=332
        )
        field_by_frame[int(ts.frame)] = add_totals_to_result(frame_result)

    with open(output_path, "w") as f:
        json.dump(field_by_frame, f, indent=2)

    print(f"✅ Saved electrostatic field analysis to {output_path}")

# Example usage:
# if __name__ == "__main__":
#     run_field_analysis(
#         topology="com_4fs.prmtop",
#         trajectory="protein.nc",
#         sf_residues=[100, 428, 756, 1084],
#         glu_residues=[98, 426, 754, 1082],
#         asn_residues=[130, 458, 786, 1114],
#         pip2_resname="PIP",
#         headgroup_atoms={"P4", "P5", "O4P", "O5P", "O41", "O42", "O43", "O51", "O52", "O53"},
#         output_path="sf_field_results.json",
#         point_strategy="sf_com"
#     )

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def smooth(values, window_size=20):
    values = np.array(values)
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='same')

def plot_field_magnitudes_from_json(field_json_path, ion_events, output_dir="field_magnitude_plots", channel_type="G2"):
    with open(field_json_path) as f:
        data = json.load(f)

    # with open(startframes_json_path) as f:
    #     ion_events = json.load(f)
    start_frames = [event["start_frame"] for event in ion_events]

    os.makedirs(output_dir, exist_ok=True)
    frames = sorted(data.keys(), key=lambda x: int(x))

    glu_data, asn_data, pip_data = {}, {}, {}
    ion_up_data, ion_down_data = {}, {}
    ion_up_total, ion_down_total = [], []
    total_glu_list, total_asn_list, total_all_list, total_no_ions_list = [], [], [], []

    for frame in frames:
        frame_data = data[frame]

        for resid, entry in frame_data.get("glu_residues", {}).items():
            glu_data.setdefault(resid, []).append(entry["magnitude"])
        for resid, entry in frame_data.get("asn_residues", {}).items():
            asn_data.setdefault(resid, []).append(entry["magnitude"])
        for resid, entry in frame_data.get("pip2_residues", {}).items():
            pip_data.setdefault(resid, []).append(entry["magnitude"])

        for ion in frame_data.get("ionic_up_fields", []):
            ion_id = ion["ion_id"]
            ion_up_data.setdefault(ion_id, []).append(ion["magnitude"])
        for ion in frame_data.get("ionic_down_fields", []):
            ion_id = ion["ion_id"]
            ion_down_data.setdefault(ion_id, []).append(ion["magnitude"])

        ion_up_total.append(frame_data["ionic_up_total_field"]["magnitude"])
        ion_down_total.append(frame_data["ionic_down_total_field"]["magnitude"])
        total_glu_list.append(frame_data["total_glu"]["magnitude"])
        total_asn_list.append(frame_data["total_asn"]["magnitude"])
        total_all_list.append(frame_data["total"]["magnitude"])
        total_no_ions_list.append(frame_data.get("total_no_ions", {}).get("magnitude", None))

    def plot_dict(data_dict, title, filename_base, channel_type):
        for with_lines in [False, True]:
            for smooth_flag in [False, True]:
                plt.figure(figsize=(12, 6))
                for key, values in data_dict.items():
                    y = smooth(values) if smooth_flag else values
                    if channel_type:
                        plt.plot(y, label=convert_to_pdb_numbering(key, channel_type), alpha=0.8, linewidth=1.2)
                    else:
                        plt.plot(y, label=str(key), alpha=0.8, linewidth=1.2)
                if with_lines:
                    for i, x in enumerate(start_frames):
                        if i == 0:
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8, label="Ion leaves SF")
                        else:
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8)
                smooth_tag = "_smoothed" if smooth_flag else ""
                suffix = f"{smooth_tag}_start_lines" if with_lines else f"{smooth_tag}"
                plt.title(f"{title}{' (Smoothed)' if smooth_flag else ''}{' + Start Lines' if with_lines else ''}")
                plt.xlabel("Frame")
                plt.ylabel("Field Magnitude")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{filename_base}{suffix}.png"))
                plt.close()

    def plot_single(data_list, labels, title, filename_base):
        for with_lines in [False, True]:
            for smooth_flag in [False, True]:
                plt.figure(figsize=(12, 6))
                for i, values in enumerate(data_list):
                    y = smooth(values) if smooth_flag else values
                    plt.plot(y, label=labels[i], alpha=0.8, linewidth=1.2)
                if with_lines:
                    for i, x in enumerate(start_frames):
                        if i == 0:
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8, label="Ion leaves SF")
                        else:
                            plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8)
                smooth_tag = "_smoothed" if smooth_flag else ""
                suffix = f"{smooth_tag}_start_lines" if with_lines else f"{smooth_tag}"
                plt.title(f"{title}{' (Smoothed)' if smooth_flag else ''}{' + Start Lines' if with_lines else ''}")
                plt.xlabel("Frame")
                plt.ylabel("Field Magnitude")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{filename_base}{suffix}.png"))
                plt.close()

    # === Plot All Variants ===
    plot_dict(glu_data, "GLU Residue Field Magnitudes", "glu_field_magnitudes", channel_type)
    plot_dict(asn_data, "ASN Residue Field Magnitudes", "asn_field_magnitudes", channel_type)
    plot_dict(pip_data, "PIP2 Residue Field Magnitudes", "pip2_field_magnitudes", None)
    plot_dict(ion_up_data, "Ionic Up Field Magnitudes (Individual Ions)", "ion_up_fields", None)
    plot_dict(ion_down_data, "Ionic Down Field Magnitudes (Individual Ions)", "ion_down_fields", None)

    plot_single([ion_up_total, ion_down_total], ["Up Total", "Down Total"], "Total Field from Ions (Up vs Down)", "ion_total_fields")
    plot_single([total_glu_list], ["Total GLU Field"], "Total Electric Field from All GLU Residues", "total_glu")
    plot_single([total_asn_list], ["Total ASN Field"], "Total Electric Field from All ASN Residues", "total_asn")
    plot_single([total_all_list], ["Total Field (All Sources)"], "Total Electric Field from All Sources", "total_all_sources")
    plot_single([total_no_ions_list], ["Total Field (No Ions)"], "Total Electric Field from GLU + ASN + PIP2 (No Ions)", "total_no_ions")

    print(f"✅ All plots with and without smoothing saved in: {output_dir}")
