import numpy as np
from tqdm import tqdm
from analysis.calculate_openmm_forces import calculate_ionic_forces_all_frames

# =========================
# Utility Functions
# =========================

def compute_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)

def compute_force(q1, q2, pos1, pos2, k=332):
    """
    Compute Coulomb force vector from ion2 to ion1.
    q1, q2: Charges
    pos1, pos2: Coordinates of ions
    k: Coulomb constant (kcal·Å/(mol·e²))
    https://simtk.org/api_docs/simbody/api_docs33/Simbody/html/group__PhysConstants.html?utm_source=chatgpt.com
    It is the Coulomb constant 1/4πε0 is expressed in MD-compatible units, i.e.:
    - Distance in Ångströms
    - Energy in kcal/mol
    - Charge in units of elementary charge (e)
    """
    r_vec = pos1 - pos2
    r = np.linalg.norm(r_vec)
    if r == 0:
        return np.zeros(3)
    return k * (q1 * q2) / (r ** 2) * (r_vec / r)

def get_motion_vector(ion_positions, frame):
    """Return movement vector from frame to frame+1."""
    if frame + 1 not in ion_positions or frame not in ion_positions:
        return None
    return ion_positions[frame + 1] - ion_positions[frame]

def unit_vector(v):
    """Return unit vector in direction of v."""
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else np.zeros_like(v)

# =========================
# Data Preparation
# =========================

def build_all_positions(universe, start_frame=0, stop_frame=None, ion_selection='resname K+'):
    """
    Extract ion positions from trajectory within a frame range.

    Parameters:
        universe (MDAnalysis.Universe): MDAnalysis trajectory object
        ion_selection (str): Atom selection string (default: 'resname K+')
        start_frame (int): Starting frame index
        stop_frame (int or None): Last frame index (exclusive). If None, reads to the end.

    Returns:
        dict: {frame: {ion_id: np.array([x, y, z])}}
    """
    all_positions = {}
    ions = universe.select_atoms(ion_selection)
    # Limit frames with slicing
    trajectory_slice = universe.trajectory[start_frame:stop_frame]

    for ts in tqdm(trajectory_slice, desc=f"Extracting positions ({start_frame}:{stop_frame})"):
        # print("Frame", ts.frame, type(ions[-1].id))
        frame_dict = {ion.resid: ion.position.copy() for ion in ions}
        all_positions[ts.frame] = frame_dict
    return all_positions


def build_charge_map(universe, ion_selection='resname K+'):
    """Return {ion_id: +1.0} for all selected ions."""
    ions = universe.select_atoms(ion_selection)
    return {ion.resid: 1.0 for ion in ions}

# =========================
# Analysis Functions
# =========================

def analyze_frame(positions, permeating_ion_id, frame, other_ions, charge_map, cutoff=10.0,
                  calculate_total_force=False, total_force_data=None):
    """
    Analyze one frame: compute ionic forces, motion, and optionally total force.
    Also calculates cosine similarities between different vectors and force decomposition.
    """

    result = {
        "frame": frame,
        "ionic_force": [0.0, 0.0, 0.0],
        "ionic_force_magnitude": None,
        "motion_vector": None,
        "cosine_ionic_motion": None,
        "ionic_motion_component": None,
        "ionic_force_x": None,
        "ionic_force_y": None,
        "ionic_force_z": None,
        "radial_force": None,
        "axial_force": None,
        "contributions": []
    }

    permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
    if permeating_pos is None:
        return result

    ionic_force = np.zeros(3)
    contributions = []

    for ion_id, pos in positions.get(frame, {}).items():
        if ion_id == permeating_ion_id or ion_id not in other_ions:
            continue
        distance = compute_distance(permeating_pos, pos)
        if distance <= cutoff:
            force = compute_force(charge_map[permeating_ion_id], charge_map[ion_id], permeating_pos, pos)
            ionic_force += force
            magnitude = np.linalg.norm(force)
            contributions.append({
                "ion": int(ion_id),
                "force": [float(f) for f in force.tolist()],
                "magnitude": float(magnitude),
                "distance": float(distance)
            })

    result["ionic_force"] = ionic_force.tolist()
    result["ionic_force_magnitude"] = float(np.linalg.norm(ionic_force))

    # ======== Decompose ionic force into x, y, z components ========
    Fx, Fy, Fz = ionic_force
    result.update({
        "ionic_force_x": float(Fx),
        "ionic_force_y": float(Fy),
        "ionic_force_z": float(Fz),
        "axial_force": float(Fz),  # assuming Z is the pore axis
        "radial_force": float(np.sqrt(Fx**2 + Fy**2))
    })
    # ===============================================================

    ion_positions_over_time = {
        f: positions.get(f, {}).get(permeating_ion_id) for f in range(frame, frame + 2)
    }

    motion_vec = get_motion_vector(ion_positions_over_time, frame)
    if motion_vec is not None:
        unit_motion = unit_vector(motion_vec)

        if np.linalg.norm(ionic_force) != 0 and np.linalg.norm(motion_vec) != 0:
            cosine_ionic_motion = float(np.dot(ionic_force, motion_vec) / (np.linalg.norm(ionic_force) * np.linalg.norm(motion_vec)))
            ionic_motion_component = cosine_ionic_motion * result["ionic_force_magnitude"]

            result.update({
                "motion_vector": motion_vec.tolist(),
                "cosine_ionic_motion": cosine_ionic_motion,
                "ionic_motion_component": ionic_motion_component
            })

        for c in contributions:
            force_vec = np.array(c["force"])
            force_mag = np.linalg.norm(force_vec)

            if force_mag != 0:
                cosine = float(np.dot(force_vec, unit_motion) / force_mag)
                projection = cosine * force_mag
            else:
                cosine = 0.0
                projection = 0.0

            c["cosine_with_motion"] = cosine
            c["motion_component"] = projection


    result["contributions"] = contributions

    # Add total force if requested
    if calculate_total_force and total_force_data is not None:
        tf = total_force_data[frame].get(permeating_ion_id)
        if tf is not None:
            total_force = np.array(tf)
            total_mag = float(np.linalg.norm(total_force))
            ionic_mag = result["ionic_force_magnitude"]
            fraction = ionic_mag / total_mag if total_mag != 0 else 0.0

            result.update({
                "total_force": total_force.tolist(),
                "total_force_magnitude": total_mag,
                "ionic_fraction_of_total": fraction
            })

            # Cosine Similarities with total force
            if np.linalg.norm(ionic_force) != 0 and np.linalg.norm(total_force) != 0:
                result["cosine_ionic_total"] = float(np.dot(ionic_force, total_force) / (np.linalg.norm(ionic_force) * np.linalg.norm(total_force)))

            if motion_vec is not None and np.linalg.norm(total_force) != 0 and np.linalg.norm(motion_vec) != 0:
                result["cosine_total_motion"] = float(np.dot(total_force, motion_vec) / (np.linalg.norm(total_force) * np.linalg.norm(motion_vec)))

    return result



def analyze_permeation_events(ch2_permeation_events, universe, start_frame, end_frame, cutoff=10.0,
                              calculate_total_force=False, prmtop_file=None, nc_file=None):
    """
    Analyze all permeation events from start_frame to permeation frame.
    If `calculate_total_force=True`, loads forces via OpenMM.
    """
    positions = build_all_positions(universe, start_frame, end_frame)
    charge_map = build_charge_map(universe)
    results = []

    total_force_data = None
    if calculate_total_force and prmtop_file and nc_file:
        print("Calculating total forces with OpenMM...")
        total_force_data, atom_index_map = calculate_ionic_forces_all_frames(prmtop_file, nc_file)

    for event in ch2_permeation_events:
        # Check if event frame is within analysis window
        if not (start_frame <= event["frame"] < end_frame):
            continue

        event_result = {
            "start_frame": event["start_frame"],
            "frame": event["frame"],
            "permeated_ion": event["permeated"],
            "analysis": {}
        }

        # Build list of frames to analyze: from start_frame to frame (inclusive)
        frames_to_check = list(range(event["start_frame"], event["frame"] + 1))

        for frame in frames_to_check:
            frame_result = analyze_frame(
                positions=positions,
                permeating_ion_id=event["permeated"],
                frame=frame,
                other_ions=positions.get(frame, {}).keys(),
                charge_map=charge_map,
                cutoff=cutoff,
                calculate_total_force=calculate_total_force,
                total_force_data=total_force_data
            )
            event_result["analysis"][frame] = frame_result

        results.append(event_result)

    return results


def find_top_cosine_frames(event_data, top_n=5):
    """
    Find the top N frames with the highest cosine_ionic_motion for each ion.
    Also report if the frame is the permeation frame.

    Args:
        event_data: list of event dictionaries (each has 'permeated_ion', 'frame', 'analysis')
        top_n: number of top frames to return per ion

    Returns:
        results: dict {ion_id: list of dicts with frame, cosine_ionic_motion, is_permeation_frame}
    """
    ion_results = {}

    for event in event_data:
        permeated_ion = event["permeated_ion"]
        permeation_frame = event["frame"]
        analysis = event["analysis"]

        # Collect all frames and their cosine_ionic_motion
        frame_cosine_list = []
        for frame, frame_data in analysis.items():
            cosine = frame_data.get("cosine_ionic_motion")
            if cosine is not None:
                frame_cosine_list.append((frame, cosine))

        # Sort by descending cosine value
        sorted_frames = sorted(frame_cosine_list, key=lambda x: -x[1])

        # Pick top N
        top_frames = sorted_frames[:top_n]

        # Build the result for this ion
        top_info = []
        for frame, cosine in top_frames:
            top_info.append({
                "frame": frame,
                "cosine_ionic_motion": cosine,
                "is_permeation_frame": (frame == permeation_frame)
            })

        ion_results[permeated_ion] = top_info

    return ion_results


def collect_sorted_cosines_until_permeation(event_data, closest_residues_by_ion):
    ion_results = {}

    for event in event_data:
        ion_id = str(event["permeated_ion"])  # ensure match with input keys
        permeation_frame = event["frame"]
        analysis = event["analysis"]

        frame_cosine_list = []
        for frame, frame_data in analysis.items():
            cosine = frame_data.get("cosine_ionic_motion")
            if cosine is not None:
                frame_cosine_list.append((frame, cosine))

        sorted_frames = sorted(frame_cosine_list, key=lambda x: -x[1])
        residue_track = closest_residues_by_ion.get(int(ion_id),[])
        collected_frames = []
        for frame, cosine in sorted_frames:
            # Initialize as None in case not found
            closest_now = None
            closest_next = None
            closest_before = None

            # Search for closest residue in current frame
            for r in residue_track:
                if r["frame"] == frame:
                    closest_now = r["residue"]
                    break

            # Search for closest residue in next frame
            for r in residue_track:
                if r["frame"] == frame + 1:
                    closest_next = r["residue"]
                    break

            # Search for closest residue in next frame
            for r in residue_track:
                if r["frame"] == frame - 1:
                    closest_before = r["residue"]
                    break


            collected_frames.append({
                "frame": frame,
                "ionic_force": analysis[frame].get("ionic_force"),
                "ionic_force_magnitude": analysis[frame].get("ionic_force_magnitude"),
                "ionic_motion_component": analysis[frame].get("ionic_motion_component"),
                "cosine_ionic_motion": cosine,
                "is_permeation_frame": (frame == permeation_frame),
                "radial_force": analysis[frame].get("radial_force"),
                "axial_force": analysis[frame].get("axial_force"),
                "before_closest_residue": closest_before,
                "closest_residue": closest_now,
                "next_closest_residue": closest_next,
                "contributions": analysis[frame].get("contributions", [])
            })
            if frame == permeation_frame:
                break

        ion_results[ion_id] = collected_frames

    return ion_results


import pandas as pd

def extract_permeation_frames(data):
    """
    Extracts permeation frame information for each ion from the JSON dict.
    Returns both a detailed contributor-expanded DataFrame and a summary DataFrame.

    Returns:
        tuple of pd.DataFrame: (df_expanded, df_summary)
    """
    expanded_rows = []
    summary_rows = []

    for ion_id, entries in data.items():
        for entry in entries:
            if entry.get("is_permeation_frame", False):
                # --- Summary row (no contributions) ---
                summary_rows.append({
                    "ion_id": ion_id,
                    "frame": entry["frame"],
                    "ionic_force": entry["ionic_force"],
                    "ionic_force_magnitude": entry["ionic_force_magnitude"],
                    "ionic_motion_component": entry["ionic_motion_component"],
                    "cosine_ionic_motion": entry["cosine_ionic_motion"],
                    "radial_force": entry["radial_force"],
                    "axial_force": entry["axial_force"],
                    "before_closest_residue": entry["before_closest_residue"],
                    "closest_residue": entry["closest_residue"],
                    "next_closest_residue": entry["next_closest_residue"]
                })

                # --- Expanded rows (one per contributor) ---
                contributions = entry.get("contributions", [])
                contributions.sort(key=lambda c: c.get("cosine_with_motion", -1), reverse=True)

                for c in contributions:
                    expanded_rows.append({
                        "ion_id": ion_id,
                        "frame": entry["frame"],
                        "ionic_force": entry["ionic_force"],
                        "ionic_force_magnitude": entry["ionic_force_magnitude"],
                        "ionic_motion_component": entry["ionic_motion_component"],
                        "cosine_ionic_motion": entry["cosine_ionic_motion"],
                        "radial_force": entry["radial_force"],
                        "axial_force": entry["axial_force"],
                        "before_closest_residue": entry["before_closest_residue"],
                        "closest_residue": entry["closest_residue"],
                        "next_closest_residue": entry["next_closest_residue"],
                        # Contributor details
                        "contributing_ion": c.get("ion"),
                        "contrib_force": c.get("force"),
                        "contrib_magnitude": c.get("magnitude"),
                        "contrib_distance": c.get("distance"),
                        "contrib_cosine_with_motion": c.get("cosine_with_motion"),
                        "contrib_motion_component": c.get("motion_component")
                    })
                break  # Only one permeation frame per ion

    df_expanded = pd.DataFrame(expanded_rows)
    df_summary = pd.DataFrame(summary_rows)

    return df_expanded, df_summary

