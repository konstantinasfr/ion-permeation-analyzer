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
    Also calculates cosine similarities between different vectors.
    """

    result = {
        "frame": frame,
        "ionic_force": [0.0, 0.0, 0.0],
        "ionic_force_magnitude": None,
        "total_force": None,
        "total_force_magnitude": None,
        "ionic_fraction_of_total": None,
        "motion_vector": None,
        "alignment_with_motion": None,
        "alignment_ratio": None,
        "cosine_ionic_total": None,
        "cosine_total_motion": None,
        "cosine_ionic_motion": None,
        "top_by_magnitude": [],
        "top_by_directional_contribution": []
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

    ion_positions_over_time = {
        f: positions.get(f, {}).get(permeating_ion_id) for f in range(frame, frame + 2)
    }

    motion_vec = get_motion_vector(ion_positions_over_time, frame)
    if motion_vec is not None:
        unit_motion = unit_vector(motion_vec)
        alignment = float(np.dot(ionic_force, unit_motion))
        net_magnitude = np.linalg.norm(ionic_force)
        alignment_ratio = alignment / net_magnitude if net_magnitude != 0 else 0.0

        result.update({
            "motion_vector": motion_vec.tolist(),
            "alignment_with_motion": alignment,
            "alignment_ratio": alignment_ratio
        })

        for c in contributions:
            c["directional_contribution"] = float(np.dot(c["force"], unit_motion))

        result["top_by_magnitude"] = sorted(contributions, key=lambda x: -x["magnitude"])
        result["top_by_directional_contribution"] = sorted(contributions, key=lambda x: -x["directional_contribution"])

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

            # ============================
            # Cosine Similarity Calculations
            # ============================

            if np.linalg.norm(ionic_force) != 0 and np.linalg.norm(total_force) != 0:
                cosine_ionic_total = float(np.dot(ionic_force, total_force) / (np.linalg.norm(ionic_force) * np.linalg.norm(total_force)))
                result["cosine_ionic_total"] = cosine_ionic_total

            if motion_vec is not None and np.linalg.norm(total_force) != 0 and np.linalg.norm(motion_vec) != 0:
                cosine_total_motion = float(np.dot(total_force, motion_vec) / (np.linalg.norm(total_force) * np.linalg.norm(motion_vec)))
                result["cosine_total_motion"] = cosine_total_motion

            if motion_vec is not None and np.linalg.norm(ionic_force) != 0 and np.linalg.norm(motion_vec) != 0:
                cosine_ionic_motion = float(np.dot(ionic_force, motion_vec) / (np.linalg.norm(ionic_force) * np.linalg.norm(motion_vec)))
                result["cosine_ionic_motion"] = cosine_ionic_motion

    return result


def analyze_permeation_events(ch2_permeation_events, universe, start_frame, end_frame, cutoff=10.0,
                              calculate_total_force=False, prmtop_file=None, nc_file=None):
    """
    Analyze all permeation events over ±2 frames. If `calculate_total_force=True`, loads forces via OpenMM.
    """
    positions = build_all_positions(universe, start_frame, end_frame)
    charge_map = build_charge_map(universe)
    results = []

    total_force_data = None
    if calculate_total_force and prmtop_file and nc_file:
        print("Calculating total forces with OpenMM...")
        total_force_data, atom_index_map = calculate_ionic_forces_all_frames(prmtop_file, nc_file)

    for event in ch2_permeation_events:
        if not (start_frame <= event["frame"] < end_frame):
            continue

        event_result = {
            "frame": event["frame"],
            "permeated_ion": event["permeated"],
            "analysis": {}
        }

        frames_to_check = [
            event["frame"] - 2,
            event["frame"] - 1,
            event["frame"],
            event["frame"] + 1,
            event["frame"] + 2
        ]

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
