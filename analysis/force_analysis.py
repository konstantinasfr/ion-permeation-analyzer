import numpy as np
from tqdm import tqdm

# =========================
# Utility Functions
# =========================

def compute_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)

def compute_force(q1, q2, pos1, pos2, k=138.935456):
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

def analyze_frame(positions, permeating_ion_id, frame, other_ions, charge_map, cutoff=10.0):
    """
    Analyze one frame of a permeation event.
    Calculates Coulomb forces from surrounding ions, the motion of the permeating ion,
    and how well the force aligns with the ion's movement direction.
    
    Returns a dictionary with all computed quantities.
    """
    
    # Initialize result dictionary to store outputs for this frame
    result = {
        "frame": frame,  # Current frame number
        "net_force": [0.0, 0.0, 0.0],  # Placeholder for net Coulomb force vector
        "motion_vector": None,  # Ion motion vector (frame to frame+1)
        "alignment_with_motion": None,  # Dot product of net force and motion direction
        "alignment_ratio": None,  # How well aligned the force is with motion (cosine of angle)
        "top_by_magnitude": [],  # All contributing ions sorted by force magnitude
        "top_by_directional_contribution": []  # All contributing ions sorted by force along motion direction
    }

    # Get 3D position of the permeating ion at this frame
    # print("PERMEATING ION ID", type(permeating_ion_id))
    permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
    print(positions[5331].get(59324))
    print(positions[5331].keys())
    if permeating_pos is None:
        print("HEREE")
        return result  # Exit early if position is missing
    
    net_force = np.zeros(3)  # Accumulate total force vector here
    contributions = []  # Store per-ion force details here

    # Loop through all ions in the same frame
    for ion_id, pos in positions.get(frame, {}).items():
        if ion_id == permeating_ion_id or ion_id not in other_ions:
            continue  # Skip the ion itself or unrelated ions

        # Compute distance between the ion and the permeating ion
        distance = compute_distance(permeating_pos, pos)
        # print(distance)
        if distance <= cutoff:
            print(f"Analyzing ion {ion_id} at distance {distance:.2f} from permeating ion {permeating_ion_id}")
            # Compute Coulomb force from this ion on the permeating ion
            force = compute_force(charge_map[permeating_ion_id], charge_map[ion_id], permeating_pos, pos)
            net_force += force  # Add to total net force

            # Record this ion's force info
            magnitude = np.linalg.norm(force)
            contributions.append({
                "ion": int(ion_id),
                "force": [float(f) for f in force.tolist()],  # ensure elements are native floats
                "magnitude": float(magnitude),
                "distance": float(distance)
            })


    # Gather positions at this frame and the next (for motion vector)
    ion_positions_over_time = {
        f: positions.get(f, {}).get(permeating_ion_id) for f in range(frame, frame + 2)
    }
    
    # Calculate motion vector from current to next frame
    motion_vec = get_motion_vector(ion_positions_over_time, frame)
    if motion_vec is not None:
        unit_motion = unit_vector(motion_vec)  # Normalize it to unit length

        # Project net force onto motion direction (dot product)
        alignment = float(np.dot(net_force, unit_motion))

        # Compute ratio: how much of the net force is aligned with motion
        net_magnitude = np.linalg.norm(net_force)
        alignment_ratio = alignment / net_magnitude if net_magnitude != 0 else 0.0

        # Save motion and alignment data in the result
        result.update({
            "motion_vector": motion_vec.tolist(),
            "alignment_with_motion": alignment,
            "alignment_ratio": alignment_ratio
        })

        # For each contributing ion, calculate how much it helps/hurts motion
        for c in contributions:
            c["directional_contribution"] = float(np.dot(c["force"], unit_motion))

        # Sort all contributors by strength and by alignment with motion
        result["top_by_magnitude"] = sorted(contributions, key=lambda x: -x["magnitude"])
        result["top_by_directional_contribution"] = sorted(contributions, key=lambda x: -x["directional_contribution"])

    # Save total net force (even if motion vector is missing)
    result["net_force"] = net_force.tolist()
    return result

def analyze_permeation_events(ch2_permeation_events, universe, start_frame, end_frame, cutoff=10.0):
    """
    Analyze all permeation events over ±2 frames.
    Returns structured results for each event.
    """
    positions = build_all_positions(universe, start_frame, end_frame)
    charge_map = build_charge_map(universe)
    results = []


    for event in ch2_permeation_events:
        # Only analyze if the permeation frame is in the specified range
        if not (start_frame <= event["frame"] < end_frame):
            continue  # Skip this event

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
            other_ions=positions.get(frame, {}).keys(),  # ✅ use all ions in this frame
            # other_ions = event["ions"].keys()
            charge_map=charge_map,
            cutoff=cutoff
        )
        event_result["analysis"][frame] = frame_result


        results.append(event_result)

    return results
