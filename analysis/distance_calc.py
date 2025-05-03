import pandas as pd
import numpy as np

# total_distances_dict = {}

def get_overlapping_ions(ion_id, target_start, target_end, event_list):
    overlapping_ions = []
    for event in event_list:
        if event['ion_id'] == ion_id:
            continue
        start = event['start_frame']
        end = event['exit_frame']
        if not (target_end < start or end < target_start):
            overlapping_ions.append(int(event['ion_id']))
    return overlapping_ions

def calculate_distances(ion_permeated, analyzer, use_ca_only=True):
    """
    Calculates the distances between a given ion and selected residues across relevant frames.
    Args:
        ion_permeated: dictionary with 'ion_id'
        analyzer: object with trajectory and permeation event lists
        use_ca_only: if True, uses only CA atoms; if False, uses all atoms (min distance)
    Returns:
        Dictionary of distances per frame for that ion
    """

    u = analyzer.u
    ch1 = analyzer.permeation_events1
    ch2 = analyzer.permeation_events2
    ch3 = analyzer.permeation_events3

    ion_id = int(ion_permeated['ion_id'])

    # Get the start/exit frame from ch2 (the source event)
    start_frame, exit_frame = next(
        (event['start_frame'], event['exit_frame'])
        for event in ch2 if event['ion_id'] == ion_id
    )

    temp_distances_dict = {}
    ions_to_test = []
    ions_to_test += get_overlapping_ions(ion_id, start_frame, exit_frame, ch1)
    ions_to_test += get_overlapping_ions(ion_id, start_frame, exit_frame, ch2)
    ions_to_test += get_overlapping_ions(ion_id, start_frame, exit_frame, ch3)

    temp_distances_dict[ion_id] = []

    residue_ids = [98, 423, 748, 1073, 130, 455, 780, 1105]
    sf_residues = [100, 425, 750, 1075]  # Selectivity filter residues
    all_residues = residue_ids + sf_residues

    # Select either CA atoms or all atoms
    if use_ca_only:
        residue_atoms = {resid: u.select_atoms(f"resid {resid} and name CA") for resid in all_residues}
    else:
        residue_atoms = {resid: u.select_atoms(f"resid {resid}") for resid in all_residues}

    # Select the ion
    ion = u.select_atoms(f"resname K+ and resid {ion_id}")
    if len(ion) != 1:
        print(f"Warning: Ion resid {ion_id} not found uniquely.")
        return

    for ts in u.trajectory[start_frame:exit_frame + 1]:
        frame_data = {'frame': ts.frame, 'residues': {}, 'ions': {}}
        ion_pos = ion.positions[0]

        # Distances to selected residues
        for resid in residue_ids:
            atomgroup = residue_atoms[resid]
            if use_ca_only:
                dist = float(np.linalg.norm(ion_pos - atomgroup.positions[0]))
            else:
                dists = np.linalg.norm(atomgroup.positions - ion_pos, axis=1)
                dist = float(np.min(dists))
            frame_data['residues'][resid] = dist

        # Selectivity filter: average of min distances
        sf_distances = []
        for sf_resid in sf_residues:
            sf_atoms = residue_atoms[sf_resid]
            if use_ca_only:
                dist = float(np.linalg.norm(ion_pos - sf_atoms.positions[0]))
            else:
                dists = np.linalg.norm(sf_atoms.positions - ion_pos, axis=1)
                dist = float(np.min(dists))
            sf_distances.append(dist)

        frame_data['residues']["SF"] = float(np.mean(sf_distances))

        # Distances to overlapping ions
        for ion_to_test in ions_to_test:
            other_ion = u.select_atoms(f"resname K+ and resid {ion_to_test}")
            if len(other_ion) == 1:
                other_pos = other_ion.positions[0]
                dist = float(np.linalg.norm(ion_pos - other_pos))
                frame_data['ions'][ion_to_test] = dist

        temp_distances_dict[ion_id].append(frame_data)

    return temp_distances_dict

