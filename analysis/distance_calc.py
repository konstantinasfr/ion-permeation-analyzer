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

def calculate_distances(ion_permeated, analyzer, use_ca_only=True, use_min_distances=False, use_charges=False):
    """
    Calculates the distances between a given ion and selected residues across relevant frames.
    Depending on the flags, uses CA atoms, all atoms (min), or functional atom-based charge centers.

    Args:
        ion_permeated: dictionary with 'ion_id'
        analyzer: object with trajectory and permeation event lists
        use_ca_only: if True, uses only CA atoms
        use_min_distances: if True, uses all atoms and takes min distance
        use_charges: if True, uses midpoint of charged/polar atoms (OE1/OE2 for Glu, OD1/ND2 for Asn)

    Returns:
        Dictionary of distances per frame for that ion
    """

    u = analyzer.u
    ch1, ch2, ch3 = analyzer.permeation_events1, analyzer.permeation_events2, analyzer.permeation_events3

    ion_id = int(ion_permeated['ion_id'])

    # Get start/exit frame for this ion from Channel 2 event list
    start_frame, exit_frame = next(
        (event['start_frame'], event['exit_frame']) for event in ch2 if event['ion_id'] == ion_id
    )

    temp_distances_dict = {ion_id: []}
    ions_to_test = get_overlapping_ions(ion_id, start_frame, exit_frame, ch1) + \
                   get_overlapping_ions(ion_id, start_frame, exit_frame, ch2) + \
                   get_overlapping_ions(ion_id, start_frame, exit_frame, ch3)

    # Residue definitions
    glu_residues = [98, 423, 748, 1073]
    asn_residues = [130, 455, 780, 1105]
    sf_residues = [100, 425, 750, 1075]
    all_residues = glu_residues + asn_residues + sf_residues

    if sum([use_ca_only, use_min_distances, use_charges]) != 1:
        raise ValueError("You must set exactly one of use_ca_only, use_min_distances, or use_charges to True.")

    residue_atoms = {}
    charge_centers = {}

    for resid in all_residues:
        if use_ca_only:
            residue_atoms[resid] = u.select_atoms(f"resid {resid} and name CA")

        elif use_min_distances:
            residue_atoms[resid] = u.select_atoms(f"resid {resid}")

        elif use_charges:
            if resid in glu_residues:
                atoms = u.select_atoms(f"resid {resid} and name OE1 OE2")
                if atoms.n_atoms == 2:
                    charge_centers[resid] = 0.5 * (atoms.positions[0] + atoms.positions[1])
                else:
                    print(f"Glu {resid} missing OE1 or OE2")
                    charge_centers[resid] = None

            elif resid in asn_residues:
                atoms = u.select_atoms(f"resid {resid} and name OD1 ND2")
                if atoms.n_atoms == 2:
                    charge_centers[resid] = 0.5 * (atoms.positions[0] + atoms.positions[1])
                else:
                    print(f"Asn {resid} missing OD1 or ND2")
                    charge_centers[resid] = None

            elif resid in sf_residues:
                residue_atoms[resid] = u.select_atoms(f"resid {resid}")

    # Select the ion of interest
    ion = u.select_atoms(f"resname K+ K and resid {ion_id}")
    if len(ion) != 1:
        print(f"Warning: Ion resid {ion_id} not found uniquely.")
        return

    for ts in u.trajectory[start_frame:exit_frame + 1]:
        frame_data = {'frame': ts.frame, 'residues': {}, 'ions': {}}
        ion_pos = ion.positions[0]

        for resid in glu_residues + asn_residues:
            if use_ca_only or use_min_distances:
                atomgroup = residue_atoms[resid]
                if use_ca_only:
                    dist = float(np.linalg.norm(ion_pos - atomgroup.positions[0]))
                else:
                    dists = np.linalg.norm(atomgroup.positions - ion_pos, axis=1)
                    dist = float(np.min(dists))

            elif use_charges:
                if resid in charge_centers and charge_centers[resid] is not None:
                    dist = float(np.linalg.norm(ion_pos - charge_centers[resid]))
                else:
                    print(f"Skipping resid {resid} due to missing charge center. {ts}")
                    dist = float('nan')

            frame_data['residues'][resid] = dist

        # Selectivity filter residues (always use atom-based min distance)
        sf_distances = []
        for sf_resid in sf_residues:
            atomgroup = residue_atoms[sf_resid]
            if use_ca_only:
                dist = float(np.linalg.norm(ion_pos - atomgroup.positions[0]))
            else:
                dists = np.linalg.norm(atomgroup.positions - ion_pos, axis=1)
                dist = float(np.min(dists))
            sf_distances.append(dist)

        frame_data['residues']['SF'] = float(np.mean(sf_distances))

        # Distances to overlapping ions
        for ion_to_test in ions_to_test:
            other_ion = u.select_atoms(f"resname K+ K and resid {ion_to_test}")
            if len(other_ion) == 1:
                other_pos = other_ion.positions[0]
                dist = float(np.linalg.norm(ion_pos - other_pos))
                frame_data['ions'][ion_to_test] = dist

        temp_distances_dict[ion_id].append(frame_data)

    return temp_distances_dict

