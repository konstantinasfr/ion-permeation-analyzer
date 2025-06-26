import numpy as np
import os
import json
from tqdm import tqdm

def find_closest_unentered_ion_to_upper_gate(u, upper_gate_residues, output_path=None, use_CA=False):
    """
    Finds the K+ ion closest to the upper gate center at each frame,
    but only among those that haven't crossed the gate (i.e., z >= gate center z).
    
    Parameters:
    - u: MDAnalysis Universe
    - upper_gate_residues: list of resid integers
    - gate_num: 4 or 5, used to determine gate geometry logic
    - output_path: folder to save JSON output, if desired
    
    Returns:
    - result: dictionary of frame -> {"resid": ion_resid, "distance": dist}
    """
    
    result = {}

    for ts in tqdm(u.trajectory, desc="Scanning for closest unentered ions"):
        # Compute upper gate center dynamically
        atom_indices = []
        for resid in upper_gate_residues:
            if use_CA:
                atoms = u.select_atoms(f"resid {resid} and name CA")
            else:
                atoms = u.select_atoms(f"resid {resid}")
                coords = atoms.positions
                upper_index = coords[:, 2].argsort()[0]  # highest atom (lowest z)
                atoms = atoms[[upper_index]]
            atom_indices.append(atoms[0].index)

        upper_center = u.atoms[atom_indices].center_of_mass()
        gate_z = upper_center[2]

        # Get all K+ ions
        ions = u.select_atoms("resname K+ or name K")
        
        # Filter ions above the gate
        unentered_ions = [ion for ion in ions if ion.position[2] >= gate_z]

        if not unentered_ions:
            continue  # No ions above the gate

        # Find the closest one
        distances = [np.linalg.norm(ion.position - upper_center) for ion in unentered_ions]
        min_index = np.argmin(distances)
        closest_ion = unentered_ions[min_index]

        result[ts.frame] = {
            "resid": int(closest_ion.resid),
            "distance": float(distances[min_index])
        }

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "closest_sf_unentered_ions.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved results to {output_path}/closest_sf_unentered_ions.json")

    return result
