
import numpy as np

def compute_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)

def analyze_close_residues(positions, permeating_ion_id, frame, other_ions,
                  close_contacts_dict, cutoff=15.0):
    """
    Analyze one frame: compute ionic forces, motion, and optionally total force.
    Also calculates cosine similarities between different vectors and force decomposition.
    """
    result = {
        permeating_ion_id: None
    }

    permeating_pos = positions.get(frame, {}).get(permeating_ion_id)
    if permeating_pos is None:
        return result

    for ion_id, pos in positions.get(frame, {}).items():
        # if ion_id == permeating_ion_id or ion_id not in other_ions:
        #     continue
        distance = compute_distance(permeating_pos, pos)


        if distance <= cutoff:
            if ion_id not in close_contacts_dict:
                result[int(ion_id)] = ["SF"]
            elif frame not in close_contacts_dict[ion_id]:
                result[int(ion_id)] = ["SF"]
            else:
                result[int(ion_id)] = close_contacts_dict[ion_id][frame]

    return result



###################################################################

import os
import json
import pandas as pd

def convert_to_pdb_numbering(residue_id: int) -> str:
    """
    Converts a residue ID to a PDB-style numbering.
    """
    if isinstance(residue_id, int):
        chain_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
        chain_number = int(residue_id) // 325
        pdb_number = residue_id - 325 * chain_number + 49
        return f"{pdb_number}.{chain_dict[chain_number]}"
    else:
        return residue_id


def get_last_nth_frame_close_residues(event, n=-1, use_pdb_format=True, sort_residues=True):
    """
    Extract close residues at a specific frame from a permeation event.

    Behavior:
    - If n < 0: counts from the end of the sorted frame list (e.g., -1 = last, -2 = second-last)
    - If n >= 0: directly uses frame number `n` as a key in event["analysis"]

    Parameters:
        event (dict): Contains 'analysis' with frame: {ion_id: residues}
        n (int): Frame position or frame number depending on sign
        use_pdb_format (bool): Whether to convert residues to PDB-style notation
        sort_residues (bool): Whether to sort residues alphabetically

    Returns:
        dict: {frame_number: {ion_id: "res1_res2_..."}}
    """
    frames = sorted(event["analysis"].keys(), key=lambda x: int(x))

    if n < 0:
        if abs(n) > len(frames):
            raise ValueError(f"Frame index {n} is out of range. Event has {len(frames)} frames.")
        selected_frame_key = frames[n]
    else:
        if int(n) not in event["analysis"]:
            raise ValueError(f"Frame {n} not found in event['analysis'].")
        selected_frame_key = int(n)

    original_data = event["analysis"][selected_frame_key]

    converted_data = {}
    for ion_id, residues in original_data.items():
        if sort_residues:
            residues = sorted(residues, key=lambda r: str(r))

        formatted_residues = [
            convert_to_pdb_numbering(res) if use_pdb_format else str(res)
            for res in residues
        ]
        converted_data[ion_id] = "_".join(formatted_residues)

    return {selected_frame_key: converted_data}



# def closest_residues_comb_before_permeation(close_residues_results, output_base_dir, n=-1, use_pdb_format=False, sort_residues=True):
#     """
#     Loop through all permeation events and apply get_last_nth_frame_close_residues.
#     Saves both JSON and CSV outputs.
#     """
#     output_dir = os.path.join(output_base_dir, "closest_residues_comb")
#     os.makedirs(output_dir, exist_ok=True)

#     summary = []
#     for i, event in enumerate(close_residues_results):
#         try:
#             frame_data = get_last_nth_frame_close_residues(
#                 event, n=n, use_pdb_format=use_pdb_format, sort_residues=sort_residues
#             )
#             summary.append(frame_data)
#         except Exception as e:
#             print(f"Skipping event {i} due to error: {e}")

#     # Save JSON
#     with open(os.path.join(output_dir, f"closest_residues_n_{n}.json"), "w") as f:
#         json.dump(summary, f, indent=2)

#     # Save CSV
#     flat_rows = []
#     for event_summary in summary:
#         for frame, ion_data in event_summary.items():
#             for ion_id, residue_str in ion_data.items():
#                 flat_rows.append({
#                     "frame": frame,
#                     "ion_id": ion_id,
#                     "residues": residue_str
#                 })

#     df = pd.DataFrame(flat_rows)
#     df.to_csv(os.path.join(output_dir, f"closest_residues_n_{n}.csv"), index=False)
