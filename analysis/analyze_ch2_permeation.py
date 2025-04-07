import json
from typing import Dict, List, Any
from collections import Counter
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from typing import List, Dict, Any

"""
analyze_ch2_permeation.py

This script identifies which ions are present and what residues they are closest to
at the exact frame where another ion permeates (i.e., exits the channel).
"""

def get_residues_at_frame(min_results_per_frame: Dict[str, List[Dict[str, Any]]], target_frame: int) -> Dict[int, int]:
    """
    Returns a dictionary of ions and their closest residue at a specific frame.
    Only includes ions that have data at that frame.

    Example:
    {
        2433: 130,
        1313: 780,
        ...
    }
    """
    residues_at_frame = {}

    for ion_id, entries in min_results_per_frame.items():
        for entry in entries:
            if entry.get("frame") == target_frame:
                residues_at_frame[int(ion_id)] = entry["residue"]
                break

    return residues_at_frame


def analyze_ch2_permation_residues(min_results_per_frame: Dict[str, List[Dict[str, Any]]], end_frame) -> List[Dict[str, Any]]:
    """
    For each ion, find the frame where it permeates (i.e., its last frame).
    Then collect the residues of all other ions present at that same frame.

    Returns a list of dictionaries like:
    {
        "frame": 3627,
        "ions": {1313: 780, ...},
        "permeated": 2433
    }
    """
    ch2_permation_residues = []

    for ion_id, entries in min_results_per_frame.items():
        if not entries:
            continue

        # The last frame is considered the permeation frame
        ion_last_frame = max(entry["frame"] for entry in entries if "frame" in entry)

        if ion_last_frame == end_frame:
            continue
        
        residues_at_frame = get_residues_at_frame(min_results_per_frame, ion_last_frame)

        ch2_permation_residues.append({
            "frame": ion_last_frame,
            "ions": residues_at_frame,
            "permeated": int(ion_id)
        })

    return ch2_permation_residues

def count_residue_combinations_with_duplicates(permeation_events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each permeation event:
    - Sort the residues (including duplicates)
    - Convert to a comma-separated string key
    - Track how often each combo appears
    - Track which residue the permeated ion was closest to
    """
    summary = defaultdict(lambda: {"count": 0, "permeated_residues": []})

    for event in permeation_events:
        residues = list(event["ions"].values())              # [98, 130, 98, 455]
        sorted_residues = sorted(residues)                   # [98, 98, 130, 455]
        residue_key = ", ".join(str(r) for r in sorted_residues)  # "98, 98, 130, 455"

        permeated_ion = event["permeated"]
        permeated_residue = event["ions"].get(permeated_ion)

        if permeated_residue is not None:
            summary[residue_key]["count"] += 1
            summary[residue_key]["permeated_residues"].append(permeated_residue)

    return summary