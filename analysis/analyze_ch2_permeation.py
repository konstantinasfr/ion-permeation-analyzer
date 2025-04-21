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

        if ion_last_frame == end_frame-1:
            continue
        
        residues_at_frame = get_residues_at_frame(min_results_per_frame, ion_last_frame)

        ch2_permation_residues.append({
            "frame": ion_last_frame,
            "ions": residues_at_frame,
            "permeated": int(ion_id)
        })

    return ch2_permation_residues

from collections import defaultdict
from typing import List, Dict, Any, Union

def count_residue_combinations_with_duplicates(permeation_events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each permeation event:
    - Sort numeric residues (keep 'SF' at the end)
    - Create a residue key string
    - Track how often each combination appears
    - Record the permeated residues and frames
    """
    summary = defaultdict(lambda: {
        "count": 0,
        "permeated_residues": [],
        "frames": []
    })

    for event in permeation_events:
        residues_raw = list(event["ions"].values())

        numeric_residues = [int(r) for r in residues_raw if str(r).isdigit()]
        non_numeric_residues = [str(r) for r in residues_raw if not str(r).isdigit()]

        sorted_residues = sorted(numeric_residues) + non_numeric_residues
        residue_key = ", ".join(str(r) for r in sorted_residues)

        permeated_ion = event["permeated"]
        permeated_residue = event["ions"].get(permeated_ion)
        frame = event.get("frame")

        if permeated_residue is not None:
            summary[residue_key]["count"] += 1
            summary[residue_key]["permeated_residues"].append(permeated_residue)
            if frame is not None:
                summary[residue_key]["frames"].append(frame)

    return summary


import pandas as pd
from pathlib import Path
from typing import Dict, Any

def save_residue_combination_summary_to_excel(
    data: Dict[str, Dict[str, Any]],
    results_dir: Path,
    filename: str = "ch2_permation_residue_comb.xlsx"
):
    """
    Saves the residue combination summary to an Excel file.
    Each row includes:
    - Residue combination (as string)
    - Count
    - Permeated residues (comma-separated string)
    - Permeation frames (comma-separated string)
    """
    records = []

    for residue_key, info in data.items():
        record = {
            "Residue Combination": residue_key,
            "Count": info.get("count", 0),
            "Permeated Residues": ", ".join(str(r) for r in info.get("permeated_residues", [])),
            "Frames": ", ".join(str(f) for f in info.get("frames", []))
        }
        records.append(record)

    df = pd.DataFrame(records)
    output_path = results_dir / filename
    df.to_excel(output_path, index=False)
    print(f"✅ Excel file saved to: {output_path}")



from collections import Counter
from typing import List, Dict, Any

def count_last_residues(permeation_events: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Counts how many times each residue appears as the last residue 
    before an ion permeates.
    """
    counter = Counter()

    for event in permeation_events:
        permeated_ion = event["permeated"]
        last_residue = event["ions"].get(str(permeated_ion)) or event["ions"].get(int(permeated_ion))

        if last_residue is not None:
            counter[str(last_residue)] += 1  # convert to string for consistent plotting

    return dict(counter)

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def plot_last_residue_bar_chart(residue_counts: Dict[str, int], results_dir: Path, filename: str = "last_residues_barplot.png"):
    """
    Plots and saves a bar chart of how often each residue was the last before permeation.
    Bars are sorted by count (descending), with larger fonts for axes and labels.
    """
    # Sort residues by count descending
    sorted_items = sorted(residue_counts.items(), key=lambda x: x[1], reverse=True)
    residues = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Plot setup
    plt.figure(figsize=(12, 7))
    bars = plt.bar(residues, counts, color="steelblue")

    # Set bigger fonts
    plt.xlabel("Residue", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.title("Last Residue Before Permeation", fontsize=20)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)

    # Annotate counts above bars with bigger font
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{int(height)}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=14)

    plt.tight_layout()

    # Save figure
    output_path = results_dir / filename
    plt.savefig(output_path)
    print(f"✅ Plot saved to: {output_path}")
    plt.close()

