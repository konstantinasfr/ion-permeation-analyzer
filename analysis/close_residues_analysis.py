from analysis.converter import convert_to_pdb_numbering
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


def get_last_nth_frame_close_residues(event, n=-1, use_pdb_format=True, sort_residues=True, channel_type="G2"):
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
            if  any(isinstance(r, int) for r in residues):
            # Sort only if there are integers in the list
                residues = sorted(residues, key=lambda r: int(r))
            
        formatted_residues = [
            convert_to_pdb_numbering(res, channel_type) if use_pdb_format else str(res)
            for res in residues
        ]
        converted_data[ion_id] = "_".join(formatted_residues)

    return {selected_frame_key: converted_data}


import matplotlib.pyplot as plt
from collections import Counter
import os

def plot_residue_counts(data, output_dir, filename="residue_counts.png", exclude=(), duplicates=True):
    """
    Plots and saves a bar chart of residue string occurrences, sorted by count.

    Parameters:
    - data (dict): {frame: {ion_id: residue_string}}
    - output_dir (str): Folder where the plot will be saved
    - filename (str): Name of output image file
    - exclude (tuple): Residue values to ignore (e.g. "SF", "no_close_residues")
    - duplicates (bool): 
        - If True (default), count all appearances
        - If False, count each unique residue string at most once per frame
    """
    # Step 1: Count residue strings
    residue_counter = Counter()
    os.makedirs(output_dir, exist_ok=True)

    for frame_dict in data.values():
        seen_in_frame = set()
        for residue_string in frame_dict.values():
            if residue_string in exclude:
                continue
            if not duplicates:
                if residue_string in seen_in_frame:
                    continue
                seen_in_frame.add(residue_string)
            residue_counter[residue_string] += 1

    if not residue_counter:
        print("⚠️ No residue strings to plot.")
        return

    # Step 2: Sort by frequency
    sorted_items = sorted(residue_counter.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Step 3: Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Plot
    plt.figure(figsize=(max(12, len(labels) * 0.5), 6))  # auto-width for many bars
    bars = plt.bar(labels, counts, color='steelblue')

    # Add count labels above each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count),
                 ha='center', va='bottom', fontsize=9)

    # Format y-axis: only integers and some top margin
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(False)

    # Labels and layout
    plt.title(f"Residue Combinations Occurrences in {len(data)} Events")
    plt.xlabel("Residue Combinations")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Step 5: Save figure
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Clean plot saved to: {plot_path}")


import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

def analyze_residue_combinations(data, output_dir, top_n_plot=20):
    """
    Analyzes how often all possible residue string combinations appear across frames.

    Parameters:
    - data (dict): {frame: {ion_id: residue_string}}
    - output_dir (str): Folder to save CSV and plot
    - top_n_plot (int): Number of top combinations to show in the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build set of residues per frame (excluding unwanted ones)
    frame_residue_sets = {
        frame: set(val for val in ion_dict.values() if val not in ("SF"))
        for frame, ion_dict in data.items()
    }

    # Step 2: Determine max combination size automatically
    max_comb_size = max(len(s) for s in frame_residue_sets.values())

    # Step 3: Count how many frames each combination appears in
    combination_counter = defaultdict(int)

    for size in range(1, max_comb_size + 1):
        for residue_set in frame_residue_sets.values():
            for combo in combinations(sorted(residue_set), size):
                combination_counter[combo] += 1

    # Step 4: Save to CSV
    csv_path = os.path.join(output_dir, "residue_combination_frequencies.csv")
    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Combination", "Count"])
        for combo, count in sorted(combination_counter.items(), key=lambda x: (-len(x[0]), -x[1])):
            writer.writerow(["+".join(combo), count])
    print(f"✅ CSV saved to: {csv_path}")

    # Step 5: Plot top N combinations
    sorted_combos = sorted(combination_counter.items(), key=lambda x: x[1], reverse=True)
    top_combos = sorted_combos[:top_n_plot]
    labels = ["+".join(combo) for combo, _ in top_combos]
    counts = [count for _, count in top_combos]

    plt.figure(figsize=(max(10, len(labels) * 0.5), 6))
    bars = plt.bar(labels, counts, color="mediumseagreen")

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, count + 0.5, str(count),
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frames Appeared")
    plt.title(f"Top {top_n_plot} Residue Combinations Across {len(data)} Frames")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "residue_combination_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved to: {plot_path}")


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
