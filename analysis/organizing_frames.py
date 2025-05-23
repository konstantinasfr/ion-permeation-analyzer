import numpy as np

def convert_to_pdb_numbering(residue_id: int) -> str:
    """
    Converts a residue ID to a PDB-style numbering.
    """
    if residue_id != "SF":
        chain_number = int(residue_id)//325
        chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        pdb_number = residue_id-325*chain_number+49
        return f"{pdb_number}.{chain_dict[chain_number]}"
    else:
        return "SF"


def cluster_frames_by_closest_residue(distance_data):
    clustered_results = {}
    min_results_per_frame = {}
    close_contacts_dict = {}

    for ion_id, frame_list in distance_data.items():
        clusters = []
        prev_residue = None
        start_frame = None
        distances = []
        close_contacts_dict[ion_id] = {}
        min_results_per_frame[ion_id] = []
        not_sf_starting = False

        for frame_data in frame_list[:-1]:
            frame = frame_data["frame"]
            residues = frame_data["residues"]

            # Find the closest residue (key with smallest value)
            closest_residue, closest_distance = min(residues.items(), key=lambda item: item[1])

            if closest_residue != "SF" or not_sf_starting:
                min_results_per_frame[ion_id].append({
                    "frame":frame,
                    "residue":closest_residue,
                    "min_distance":closest_distance
                })
                not_sf_starting = True

                ######### find close contact residues, filter residues with distance < 6 #####################################3
                sorted_close_contacts = dict(
                                                sorted(
                                                    {resid: dist for resid, dist in residues.items() if dist < 6}.items(),
                                                    key=lambda item: item[1]
                                                )
                                            )
                close_residues = []
                no_close_contacts = True
                for resid, dist in sorted_close_contacts.items():
                    no_close_contacts = False
                    if resid == "SF":
                        # if closest is SF then we consider that ion is still in SF
                        if not close_residues:
                            close_residues.append(resid)
                            break
                        else:
                            continue
                    else:
                        close_residues.append(resid)

                if no_close_contacts:
                    close_residues.append("no_close_residues")

                close_contacts_dict[ion_id][frame] = close_residues

                ################################## Cluster creation per residue ########################
                if closest_residue != prev_residue:
                    # If ending a previous cluster, store it
                    if prev_residue is not None:
                        clusters.append({
                            "residue": prev_residue,
                            "start": start_frame,
                            "end": prev_frame,
                            "frames": prev_frame - start_frame + 1,
                            "mean_distance": sum(distances) / len(distances)
                        })
                    # Start a new cluster
                    start_frame = frame
                    prev_residue = closest_residue
                    distances = [closest_distance]
                else:
                    distances.append(closest_distance)

                prev_frame = frame

        # Add final cluster
        if prev_residue is not None:
            clusters.append({
                "residue": prev_residue,
                "start": start_frame,
                "end": prev_frame,
                "frames": prev_frame - start_frame + 1,
                "mean_distance": sum(distances) / len(distances)
            })

        clustered_results[ion_id] = clusters

    return clustered_results, min_results_per_frame, close_contacts_dict


import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

def close_contact_residues_analysis(data, main_path, max_bar_number=20):
    """
    For each ion, plots and saves a bar chart of residue combinations (unordered)
    that are close during trajectory frames, and writes full CSV summary.

    Parameters:
        data (dict): ion_id → frame_id → list of close residues
        results_dir (str): directory where two subfolders will be created:
                           - close_contact_residues/plots
                           - close_contact_residues/csv
        max_bar_number (int): max number of bars in each plot
    """

    main_path = os.path.join(main_path, "")
    plot_dir = os.path.join(main_path, "plots")
    csv_dir = os.path.join(main_path, "csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    total_residue_comb_over_all_frames = {}

    def normalize_combo(combo):
        return tuple(sorted(combo))

    total_combo_counts = Counter()

    for ion_id, frames in data.items():
        combo_counts = Counter()

        for frame, residues in frames.items():
            if frame not in total_residue_comb_over_all_frames:
                total_residue_comb_over_all_frames[frame] = {}
            if residues != ["SF"] and residues != ["no_close_residues"]:
                norm_combo = normalize_combo(residues)
                combo_counts[norm_combo] += 1
                total_combo_counts[norm_combo] += 1
                total_residue_comb_over_all_frames[frame][ion_id] = '_'.join(map(convert_to_pdb_numbering, norm_combo))

        if not combo_counts:
            continue

        combo_data = [{"residue_combination": '_'.join(map(convert_to_pdb_numbering, combo)), "count": count}
                      for combo, count in combo_counts.items()]
        df = pd.DataFrame(combo_data).sort_values(by="count", ascending=False)
        csv_path = os.path.join(csv_dir, f"{ion_id}.csv")
        df.to_csv(csv_path, index=False)

        top_combos = combo_counts.most_common(max_bar_number)
        labels = ['_'.join(map(convert_to_pdb_numbering, combo)) for combo, _ in top_combos]
        counts = [count for _, count in top_combos]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, counts)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.title(f"Ion {ion_id} — Top {max_bar_number} combos")
        plt.xlabel("Residue combination")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = os.path.join(plot_dir, f"{ion_id}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"✅ Ion {ion_id}: plot saved to {plot_path}, data to {csv_path}")

    # Global summary across all ions
    if total_combo_counts:
        total_combo_data = [{"residue_combination": '_'.join(map(convert_to_pdb_numbering, combo)), "count": count}
                            for combo, count in total_combo_counts.items()]
        df_total = pd.DataFrame(total_combo_data).sort_values(by="count", ascending=False)
        total_csv_path = os.path.join(main_path, "ALL_ions_combined.csv")
        df_total.to_csv(total_csv_path, index=False)

        top_total_combos = total_combo_counts.most_common(max_bar_number)
        labels = ['_'.join(map(convert_to_pdb_numbering, combo)) for combo, _ in top_total_combos]
        counts = [count for _, count in top_total_combos]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(labels, counts)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.title(f"All Ions — Top {max_bar_number} Residue Combinations")
        plt.xlabel("Residue combination")
        plt.ylabel("Total Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()

        total_plot_path = os.path.join(main_path, "ALL_ions_combined.png")
        plt.savefig(total_plot_path)
        plt.close()

        print(f"📊 Combined plot saved to {total_plot_path}, data to {total_csv_path}")

    return total_residue_comb_over_all_frames
        



def get_ions_for_frame(data, target_frame):
    for frame_info in data:
        if frame_info["frame"] == target_frame:
            return frame_info["ions"]
    return None  # If not found

def tracking_ion_distances(
        permeation_events,        # list of dicts: {frame, ions, permeated}
        distances,               # list of dicts: {frame, ions: {ion_id: distance}, ...}
        ch2_entry_exit_list       # list of dicts: {ion_id, start_frame, exit_frame}
    ):
    from collections import defaultdict

    results = {}

    # Convert a flat list of CH2 entry/exit records into a dictionary grouped by ion_id
    ch2_entry_exit_dict = defaultdict(list)
    for entry in ch2_entry_exit_list:
        ion_id = int(entry['ion_id'])
        ch2_entry_exit_dict[ion_id].append({
            'start_frame': entry['start_frame'],
            'exit_frame': entry['exit_frame']
        })

    # Keep only the latest event per ion
    latest_permeation_bounds = {
        ion_id: sorted(ranges, key=lambda x: x['exit_frame'], reverse=True)[0]
        for ion_id, ranges in ch2_entry_exit_dict.items()
    }

    for event in permeation_events:

        target_ion = int(event['permeated'])
        results[target_ion] = []
        end_frame = event['frame']
        start_frame = latest_permeation_bounds[target_ion]['start_frame']

        # for ions_in_ch2 in event['ions']:
        #     for ion_ion_distance in distances[target_ion]["ions"]:
        #         if ion_ion_distance

        for f in range(start_frame, end_frame + 1):
            ion_ion_dist = get_ions_for_frame(distances[target_ion], f)
            proximate_ions = {}
            for ion_in_ch2 in event["ions"]:
                if ion_in_ch2 != target_ion:
                    try:
                        proximate_ions[int(ion_in_ch2)] = ion_ion_dist[ion_in_ch2]
                    except KeyError:
                        continue
            results[target_ion].append({
                "frame": f,
                "distances": proximate_ions
            })

    return results


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ion_distance_traces(distance_data, results_dir):
    """
    Takes a dictionary of distance traces and saves two line plots per target ion:
    1. Full timeline of distances
    2. Last 15 frames before permeation (showing true frame numbers on x-axis)

    Parameters:
    - distance_data: dict
        {
            "2433": [
                {"frame": 3185, "distances": {"1313": 6.89, "1460": 9.15}},
                {"frame": 3186, "distances": {"1313": 8.06, "1460": 9.32}},
                ...
            ],
            ...
        }
    - results_dir: Path or str
        Base directory where plots should be saved.
    """
    # Ensure output folders exist
    results_dir = Path(results_dir)
    full_dir = results_dir / "ion_distances"
    last15_dir = results_dir / "ion_distances_last15"
    full_dir.mkdir(parents=True, exist_ok=True)
    last15_dir.mkdir(parents=True, exist_ok=True)

    # Flatten the nested dictionary into a long-format DataFrame
    records = []
    for target_ion, frames in distance_data.items():
        for entry in frames:
            frame = entry["frame"]
            for other_ion, distance in entry["distances"].items():
                records.append({
                    "target_ion": int(target_ion),
                    "other_ion": int(other_ion),
                    "frame": int(frame),
                    "distance": distance
                })

    df = pd.DataFrame(records)

    # Create and save one plot per target ion (full timeline + last 15 frames per other ion)
    for target_ion in df["target_ion"].unique():
        subset = df[df["target_ion"] == target_ion].sort_values("frame")

        # === Full plot ===
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x="frame", y="distance", hue="other_ion", marker="o", palette="tab10")
        plt.title(f"Distances from Ion {target_ion} to Other Ions (Full Timeline)")
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.legend(title="Other Ion")
        plt.xticks(subset["frame"].unique())
        plt.tight_layout()
        plt.savefig(full_dir / f"{target_ion}.png")
        plt.close()

        # === Last 15 frame plot per other_ion, then merge ===
        last_15_frames = (
            subset.groupby("other_ion", group_keys=False)
            .apply(lambda x: x.sort_values("frame").tail(15))
        )

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=last_15_frames, x="frame", y="distance", hue="other_ion", marker="o", palette="tab10")
        plt.title(f"Distances from Ion {target_ion} (Last 15 Frames per Ion Before Permeation)")
        plt.xlabel("Frame")
        plt.ylabel("Distance")
        plt.legend(title="Other Ion")
        plt.xticks(last_15_frames["frame"].unique())
        plt.tight_layout()
        plt.savefig(last15_dir / f"{target_ion}.png")
        plt.close()

