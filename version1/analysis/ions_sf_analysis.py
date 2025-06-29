import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def analyze_resid_changes_and_plot(data, output_folder, permeation_events2, filename="residues_permeating_sf.png"):
    # Sort keys as integers for sequential frame order
    sorted_keys = sorted(data.keys(), key=int)
    previous_resid = None
    resid_changes = 0
    resid_set = set()

    for key in sorted_keys:
        current_resid = data[key]["resid"]
        resid_set.add(current_resid)

        if previous_resid is not None and current_resid != previous_resid:
            resid_changes += 1

        previous_resid = current_resid

    unique_resid_count = len(permeation_events2)

    # Plotting
    labels = ['Residues Changes', 'Permeating Residues']
    values = [resid_changes, unique_resid_count]

    fig, ax = plt.subplots(figsize=(4, 6))  # Tall and narrow
    bar_width = 0.3  # Slim bars
    x = range(len(values))

    bars = ax.bar(x, values, width=bar_width, color=['skyblue', 'lightgreen'])

    # Add number labels on top of bars
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, int(yval),
                ha='center', va='bottom', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, rotation=45)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('Residue Changes in SF end vs Residues leaving SF', fontsize=18)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    plt.close()

    # return resid_changes, unique_resid_count, output_path

# import json
# import os
# import matplotlib.pyplot as plt

# def plot_field_histograms_from_json(json_path, output_folder):
#     target_keys = [
#         "total_no_ions", "total_asn_glu", "total_pip",
#         "total", "total_asn", "total_glu"
#     ]

#     with open(json_path, "r") as f:
#         data = json.load(f)

#     magnitudes = {key: [] for key in target_keys}
#     axials = {key: [] for key in target_keys}

#     # Collect data
#     for frame_data in data.values():
#         for key in target_keys:
#             if key in frame_data:
#                 magnitudes[key].append(frame_data[key]["magnitude"])
#                 axials[key].append(frame_data[key]["axial"])

#     os.makedirs(output_folder, exist_ok=True)
#     plot_paths = []

#     for key in target_keys:
#         # Magnitude plot
#         plt.figure(figsize=(6, 4))
#         plt.hist(magnitudes[key], bins=100, color="skyblue", edgecolor='black')
#         plt.title(f"{key} - Magnitude Distribution")
#         plt.xlabel("Magnitude")
#         plt.ylabel("Frequency")
#         plt.tight_layout()
#         mag_path = os.path.join(output_folder, f"{key}_magnitude.png")
#         plt.savefig(mag_path)
#         plt.close()
#         plot_paths.append(mag_path)

#         # Axial plot
#         plt.figure(figsize=(6, 4))
#         plt.hist(axials[key], bins=100, color="salmon", edgecolor='black')
#         plt.title(f"{key} - Axial Field Distribution")
#         plt.xlabel("Axial (Z-component)")
#         plt.ylabel("Frequency")
#         plt.tight_layout()
#         axial_path = os.path.join(output_folder, f"{key}_axial.png")
#         plt.savefig(axial_path)
#         plt.close()
#         plot_paths.append(axial_path)

#     return plot_paths


import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_field_histograms_from_json(json_path, output_folder):
    import json
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    target_keys = [
        "total_no_ions", "total_asn_glu", "total_pip",
        "total", "total_asn", "total_glu"
    ]

    with open(json_path, "r") as f:
        data = json.load(f)

    magnitudes = {key: [] for key in target_keys}
    axials = {key: [] for key in target_keys}

    for frame_data in data.values():
        for key in target_keys:
            if key in frame_data:
                magnitudes[key].append(frame_data[key]["magnitude"])
                axials[key].append(frame_data[key]["axial"])

    os.makedirs(output_folder/"histograms", exist_ok=True)
    plot_paths = []

    for key in target_keys:
        # Magnitude plot
        plt.figure(figsize=(6, 4))
        plt.hist(magnitudes[key], bins=100, color="skyblue", edgecolor='black')
        plt.title(f"{key} - Magnitude Distribution", fontsize=16)
        plt.xlabel("Magnitude", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        mag_path = os.path.join(output_folder, f"{key}_magnitude.png")
        plt.savefig(mag_path)
        plt.close()
        plot_paths.append(mag_path)

        # Axial plot
        axial_data = np.array(axials[key])
        if key == "total":
            axial_data = axial_data[(axial_data >= -40) & (axial_data <= 40)]

        plt.figure(figsize=(6, 4))
        plt.hist(axial_data, bins=100, color="salmon", edgecolor='black')
        plt.title(f"{key} - Axial Field Distribution", fontsize=16)
        plt.xlabel("Axial (Z-component)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        axial_path = os.path.join(output_folder, f"{key}_axial.png")
        plt.savefig(axial_path)
        plt.close()
        plot_paths.append(axial_path)

    return plot_paths




##################################################################

def extract_permeation_frames_by_mode(data, mode="all"):
    """
    Detects permeation frames using different strategies:
    - "all": all resid changes are counted (frame-by-frame)
    - "last": only the last frame each ion is closest (i.e., just before it's replaced)
    - "first": only the first frame each ion appears (i.e., first entry)

    Parameters:
    - json_path: path to the closest_sf_unentered_ions.json
    - mode: "all", "last", or "first"

    Returns:
    - Set of frame indices (as integers)
    """
    import json

    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    sorted_frames = sorted(int(k) for k in data.keys())
    prev_resid = None

    permeation_frames = set()
    resid_to_frames = {}  # resid → list of frames

    for frame in sorted_frames:
        resid = data[int(frame)]["resid"]

        if mode == "all":
            if prev_resid is not None and resid != prev_resid:
                permeation_frames.add(frame - 1)  # frame where the previous ion exited
            prev_resid = resid

        elif mode in ("last", "first"):
            if resid not in resid_to_frames:
                resid_to_frames[resid] = []
            resid_to_frames[resid].append(frame)

    if mode == "last":
        for frames in resid_to_frames.values():
            permeation_frames.add(frames[-1])  # keep only the last frame for each ion

    elif mode == "first":
        for frames in resid_to_frames.values():
            permeation_frames.add(frames[0])  # keep only the first frame for each ion

    return permeation_frames


def analyze_field_at_permeation(json_path, closest_unentered_ion_to_upper_gate, permation_mode , output_path, key="total", field_component="axial"):
    """
    Compares the electric field component values (e.g., axial or magnitude) between
    frames with ion permeation and those without.

    Parameters:
    - json_path: path to the JSON file containing frame-wise field data
    - permeation_frames: set of integers representing frames with SF exit/permeation
    - output_path: folder to save the boxplot
    - key: which key from the JSON to use ("total", "total_glu", "total_asn", etc.)
    - field_component: either "axial" or "magnitude"

    Returns:
    - A dictionary with lists of values for "with_permeation" and "without_permeation"
    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    permeation_frames = extract_permeation_frames_by_mode(closest_unentered_ion_to_upper_gate, permation_mode)
    with_permeation = []
    without_permeation = []

    for frame_str, frame_data in data.items():
        frame = int(frame_str)
        if key not in frame_data:
            continue
        value = frame_data[key][field_component]
        if frame in permeation_frames:
            with_permeation.append(value)
        else:
            without_permeation.append(value)

    # Plot
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(6, 5))
    box = plt.boxplot([with_permeation, without_permeation],
                  labels=["Permeation", "No Permeation"],
                  showfliers=False)


    # Labels and titles
    plt.ylabel(f"{key} - {field_component.capitalize()} Field", fontsize=14)
    plt.title(f"{field_component.capitalize()} Field During vs Not During Permeation", fontsize=15)
    plt.xticks([1, 2], ["Permeation", "No Permeation"], fontsize=13)
    plt.yticks(fontsize=13)



    # Compute stats
    mean_perm = np.mean(with_permeation)
    median_perm = np.median(with_permeation)
    min_perm = np.min(with_permeation)
    max_perm = np.max(with_permeation)

    mean_no = np.mean(without_permeation)
    median_no = np.median(without_permeation)
    min_no = np.min(without_permeation)
    max_no = np.max(without_permeation)

    # Mann–Whitney U test
    stat, p_value = mannwhitneyu(with_permeation, without_permeation, alternative='two-sided')

    # Format stat text
    def format_stats(mean, median, min_val, max_val):
        return f"Mean: {mean:.2f}\nMedian: {median:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"

    y_bottom = plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0])

    plt.text(1, y_bottom, format_stats(mean_perm, median_perm, min_perm, max_perm), ha='center', fontsize=10)
    plt.text(2, y_bottom, format_stats(mean_no, median_no, min_no, max_no), ha='center', fontsize=10)

    # Add Mann–Whitney p-value below both boxes
    plt.text(1.5, y_bottom - 0.1 * (plt.ylim()[1] - plt.ylim()[0]),
            f"Mann–Whitney U Test p = {p_value:.3e}", ha='center', fontsize=11, fontweight='bold')



    plt.subplots_adjust(bottom=0.35)  # or tweak this value as needed
    plot_path = os.path.join(output_path, f"{key}_{field_component}_permeation_comparison.png")
    plt.savefig(plot_path)
    plt.close()



def run_all_field_permeation_analyses(json_path, closest_ion_dict, output_base_path):
    import os

    modes = ["all", "last", "first"]
    target_keys = [
        "total_no_ions", "total_asn_glu", "total_pip",
        "total", "total_asn", "total_glu"
    ]
    components = ["axial", "magnitude"]

    all_results = {}

    for mode in modes:
        mode_path = os.path.join(output_base_path, f"mode_{mode}")
        os.makedirs(mode_path, exist_ok=True)

        for key in target_keys:
            for component in components:
                result = analyze_field_at_permeation(
                    json_path=json_path,
                    closest_unentered_ion_to_upper_gate=closest_ion_dict,
                    permation_mode=mode,
                    output_path=mode_path,
                    key=key,
                    field_component=component
                )
                # Save result in a structured dict
                all_results[(mode, key, component)] = result

    return all_results
