import os
import matplotlib.pyplot as plt

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
                ha='center', va='bottom', fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Count')
    ax.set_title('Residue Changes in SF end vs Residues leaving SF')
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

    os.makedirs(output_folder, exist_ok=True)
    plot_paths = []

    for key in target_keys:
        # Magnitude plot (no filtering)
        plt.figure(figsize=(6, 4))
        plt.hist(magnitudes[key], bins=100, color="skyblue", edgecolor='black')
        plt.title(f"{key} - Magnitude Distribution")
        plt.xlabel("Magnitude")
        plt.ylabel("Frequency")
        plt.tight_layout()
        mag_path = os.path.join(output_folder, f"{key}_magnitude.png")
        plt.savefig(mag_path)
        plt.close()
        plot_paths.append(mag_path)

        # Axial plot (filter -40 to 40 for 'total')
        axial_data = np.array(axials[key])
        if key == "total":
            axial_data = axial_data[(axial_data >= -40) & (axial_data <= 40)]

        plt.figure(figsize=(6, 4))
        plt.hist(axial_data, bins=100, color="salmon", edgecolor='black')
        plt.title(f"{key} - Axial Field Distribution")
        plt.xlabel("Axial (Z-component)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        axial_path = os.path.join(output_folder, f"{key}_axial.png")
        plt.savefig(axial_path)
        plt.close()
        plot_paths.append(axial_path)

    return plot_paths
