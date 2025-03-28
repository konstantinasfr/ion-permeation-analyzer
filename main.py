import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import argparse
import warnings
from MDAnalysis.core.groups import AtomGroup
warnings.filterwarnings("ignore")
from pathlib import Path
from analysis.channels import Channel
from analysis.ion_analysis import IonPermeationAnalysis
from analysis.distance_calc import calculate_distances
from analysis.organizing_frames import cluster_frames_by_closest_residue
import json
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Run dual-channel ion permeation analysis.")
    parser.add_argument("--top_file", default="/media/konsfr/Intenso/Nousheen/com_4fs.prmtop")
    parser.add_argument("--traj_file", default="/media/konsfr/Intenso/Nousheen/protein.nc")
    args = parser.parse_args()

    u = mda.Universe(args.top_file, args.traj_file)

    upper1 = [106, 431, 756, 1081]
    lower1 = [100, 425, 750, 1075]

    # upper2 = [98, 423, 748, 1073]
    upper2 = [100, 425, 750, 1075]
    lower2 = [130, 455, 780, 1105]
    # lower2 = [138, 463, 788, 1113]

    upper3 = [130, 455, 780, 1105]
    lower3 = [138, 463, 788, 1113]

    ch1 = Channel(u, upper1, lower1, radius=8.0)
    ch2 = Channel(u, upper2, lower2, radius=15.0)
    ch3 = Channel(u, upper3, lower3, radius=15.0)

    analyzer = IonPermeationAnalysis(u, ion_selection="resname K+", channel1=ch1, channel2=ch2, channel3=ch3)
    analyzer.run_analysis()
    analyzer.print_results()

    total_distances_dict = {}
                       
    for ion_permpeated in tqdm(analyzer.permeation_events3,total=len(analyzer.permeation_events3),
                       desc="Calculating Distances", unit="ion"):
        temp_distances_dict = calculate_distances(ion_permpeated, analyzer)
        total_distances_dict.update(temp_distances_dict)


    # Create 'results' directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save total_distances_dict to JSON
    with open(results_dir / "distances.json", "w") as f:
        json.dump(total_distances_dict, f, indent=2)

    residue_clusters = cluster_frames_by_closest_residue(total_distances_dict)

    with open(results_dir / "residue_clusters.json", "w") as f:
        json.dump(residue_clusters, f, indent=2)

    print("Saved residue clustering to results/residue_clusters.json")

    # Create an ExcelWriter to hold multiple sheets
    with pd.ExcelWriter(results_dir / "residue_clusters.xlsx") as writer:
        for ion_id, intervals in residue_clusters.items():
            df = pd.DataFrame(intervals)
            df.insert(0, "residue", df.pop("residue"))
            df.to_excel(writer, sheet_name=str(ion_id), index=False)

if __name__ == "__main__":
    main()

