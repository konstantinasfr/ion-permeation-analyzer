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
from analysis.organizing_frames import cluster_frames_by_closest_residue, tracking_ion_distances, plot_ion_distance_traces
from analysis.frames_frequencies_plots import plot_top_intervals_by_frames
from analysis.analyze_ch2_permeation import analyze_ch2_permation_residues, count_residue_combinations_with_duplicates
from analysis.analyze_ch2_permeation import count_last_residues,plot_last_residue_bar_chart, save_residue_combination_summary_to_excel
from analysis.force_analysis import analyze_permeation_events
import json
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description="Run dual-channel ion permeation analysis.")
    # parser.add_argument("--top_file", default="/media/konsfr/KINGSTON/trajectory/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="/media/konsfr/KINGSTON/trajectory/protein.nc")
    parser.add_argument("--top_file", default="../com_4fs.prmtop")
    parser.add_argument("--traj_file", default="../protein.nc")
    args = parser.parse_args()

    u = mda.Universe(args.top_file, args.traj_file)

    upper1 = [106, 431, 756, 1081]
    lower1 = [100, 425, 750, 1075]
    # lower1 = [98, 423, 748, 1073]

    # upper2 = [98, 423, 748, 1073]
    upper2 = [100, 425, 750, 1075]
    lower2 = [130, 455, 780, 1105]
    # lower2 = [138, 463, 788, 1113]

    upper3 = [130, 455, 780, 1105]
    lower3 = [138, 463, 788, 1113]

    # start_frame = 5414
    # end_frame = 5553
    # start_frame = 5000
    # end_frame = 6800
    # start_frame = 6500
    start_frame = 3000
    start_frame = 5300
    end_frame = 5400
    # end_frame = 6799

    ch1 = Channel(u, upper1, lower1, radius=11)
    ch2 = Channel(u, upper2, lower2, radius=15.0)
    ch3 = Channel(u, upper3, lower3, radius=15.0)

    analyzer = IonPermeationAnalysis(u, ion_selection="resname K+", start_frame=start_frame, end_frame=end_frame, channel1=ch1, channel2=ch2, channel3=ch3)
    analyzer.run_analysis()
    analyzer.print_results()

    total_distances_dict = {}
                       
    for ion_in_ch2 in tqdm(analyzer.permeation_events2,total=len(analyzer.permeation_events2),
                       desc="Calculating Distances", unit="ion"):
        temp_distances_dict = calculate_distances(ion_in_ch2, analyzer)
        total_distances_dict.update(temp_distances_dict)


    # Create 'results' directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "hbc_diameters.json", "w") as f:
        json.dump(analyzer.hbc_diameters, f, indent=2)
    
    with open(results_dir / "ch1.json", "w") as f:
        json.dump(analyzer.permeation_events1, f, indent=2)

    with open(results_dir / "ch2.json", "w") as f:
        json.dump(analyzer.permeation_events2, f, indent=2)

    with open(results_dir / "ch3.json", "w") as f:
        json.dump(analyzer.permeation_events3, f, indent=2)

    # Save total_distances_dict to JSON
    with open(results_dir / "distances.json", "w") as f:
        json.dump(total_distances_dict, f, indent=2)

    residue_clusters, min_results_per_frame = cluster_frames_by_closest_residue(total_distances_dict)

    ch2_permeations = analyzer.fix_permeations(residue_clusters)

    with open(results_dir / "ch2_fixed.json", "w") as f:
        json.dump(ch2_permeations, f, indent=2)

    with open(results_dir / "residue_clusters.json", "w") as f:
        json.dump(residue_clusters, f, indent=2)

    with open(results_dir / "min_results_per_frame.json", "w") as f:
        json.dump(min_results_per_frame, f, indent=2)

    print("Saved residue clustering to results/residue_clusters.json")

    ch2_permation_residues = analyze_ch2_permation_residues(min_results_per_frame, end_frame)

    ch2_permation_residue_comb = count_residue_combinations_with_duplicates(ch2_permation_residues)

    for residues, count in ch2_permation_residue_comb.items():
        print(f"Residues {residues} appear {count} time(s)")

    with open(results_dir / "ch2_permation_residues.json", "w") as f:
        json.dump(ch2_permation_residues, f, indent=2)

    with open(results_dir / "ch2_permation_residue_comb.json", "w") as f:
        json.dump(ch2_permation_residue_comb, f, indent=2)
    save_residue_combination_summary_to_excel(ch2_permation_residue_comb, results_dir)


    final_residue_counts = count_last_residues(ch2_permation_residues)
    plot_last_residue_bar_chart(final_residue_counts, results_dir, filename="ch2_permeation_last_residues.png")

    ion_distances = tracking_ion_distances(ch2_permation_residues, total_distances_dict, ch2_permeations)
    with open(results_dir / "ion_distances.json", "w") as f:
        json.dump(ion_distances, f, indent=2)
    plot_ion_distance_traces(ion_distances, results_dir)

    # Create an ExcelWriter to hold multiple sheets
    with pd.ExcelWriter(results_dir / "residue_clusters.xlsx") as writer:
        for ion_id, intervals in residue_clusters.items():
            df = pd.DataFrame(intervals)
            df.insert(0, "residue", df.pop("residue"))
            df.to_excel(writer, sheet_name=str(ion_id), index=False)

        # Create an ExcelWriter to hold multiple sheets
    with pd.ExcelWriter(results_dir / "min_results_per_frame.xlsx") as writer:
        for ion_id, intervals in min_results_per_frame.items():
            df = pd.DataFrame(intervals)
            df.insert(0, "residue", df.pop("residue"))
            df.to_excel(writer, sheet_name=str(ion_id), index=False)

    
    plot_top_intervals_by_frames(residue_clusters, max_bar_number=20)


    forces_results = analyze_permeation_events(ch2_permation_residues, u, start_frame, end_frame, cutoff=15.0, calculate_total_force=True, 
                                               prmtop_file=args.top_file, nc_file=args.traj_file)

    # Save to JSON
    with open(results_dir / "permeation_force_results.json", "w") as f:
        json.dump(forces_results, f, indent=2)

if __name__ == "__main__":
    main()

