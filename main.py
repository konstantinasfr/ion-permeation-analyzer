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
from analysis.organizing_frames import close_contact_residues_analysis, get_clean_ion_coexistence_table
from analysis.frames_frequencies_plots import plot_top_intervals_by_frames
from analysis.analyze_ch2_permeation import analyze_ch2_permation_residues, count_residue_combinations_with_duplicates, find_all_pre_permeation_patterns
from analysis.analyze_ch2_permeation import count_last_residues,plot_last_residue_bar_chart, save_residue_combination_summary_to_excel
from analysis.force_analysis import collect_sorted_cosines_until_permeation
from analysis.force_analysis import extract_permeation_frames, extract_last_frame_analysis, extract_permeation_forces
import json
import pandas as pd
from analysis.permation_profile_creator import PermeationAnalyzer
from analysis.close_residues_analysis import plot_residue_counts, analyze_residue_combinations, find_closest_residues_percentage
from analysis.close_residues_analysis import count_frames_residue_closest, extract_min_mean_distance_pairs, count_frames_pair_closest, plot_start_frame_residue_distribution
from analysis.significant_forces import significant_forces
from analysis.find_clean_stuck_frames import find_clean_stuck_frames


def main():
    parser = argparse.ArgumentParser(description="Run dual-channel ion permeation analysis.")
    parser.add_argument("--do_permeation_analysis", default=True)
    # parser.add_argument("--top_file", default="/media/konsfr/KINGSTON/trajectory/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="/media/konsfr/KINGSTON/trajectory/protein.nc")

    # parser.add_argument("--top_file", default="../com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../protein.nc")
    # parser.add_argument("--channel_type", default="G4")

    # parser.add_argument("--top_file", default="../../G4-homotetramer/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../../G4-homotetramer/protein.nc")

    # parser.add_argument("--top_file", default="../Rep0/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../Rep0/protein.nc")

    # parser.add_argument("--top_file", default="../GIRK12_WT/RUN2/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../GIRK12_WT/RUN2/protein.nc")
    

    # parser.add_argument("--top_file", default="../GIRK12_WT/RUN1/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="../GIRK12_WT/RUN1/protein.nc")

    # parser.add_argument("--top_file", default="/media/konsfr/KINGSTON/trajectory/Rep0/com_4fs.prmtop")
    # parser.add_argument("--traj_file", default="/media/konsfr/KINGSTON/trajectory/Rep0/GIRK_4kfm_NoCHL_Rep0_500ns.nc")
    # parser.add_argument("--channel_type", default="G12")
    parser.add_argument("--channel_type", default="G2")
    args = parser.parse_args()

    data_path = "/home/data/Konstantina/ion-permeation-analyzer-results"

    

    if args.channel_type == "G4":
        upper1 = [106, 431, 756, 1081]
        lower1 = [100, 425, 750, 1075]  #sf_residues
  
        upper2 = [100, 425, 750, 1075]  #sf_residues
        lower2 = [130, 455, 780, 1105]  #asn_residues

        upper3 = [130, 455, 780, 1105]
        lower3 = [138, 463, 788, 1113]

        upper4 = [138, 463, 788, 1113]  #hbc_residues
        lower4 = [265, 590, 915 ,1240]

        upper5 = [265, 590, 915 ,1240]
        lower5 = [259, 584, 909, 1234]

        hbc_residues = [138, 463, 788, 1113]
        hbc_diagonal_pairs = [(138, 788), (463, 1113)]

        sf_low_res_residues = [100, 425, 750, 1075]
        sf_low_res_diagonal_pairs = [(100, 750), (425, 1075)]

        glu_residues = [98, 423, 748, 1073]
        asn_residues = [130, 455, 780, 1105]
        sf_residues = [100, 425, 750, 1075]

        start_frame = 0
        # start_frame = 5550
        # start_frame = 6500
        end_frame = 6799

    elif args.channel_type == "G2":
        upper1 = [106, 434, 762, 1090]
        lower1 = [100, 428, 756, 1084]

        upper2 = [100, 428, 756, 1084]
        lower2 = [130, 458, 786, 1114] #asn_residues

        upper3 = [130, 458, 786, 1114] #asn_residues
        lower3 = [138, 466, 794, 1122] #hbc_residues

        upper4 = [138, 466, 794, 1122] #hbc_residues
        lower4 = [265, 593, 921, 1249]

        upper5 = [265, 593, 921, 1249] #upper gloop
        lower5 = [259, 587, 915, 1243] #lower gloop

        hbc_residues = [138, 466, 794, 1122]
        hbc_diagonal_pairs = [(138, 794), (466, 1122)]

        glu_residues = [98, 426, 754, 1082]
        asn_residues = [130, 458, 786, 1114]
        sf_residues = [100, 428, 756, 1084]

        sf_low_res_residues = [100, 428, 756, 1084]
        sf_low_res_diagonal_pairs = [(100, 756), (428, 1084)]

        start_frame = 0
        # start_frame = 800
        # start_frame = 5550
        # start_frame = 6500
        # end_frame = 1250
        end_frame = 5000

        results_dir = Path(f"{data_path}/results_G2")
        results_dir = Path(f"{data_path}/results_G2_5000_frames")

        top_file = Path("/home/data/Konstantina/Rep0/com_4fs.prmtop")
        traj_file = Path("/home/data/Konstantina/Rep0/protein.nc")

    elif args.channel_type == "G12":
        upper1 = [107, 432, 757, 1082]
        lower1 = [101, 426, 751, 1076]  #sf_residues
  
        upper2 = [101, 426, 751, 1076]  #sf_residues
        lower2 = [131, 456, 781, 1106]  #asn_residues

        upper3 = [131, 456, 781, 1106]
        lower3 = [139, 464, 789, 1114]

        upper4 = [139, 464, 789, 1114]  #hbc_residues
        lower4 = [266, 591, 916 ,1241]

        upper5 = [266, 591, 916 ,1241]
        lower5 = [260, 585, 910, 1235]

        hbc_residues = [139, 464, 789, 1114]
        hbc_diagonal_pairs = [(139, 789), (464, 1114)]

        sf_low_res_residues = [101, 426, 751, 1076] 
        sf_low_res_diagonal_pairs = [(101, 751), (426, 1076)]

        glu_residues = [99, 424, 749, 1074]
        asn_residues = [131, 456, 781, 1106]
        sf_residues = [101, 426, 751, 1076] 

        start_frame = 0
        # start_frame = 3550
        # end_frame = 1000
        # end_frame = 6800
        end_frame = 1250
        # end_frame = 3550

        top_file = Path("/home/data/Konstantina/GIRK12_WT/RUN2/com_4fs.prmtop")
        traj_file = Path("/home/data/Konstantina/GIRK12_WT/RUN2/protein.nc")

        results_dir = Path(f"{data_path}/results_G12_RUN1")
        results_dir = Path(f"{data_path}/results_G12_3500_6800")
        results_dir = Path(f"{data_path}/results_G12_3550_6800_duplicates")
        results_dir = Path(f"{data_path}/results_G12_0_1250")

    # start_frame = 5414
    # end_frame = 5553
    # start_frame = 5000
    # end_frame = 6800
    # start_frame = 6500
    # start_frame = 0
    # # start_frame = 5550
    # # start_frame = 6500
    # end_frame = 6799
    # end_frame = 6562
    u = mda.Universe(top_file, traj_file)
    results_dir.mkdir(exist_ok=True)
    
    ch1 = Channel(u, upper1, lower1, num=1, radius=11)
    ch2 = Channel(u, upper2, lower2, num=2, radius=15.0)
    ch3 = Channel(u, upper3, lower3, num=3, radius=15.0)
    ch4 = Channel(u, upper4, lower4, num=4, radius=15.0)
    ch5 = Channel(u, upper5, lower5, num=5, radius=15.0)

    analyzer = IonPermeationAnalysis(u, ion_selection="resname K+ K", start_frame=start_frame, end_frame=end_frame, channel1=ch1, channel2=ch2, channel3=ch3, channel4=ch4, channel5=ch5,
                                     hbc_residues=hbc_residues, hbc_diagonal_pairs=hbc_diagonal_pairs,
                                     sf_low_res_residues=sf_low_res_residues, sf_low_res_diagonal_pairs=sf_low_res_diagonal_pairs)

    analyzer.run_analysis()
    # analyzer.rename_all_permeation_ion_ids()
    analyzer.print_results()

    total_distances_dict = {}
                       
    distances_path = results_dir / "distances.json"

    if distances_path.exists():
        print(f"✅ File {distances_path} already exists. Skipping distance calculation.")
        with open(distances_path) as f:
            raw_data = json.load(f)

        total_distances_dict = {}

        for ion_id_str, frame_data in raw_data.items():
            ion_id = str(ion_id_str)
            total_distances_dict[ion_id] = []

            for entry in frame_data:
                # Fix inner structure so it matches calculate_distances output
                cleaned_entry = {
                    "frame": int(entry["frame"]),
                    "residues": {
                        int(k) if k.isdigit() else k: float(v)
                        for k, v in entry["residues"].items()
                    },
                    "ions": {
                        str(k): float(v) for k, v in entry["ions"].items()
                    }
                }
                total_distances_dict[ion_id].append(cleaned_entry)

    else:
        print("🚀 Calculating distances...")
        total_distances_dict = {}

        for ion_in_ch2 in tqdm(analyzer.permeation_events2, total=len(analyzer.permeation_events2),
                            desc="Calculating Distances", unit="ion"):
            temp_distances_dict = calculate_distances(
                ion_in_ch2, analyzer, use_ca_only=False, use_min_distances=False, use_charges=True,
                glu_residues=glu_residues, asn_residues=asn_residues, sf_residues=sf_residues
            )
            total_distances_dict.update(temp_distances_dict)

        # Save with string keys (required by JSON)
        json_ready = {str(k): v for k, v in total_distances_dict.items()}
        with open(distances_path, "w") as f:
            json.dump(json_ready, f, indent=2)
        print(f"💾 Saved distances to {distances_path}")




    total_distances_dict_ca = {}
                       

                       
    # for ion_in_ch2 in tqdm(analyzer.permeation_events2,total=len(analyzer.permeation_events2),
    #                    desc="Calculating Distances", unit="ion"):
    #     temp_distances_dict = calculate_distances(ion_in_ch2, analyzer, use_ca_only=True)
    #     total_distances_dict_ca.update(temp_distances_dict)


    # Create 'results' directory if it doesn't exist
    # results_dir = Path("results_no_mutations")
    
    # results_dir = Path("results_test")
    
    force_results_dir = Path(f"{results_dir}/forces")
    force_results_dir.mkdir(exist_ok=True)
    coexisting_ions_results_dir = Path(f"{results_dir}/coexisting_ions_in_channel2")
    coexisting_ions_results_dir.mkdir(exist_ok=True)
    force_per_ion_results_dir = Path(f"{force_results_dir}/forces_per_ion")
    force_per_ion_results_dir.mkdir(exist_ok=True)
    ch2_permeation_characteristics_dir = Path(f"{results_dir}/ch2_permeation_characteristics")
    ch2_permeation_characteristics_dir.mkdir(exist_ok=True)
    close_contact_residues_dir = Path(f"{results_dir}/close_contact_residues")
    close_contact_residues_dir.mkdir(exist_ok=True)
    last_frame_forces_dir = Path(f"{ch2_permeation_characteristics_dir}/forces_last_frame")
    last_frame_forces_dir.mkdir(exist_ok=True)

    with open(results_dir / "hbc_diameters.json", "w") as f:
        json.dump(analyzer.hbc_diameters, f, indent=2)
    
    with open(results_dir / "sf_low_res_diameters.json", "w") as f:
        json.dump(analyzer.sf_low_res_diameters, f, indent=2)

    with open(results_dir / "ch1.json", "w") as f:
        json.dump(analyzer.permeation_events1, f, indent=2)

    with open(results_dir / "ch2.json", "w") as f:
        json.dump(analyzer.permeation_events2, f, indent=2)

    with open(results_dir / "ch3.json", "w") as f:
        json.dump(analyzer.permeation_events3, f, indent=2)

    with open(results_dir / "ch4.json", "w") as f:
        json.dump(analyzer.permeation_events4, f, indent=2)

    with open(results_dir / "ch5.json", "w") as f:
        json.dump(analyzer.permeation_events5, f, indent=2)

    #     # Save total_distances_dict to JSON
    # with open(results_dir / "distances_ca.json", "w") as f:
    #     json.dump(total_distances_dict_ca, f, indent=2)

    residue_clusters, min_results_per_frame, close_contacts_dict = cluster_frames_by_closest_residue(total_distances_dict)

    total_residue_comb_over_all_frames = close_contact_residues_analysis(close_contacts_dict, close_contact_residues_dir, args.channel_type, max_bar_number=20)
    plot_residue_counts(total_residue_comb_over_all_frames, close_contact_residues_dir, filename=f"residue_counts_all_frames.png", exclude=(), duplicates=False)
    analyze_residue_combinations(total_residue_comb_over_all_frames, close_contact_residues_dir, top_n_plot=20)
    find_closest_residues_percentage(min_results_per_frame, close_contact_residues_dir, args.channel_type)
    count_frames_residue_closest(min_results_per_frame, close_contact_residues_dir, end_frame, args.channel_type)
    min_mean_distance_pairs = extract_min_mean_distance_pairs(total_distances_dict)
    count_frames_pair_closest(min_mean_distance_pairs, close_contact_residues_dir, end_frame, args.channel_type)
    plot_start_frame_residue_distribution(min_results_per_frame, analyzer.permeation_events2, close_contact_residues_dir, args.channel_type)

    with open(results_dir / "min_mean_distance_pairs.json", "w") as f:
        json.dump(min_mean_distance_pairs, f, indent=2)

    with open(close_contact_residues_dir / "total_residue_comb_over_all_frames.json", "w") as f:
        json.dump(total_residue_comb_over_all_frames, f, indent=2)

    with open(results_dir / "residue_clusters.json", "w") as f:
        json.dump(residue_clusters, f, indent=2)

    ch2_permeations = analyzer.fix_permeations(residue_clusters)

    get_clean_ion_coexistence_table(ch2_permeations, coexisting_ions_results_dir)

    with open(results_dir / "ch2_fixed.json", "w") as f:
        json.dump(ch2_permeations, f, indent=2)

    with open(results_dir / "min_results_per_frame.json", "w") as f:
        json.dump(min_results_per_frame, f, indent=2)

    with open(results_dir / "close_contacts_dict.json", "w") as f:
        json.dump(close_contacts_dict, f, indent=2)

    print("Saved residue clustering to results/residue_clusters.json")

    if len(analyzer.permeation_events3) > 0:
        print(f"Found {len(ch2_permeations)} permeation events in channel 2")
        # ch2_permation_residues, frame here is the last frame before the permeation
        ch2_permation_residues,  ch2_permation_residues_pdb = analyze_ch2_permation_residues(min_results_per_frame, ch2_permeations, end_frame, args.channel_type)

        ch2_permation_residue_comb = count_residue_combinations_with_duplicates(ch2_permation_residues)

        for residues, count in ch2_permation_residue_comb.items():
            print(f"Residues {residues} appear {count} time(s)")

        with open(results_dir / "ch2_permation_residues.json", "w") as f:
            json.dump(ch2_permation_residues, f, indent=2)

        with open(results_dir / "ch2_permation_residues_pdb.json", "w") as f:
            json.dump(ch2_permation_residues_pdb, f, indent=2)

        with open(results_dir / "ch2_permation_residue_comb.json", "w") as f:
            json.dump(ch2_permation_residue_comb, f, indent=2)
        save_residue_combination_summary_to_excel(ch2_permation_residue_comb, results_dir)


        final_residue_counts = count_last_residues(ch2_permation_residues_pdb)
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


        pre_permeation_pattern_results = find_all_pre_permeation_patterns(ch2_permation_residues, min_results_per_frame)

        with open(results_dir / "pre_permeation_patterns.json", "w") as f:
            json.dump(pre_permeation_pattern_results, f, indent=2)

        
        plot_top_intervals_by_frames(residue_clusters, max_bar_number=20)
        
        if args.do_permeation_analysis:
            permeation_analysis = PermeationAnalyzer(
                ch2_permation_residues=ch2_permation_residues,
                ch1_permeation_events=analyzer.permeation_events1,
                ch2_permeation_events=analyzer.permeation_events2,
                u=u,
                start_frame=start_frame,
                end_frame=end_frame,
                min_results_per_frame=min_results_per_frame,
                ch2=ch2,
                close_contacts_dict=close_contacts_dict,
                total_residue_comb_over_all_frames=total_residue_comb_over_all_frames,
                glu_residues = glu_residues,
                asn_residues = asn_residues,
                sf_residues= sf_residues,
                cutoff=15.0,
                calculate_total_force=False,
                prmtop_file=top_file,
                nc_file=traj_file,
                output_base_dir=ch2_permeation_characteristics_dir
            )



            # === Define cached result file paths ===
            force_results_path = force_results_dir / "force_results.json"
            radial_distances_path = ch2_permeation_characteristics_dir / "radial_distances_results.json"
            close_residues_path = ch2_permeation_characteristics_dir / "close_residues_results.json"
            force_intervals_path = last_frame_forces_dir / "force_intervals_results.json"

            # === Try to load only the 4 core results ===
            if (
                force_results_path.exists()
                and radial_distances_path.exists()
                and close_residues_path.exists()
                and force_intervals_path.exists()
            ):
                print("✅ Loaded precomputed core results (forces, radial, close, intervals).")

                with open(force_results_path) as f:
                    forces_results = json.load(f)
                    permeation_analysis.force_results = forces_results

                with open(radial_distances_path) as f:
                    radial_distances_results = json.load(f)
                    permeation_analysis.radial_distances_results = radial_distances_results

                with open(close_residues_path) as f:
                    close_residues_results = json.load(f)
                    permeation_analysis.close_residues_results = close_residues_results

                with open(force_intervals_path) as f:
                    force_intervals_results = json.load(f)
                    permeation_analysis.force_intervals_results = force_intervals_results

            else:
                print("⚙️ Running full `run_permeation_analysis()`...")
                forces_results, radial_distances_results, close_residues_results, force_intervals_results = permeation_analysis.run_permeation_analysis()

                permeation_analysis.force_results = forces_results
                permeation_analysis.radial_distances_results = radial_distances_results
                permeation_analysis.close_residues_results = close_residues_results
                permeation_analysis.force_intervals_results = force_intervals_results

                # Save the 4 core results
                with open(force_results_path, "w") as f:
                    json.dump(forces_results, f, indent=2)
                with open(radial_distances_path, "w") as f:
                    json.dump(radial_distances_results, f, indent=2)
                with open(close_residues_path, "w") as f:
                    json.dump(close_residues_results, f, indent=2)
                with open(force_intervals_path, "w") as f:
                    json.dump(force_intervals_results, f, indent=2)

                for ion_forces in forces_results:
                    ion_id = ion_forces["permeated_ion"]
                    with open(force_per_ion_results_dir / f"{ion_id}.json", "w") as f:
                        json.dump(ion_forces, f, indent=2)        

            
            # === Recalculate everything else, even if loaded ===
            last_frame_forces = extract_last_frame_analysis(forces_results)
            permeation_analysis.last_frame_forces = last_frame_forces
            extract_permeation_forces(data=last_frame_forces, output_dir=last_frame_forces_dir)

            with open(last_frame_forces_dir / "force_results_last_frame.json", "w") as f:
                json.dump(last_frame_forces, f, indent=2)



            forces_df = pd.DataFrame(forces_results)
            # forces_df.to_excel(force_results_dir / "force_results.xlsx", index=False)

            top_cosine_ionic_motion = collect_sorted_cosines_until_permeation(forces_results)
            with open(force_results_dir / "top_cosine_ionic_motion.json", "w") as f:
                json.dump(top_cosine_ionic_motion, f, indent=2)

            df_permeation_frames_forces_with_ions, df_permeation_frames_forces = extract_permeation_frames(forces_results, offset_from_end=1)
            df_permeation_frames_forces.to_csv(force_results_dir / "permeation_frames_forces.csv", index=False)
            df_permeation_frames_forces.to_excel(force_results_dir / "permeation_frames_forces.xlsx", index=False)
            df_permeation_frames_forces_with_ions.to_csv(force_results_dir / "permeation_frames_forces_with_ions.csv", index=False)
            df_permeation_frames_forces_with_ions.to_excel(force_results_dir / "permeation_frames_forces_with_ions.xlsx", index=False)

            print("Saved force results and recalculated dependent outputs.")

            # === Run post-analysis ===
            permeation_analysis.closest_residues_comb_before_permeation(n=-1, use_pdb_format=True, sort_residues=True)
            # permeation_analysis.analyze_cosine_significance(force_results_dir)
            # permeation_analysis.analyze_radial_significance()

            find_clean_stuck_frames(force_per_ion_results_dir, force_results_dir, u, sf_residues)
            significant_forces(args.channel_type, force_results_dir)
            
    else:
        print("No permeation events found in channel 2")


if __name__ == "__main__":
    main()

