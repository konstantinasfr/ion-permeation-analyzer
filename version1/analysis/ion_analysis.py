import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import os
import numpy as np

class IonPermeationAnalysis:
    def __init__(self, universe, ion_selection, start_frame, end_frame, channel1, channel2, channel3, channel4, channel5,
                 hbc_residues, hbc_diagonal_pairs, sf_low_res_residues, sf_low_res_diagonal_pairs, results_dir):
        
        self.u = universe
        self.ion_selection = ion_selection
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.channel4 = channel4
        self.channel5 = channel5
        self.hbc_residues =hbc_residues
        self.hbc_diagonal_pairs = hbc_diagonal_pairs
        self.sf_low_res_residues =sf_low_res_residues
        self.sf_low_res_diagonal_pairs = sf_low_res_diagonal_pairs
        self.ion_states1 = {}
        self.ion_states2 = {}
        self.ion_states3 = {}
        self.ion_states4 = {}
        self.ion_states5 = {}
        self.permeation_events1 = []
        self.permeation_events2 = []
        self.permeation_events3 = []
        self.permeation_events4 = []
        self.permeation_events5 = []
        self.results_dir = results_dir
        self.ions = self.u.select_atoms(self.ion_selection)

    def _check_ion_position(self, ion_id, ion_pos, channel, states, events, frame, keep_first_permeation, keep_first_insertion):
        ion_vec = ion_pos - channel.channel_center
        ion_z = np.dot(ion_vec, channel.channel_axis)
        channel_number = channel.channel_number
        

        if ion_id not in states:
            states[ion_id] = {'upper_flag': 0, 'lower_flag': 0, 'upper_flag_frame': 0, 'lower_flag_frame': 0, 'prev_ion_z': None}

        upper_z = np.dot(channel.upper_center - channel.channel_center, channel.channel_axis)
        lower_z = np.dot(channel.lower_center - channel.channel_center, channel.channel_axis)
        in_cylinder = channel.is_within_cylinder(ion_pos)

        # if channel_number==5 and ion_id==2257:
        #     print(f"Debugging ion_id: {ion_id}, frame: {frame}, in_cylinder: {in_cylinder}")
        if in_cylinder:
            if states[ion_id]['upper_flag'] == 0:
                states[ion_id]['upper_flag'] = 1
                if keep_first_insertion:
                    if states[ion_id]['upper_flag_frame'] == 0:
                        states[ion_id]['upper_flag_frame'] = frame
                else:
                    states[ion_id]['upper_flag_frame'] = frame
            elif states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 1:

                if states[ion_id]['prev_ion_z'] < upper_z:
                    #ion permeates again
                    start_frame = states[ion_id]['upper_flag_frame']
                    exit_frame = states[ion_id]['lower_flag_frame']
                    total_time = exit_frame - start_frame
                    events.append({
                        'ion_id': int(ion_id),
                        'start_frame': int(start_frame),
                        'exit_frame': int(exit_frame),
                        'total_time': int(total_time)
                    })
                    states[ion_id]['upper_flag_frame'] = frame
                states[ion_id]['lower_flag'] = 0

       
        if not in_cylinder:
             #If the ion passes the lower gate while it had previously entered (upper_flag = 1), mark the exit frame.
        # the ion permeated
            if ion_z > lower_z and states[ion_id]['upper_flag'] == 1:
                if states[ion_id]['lower_flag'] == 0:
                        # --- Check Asn dipole condition for Channel 2 ---
                    close_to_dipole = False
                    if channel_number == 2:
                        for resid in channel.lower_gate_residues:  # should be [130, 455, 780, 1105]
                            # Select the five key atoms in the ASN side chain that contribute to electrostatic interactions CG OD1 ND2 HD21 HD22
                            asn_atoms = self.u.select_atoms(
                                f"resid {resid} and not name N CA C O HA H"
                            )

                            # Ensure that all 5 atoms are present (sometimes an atom might be missing in a corrupted frame)
                            if len(asn_atoms) != 0:
                                # Compute the distance between the ion and each ASN sidechain atom
                                distances = np.linalg.norm(asn_atoms.positions - ion_pos, axis=1)

                                # Take the minimum of those distances
                                min_distance = np.min(distances)

                                # If the ion is within 6 Å of any of the atoms, flag it
                                if min_distance < 6.0:
                                    close_to_dipole = True  # Flag this residue as relevant for force calculation
                                    break  # No need to check more residues — one nearby ASN is enough

                    if not close_to_dipole or channel.channel_number != 2:
                        states[ion_id]['lower_flag'] = 1
                        if keep_first_permeation:
                            if states[ion_id]['lower_flag_frame'] == 0:
                                states[ion_id]['lower_flag_frame'] = frame
                        else:
                            states[ion_id]['lower_flag_frame'] = frame

            #If the ion was in the channel but left before exiting, reset the flags, maybe left from the sides
            elif states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 0:
                states[ion_id]['upper_flag'] = 0

        states[ion_id]['prev_ion_z'] = ion_z


        # if frame == self.end_frame-1 and states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 0:
        if frame == self.end_frame and states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 0:
            print(f"Warning: Ion {ion_id} was still in the channel at the end of the simulation. ")
            states[ion_id]['lower_flag'] = 1
            states[ion_id]['lower_flag_frame'] = frame

        # if ion_id == 2271 and channel_number == 2:
        #    print(f"'Frame: {frame}, channel_num: {channel_number}, upper_flag: {states[ion_id]['upper_flag']}, lower_flag: {states[ion_id]['lower_flag']}")



    def compute_constriction_point_diameters(self, frame, atoms, diagonal_pairs):
        """
        Computes the mean distance between pairs of HBC residues across the specified frames.
        """
        distances = []
        for res1, res2 in diagonal_pairs:
            pos1 = atoms[res1].positions
            pos2 = atoms[res2].positions
            # Compute all pairwise distances and take the minimum
            pairwise_dists = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)
            dist = np.min(pairwise_dists)
            distances.append(dist)

        mean_diameter = np.mean(distances)
        consiction_point_diameters_dict = {
            "frame": int(frame),
            "mean": float(mean_diameter),
            "A_C": float(distances[0]),
            "B_D": float(distances[1])
        }
        return consiction_point_diameters_dict
    

    def run_analysis(self):
        print("Starting analysis...")


       

        self.hbc_diameters = []

        
        self.sf_low_res_diameters = []


        for ts in tqdm(self.u.trajectory[self.start_frame:self.end_frame+1],
                    total=(self.end_frame - self.start_frame),
                    desc="Processing Frames", unit="frame"):
            
             # Select all atoms for each HBC residue
            hbc_atoms = {resid: self.u.select_atoms(f"resid {resid}") for resid in self.hbc_residues}

            self.hbc_diameters.append(self.compute_constriction_point_diameters(ts.frame, hbc_atoms, self.hbc_diagonal_pairs))

            sf_low_res_atoms = {resid: self.u.select_atoms(f"resid {resid}") for resid in self.sf_low_res_residues}

            self.sf_low_res_diameters.append(self.compute_constriction_point_diameters(ts.frame, sf_low_res_atoms, self.sf_low_res_diagonal_pairs))

            self.channel1.compute_geometry(1)
            self.channel2.compute_geometry(2)
            self.channel3.compute_geometry(3)
            self.channel4.compute_geometry(4)
            self.channel5.compute_geometry(5)

            for ion in self.ions:
                ion_id = ion.resid
                pos = ion.position
                self._check_ion_position(ion_id, pos, self.channel1, self.ion_states1, self.permeation_events1, ts.frame, False, False)
                self._check_ion_position(ion_id, pos, self.channel2, self.ion_states2, self.permeation_events2, ts.frame, False, True)
                self._check_ion_position(ion_id, pos, self.channel3, self.ion_states3, self.permeation_events3, ts.frame, False, False)
                self._check_ion_position(ion_id, pos, self.channel4, self.ion_states4, self.permeation_events4, ts.frame, False, False)
                self._check_ion_position(ion_id, pos, self.channel5, self.ion_states5, self.permeation_events5, ts.frame, False, False)


    def rename_all_permeation_ion_ids(self):
        self.permeation_events1 = self.rename_duplicate_ion_ids(self.permeation_events1)
        self.permeation_events2 = self.rename_duplicate_ion_ids(self.permeation_events2)
        self.permeation_events3 = self.rename_duplicate_ion_ids(self.permeation_events3)
        self.permeation_events4 = self.rename_duplicate_ion_ids(self.permeation_events4)
        self.permeation_events5 = self.rename_duplicate_ion_ids(self.permeation_events5)

    def filter_permeation_events(self, permeation_events_previous, permeation_events_current):
        """
        Filters the permeation events to keep only those that are present in the previous events.
        """
        previous_ids = {event['ion_id'] for event in permeation_events_previous}
        filtered_events = [event for event in permeation_events_current if event['ion_id'] in previous_ids]
        return filtered_events


    # def filter_permeation_events(self, permeation_events_previous, permeation_events_current, allow_start_at_zero=False):
    #     """
    #     Filters the permeation events.
        
    #     If allow_start_at_zero is False:
    #         - Only keeps events whose ion_id is present in the previous list.
        
    #     If allow_start_at_zero is True:
    #         - Keeps events that are in the previous list
    #         - OR events not in the previous list but with start_frame == 0
    #     """
    #     previous_ids = {event['ion_id'] for event in permeation_events_previous}

    #     filtered_events = []
    #     for event in permeation_events_current:
    #         if event['ion_id'] in previous_ids:
    #             filtered_events.append(event)
    #         elif allow_start_at_zero and event.get('start_frame') == 0:
    #             filtered_events.append(event)

    #     return filtered_events
       
    def keep_ions_that_pass_all_channels(self):
        """
        Filters the ion states to keep only those ions that have passed through all channels.
        """
        # self.permeation_events2 = self.filter_permeation_events(self.permeation_events1, self.permeation_events2)
        self.permeation_events3 = self.filter_permeation_events(self.permeation_events2, self.permeation_events3)
        self.permeation_events4 = self.filter_permeation_events(self.permeation_events3, self.permeation_events4)
        self.permeation_events5 = self.filter_permeation_events(self.permeation_events4, self.permeation_events5)

    # def print_results(self):
    #     def print_channel_results(channel_name, ion_states, permeation_events):
    #         print(f"\nFinal Permeation Events for {channel_name} (1,1 Flags):")
    #         print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
    #         print("-" * 55)

    #         for ion_id, state in ion_states.items():
    #             if state['upper_flag'] == 1 and state['lower_flag'] == 1:
    #                 start_frame = state['upper_flag_frame']
    #                 exit_frame = state['lower_flag_frame']
    #                 total_time = exit_frame - start_frame
    #                 permeation_events.append({
    #                     'ion_id': int(ion_id),
    #                     'start_frame': int(start_frame),
    #                     'exit_frame': int(exit_frame),
    #                     'total_time': int(total_time)
    #                 })

    #         permeation_events.sort(key=lambda x: x['start_frame'])

    #         for event in permeation_events:
    #             print(f"{int(event['ion_id']):6d} | {int(event['start_frame']):11d} | {int(event['exit_frame']):10d} | {int(event['total_time']):10d}")

    #         print(f"\nTotal forward permeation events: {len(permeation_events)}")


    #     print_channel_results("Channel 1", self.ion_states1, self.permeation_events1)
    #     print_channel_results("Channel 2", self.ion_states2, self.permeation_events2)
    #     print_channel_results("Channel 3", self.ion_states3, self.permeation_events3)
    #     print_channel_results("Channel 4", self.ion_states4, self.permeation_events4)
    #     print_channel_results("Channel 5", self.ion_states5, self.permeation_events5)

    #     self.rename_all_permeation_ion_ids()



    def print_results(self):
        def create_ch_permeation_dict(ion_states, permeation_events):
            for ion_id, state in ion_states.items():
                # print(f"Processing ion_id: {ion_id}, state: {state}")
                if int(ion_id) == 2400:
                    print(f"Debugging ion_id: {ion_id}, state: {state}")
                if state['upper_flag'] == 1 and state['lower_flag'] == 1:
                    start_frame = state['upper_flag_frame']
                    exit_frame = state['lower_flag_frame']
                    total_time = exit_frame - start_frame
                    permeation_events.append({
                        'ion_id': int(ion_id),
                        'start_frame': int(start_frame),
                        'exit_frame': int(exit_frame),
                        'total_time': int(total_time)
                    })

            permeation_events.sort(key=lambda x: x['start_frame'])

        def print_channel_results(channel_name, permeation_events):
            print(f"\nFinal Permeation Events for {channel_name} (1,1 Flags):")
            print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
            print("-" * 55)

            for event in permeation_events:
                print(f"{str(event['ion_id'])} | {int(event['start_frame']):11d} | {int(event['exit_frame']):10d} | {int(event['total_time']):10d}")

            print(f"\nTotal forward permeation events: {len(permeation_events)}")


        create_ch_permeation_dict(self.ion_states1, self.permeation_events1)
        create_ch_permeation_dict(self.ion_states2, self.permeation_events2)
        create_ch_permeation_dict(self.ion_states3, self.permeation_events3)
        create_ch_permeation_dict(self.ion_states4, self.permeation_events4)
        create_ch_permeation_dict(self.ion_states5, self.permeation_events5)
        self.rename_all_permeation_ion_ids()
        self.keep_ions_that_pass_all_channels()
        print_channel_results("Channel 1", self.permeation_events1)
        print_channel_results("Channel 2", self.permeation_events2)
        print_channel_results("Channel 3", self.permeation_events3)
        print_channel_results("Channel 4", self.permeation_events4)
        print_channel_results("Channel 5", self.permeation_events5)

        self.plot_residue_distances(self.hbc_diameters, self.results_dir, "hbc_pairs_distances.png", "HBC Residue Pair Distances Over Time", "exit_frame")
        self.plot_residue_distances(self.sf_low_res_diameters, self.results_dir, "sf_pairs_distances.png", "SF Residue Pair Distances Over Time", "start_frame")
         

    def rename_duplicate_ion_ids(self, events):
        """
        Renames ion_id fields in a list of dictionaries by appending _1, _2, etc.
        if the same ion_id appears multiple times.
        """
        from collections import defaultdict

        counter = defaultdict(int)
        renamed_events = []

        for event in events:
            ion_id = event["ion_id"]
            counter[ion_id] += 1
            new_ion_id = f"{ion_id}_{counter[ion_id]}"
            new_event = event.copy()
            new_event["ion_id"] = new_ion_id
            renamed_events.append(new_event)

        return renamed_events


    def fix_permeations(self, residue_clusters):
        def print_fixed_channel_results(ch2_fixed):
            print(f"\nFixed Permeation Events for Channel 2 (after residue clustering):")
            print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
            print("-" * 55)

            ch2_fixed_sorted = sorted(ch2_fixed, key=lambda x: x['start_frame'])

            for event in ch2_fixed_sorted:
                print(f"{int(event['ion_id']):6d} | {int(event['start_frame']):11d} | {int(event['exit_frame']):10d} | {int(event['total_time']):10d}")


            print(f"\nTotal fixed permeation events: {len(ch2_fixed_sorted)}")
    
        ch2_fixed = []
        for ion_id, ion_grouped_frames in residue_clusters.items():
            
            
            sorted_ion_grouped_frames = sorted(ion_grouped_frames, key=lambda x: x['start'])

            if len(sorted_ion_grouped_frames) == 1:
                ch2_fixed.append({
                    "ion_id": ion_id,
                    "start_frame": sorted_ion_grouped_frames[0]["start"],
                    "exit_frame": sorted_ion_grouped_frames[0]["end"],
                    "total_time": sorted_ion_grouped_frames[0]["end"] - sorted_ion_grouped_frames[0]["start"] + 1
                })
                continue
            
            if sorted_ion_grouped_frames == []:
                print(ion_id, ion_grouped_frames)
            # if sorted_ion_grouped_frames[0]["residue"] == "SF":
            #     ch2_start = sorted_ion_grouped_frames[0]["end"]+1
            #     ######################################################
            #     #### make the ch2 json start frame correct too
            #     for item in self.permeation_events2:
            #         if item["ion_id"] == ion_id:
            #             item["start_frame"] = ch2_start  
            #     ######################################################

            else:
                ch2_start = sorted_ion_grouped_frames[0]["start"]

            previous_mean_distance = 0
            for group in sorted_ion_grouped_frames[1:]:
                if group["residue"] == "SF":
                    # if group["end"]-group["start"]+1>10:
                    if group["end"]-group["start"]+1>10:
                        ch2_fixed.append({
                                    "ion_id": ion_id,
                                    "start_frame": ch2_start,
                                    "exit_frame": group["start"]-1,
                                    "total_time": group["start"]-ch2_start
                                })
                        ch2_start = group["end"]+1

                elif group["mean_distance"] > 10.0:
                    if previous_mean_distance < 10.0:
                        ch2_fixed.append({
                            "ion_id": ion_id,
                            "start_frame": ch2_start,
                            "exit_frame": group["start"]-1,
                            "total_time": group["start"]-ch2_start
                        })
                    ch2_start = group["end"]+1
                previous_mean_distance = group["mean_distance"]

            ch2_fixed.append({
                        "ion_id": ion_id,
                        "start_frame": ch2_start,
                        "exit_frame": group["end"],
                        "total_time": group["end"] - ch2_start + 1
                    })
            
        print_fixed_channel_results(ch2_fixed)

        return ch2_fixed
    

    def tracking_ion_distances(
        permeation_events,        # list of dicts: {frame, ions, permeated}
        frame_data,               # list of dicts: {frame, ions: {ion_id: distance}, ...}
        ch2_entry_exit_dict       # dict: {ion_id: [{start_frame, exit_frame}, ...]}
    ):
        results = []

        # Keep only the latest event per ion from ch2_entry_exit_dict
        latest_permeation_bounds = {
            int(ion_id): sorted(ranges, key=lambda x: x['exit_frame'], reverse=True)[0]
            for ion_id, ranges in ch2_entry_exit_dict.items()
        }

        for event in permeation_events:
            target_ion = int(event['permeated'])
            frame = event['frame']

            if target_ion not in latest_permeation_bounds:
                continue  # skip if no ch2 window for this ion

            ch2_window = latest_permeation_bounds[target_ion]
            start_frame = ch2_window['start_frame']
            end_frame = ch2_window['exit_frame']

            for f in frame_data:
                if f['frame'] < start_frame or f['frame'] > end_frame:
                    continue

                if 'ions' not in f:
                    continue

                ion_positions = f['ions']
                if str(target_ion) not in ion_positions:
                    continue

                distances = {}
                for other_ion, other_dist in ion_positions.items():
                    if int(other_ion) != target_ion:
                        # compute absolute distance difference between target and other ion
                        if str(other_ion) in ion_positions:
                            d = abs(ion_positions[str(target_ion)] - ion_positions[str(other_ion)])
                            distances[int(other_ion)] = d

                results.append({
                    "frame": f['frame'],
                    "target_ion": target_ion,
                    "distances": distances
                })

        return results


    def summarize_coexistence_blocks(self, df):
        """
        Takes a DataFrame of coexistence blocks and returns:
        - ion_count
        - num_states
        - total_frames
        - percent_time
        - mean_frames
        """
        # Compute total simulation time from first start to last end
        total_simulation_frames = df["end"].max() - df["start"].min() + 1

        summary = (
            df.groupby("num_ions")
            .agg(
                num_states=("num_ions", "count"),
                total_frames=("duration", "sum"),
                mean_frames=("duration", "mean")
            )
            .reset_index()
            .rename(columns={"num_ions": "ion_count"})
        )

        summary["percent_time"] = (summary["total_frames"] / total_simulation_frames * 100).round(2)
        summary["mean_frames"] = summary["mean_frames"].round(2)
        return summary
    



    def moving_average(self, values, window=50):
        """Simple moving average for smoothing."""
        return np.convolve(values, np.ones(window)/window, mode='same')

    def plot_residue_distances(self, data, output_dir="plots", filename_base="residue_distances",
                            title_base="Residue Pair Distances Over Time", frame_lines="start_frame"):
        # Extract distance data
        frames = [entry["frame"] for entry in data]
        mean_values = [entry["mean"] for entry in data]
        ac_values = [entry["A_C"] for entry in data]
        bd_values = [entry["B_D"] for entry in data]

        # Extract start frames from permeation events (if provided)
        start_frames = []
        if self.permeation_events2:
            start_frames = [entry[frame_lines] for entry in self.permeation_events2]

        os.makedirs(output_dir, exist_ok=True)

        for smooth in [False, True]:
            suffix = "_smoothed" if smooth else "_raw"
            title = title_base + (" (Smoothed)" if smooth else "")
            filename = f"{filename_base}{suffix}.png"
            filepath = os.path.join(output_dir, filename)

            # Apply smoothing if needed
            if smooth:
                mean_plot = self.moving_average(mean_values)
                ac_plot = self.moving_average(ac_values)
                bd_plot = self.moving_average(bd_values)
            else:
                mean_plot = mean_values
                ac_plot = ac_values
                bd_plot = bd_values

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(frames, mean_plot, label="Mean", linewidth=2)
            plt.plot(frames, ac_plot, label="A_C", linestyle="--")
            plt.plot(frames, bd_plot, label="B_D", linestyle=":")

            # Add vertical lines at start frames
            for i, x in enumerate(start_frames):
                if i == 0:
                    plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8, label="Ion leaves SF")
                else:
                    plt.axvline(x=x, linestyle="--", color="green", linewidth=0.8)

            plt.xlabel("Frame")
            plt.ylabel("Distance (Å)")
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"Plot saved to: {filepath}")
