import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

class IonPermeationAnalysis:
    def __init__(self, universe, ion_selection, start_frame, end_frame, channel1, channel2, channel3, channel4, channel5,
                 hbc_residues, hbc_diagonal_pairs, sf_low_res_residues, sf_low_res_diagonal_pairs):
        
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
                            # Select the five key atoms in the ASN side chain that contribute to electrostatic interactions
                            asn_atoms = self.u.select_atoms(
                                f"resid {resid} and name CG OD1 ND2 HD21 HD22"
                            )

                            # Ensure that all 5 atoms are present (sometimes an atom might be missing in a corrupted frame)
                            if len(asn_atoms) == 5:
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

 #       if ion_id == 2231:
 #           print(f"'Frame: {frame}, channel_num: {channel_number}, upper_flag: {states[ion_id]['upper_flag']}, lower_flag: {states[ion_id]['lower_flag']}")

        if frame == self.end_frame and states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 0:
            states[ion_id]['lower_flag'] = 1
            states[ion_id]['lower_flag_frame'] = frame


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
                self._check_ion_position(ion_id, pos, self.channel1, self.ion_states1, self.permeation_events1, ts.frame, True, False)
                self._check_ion_position(ion_id, pos, self.channel2, self.ion_states2, self.permeation_events2, ts.frame, False, True)
                self._check_ion_position(ion_id, pos, self.channel3, self.ion_states3, self.permeation_events3, ts.frame, False, False)
                self._check_ion_position(ion_id, pos, self.channel4, self.ion_states4, self.permeation_events4, ts.frame, False, False)
                self._check_ion_position(ion_id, pos, self.channel5, self.ion_states5, self.permeation_events5, ts.frame, False, False)

    def print_results(self):
        def print_channel_results(channel_name, ion_states, permeation_events):
            print(f"\nFinal Permeation Events for {channel_name} (1,1 Flags):")
            print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
            print("-" * 55)

            for ion_id, state in ion_states.items():
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

            for event in permeation_events:
                print(f"{event['ion_id']:6d} | {event['start_frame']:11d} | {event['exit_frame']:10d} | {event['total_time']:10d}")

            print(f"\nTotal forward permeation events: {len(permeation_events)}")

        print_channel_results("Channel 1", self.ion_states1, self.permeation_events1)
        print_channel_results("Channel 2", self.ion_states2, self.permeation_events2)
        print_channel_results("Channel 3", self.ion_states3, self.permeation_events3)
        print_channel_results("Channel 4", self.ion_states4, self.permeation_events4)
        print_channel_results("Channel 5", self.ion_states5, self.permeation_events5)





    def fix_permeations(self, residue_clusters):
        def print_fixed_channel_results(ch2_fixed):
            print(f"\nFixed Permeation Events for Channel 2 (after residue clustering):")
            print("Ion ID | Start Frame | Exit Frame | Total Time (frames)")
            print("-" * 55)

            ch2_fixed_sorted = sorted(ch2_fixed, key=lambda x: x['start_frame'])

            for event in ch2_fixed_sorted:
                print(f"{event['ion_id']:6d} | {event['start_frame']:11d} | {event['exit_frame']:10d} | {event['total_time']:10d}")

            print(f"\nTotal fixed permeation events: {len(ch2_fixed_sorted)}")
    
        ch2_fixed = []
        for ion_id, ion_grouped_frames in residue_clusters.items():
            
            
            sorted_ion_grouped_frames = sorted(ion_grouped_frames, key=lambda x: x['start'])
            if sorted_ion_grouped_frames == []:
                print(ion_id, ion_grouped_frames)
            if sorted_ion_grouped_frames[0]["residue"] == "SF":
                ch2_start = sorted_ion_grouped_frames[0]["end"]+1
            else:
                ch2_start = sorted_ion_grouped_frames[0]["start"]

            for group in sorted_ion_grouped_frames[1:]:
                if group["residue"] == "SF":
                    if group["end"]-group["start"]+1>3:
                        ch2_fixed.append({
                                    "ion_id": ion_id,
                                    "start_frame": ch2_start,
                                    "exit_frame": group["start"]-1,
                                    "total_time": group["start"]-ch2_start
                                })
                        ch2_start = group["end"]+1

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
