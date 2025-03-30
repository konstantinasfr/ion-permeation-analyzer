import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

class IonPermeationAnalysis:
    def __init__(self, universe, ion_selection, start_frame, end_frame, channel1, channel2, channel3):
        self.u = universe
        self.ion_selection = ion_selection
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.channel1 = channel1
        self.channel2 = channel2
        self.channel3 = channel3
        self.ion_states1 = {}
        self.ion_states2 = {}
        self.ion_states3 = {}
        self.permeation_events1 = []
        self.permeation_events2 = []
        self.permeation_events3 = []
        self.ions = self.u.select_atoms(self.ion_selection)

    def _check_ion_position(self, ion_id, ion_pos, channel, states, events, frame, keep_first_permeation, keep_first_insertion):
        ion_vec = ion_pos - channel.channel_center
        ion_z = np.dot(ion_vec, channel.channel_axis)

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

        if ion_z > lower_z and states[ion_id]['upper_flag'] == 1:
            if states[ion_id]['lower_flag'] == 0:
                states[ion_id]['lower_flag'] = 1
                if keep_first_permeation:
                    if states[ion_id]['lower_flag_frame'] == 0:
                        states[ion_id]['lower_flag_frame'] = frame
                else:
                    states[ion_id]['lower_flag_frame'] = frame


        if not in_cylinder and states[ion_id]['upper_flag'] == 1 and states[ion_id]['lower_flag'] == 0:
            states[ion_id]['upper_flag'] = 0

        states[ion_id]['prev_ion_z'] = ion_z

        # if ion_id == 1469 and keep_first_permeation:
        #     print(frame, in_cylinder, ion_z, lower_z)

    def run_analysis(self):
        print("Starting analysis...")

        for ts in tqdm(self.u.trajectory[self.start_frame:self.end_frame],
                       total=(self.end_frame - self.start_frame),
                       desc="Processing Frames", unit="frame"):
            # if ts.frame>self.start_frame+1:
            #     # print(ts.frame, self.ion_states1[2433])
            #     print(ts.frame, self.ion_states1[1469])
            #     # print(self.channel1.lower_center)
            #     # print(self.channel2.upper_center)
            self.channel1.compute_geometry(1)
            self.channel2.compute_geometry(2)
            self.channel3.compute_geometry(3)

            for ion in self.ions:
                ion_id = ion.resid
                pos = ion.position
                self._check_ion_position(ion_id, pos, self.channel1, self.ion_states1, self.permeation_events1, ts.frame, True, False)
                self._check_ion_position(ion_id, pos, self.channel2, self.ion_states2, self.permeation_events2, ts.frame, False, True)
                self._check_ion_position(ion_id, pos, self.channel3, self.ion_states3, self.permeation_events3, ts.frame, False, False)

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