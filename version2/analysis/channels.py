import MDAnalysis as mda
import numpy as np

class Channel:
    def __init__(self, universe, upper_gate_residues, lower_gate_residues, num, radius=8.0):
        self.u = universe
        self.upper_gate_residues = upper_gate_residues
        self.lower_gate_residues = lower_gate_residues
        self.radius = radius
        self.channel_axis = None
        self.channel_center = None
        self.channel_length = None
        self.upper_center = None
        self.lower_center = None
        self.channel_number = num

    # def compute_geometry(self, gate_num):
    #     """
    #     1. Compute the channel axis using CA atoms of the upper and lower gate residues.
    #     2. Then, for each gate residue, find the atom with the lowest projection along that axis.
    #     3. Compute upper and lower centers based on those atoms.
    #     """

    #     # === Step 1: Get CA atoms to define the channel axis ===
    #     upper_ca_indices = []
    #     for resid in self.upper_gate_residues:
    #         residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #         upper_ca_indices.append(residue_atoms[0].index)

    #     lower_ca_indices = []
    #     for resid in self.lower_gate_residues:
    #         residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #         lower_ca_indices.append(residue_atoms[0].index)

    #     # Compute channel axis from CA atoms
    #     upper_ca_atoms = self.u.atoms[upper_ca_indices]
    #     lower_ca_atoms = self.u.atoms[lower_ca_indices]
    #     upper_ca_center = upper_ca_atoms.center_of_mass()
    #     lower_ca_center = lower_ca_atoms.center_of_mass()

    #     self.channel_vector = lower_ca_center - upper_ca_center
    #     self.channel_length = np.linalg.norm(self.channel_vector)
    #     self.channel_axis = self.channel_vector / self.channel_length
    #     self.channel_center = (upper_ca_center + lower_ca_center) / 2

    #     # === Step 2: Find lowest-projection atom per residue (relative to its own CA) ===

    #     # ---- Upper gate ----
    #     upper_indices = []
    #     for resid in self.upper_gate_residues:
    #         residue_atoms = self.u.select_atoms(f"resid {resid}")
    #         ca_center = self.u.select_atoms(f"resid {resid} and name CA").center_of_mass()
    #         coords = residue_atoms.positions
    #         projections = np.dot(coords - ca_center, self.channel_axis)
    #         lowest_index = np.argmin(projections)
    #         residue_atoms = residue_atoms[[lowest_index]]
    #         upper_indices.append(residue_atoms[0].index)
    #     upper_atoms = self.u.atoms[upper_indices]
    #     self.upper_center = upper_atoms.center_of_mass()

    #     # ---- Lower gate ----
    #     lower_indices = []
    #     for resid in self.lower_gate_residues:
    #         residue_atoms = self.u.select_atoms(f"resid {resid}")
    #         ca_center = self.u.select_atoms(f"resid {resid} and name CA").center_of_mass()
    #         coords = residue_atoms.positions
    #         projections = np.dot(coords - ca_center, self.channel_axis)
    #         lowest_index = np.argmin(projections)
    #         residue_atoms = residue_atoms[[lowest_index]]
    #         lower_indices.append(residue_atoms[0].index)
    #     lower_atoms = self.u.atoms[lower_indices]
    #     self.lower_center = lower_atoms.center_of_mass()

    #     # === Step 3: Final center of the whole channel
    #     self.channel_center = (self.upper_center + self.lower_center) / 2

    def compute_geometry(self, gate_num):
        # offset = 1.33  # adjust this value as needed (in Ångströms)
        
        atom_indices = []
        for resid in self.upper_gate_residues:
            if gate_num in [1, 3, 4]:
                residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
            else:
                residue_atoms = self.u.select_atoms(f"resid {resid}")
                coords = residue_atoms.positions
                sorted_indices = coords[:, 2].argsort()
                upper_index = sorted_indices[0]
                residue_atoms = residue_atoms[[upper_index]]
            atom_indices.append(residue_atoms[0].index)

        upper_atoms = self.u.atoms[atom_indices]
        self.upper_center = upper_atoms.center_of_mass()

        
        # in channel 5 by choosing ca for upper gate and min for lower gate i maximize the distance between upper and lower gates
        pos = 0  # use pos if you want lowest or second-lowest atom; adjust as needed
        atom_indices = []
        for resid in self.lower_gate_residues:
            if gate_num in [2, 3, 4]:
                residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
            else:
                residue_atoms = self.u.select_atoms(f"resid {resid}")
                coords = residue_atoms.positions
                sorted_indices = coords[:, 2].argsort()
                lowest_index = sorted_indices[pos]
                residue_atoms = residue_atoms[[lowest_index]]
            atom_indices.append(residue_atoms[0].index)

        lowest_atoms = self.u.atoms[atom_indices]
        self.lower_center = lowest_atoms.center_of_mass()
        # if gate_num == 2:
        #     print(gate_num, self.upper_center)
        # if gate_num == 1:
        #     print(gate_num, self.lower_center)

        # if gate_num in [5]:
        #     self.lower_center[2] -= offset

        self.channel_vector = self.lower_center - self.upper_center
        self.channel_length = np.linalg.norm(self.channel_vector)
        self.channel_axis = self.channel_vector / self.channel_length
        self.channel_center = (self.upper_center + self.lower_center) / 2






    def is_within_cylinder(self, pos):
        rel_vector = pos - self.channel_center
        proj = np.dot(rel_vector, self.channel_axis) * self.channel_axis
        radial = rel_vector - proj
        radial_dist = np.linalg.norm(radial)
        axial_pos = np.dot(rel_vector, self.channel_axis)
        lower_z = np.dot(self.lower_center - self.channel_center, self.channel_axis)
        upper_z = np.dot(self.upper_center - self.channel_center, self.channel_axis)
        return radial_dist <= self.radius and upper_z <= axial_pos <= lower_z
