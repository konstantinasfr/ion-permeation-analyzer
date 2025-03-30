import MDAnalysis as mda
import numpy as np

class Channel:
    def __init__(self, universe, upper_gate_residues, lower_gate_residues, radius=8.0):
        self.u = universe
        self.upper_gate_residues = upper_gate_residues
        self.lower_gate_residues = lower_gate_residues
        self.radius = radius
        self.channel_axis = None
        self.channel_center = None
        self.channel_length = None
        self.upper_center = None
        self.lower_center = None

    # def compute_geometry(self, gate_num):
    #     if gate_num == 1:
    #         upper_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.upper_gate_residues))}")
    #         self.upper_center = upper_sel.center_of_mass()
    #         # atom_indices = []
    #         # for resid in self.upper_gate_residues:
    #         #     residue_atoms = self.u.select_atoms(f"resid {resid}")
    #         #     coords = residue_atoms.positions
    #         #     sorted_indices = coords[:, 2].argsort()  # sort Z-values
    #         #     upper_index = sorted_indices[-1]  # second from the end
    #         #     atom_indices.append(residue_atoms[upper_index].index)

    #         # upper_atoms = self.u.atoms[atom_indices]
    #         # self.upper_center = upper_atoms.center_of_mass()

    #     if gate_num == 2:
    #         atom_indices = []
    #         for resid in self.upper_gate_residues:
    #             residue_atoms = self.u.select_atoms(f"resid {resid}")
    #             coords = residue_atoms.positions
    #             sorted_indices = coords[:, 2].argsort()  # sort Z-values
    #             upper_index = sorted_indices[2]  # second from the end
    #             atom_indices.append(residue_atoms[upper_index].index)

    #         upper_atoms = self.u.atoms[atom_indices]
    #         self.upper_center = upper_atoms.center_of_mass()
        
    #     if gate_num == 3:
    #         atom_indices = []
    #         for resid in self.upper_gate_residues:
    #             residue_atoms = self.u.select_atoms(f"resid {resid}")
    #             coords = residue_atoms.positions
    #             upper_index = coords[:, 2].argmin()
    #             atom_indices.append(residue_atoms[upper_index].index)

    #         upper_atoms = self.u.atoms[atom_indices]
    #         self.upper_center = upper_atoms.center_of_mass()

    #     if gate_num  == 3:
    #         # atom_indices = []
    #         # for resid in self.lower_gate_residues:
    #         #     residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #         #     coords = residue_atoms.positions
    #         #     sorted_indices = coords[:, 2].argsort()  # sort Z-values
    #         #     lowest_index = sorted_indices[0]  # second from the end
    #         #     atom_indices.append(residue_atoms[lowest_index].index)

    #         # lowest_atoms = self.u.atoms[atom_indices]
    #         # self.lower_center = lowest_atoms.center_of_mass()
    #         atom_indices = []
    #         for resid in self.lower_gate_residues:
    #             residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #             if len(residue_atoms) > 0:
    #                 atom_indices.append(residue_atoms[0].index)

    #         lowest_atoms = self.u.atoms[atom_indices]
    #         self.lower_center = lowest_atoms.center_of_mass()


    #     elif gate_num == 2:
    #         atom_indices = []
    #         for resid in self.lower_gate_residues:
    #             residue_atoms = self.u.select_atoms(f"resid {resid}")
    #             coords = residue_atoms.positions
    #             lowest_index = coords[:, 2].argmin()
    #             atom_indices.append(residue_atoms[lowest_index].index)

    #         lowest_atoms = self.u.atoms[atom_indices]
    #         self.lower_center = lowest_atoms.center_of_mass()

    #     elif gate_num == 1:
    #         atom_indices = []
    #         for resid in self.lower_gate_residues:
    #             residue_atoms = self.u.select_atoms(f"resid {resid}")
    #             coords = residue_atoms.positions
    #             sorted_indices = coords[:, 2].argsort()  # sort Z-values
    #             lowest_index = sorted_indices[2]  # second from the end
    #             atom_indices.append(residue_atoms[lowest_index].index)

    #         lowest_atoms = self.u.atoms[atom_indices]
    #         self.lower_center = lowest_atoms.center_of_mass()

    #     self.channel_vector = self.lower_center - self.upper_center
    #     self.channel_length = np.linalg.norm(self.channel_vector)
    #     self.channel_axis = self.channel_vector / self.channel_length
    #     self.channel_center = (self.upper_center + self.lower_center) / 2

    def compute_geometry(self, gate_num):
        offset = 1.2  # adjust this value as needed (in Ångströms)
        if gate_num == 1:
            upper_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.upper_gate_residues))}")
            self.upper_center = upper_sel.center_of_mass()
        else:

            atom_indices = []
            for resid in self.upper_gate_residues:
                residue_atoms = self.u.select_atoms(f"resid {resid}")
                coords = residue_atoms.positions
                upper_index = coords[:, 2].argmin()
                atom_indices.append(residue_atoms[upper_index].index)

            upper_atoms = self.u.atoms[atom_indices]
            self.upper_center = upper_atoms.center_of_mass()
        
        if gate_num == 2:
            self.upper_center[2] -= offset

        if gate_num == 1:
            pos = 0
        if gate_num == 2:
            pos = 0
        if gate_num == 3:
            pos = 0

        if gate_num  == 3:
            atom_indices = []
            for resid in self.lower_gate_residues:
                residue_atoms = self.u.select_atoms(f"resid {resid} and name CA")
                if len(residue_atoms) > 0:
                    atom_indices.append(residue_atoms[0].index)

            lowest_atoms = self.u.atoms[atom_indices]
            self.lower_center = lowest_atoms.center_of_mass()
        else:
            atom_indices = []
            for resid in self.lower_gate_residues:
                residue_atoms = self.u.select_atoms(f"resid {resid}")
                coords = residue_atoms.positions
                sorted_indices = coords[:, 2].argsort()  # sort Z-values
                lowest_index = sorted_indices[pos]  # second from the end
                atom_indices.append(residue_atoms[lowest_index].index)

            lowest_atoms = self.u.atoms[atom_indices]
            self.lower_center = lowest_atoms.center_of_mass()
            self.lower_center[2] -= offset

        self.channel_vector = self.lower_center - self.upper_center
        self.channel_length = np.linalg.norm(self.channel_vector)
        self.channel_axis = self.channel_vector / self.channel_length
        self.channel_center = (self.upper_center + self.lower_center) / 2


    # def compute_geometry(self, gate_num):
    #     # Calculate upper gate center from CA atoms
    #     upper_indices = []
    #     for resid in self.upper_gate_residues:
    #         atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #         if len(atoms) > 0:
    #             upper_indices.append(atoms[0].index)
    #     upper_atoms = self.u.atoms[upper_indices]
    #     self.upper_center = upper_atoms.center_of_mass()

    #     # Calculate lower gate center from CA atoms
    #     lower_indices = []
    #     for resid in self.lower_gate_residues:
    #         atoms = self.u.select_atoms(f"resid {resid} and name CA")
    #         if len(atoms) > 0:
    #             lower_indices.append(atoms[0].index)
    #     lower_atoms = self.u.atoms[lower_indices]
    #     self.lower_center = lower_atoms.center_of_mass()

    #     # Channel geometry
    #     self.channel_vector = self.lower_center - self.upper_center
    #     self.channel_length = np.linalg.norm(self.channel_vector)
    #     self.channel_axis = self.channel_vector / self.channel_length
    #     self.channel_center = (self.upper_center + self.lower_center) / 2



    def is_within_cylinder(self, pos):
        rel_vector = pos - self.channel_center
        proj = np.dot(rel_vector, self.channel_axis) * self.channel_axis
        radial = rel_vector - proj
        radial_dist = np.linalg.norm(radial)
        axial_pos = np.dot(rel_vector, self.channel_axis)
        return radial_dist <= self.radius and abs(axial_pos) <= self.channel_length / 2