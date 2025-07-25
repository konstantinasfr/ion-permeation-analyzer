import json
import numpy as np
import pandas as pd

def convert_to_pdb_numbering(residue_id, channel_type):
    """
    Converts a residue ID to a PDB-style numbering.
    """
    if channel_type == "G4":
        residues_per_chain = 325
        offset = 49
    elif channel_type == "G2" or channel_type == "G2_FD":
        residues_per_chain = 328
        offset = 54
    elif channel_type == "G12":
        residues_per_chain = 325
        offset = 53

    amino_acid_names = {152:"E",
                       184:"N",
                       141:"E",
                       173:"D",
                       }
    if channel_type == "G2_FD":
            amino_acid_names = {152:"E",
                       184:"N",
                       141:"E",
                       173:"D",
                       }
            
    if residue_id != "SF":
        residue_id = int(residue_id)
        chain_number = int(residue_id)//residues_per_chain
        if channel_type == "G2" or channel_type == "G2_FD":
            chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
        elif channel_type == "G12":
            chain_dict = {0:"D", 1:"C", 2:"B", 3:"A"}
        pdb_number = residue_id-residues_per_chain*chain_number+offset
        if channel_type == "G12" and residue_id<=325:
            pdb_number = residue_id+42
        if channel_type == "G2_FD" and pdb_number==184 and chain_number==0:
            return "D184.A"
        else:
            return f"{amino_acid_names[pdb_number]}{pdb_number}.{chain_dict[chain_number]}"
    else:
        return "SF"
        

def get_first_ion_id_part(ion_id):
    return ion_id.split('_')[0]
