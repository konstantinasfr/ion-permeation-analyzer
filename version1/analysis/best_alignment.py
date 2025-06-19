import pandas as pd
import numpy as np
from itertools import combinations
from analysis.converter import convert_to_pdb_numbering

def get_best_aligned_residue_combo(df, frame_number, max_combo_size=None, source_filter=None, channel_type="G2"):
    """
    Find the combination of sources whose summed force vector aligns best with the motion vector.

    Parameters:
        df (pd.DataFrame): Full force/motion DataFrame.
        frame_number (int): The frame to analyze.
        max_combo_size (int or None): Maximum combo size. If None, tries all sizes.
        source_filter (str or None): 'residue', 'pip', etc., or None to include all.
        channel_type (str): 'G2', 'G4', or 'G12'.

    Returns:
        dict with best combination info.
    """

    def cosine_similarity(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    # Filter dataframe
    frame_df = df[df['frame'] == frame_number].copy()
    if source_filter:
        frame_df = frame_df[frame_df['source_type'] == source_filter]
    if frame_df.empty:
        return {"error": f"No data for frame {frame_number} with source_type '{source_filter}'."}

    # Motion vector
    motion_vec = np.array([
        frame_df["motion_vec_x"].iloc[0],
        frame_df["motion_vec_y"].iloc[0],
        frame_df["motion_vec_z"].iloc[0]
    ])

    # Collect candidate force vectors
    sources = list(frame_df[['resid', 'Fx', 'Fy', 'Fz', 'resname']].itertuples(index=False, name=None))
    max_size = max_combo_size if max_combo_size else len(sources)

    best_cosine = -1
    best_combo = None
    best_vector = None

    for r in range(1, max_size + 1):
        for combo in combinations(sources, r):
            vector_sum = np.sum([[x[1], x[2], x[3]] for x in combo], axis=0)
            cos_sim = cosine_similarity(vector_sum, motion_vec)

            if cos_sim > best_cosine:
                best_cosine = cos_sim
                best_vector = vector_sum

                best_combo = []
                best_combo_numbers = [x[0] for x in combo]
                for resid, _, _, _, resname in combo:
                    try:
                        value = int(resid)
                        if value < 1300 and resname != "PIP":
                            label = convert_to_pdb_numbering(value, channel_type)
                        else:
                            label = f"{resname}_{value}"
                        best_combo.append(label)
                    except Exception:
                        best_combo.append(str(resid))

    return {
        "frame": frame_number,
        "best_cosine_similarity": best_cosine,
        "residue_combination": best_combo,
        "residue_combination_numb":best_combo_numbers,
        "vector_sum": best_vector,
        "magnitude": np.linalg.norm(best_vector)
    }

import os
import pandas as pd
from collections import defaultdict

import os
import json
from tqdm import tqdm
import os


def run_best_combo_per_ion_from_json(csv_folder, frame_data, output_csv, source_filter="residue", channel_type="G2"):
    """
    Runs best-aligned force combo analysis for each ion using start_frame from a JSON list.

    Parameters:
    - csv_folder: folder with CSVs like 2223_1.csv
    - json_path: path to the JSON file with ion_id and start_frame
    - source_filter: 'residue', 'pip', etc.
    - channel_type: 'G2', 'G4', or 'G12'

    Returns:
    - List of result dictionaries
    """
    # Load JSON into a dict for quick lookup
    # with open(json_path, "r") as f:
    #     frame_data = json.load(f)
    frame_map = {entry["ion_id"]: entry["start_frame"]-1 for entry in frame_data}

    results = []
    for fname in tqdm(os.listdir(csv_folder), desc="Processing CSV files"):
        if not fname.endswith(".csv"):
            continue

        ion_id = fname[:-4]  # remove .csv
        if ion_id not in frame_map:
            print(f"⚠️ Skipping {ion_id}, no start_frame in JSON.")
            continue

        frame_number = frame_map[ion_id]
        filepath = os.path.join(csv_folder, fname)
        df = pd.read_csv(filepath)

        result = get_best_aligned_residue_combo(
            df,
            frame_number=frame_number,
            source_filter=source_filter,
            channel_type=channel_type
        )
        result["ion_id"] = ion_id
        result["start_frame"] = frame_number
        results.append(result)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_csv}/best_force_alignment_results.csv", index=False)
    print(f"✅ Saved results to {output_csv}")

    return results_df