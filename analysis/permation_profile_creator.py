import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import wilcoxon
import json

class FrameAnalyzer:
    def __init__(self, close_residues_results=None, radial_results=None, force_results=None, output_base_dir=None):
        """
        Initialize FrameAnalyzer with optional data.
        """
        self.close_residues_results = close_residues_results
        self.radial_results = radial_results
        self.force_results = force_results
        self.output_base_dir = output_base_dir

    @staticmethod
    def convert_to_pdb_numbering(residue_id: int) -> str:
        """
        Converts a residue ID to a PDB-style numbering.
        """
        if not isinstance(residue_id, int):
            chain_number = int(residue_id)//325
            chain_dict = {0:"A", 1:"B", 2:"C", 3:"D"}
            pdb_number = residue_id-325*chain_number+49
            return f"{pdb_number}.{chain_dict[chain_number]}"
        else:
            return residue_id
    
    def get_last_nth_frame_close_residues(self, event, n=-1, use_pdb_format=False, sort_residues=True):
        """
        Extract close residues at the n-th frame from the end of a single permeation event.

        Parameters:
        - event (dict): single permeation event with 'analysis'
        - n (int): frame index from the end
        - use_pdb_format (bool): apply PDB-style numbering
        - sort_residues (bool): sort residues before joining

        Returns:
        - dict: {frame_number: {ion_id: "res1_res2_..."}}
        """
        frames = sorted(event["analysis"].keys(), key=lambda x: int(x))

        if abs(n) > len(frames):
            raise ValueError(f"Frame index {n} is out of range. Only {len(frames)} frames available.")

        selected_frame_key = frames[n]
        original_data = event["analysis"][selected_frame_key]

        converted_data = {}
        for ion_id, residues in original_data.items():
            if sort_residues:
                residues = sorted(residues, key=lambda r: str(r))

            formatted_residues = [
                self.convert_to_pdb_numbering(res) if use_pdb_format else str(res)
                for res in residues
            ]
            converted_data[ion_id] = "_".join(formatted_residues)

        return {selected_frame_key: converted_data}
    
    def closest_residues_comb_before_permeation(self, n=-1, use_pdb_format=False, sort_residues=True):
        """
        Loop through all permeation events and apply get_last_nth_frame_close_residues.

        Parameters:
        - all_events (list): list of event dicts
        - n (int): frame index from end
        - use_pdb_format (bool): apply PDB-style numbering
        - sort_residues (bool): sort residues before joining

        Returns:
        - list of dicts: each dict is the formatted output per event
        """
        self.output_dir = os.path.join(self.output_base_dir, "closest_residues_comb")
        os.makedirs(self.output_dir, exist_ok=True)

        summary = []
        for i, event in enumerate(self.close_residues_results):
            try:
                frame_data = self.get_last_nth_frame_close_residues(
                    event, n=n, use_pdb_format=use_pdb_format, sort_residues=sort_residues
                )
                summary.append(frame_data)
            except Exception as e:
                print(f"Skipping event {i} due to error: {e}")


        with open(os.path.join(self.output_dir, f"closest_residues_n_{n}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        flat_rows = []
        for event_summary in summary:
            for frame, ion_data in event_summary.items():
                for ion_id, residue_str in ion_data.items():
                    flat_rows.append({
                        "frame": frame,
                        "ion_id": ion_id,
                        "residues": residue_str
                    })

        df = pd.DataFrame(flat_rows)
        df.to_csv(os.path.join(self.output_dir, f"closest_residues_n_{n}.csv"), index=False)

        # return summary



    

    
    def analyze_radial_significance(self):
        """
        Requires self.radial_results to be set.
        """
        if self.radial_results is None:
            raise ValueError("radial_results not set in FrameAnalyzer.")

        self.output_dir = os.path.join(self.output_base_dir, "radial_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        permeation_radials = []
        avg_nonpermeation_radials = []

        for event in tqdm(self.radial_results, desc="Analyzing Radial Significance"):
            ion_id = str(event["permeated_ion"])
            permeation_frame = int(event["frame"])
            analysis = {int(k): v for k, v in event["analysis"].items()}

            frames = sorted(analysis.keys())
            radial_values = [analysis[f] for f in frames]

            if permeation_frame not in analysis:
                continue

            perm_radial = analysis[permeation_frame]
            non_perm_radials = [v for f, v in analysis.items() if f != permeation_frame]
            if not non_perm_radials:
                continue

            permeation_radials.append(perm_radial)
            avg_nonpermeation_radials.append(np.mean(non_perm_radials))

            count_extreme = sum(1 for v in non_perm_radials if v >= perm_radial)
            empirical_p = (count_extreme + 1) / (len(non_perm_radials) + 1)

            results.append({
                "ion_id": ion_id,
                "start_frame": event["start_frame"],
                "permeation_frame": permeation_frame,
                "permeation_radial": round(perm_radial, 3),
                "avg_nonpermeation_radial": round(np.mean(non_perm_radials), 3),
                "empirical_p": round(empirical_p, 3),
                "total_frames": len(frames),
            })

            self._plot_radial_traces(ion_id, frames, radial_values, perm_radial, permeation_frame, self.output_dir)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "radial_significance_results.csv"), index=False)

        self._plot_global_radial_distribution(permeation_radials, avg_nonpermeation_radials, self.output_dir)

        stat, p_value = wilcoxon(permeation_radials, avg_nonpermeation_radials)
        with open(os.path.join(self.output_dir, "wilcoxon_test_results.txt"), "w") as f:
            f.write(f"Wilcoxon signed-rank test result:\nStatistic = {stat}\nP-value = {p_value}\n")

    def analyze_cosine_significance(self, force_dir):
        """
        Requires self.force_results to be set.
        """
        if self.force_results is None:
            raise ValueError("force_results not set in FrameAnalyzer.")

        output_dir = os.path.join(force_dir, "cosine_analysis")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        permeation_cosines = []
        avg_nonpermeation_cosines = []

        for event in tqdm(self.force_results, desc="Analyzing Cosine Significance"):
            ion_id = str(event["permeated_ion"])
            permeation_frame = event["frame"]
            analysis = event["analysis"]

            cosines = []
            frames = []
            permeation_cosine = None

            for frame_data in sorted(analysis.values(), key=lambda x: x["frame"]):
                frame = frame_data["frame"]
                cosine = frame_data["cosine_ionic_motion"]
                frames.append(frame)
                cosines.append(cosine)
                if frame == permeation_frame:
                    permeation_cosine = cosine

            if permeation_cosine is None or len(cosines) <= 1:
                continue

            permeation_cosines.append(permeation_cosine)
            avg_nonpermeation_cosines.append(np.mean([c for f, c in zip(frames, cosines) if f != permeation_frame]))

            count_extreme = sum(1 for c in cosines if c >= permeation_cosine)
            empirical_p = (count_extreme + 1) / (len(cosines) + 1)

            results.append({
                "ion_id": ion_id,
                "start_frame": event["start_frame"],
                "permeation_frame": permeation_frame,
                "permeation_cosine": round(permeation_cosine,3),
                "avg_nonpermeation_cosine": round(np.mean([c for f, c in zip(frames, cosines) if f != permeation_frame]),3),
                "empirical_p": round(empirical_p,3),
                "total_cosines": len(cosines),
            })

            self._plot_cosine_traces(ion_id, frames, cosines, permeation_cosine, permeation_frame, output_dir)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "cosine_significance_results.csv"), index=False)

        self._plot_global_cosine_distribution(permeation_cosines, avg_nonpermeation_cosines, output_dir)

        stat, p_value = wilcoxon(permeation_cosines, avg_nonpermeation_cosines)
        with open(os.path.join(output_dir, "wilcoxon_test_results.txt"), "w") as f:
            f.write(f"Wilcoxon signed-rank test result:\nStatistic = {stat}\nP-value = {p_value}\n")

        simplified_df = pd.DataFrame([
            {
                "ion_id": row["ion_id"],
                "avg_nonpermeation_cosine": round(row["avg_nonpermeation_cosine"],2),
                "permeation_cosine": round(row["permeation_cosine"],2)
            } for row in results
        ])
        simplified_df.to_csv(os.path.join(output_dir, "cosine_summary_table.csv"), index=False)

    # ---- Helper plotting methods ----
    def _plot_radial_traces(self, ion_id, frames, radials, perm_radial, perm_frame, output_dir):
        os.makedirs(f"{output_dir}/radial_trace_ion", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(frames, radials, label="Radial Distance", color='blue')
        plt.axvline(perm_frame, color='red', linestyle='--', label='Permeation Frame')
        plt.axhline(perm_radial, color='green', linestyle='--', label='Permeation Radial')
        plt.title(f"Ion {ion_id} – Radial Distance Trace")
        plt.xlabel("Frame")
        plt.ylabel("Radial Distance (Å)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radial_trace_ion/{ion_id}.png")
        plt.close()

        os.makedirs(f"{output_dir}/radial_histogram", exist_ok=True)
        plt.figure(figsize=(6, 4))
        non_perm = [r for r in radials if r != perm_radial]
        plt.hist(non_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(perm_radial, color='red', linestyle='--', linewidth=2, label='Permeation Radial')
        plt.title(f"Ion {ion_id} – Radial Histogram – Permeation: {perm_radial:.2f}")
        plt.xlabel("Radial Distance (Å)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radial_histogram/{ion_id}.png")
        plt.close()

    def _plot_global_radial_distribution(self, permeation_radials, avg_nonperm_radials, output_dir):
        plt.figure()
        plt.hist(avg_nonperm_radials, bins=20, alpha=0.7, label='Avg Non-Permeation Radials')
        plt.axvline(np.mean(permeation_radials), color='red', linestyle='--', label='Mean Permeation Radial')
        plt.xlabel("Radial Distance (Å)")
        plt.ylabel("Frequency")
        plt.title("Radial Distance Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "radial_distribution_histogram.png"))
        plt.close()

    def _plot_cosine_traces(self, ion_id, frames, cosines, perm_cosine, perm_frame, output_dir):
        os.makedirs(f"{output_dir}/cosine_trace_ion", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(frames, cosines, label="Cosine Ionic–Motion", color='blue')
        plt.axvline(perm_frame, color='red', linestyle='--', label='Permeation Frame')
        plt.axhline(perm_cosine, color='green', linestyle='--', label='Permeation Cosine')
        plt.title(f"Ion {ion_id} – Cosine Trajectory")
        plt.xlabel("Frame")
        plt.ylabel("Cosine(Ionic Force, Motion)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cosine_trace_ion/{ion_id}.png")
        plt.close()

        os.makedirs(f"{output_dir}/cosine_histogram", exist_ok=True)
        plt.figure(figsize=(6, 4))
        non_perm = [c for c in cosines if c != perm_cosine]
        plt.hist(non_perm, bins=50, alpha=0.7, color='gray', edgecolor='black')
        plt.axvline(perm_cosine, color='red', linestyle='--', linewidth=2, label='Permeation Cosine')
        plt.title(f"Ion {ion_id} – Cosine Histogram – Permeation: {perm_cosine:.2f}")
        plt.xlabel("Cosine Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cosine_histogram/{ion_id}.png")
        plt.close()


    def _plot_global_cosine_distribution(self, permeation_cosines, avg_nonperm_cosines, output_dir):
        plt.figure()
        plt.hist(avg_nonperm_cosines, bins=20, alpha=0.7, label='Avg Non-Permeation Cosines')
        plt.axvline(np.mean(permeation_cosines), color='red', linestyle='--', linewidth=2, label='Mean Permeation Cosine')
        plt.xlabel("Cosine Value")
        plt.ylabel("Frequency")
        plt.title("Cosine Distribution: Non-Permeation vs Permeation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cosine_distribution_histogram.png"))
        plt.close()
