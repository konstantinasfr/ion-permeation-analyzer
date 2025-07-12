import MDAnalysis as mda
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def analyze_system_relaxation(top_path, traj_path, output_dir, selection="protein and backbone", ps_per_frame=None):
    """
    Analyze MD trajectory to determine when the system is relaxed based on RMSD.

    Args:
        top_path (str): Path to topology file (e.g., .prmtop, .psf, .pdb).
        traj_path (str): Path to trajectory file (e.g., .nc, .dcd, .xtc).
        output_dir (str): Directory to save the RMSD plot.
        selection (str): Atom selection for RMSD calculation (default: protein backbone).
        ps_per_frame (float): Picoseconds per frame. If provided, x-axis will be in ns.

    Returns:
        int: Frame number after which the system is considered relaxed.
    """
    # Load the universe
    u = mda.Universe(top_path, traj_path)

    # Select atoms for RMSD
    ref_atoms = u.select_atoms(selection)
    mobile_atoms = u.select_atoms(selection)

    # Use first frame as reference
    reference = ref_atoms.positions.copy()

    rmsd_values = []
    time_values = []

    print("🔄 Calculating RMSD...")
    for ts in u.trajectory:
        # Align mobile to reference
        align.alignto(mobile_atoms, ref_atoms)

        # Compute RMSD
        rmsd = np.sqrt(np.mean(np.sum((mobile_atoms.positions - reference)**2, axis=1)))
        rmsd_values.append(rmsd)

        # Time (if provided)
        if ps_per_frame:
            time_values.append((ts.frame * ps_per_frame) / 1000.0)  # convert ps to ns
        else:
            time_values.append(ts.frame)

    # Detect relaxed frame: where RMSD plateaus
    window = 50  # Moving average window
    smoothed = np.convolve(rmsd_values, np.ones(window)/window, mode='valid')
    gradient = np.abs(np.gradient(smoothed))
    threshold = 0.002  # Smaller gradient threshold = more strict

    relaxed_index = next((i for i, g in enumerate(gradient) if g < threshold), None)

    if relaxed_index is None:
        print("⚠️ No clear plateau detected. System may not have equilibrated.")
        relaxed_frame = 0
        relaxed_time = None
    else:
        relaxed_frame = relaxed_index + window  # Offset due to moving average
        relaxed_time = time_values[relaxed_frame]
        print(f"✅ System appears relaxed after frame {relaxed_frame}.")

    # Create output directory if not exist
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    plt.figure(figsize=(10, 6))

    # Larger fonts for everything
    plt.rcParams.update({
        "font.size": 16,         # Default text size
        "axes.titlesize": 18,    # Title size
        "axes.labelsize": 18,    # X and Y labels
        "xtick.labelsize": 14,   # X tick labels
        "ytick.labelsize": 14,   # Y tick labels
        "legend.fontsize": 16    # Legend text
    })

    plt.plot(time_values, rmsd_values, color="blue", label="Backbone RMSD")

    if relaxed_index is not None:
        plt.axvline(
            x=relaxed_time, color="red", linestyle="--",
            label=f"Relaxed at frame {relaxed_frame}"
        )

    plt.xlabel("Time (ns)" if ps_per_frame else "Frame")
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD vs Time" if ps_per_frame else "RMSD vs Frame")
    plt.legend()
    plt.grid(True)

    # Tighten layout to remove extra white space
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, "rmsd_vs_frame.png")
    plt.savefig(plot_path, dpi=300)  # dpi=300 for high quality
    plt.close()
    print(f"📈 RMSD plot saved to: {plot_path}")


    return relaxed_frame
