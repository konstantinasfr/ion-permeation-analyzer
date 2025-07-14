import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

def analyze_gaussian_means(csv_path, z_threshold_moderate=2, z_threshold_strong=3):
    """
    Analyze Gaussian means from a CSV file, compute Z-scores, flag outliers,
    check for Gaussianity, plot results, and save everything in a folder.
    """
    # === Create output folder ===
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_folder = os.path.join(csv_dir, csv_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n📁 Results will be saved in: {output_folder}")

    # === Load data ===
    df = pd.read_csv(csv_path)

    # === Clean column names ===
    df.columns = [col.strip() for col in df.columns]  # remove leading/trailing spaces

    # Automatically detect the column for means and stddev
    mean_col = next((col for col in df.columns if "Gaussian_Mean" in col), None)
    std_col = next((col for col in df.columns if "Gaussian_StdDev" in col), None)
    label_col = next((col for col in df.columns if "Simulation" in col or "Label" in col), None)

    if mean_col is None or std_col is None:
        raise ValueError("❌ Could not find 'Gaussian Mean' and/or 'Gaussian StdDev' columns in the CSV.")

    means = df[mean_col]
    std_devs = df[std_col]

    # Compute overall mean and std deviation of means
    overall_mean = means.mean()
    overall_std = means.std()

    # Compute Z-scores
    df["Z-score"] = (means - overall_mean) / overall_std

    # Flag outliers
    df["Outlier"] = "No"
    df.loc[df["Z-score"].abs() > z_threshold_moderate, "Outlier"] = "Moderate"
    df.loc[df["Z-score"].abs() > z_threshold_strong, "Outlier"] = "Strong"

    # === Save updated CSV ===
    csv_out_path = os.path.join(output_folder, f"{csv_name}_with_outliers.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"✅ Updated CSV with outlier flags saved to: {csv_out_path}")

    # === Plot 1: Means with error bars ===
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(means)), means, yerr=std_devs, fmt='o', capsize=5, label='Gaussians')
    plt.axhline(overall_mean, color='green', linestyle='--', label='Mean of Means')
    plt.axhline(overall_mean + z_threshold_moderate * overall_std, color='orange', linestyle=':', label=f'|Z|={z_threshold_moderate}')
    plt.axhline(overall_mean - z_threshold_moderate * overall_std, color='orange', linestyle=':')
    plt.axhline(overall_mean + z_threshold_strong * overall_std, color='red', linestyle=':', label=f'|Z|={z_threshold_strong}')
    plt.axhline(overall_mean - z_threshold_strong * overall_std, color='red', linestyle=':')

    # Add simulation labels for moderate/strong outliers
    if label_col is not None:
        for idx, row in df.iterrows():
            if abs(row["Z-score"]) > z_threshold_moderate:
                plt.text(idx, row[mean_col] + row[std_col] + 0.1, row[label_col], fontsize=8, rotation=45, ha='center')

    plt.title("Gaussian Means with Error Bars (±SD)")
    plt.xlabel("Gaussian Index")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.grid(True)

    plot1_path = os.path.join(output_folder, f"means_with_errorbars.png")
    plt.savefig(plot1_path, dpi=300)
    print(f"✅ Plot saved to: {plot1_path}")
    plt.close()

    # === Plot 2: Histogram and Gaussian Fit ===
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(means, bins=10, alpha=0.6, color='gray', edgecolor='black', density=True, label='Histogram of Means')

    # Fit a normal distribution and plot
    mu_fit, std_fit = stats.norm.fit(means)
    x = np.linspace(bins[0], bins[-1], 100)
    p = stats.norm.pdf(x, mu_fit, std_fit)
    plt.plot(x, p, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu_fit:.2f}, σ={std_fit:.2f})')

    plt.title("Histogram of Gaussian Means with Fit")
    plt.xlabel("Mean Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plot2_path = os.path.join(output_folder, f"histogram_gaussian_fit.png")
    plt.savefig(plot2_path, dpi=300)
    print(f"✅ Plot saved to: {plot2_path}")
    plt.close()

    # === Plot 3: QQ-plot ===
    plt.figure(figsize=(6, 6))
    stats.probplot(means, dist="norm", plot=plt)
    plt.title("QQ-Plot of Gaussian Means")
    plt.grid(True)

    plot3_path = os.path.join(output_folder, f"qqplot.png")
    plt.savefig(plot3_path, dpi=300)
    print(f"✅ QQ-Plot saved to: {plot3_path}")
    plt.close()

    # === Shapiro-Wilk test ===
    shapiro_stat, shapiro_p = stats.shapiro(means)
    shapiro_summary = (
        f"📊 Shapiro-Wilk Test for Normality:\n"
        f"Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}\n"
    )
    if shapiro_p > 0.05:
        shapiro_summary += "✅ The means are likely Gaussian (fail to reject H0).\n"
    else:
        shapiro_summary += "❌ The means may not be Gaussian (reject H0).\n"

    print(shapiro_summary)

    # Save Shapiro-Wilk output to a text file
    shapiro_txt_path = os.path.join(output_folder, f"shapiro_test.txt")
    with open(shapiro_txt_path, "w") as f:
        f.write(shapiro_summary)
    print(f"✅ Shapiro-Wilk test results saved to: {shapiro_txt_path}")

    # === Print outliers ===
    print("\n=== Moderate Outliers (|Z| > {}) ===".format(z_threshold_moderate))
    print(df[df["Outlier"] == "Moderate"])

    print("\n=== Strong Outliers (|Z| > {}) ===".format(z_threshold_strong))
    print(df[df["Outlier"] == "Strong"])

    return df


csv_names = [
    "z_offset_from_sf_gaussian_parameters.csv",
    "com_to_sf_com_distance_gaussian_parameters.csv",
    "min_atom_to_sf_com_distance_gaussian_parameters.csv",
    "radial_distance_gaussian_parameters.csv"
]

relax_or_not = ["relaxation", "no_relaxation"]
sidechain_or_not = ["sidechain", "full_residue"]

for sidechain in sidechain_or_not:
    for relax in relax_or_not:
        for csv_name in csv_names:
            path = os.path.join(
                "./gaussian_analysis_results", sidechain, relax, "plots", csv_name
            )
            print(f"\n🔍 Analyzing: {path}")
            # Analyze your Gaussian means CSV
            df_results = analyze_gaussian_means(path)
            
# path = "./gaussian_analysis_results/sidechain/relaxation/plots/z_offset_from_sf_gaussian_parameters.csv"
# df_results = analyze_gaussian_means(path)
