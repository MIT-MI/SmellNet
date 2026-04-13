#!/usr/bin/env python
"""
GC-MS PCA plot with styling from analysis/gcms_analysis.ipynb.
Uses seaborn Set2 palette, larger fonts, and legend outside the plot.
Run from gcms_analysis/ or project root. Saves to gcms_analysis/plots/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reuse data loading and category mapping from gcms_paper_figures
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from gcms_paper_figures import load_gcms_data, get_category

PLOTS_DIR = SCRIPT_DIR / "plots"


def main():
    print("Loading GC-MS data...")
    X, foods, mz_min, mz_max, bin_size = load_gcms_data()
    print(f"  {len(foods)} foods, {X.shape[1]} m/z bins")

    # Build dataframe for seaborn (same structure as notebook)
    features = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df = pd.DataFrame({"food_name": foods})
    df["category"] = df["food_name"].map(get_category)

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # Styling from gcms_analysis.ipynb
    sns.set(style="whitegrid", context="notebook")
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("Set2")

    sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="category",
        palette=palette,
        s=160,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
    )

    plt.title("PCA of GC-MS Features by Ingredient Category", fontsize=28, fontweight="bold")
    plt.xlabel(f"PC1 ({explained_var[0] * 100:.1f}% variance)", fontsize=25)
    plt.ylabel(f"PC2 ({explained_var[1] * 100:.1f}% variance)", fontsize=25)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.legend(
        title="Category",
        title_fontsize=25,
        fontsize=20,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        markerscale=1.5,
        frameon=True,
    )

    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "PCA_gcms_category.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
