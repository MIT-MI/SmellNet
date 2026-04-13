#!/usr/bin/env python3
"""
Generate PCA plots from SmellNet-Base data (base_data/).
Extracted from analysis/data_analysis.ipynb (Cells 16, 18, 22).

Produces:
  1. Overall PCA by Ingredient Category
  2. Per-category PCA plots (one per category, colored by Ingredient)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"
OUTPUT_DIR = PROJECT_ROOT / "data_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# base_data uses 6 sensor columns only
SENSOR_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]

INGREDIENT_TO_CATEGORY = {
    # Nuts
    "peanuts": "Nuts",
    "cashew": "Nuts",
    "chestnuts": "Nuts",
    "pistachios": "Nuts",
    "almond": "Nuts",
    "hazelnut": "Nuts",
    "walnuts": "Nuts",
    "pecans": "Nuts",
    "brazil_nut": "Nuts",
    "pili_nut": "Nuts",
    # Spices
    "cumin": "Spices",
    "star_anise": "Spices",
    "nutmeg": "Spices",
    "cloves": "Spices",
    "ginger": "Spices",
    "allspice": "Spices",
    "chervil": "Spices",
    "mustard": "Spices",
    "cinnamon": "Spices",
    "saffron": "Spices",
    # Herbs
    "angelica": "Herbs",
    "garlic": "Herbs",
    "chives": "Herbs",
    "turnip": "Herbs",
    "dill": "Herbs",
    "mugwort": "Herbs",
    "chamomile": "Herbs",
    "coriander": "Herbs",
    "oregano": "Herbs",
    "mint": "Herbs",
    # Fruits
    "kiwi": "Fruits",
    "pineapple": "Fruits",
    "banana": "Fruits",
    "lemon": "Fruits",
    "mandarin_orange": "Fruits",
    "strawberry": "Fruits",
    "apple": "Fruits",
    "mango": "Fruits",
    "peach": "Fruits",
    "pear": "Fruits",
    # Vegetables
    "cauliflower": "Vegetables",
    "brussel_sprouts": "Vegetables",
    "broccoli": "Vegetables",
    "sweet_potato": "Vegetables",
    "asparagus": "Vegetables",
    "avocado": "Vegetables",
    "radish": "Vegetables",
    "tomato": "Vegetables",
    "potato": "Vegetables",
    "cabbage": "Vegetables",
}


def load_iclr_aggregated() -> pd.DataFrame:
    """Load all CSVs from base_data/training and base_data/testing into one aggregated DataFrame."""
    rows = []
    for split in ["training", "testing"]:
        split_path = BASE_DATA / split
        if not split_path.exists():
            continue
        for folder_name in os.listdir(split_path):
            folder_path = split_path / folder_name
            if not folder_path.is_dir() or folder_name not in INGREDIENT_TO_CATEGORY:
                continue
            category = INGREDIENT_TO_CATEGORY[folder_name]
            for f in folder_path.glob("*.csv"):
                df = pd.read_csv(f, usecols=SENSOR_COLUMNS)
                df = df[SENSOR_COLUMNS]
                df["ingredient"] = folder_name
                df["category"] = category
                rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No CSV data found in {BASE_DATA}/training or {BASE_DATA}/testing")
    return pd.concat(rows, ignore_index=True)


def plot_pca_overall(agg_df: pd.DataFrame) -> None:
    """
    Overall PCA of all sensor data, colored by Ingredient Category.
    Mirrors notebook Cell 16.
    """
    X_raw = agg_df[SENSOR_COLUMNS]
    y_category = agg_df["category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    print(f"Explained variance ratios: {explained_var}")

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Category": y_category,
    })

    sns.set(style="whitegrid", context="notebook")
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Category",
        palette=custom_palette,
        s=8,
        alpha=0.6,
        linewidth=0,
    )

    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", fontsize=25)
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", fontsize=25)
    plt.title("PCA of Sensor Data", fontsize=33, fontweight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(
        title="Ingredient Category",
        title_fontsize=25,
        fontsize=25,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.8,
        handlelength=2.5,
        markerscale=5,
    )

    out_path = OUTPUT_DIR / "PCA_sensor_data_category_iclr.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_pca_per_category(agg_df: pd.DataFrame) -> None:
    """
    One PCA plot per category, colored by Ingredient.
    Mirrors notebook Cells 18 and 22.
    """
    sns.set(style="whitegrid", context="notebook", font_scale=1.4)

    for category in agg_df["category"].unique():
        group_df = agg_df[agg_df["category"] == category]
        X = group_df[SENSOR_COLUMNS]

        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Ingredient": group_df["ingredient"].values,
        })

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="Ingredient",
            palette="tab20",
            s=8,
            alpha=0.7,
            linewidth=0,
            ax=ax,
        )

        ax.set_title(f"{category} PCA", fontsize=33, fontweight="bold")
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)", fontsize=25)
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)", fontsize=25)
        ax.tick_params(axis="both", labelsize=16)
        plt.legend(
            title="Ingredient",
            title_fontsize=25,
            fontsize=25,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            handletextpad=0.4,
            borderaxespad=0.2,
            labelspacing=0.8,
            handlelength=2.5,
            markerscale=5,
        )

        out_path = OUTPUT_DIR / f"PCA_{category.replace(' ', '_').lower()}_iclr.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")


def main() -> None:
    print("Loading base_data...")
    agg_df = load_iclr_aggregated()
    print(f"Loaded {len(agg_df)} rows, {agg_df['ingredient'].nunique()} ingredients")

    print("Generating overall PCA by category...")
    plot_pca_overall(agg_df)

    print("Generating per-category PCA plots...")
    plot_pca_per_category(agg_df)
    print("Done.")


if __name__ == "__main__":
    main()
