#!/usr/bin/env python3
"""
Generate all PCA plots using SmellNet-Base data under base_data/ (6 sensor columns only).
Produces: (1) overall PCA by Ingredient Category, (2) one PCA per category by Ingredient.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"
OUTPUT_DIR = PROJECT_ROOT / "data_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# base_data: 6 sensor columns only
SENSOR_COLUMNS = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"]

INGREDIENT_TO_CATEGORY = {
    "peanuts": "Nuts", "cashew": "Nuts", "chestnuts": "Nuts", "pistachios": "Nuts",
    "almond": "Nuts", "hazelnut": "Nuts", "walnuts": "Nuts", "pecans": "Nuts",
    "brazil_nut": "Nuts", "pili_nut": "Nuts",
    "cumin": "Spices", "star_anise": "Spices", "nutmeg": "Spices", "cloves": "Spices",
    "ginger": "Spices", "allspice": "Spices", "chervil": "Spices", "mustard": "Spices",
    "cinnamon": "Spices", "saffron": "Spices",
    "angelica": "Herbs", "garlic": "Herbs", "chives": "Herbs", "turnip": "Herbs",
    "dill": "Herbs", "mugwort": "Herbs", "chamomile": "Herbs", "coriander": "Herbs",
    "oregano": "Herbs", "mint": "Herbs",
    "kiwi": "Fruits", "pineapple": "Fruits", "banana": "Fruits", "lemon": "Fruits",
    "mandarin_orange": "Fruits", "strawberry": "Fruits", "apple": "Fruits",
    "mango": "Fruits", "peach": "Fruits", "pear": "Fruits",
    "cauliflower": "Vegetables", "brussel_sprouts": "Vegetables", "broccoli": "Vegetables",
    "sweet_potato": "Vegetables", "asparagus": "Vegetables", "avocado": "Vegetables",
    "radish": "Vegetables", "tomato": "Vegetables", "potato": "Vegetables",
    "cabbage": "Vegetables",
}

# Plot style: match reference Nuts PCA image (sans-serif, title bold only)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 12


def load_iclr_data():
    """Load all CSVs from base_data/training and base_data/testing; return one DataFrame with ingredient and category."""
    training_path = BASE_DATA / "training"
    testing_path = BASE_DATA / "testing"
    rows = []
    for split_path, split_name in [(training_path, "train"), (testing_path, "test")]:
        if not split_path.exists():
            continue
        for folder_name in os.listdir(split_path):
            folder_path = split_path / folder_name
            if not folder_path.is_dir():
                continue
            if folder_name not in INGREDIENT_TO_CATEGORY:
                continue
            category = INGREDIENT_TO_CATEGORY[folder_name]
            for f in folder_path.glob("*.csv"):
                df = pd.read_csv(f, usecols=SENSOR_COLUMNS)
                df = df[SENSOR_COLUMNS]  # ensure order
                df["ingredient"] = folder_name
                df["category"] = category
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_pca_overall(agg_df):
    """PCA of all data, colored by Ingredient Category."""
    X = agg_df[SENSOR_COLUMNS]
    y = agg_df["category"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Ingredient Category": y
    })
    # Deeper, darker colors for better differentiation (Spices, Nuts, Herbs, Vegetables, Fruits)
    palette = ["#1565C0", "#E65100", "#2E7D32", "#C62828", "#6A1B9A"]
    category_order = ["Spices", "Nuts", "Herbs", "Vegetables", "Fruits"]
    pca_df["Ingredient Category"] = pd.Categorical(pca_df["Ingredient Category"], categories=category_order, ordered=True)
    pca_df = pca_df.sort_values("Ingredient Category")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Ingredient Category", palette=palette, s=8, alpha=0.6, linewidth=0, ax=ax)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=LABEL_FONTSIZE)
    ax.set_title("PCA of Sensor Data", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, color="lightgrey", linestyle="-")
    ax.set_facecolor("white")
    ax.legend(title="Ingredient Category", title_fontsize=LEGEND_TITLE_FONTSIZE, fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 0, 0.2, 1), loc="upper left", markerscale=3, handletextpad=0.8, labelspacing=2.5, frameon=True, fancybox=False, facecolor="whitesmoke", edgecolor="lightgray")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out = OUTPUT_DIR / "PCA_sensor_data_category_iclr.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_pca_per_category(agg_df):
    """One PCA per category, colored by Ingredient."""
    for category in ["Fruits", "Nuts", "Herbs", "Spices", "Vegetables"]:
        sub = agg_df[agg_df["category"] == category]
        if len(sub) == 0:
            print(f"Skipping {category}: no data")
            continue
        X = sub[SENSOR_COLUMNS]
        y = sub["ingredient"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Ingredient": y
        })
        ingredients = sorted(pca_df["Ingredient"].unique())
        palette = sns.color_palette("Paired", n_colors=len(ingredients))

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Ingredient", palette=palette, s=8, alpha=0.6, linewidth=0, ax=ax, hue_order=ingredients)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)", fontsize=LABEL_FONTSIZE)
        ax.set_title(f"{category} PCA", fontsize=TITLE_FONTSIZE, fontweight="bold")
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, color="lightgrey", linestyle="-")
        ax.set_facecolor("white")
        ax.legend(title="Ingredient", title_fontsize=LEGEND_TITLE_FONTSIZE, fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(1.02, 0, 0.2, 1), loc="upper left", markerscale=3, handletextpad=0.8, labelspacing=2.5, frameon=True, fancybox=False, facecolor="whitesmoke", edgecolor="lightgray")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        out = OUTPUT_DIR / f"PCA_{category.lower()}_iclr.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main():
    print("Loading base_data...")
    agg_df = load_iclr_data()
    print(f"Loaded {len(agg_df)} rows, {agg_df['ingredient'].nunique()} ingredients")
    print("Overall PCA by category...")
    plot_pca_overall(agg_df)
    print("Per-category PCA...")
    plot_pca_per_category(agg_df)
    print("Done.")


if __name__ == "__main__":
    main()
