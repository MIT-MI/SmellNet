#!/usr/bin/env python3
"""
Compute and plot the correlation matrix for SmellNet-Base sensor data (base_data/).
Reference: analysis/data_analysis.ipynb Cell 15.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"
OUTPUT_DIR = PROJECT_ROOT / "data_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SmellNet-Base sensor columns (6 columns, same as generate_pca_iclr)
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
    "radish": "Vegetables", "tomato": "Vegetables", "potato": "Vegetables", "cabbage": "Vegetables",
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
        # Fallback: GC-MS table if sensor CSVs not found
        for gcms_path in (
            PROJECT_ROOT / "base_data" / "gcms_dataframe.csv",
            PROJECT_ROOT / "gcms_processed" / "gcms_dataframe.csv",
        ):
            if gcms_path.exists():
                return load_gcms_data(gcms_path)
        raise FileNotFoundError(
            f"No CSV data found in {BASE_DATA}/training or {BASE_DATA}/testing"
        )
    return pd.concat(rows, ignore_index=True)


def load_gcms_data(gcms_path: Path) -> pd.DataFrame:
    """Load GCMS chemical composition data (fallback when sensor CSVs are missing)."""
    df = pd.read_csv(gcms_path)
    # First column is food_name, rest are numeric features
    feature_cols = [c for c in df.columns if c != "food_name"]
    df = df.rename(columns={"food_name": "ingredient"})
    df["category"] = df["ingredient"].map(INGREDIENT_TO_CATEGORY)
    return df


def compute_and_plot_correlation(agg_df: pd.DataFrame) -> None:
    """Compute correlation matrix and save heatmap plot."""
    # Determine numeric columns
    numeric_cols = [c for c in agg_df.columns if c in SENSOR_COLUMNS]
    if not numeric_cols:
        # GCMS: use all numeric columns except ingredient/category
        numeric_cols = [
            c for c in agg_df.columns
            if c not in ("ingredient", "category") and pd.api.types.is_numeric_dtype(agg_df[c])
        ]
    if not numeric_cols:
        raise ValueError("No numeric columns found for correlation matrix")

    # Compute correlation matrix
    correlation_matrix = agg_df[numeric_cols].corr()

    # Plot (mirroring Cell 15 style)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 20},
    )
    plt.title("SmellNet-Base: Feature Correlation Matrix", fontsize=30, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=25)
    plt.yticks(rotation=45, fontsize=25)
    plt.tight_layout(pad=2.0)

    out_path = OUTPUT_DIR / "feature_correlation_iclr.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    # Optionally save the correlation matrix as CSV
    csv_path = OUTPUT_DIR / "feature_correlation_iclr.csv"
    correlation_matrix.to_csv(csv_path)
    print(f"Saved: {csv_path}")


def main() -> None:
    print("Loading base_data...")
    agg_df = load_iclr_aggregated()
    print(f"Loaded {len(agg_df)} rows, {agg_df['ingredient'].nunique()} ingredients")

    print("Computing correlation matrix...")
    compute_and_plot_correlation(agg_df)
    print("Done.")


if __name__ == "__main__":
    main()
