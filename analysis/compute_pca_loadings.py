#!/usr/bin/env python3
"""
Compute PC1 and PC2 feature contributions (loadings) from PCA.
Reference: analysis/data_analysis.ipynb Cell 17.
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"
OUTPUT_DIR = PROJECT_ROOT / "data_stats"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SmellNet-Base sensor columns
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
        raise FileNotFoundError(
            f"No CSV data found in {BASE_DATA}/training or {BASE_DATA}/testing"
        )
    return pd.concat(rows, ignore_index=True)


def compute_pca_loadings(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run PCA and return PC1, PC2 feature contributions (loadings) plus magnitude.
    """
    X_raw = agg_df[SENSOR_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    # Loadings: pca.components_ is (n_components, n_features), transpose to (n_features, n_components)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=SENSOR_COLUMNS,
    )

    # Magnitude of contribution (Euclidean norm across PC1 and PC2)
    loadings["Magnitude"] = (loadings[["PC1", "PC2"]] ** 2).sum(axis=1) ** 0.5

    # Sort by magnitude (descending)
    loadings = loadings.sort_values("Magnitude", ascending=False)

    return loadings, pca.explained_variance_ratio_


def main() -> None:
    print("Loading base_data...")
    agg_df = load_iclr_aggregated()
    print(f"Loaded {len(agg_df)} rows, {agg_df['ingredient'].nunique()} ingredients")

    print("Computing PCA and feature contributions...")
    loadings, explained_var = compute_pca_loadings(agg_df)

    print(f"\nExplained variance ratios: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    print("\nFeature Contributions (sorted by Magnitude):")
    print(loadings.to_string())

    # Save to CSV
    out_path = OUTPUT_DIR / "pca_loadings_iclr.csv"
    loadings.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
