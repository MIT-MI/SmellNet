#!/usr/bin/env python3
"""
Regenerate the time-series figure with CO (top) and NO2 (bottom).
Uses generator from data_analysis.ipynb and data from plot_co_no2 (oregano_6).
Style: blue lines, light grey grid, shared x-axis, Time Step on x-axis.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_PATH = DATA_ROOT / "training"
TESTING_PATH = DATA_ROOT / "testing"
SAMPLE_CSV = TESTING_PATH / "oregano" / "oregano_6.csv"
OUTPUT_DIR = PROJECT_ROOT / "data_stats" / "time_series_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times"]
plt.rcParams["font.weight"] = "bold"
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16


def load_aggregated_data():
    """
    Generator logic from data_analysis.ipynb (Cells 2, 4):
    Load training/testing CSVs, add ingredient, file_id, time_step.
    """
    training_data = defaultdict(list)
    testing_data = defaultdict(list)

    for folder_path, split in [(TRAINING_PATH, "train"), (TESTING_PATH, "test")]:
        if not folder_path.exists():
            continue
        for folder_name in os.listdir(folder_path):
            cur_folder = folder_path / folder_name
            if not cur_folder.is_dir():
                continue
            for f in cur_folder.glob("*.csv"):
                df = pd.read_csv(f).copy()
                df["ingredient"] = folder_name
                df["file_id"] = f"{folder_name}_{split}_{f.stem.split('_')[-1]}"
                df["time_step"] = range(len(df))
                if split == "train":
                    training_data[folder_name].append(df)
                else:
                    testing_data[folder_name].append(df)

    rows = [d for dfs in training_data.values() for d in dfs] + [d for dfs in testing_data.values() for d in dfs]
    if not rows:
        raise FileNotFoundError(f"No data in {TRAINING_PATH} or {TESTING_PATH}")
    return pd.concat(rows, ignore_index=True)


def main():
    if SAMPLE_CSV.exists():
        df = pd.read_csv(SAMPLE_CSV)
        df["time_step"] = range(len(df))
    else:
        agg = load_aggregated_data()
        sample_id = "oregano_test_6"
        subset = agg[agg["file_id"] == sample_id]
        if subset.empty:
            subset = agg[(agg["ingredient"] == "oregano") & (agg["file_id"].str.contains("test"))]
            if subset.empty:
                raise FileNotFoundError(f"Sample oregano_6 not found")
            sample_id = subset["file_id"].iloc[0]
            subset = agg[agg["file_id"] == sample_id]
        df = subset.copy()
    time_step = df["time_step"].values if "time_step" in df.columns else range(len(df))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(time_step, df["CO"], color="steelblue")
    ax1.set_title("CO over Time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.set_ylabel("CO", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax1.grid(True, color="lightgrey", linestyle="-")
    ax1.set_facecolor("white")
    ax2.plot(time_step, df["NO2"], color="steelblue")
    ax2.set_title("NO2 over Time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax2.set_ylabel("NO2", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax2.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax2.grid(True, color="lightgrey", linestyle="-")
    ax2.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "CO_NO2_timeseries.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
