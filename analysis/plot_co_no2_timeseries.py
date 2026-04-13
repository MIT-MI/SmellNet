#!/usr/bin/env python3
"""
Time-series figure: CO (top) and CO temporal difference with lag p (bottom).
Uses SmellNet-Base data under base_data/ (6 sensor columns). Style: blue lines, light grey grid, shared x-axis.

Temporal difference: value at t minus value at t - p (pandas diff with periods=p).

Samples: apple_6, angelica_6, cashew_6, pear_6, mandarin_orange_6 (testing set).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA = PROJECT_ROOT / "base_data"
OUTPUT_DIR = PROJECT_ROOT / "data_stats" / "time_series_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (class folder name, csv filename, output stem)
SAMPLES = [
    ("apple", "apple_6.csv", "CO_apple_6_timeseries"),
    ("angelica", "angelica_6.csv", "CO_angelica_6_timeseries"),
    ("cashew", "cashew_6.csv", "CO_cashew_6_timeseries"),
    ("pear", "pear_6.csv", "CO_pear_6_timeseries"),
    ("mandarin_orange", "mandarin_orange_6.csv", "CO_mandarin_orange_6_timeseries"),
]

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
P = 25


def plot_sample(sample_csv: Path, out_stem: str) -> None:
    df = pd.read_csv(sample_csv)
    time_step = range(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(time_step, df["CO"], color="steelblue")
    ax1.set_title("CO over Time", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax1.set_ylabel("CO", fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax1.grid(True, color="lightgrey", linestyle="-")
    ax1.set_facecolor("white")

    co_diff = df["CO"].diff(P)
    ax2.plot(time_step, co_diff, color="steelblue")
    ax2.set_title(
        f"CO temporal difference (p = {P})",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax2.set_ylabel(r"$\Delta$ CO", fontsize=LABEL_FONTSIZE)
    ax2.set_xlabel("Time Step", fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax2.grid(True, color="lightgrey", linestyle="-")
    ax2.set_facecolor("white")

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{out_stem}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}  (sample: {sample_csv.parent.name}, {sample_csv.name})")


def main():
    for class_name, fname, out_stem in SAMPLES:
        sample_csv = BASE_DATA / "testing" / class_name / fname
        plot_sample(sample_csv, out_stem)


if __name__ == "__main__":
    main()
