# iclr_per_category_barplots.py
# Per-category acc@1 bar plots: Regular vs Contrastive, using provided data.
# Requirements: matplotlib, numpy

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent

# ---- Data (Regular then Contrastive: rows = categories, cols = MLP, CNN, LSTM, ScentFormer) ----
CATEGORIES = ["Fruits", "Herbs", "Nuts", "Spices", "Vegetables"]
MODELS = ["MLP", "CNN", "LSTM", "ScentFormer (ours)"]

REGULAR = np.array([
    [18.18, 63.63, 55.55, 62.63],   # Fruits
    [27.55, 56.12, 57.14, 56.12],   # Herbs
    [31.78, 49.53, 69.16, 54.21],   # Nuts
    [34.69, 58.16, 54.08, 58.16],   # Spices
    [22.0,  37.0,  53.0,  50.0],    # Vegetable
])

CONTRASTIVE = np.array([
    [27.27, 74.75, 67.67, 69.69],   # Fruits
    [40.82, 56.12, 47.96, 60.2],    # Herbs
    [28.97, 60.75, 63.55, 63.55],   # Nuts
    [34.69, 65.31, 58.18, 71.43],   # Spices
    [23.0,  38.0,  46.0,  52.0],    # Vegetable
])

# ---- Colors (regular, contrastive) ----
COLORS = ["#da81c1", "#7dbfa7"]
REGULAR_COLOR, CONTRASTIVE_COLOR = COLORS[0], COLORS[1]

# ---- Output ----
FIG_DIR = _REPO / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = FIG_DIR / "iclr_per_category_barplots.png"
OUTPUT_PDF = FIG_DIR / "iclr_per_category_barplots.pdf"

# ---- Plot (bigger fonts, 4 panels) ----
FONT_SIZE = 20
TITLE_SIZE = 22
LABEL_SIZE = 20
TICK_SIZE = 20
LEGEND_SIZE = 20

plt.figure(figsize=(26, 6), dpi=150)
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.weight": "bold",
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE,
    "axes.labelweight": "bold",
    "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

n_cats = len(CATEGORIES)
x = np.arange(n_cats)
w = 0.38

for i, model_name in enumerate(MODELS, 1):
    ax = plt.subplot(1, 4, i)
    y_reg = REGULAR[:, i - 1]
    y_con = CONTRASTIVE[:, i - 1]

    ax.bar(x - w / 2, y_reg, width=w, label="Regular",
           color=REGULAR_COLOR, edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, y_con, width=w, label="Contrastive",
           color=CONTRASTIVE_COLOR, edgecolor="black", linewidth=0.6)

    ax.set_title(model_name, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=40, ha="right", fontsize=TICK_SIZE, fontweight="bold")
    ax.set_ylabel("Accuracy (acc@1, %)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)  # draw grid behind the bars
    ax.grid(axis="y", linestyle="-", linewidth=1.0, alpha=0.5, color="gray")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    if i == len(MODELS):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=LEGEND_SIZE, prop=dict(weight="bold", size=LEGEND_SIZE))

plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
print(f"Saved: {OUTPUT_PDF}")
