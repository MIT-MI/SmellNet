# iclr_per_category_barplots.py
# Requirements: matplotlib, pandas

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- File paths ----
REGULAR_CSV = "/mnt/data/regular.csv"
CONTRASTIVE_CSV = "/mnt/data/contrastive_learning.csv"
OUTPUT_PNG = "/mnt/data/iclr_per_category_barplots.png"

# ---- Settings ----
TARGET_GRADIENT = 25
MODELS = ["cnn", "lstm", "mlp", "transformer"]  # ensure fixed order on the row

# ---- Load & parse ----
def load_with_per_category(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # per_category is a stringified dict like:
    # {"Fruits": {"n": ..., "acc@1": ..., "acc@5": ...}, ...}
    df["pc"] = df["per_category"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("{") else {}
    )
    return df

reg = load_with_per_category(REGULAR_CSV)
con = load_with_per_category(CONTRASTIVE_CSV)

# Filter to the requested gradient
reg = reg[reg["gradient"] == TARGET_GRADIENT]
con = con[con["gradient"] == TARGET_GRADIENT]

# Collect a stable category order (from the union of all rows)
all_cats = []
for df in [reg, con]:
    for d in df["pc"]:
        for k in d.keys():
            if k not in all_cats:
                all_cats.append(k)

def acc_by_cat(df: pd.DataFrame, model: str, cats: list[str]) -> list[float]:
    """Return acc@1 per category for the specified model (NaN if missing)."""
    row = df[df["model"] == model]
    if row.empty:
        return [np.nan] * len(cats)
    pc = row.iloc[0]["pc"]
    return [float(pc.get(cat, {}).get("acc@1", np.nan)) for cat in cats]

# ---- Plot ----
plt.figure(figsize=(26, 5), dpi=160)  # wide, single row
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Subtle color contrast for Regular vs Contrastive
regular_color = "#1f77b4"
contrast_color = "#4fa3d5"

for i, m in enumerate(MODELS, 1):
    ax = plt.subplot(1, 4, i)
    y_reg = acc_by_cat(reg, m, all_cats)
    y_con = acc_by_cat(con, m, all_cats)

    x = np.arange(len(all_cats))
    w = 0.38

    ax.bar(x - w/2, y_reg, width=w, label="Regular",
           color=regular_color, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, y_con, width=w, label="Contrastive",
           color=contrast_color, edgecolor="black", linewidth=0.5)

    ax.set_title(m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=40, ha="right")
    ax.set_ylabel("Accuracy (acc@1, %)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    # Put the legend once (on the last panel) to keep things tidy
    if i == len(MODELS):
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

plt.suptitle(f"Per-Category Accuracy @1 (Gradient={TARGET_GRADIENT}) — Regular vs Contrastive",
             y=1.05, fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight")
print(f"Saved to: {OUTPUT_PNG}")
