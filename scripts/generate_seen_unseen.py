from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

models = ["MLP", "CNN", "LSTM", "ScentFormer"]

seen_top1 = [44.0, 48.1, 46.4, 50.2]
unseen_top1 = [11.7, 12.4, 11.8, 16.0]

# Palette matching your poster style
TITLE_COLOR = "#8D1818"

SEEN_FILL = "#D8C7CF"
SEEN_EDGE = "#B07A88"
SEEN_TEXT = "#9E2C2C"

UNSEEN_FILL = "#C9D8D0"
UNSEEN_EDGE = "#88B09B"
UNSEEN_TEXT = "#6FA08A"

GRID_COLOR = "#D9D9D9"
AXIS_COLOR = "#BBBBBB"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AXIS_COLOR,
})

def add_value_labels(ax, bars, color, dy=0.6, fontsize=10):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )

def single_bar_chart(values, title, ylabel, stem, fill, edge, text_color, ymax=None):
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    bars = ax.bar(
        x,
        values,
        width=0.58,
        color=fill,
        edgecolor=edge,
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, color=TITLE_COLOR, fontweight="bold")

    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    if ymax is None:
        ymax = max(values) + 10
    ax.set_ylim(0, ymax)

    add_value_labels(ax, bars, text_color)

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# Seen only
single_bar_chart(
    seen_top1,
    "Seen Mixtures",
    "Top-1@0.1 (%)",
    "seen_only_top1",
    SEEN_FILL,
    SEEN_EDGE,
    SEEN_TEXT,
    ymax=60,
)

# Unseen only
single_bar_chart(
    unseen_top1,
    "Unseen Mixtures",
    "Top-1@0.1 (%)",
    "unseen_only_top1",
    UNSEEN_FILL,
    UNSEEN_EDGE,
    UNSEEN_TEXT,
    ymax=22,
)