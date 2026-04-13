from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

models = ["MLP", "CNN", "LSTM", "ScentFormer"]
sensor_only_best_acc1 = [26.8, 52.7, 57.9, 56.1]
cross_modal_best_acc1 = [30.8, 58.9, 56.9, 63.3]
seen_metrics = {
    50: {"MAE": [0.0428, 0.0404, 0.0399, 0.0395], "Top-1@0.1": [44.0, 48.1, 46.4, 50.2], "Top-K (%)": [85.0, 86.7, 89.3, 87.9]},
    100: {"MAE": [0.0586, 0.0476, 0.0430, 0.0417], "Top-1@0.1": [33.7, 36.2, 46.5, 47.9], "Top-K (%)": [78.9, 87.0, 86.3, 89.0]},
}
unseen_w50 = {"Top-1@0.1": [11.7, 12.4, 11.8, 16.0], "Top-K (%)": [34.0, 36.4, 34.2, 38.9]}

# Updated palette to match your screenshot
TITLE_COLOR = "#8D1818"

SERIES_A_FILL = "#D8C7CF"   # pink-ish fill
SERIES_A_EDGE = "#B07A88"   # pink edge
SERIES_A_TEXT = "#9E2C2C"   # dark red labels

SERIES_B_FILL = "#C9D8D0"   # pale green fill
SERIES_B_EDGE = "#88B09B"   # green edge
SERIES_B_TEXT = "#6FA08A"   # green labels

GRID_COLOR = "#D9D9D9"
AXIS_COLOR = "#BBBBBB"
LEGEND_TEXT_COLOR = "#666666"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AXIS_COLOR,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.alpha": 0.85,
    "lines.linewidth": 1.5,
})

def _label_pad_y(ax, frac=0.012):
    """Vertical offset for bar labels as a fraction of the y-axis span."""
    lo, hi = ax.get_ylim()
    return (hi - lo) * frac


def add_value_labels(ax, bars, color, fmt="{:.1f}", fontsize=10):
    dy = _label_pad_y(ax)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def add_value_labels_small(ax, bars, color, fmt="{:.4f}", fontsize=10):
    dy = _label_pad_y(ax, frac=0.02)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def save_chart(fig, stem):
    out = FIG_DIR / f"{stem}.png"
    kwargs = dict(
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.02,
    )
    fig.savefig(out, **kwargs)
    fig.savefig(FIG_DIR / f"{stem}.pdf", **kwargs)
    plt.close(fig)

DEFAULT_FIGSIZE = (9.0, 5.2)
# Narrower single-panel for the main SmellNet-Base bar chart (e.g. two-column paper width)
MAIN_RESULTS_FIGSIZE = (6.4, 5.2)


def grouped(
    metric_name,
    values_a,
    values_b,
    label_a,
    label_b,
    ylabel,
    stem,
    lower=False,
    figsize=DEFAULT_FIGSIZE,
):
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=figsize)

    bars_a = ax.bar(
        x - width / 2,
        values_a,
        width,
        label=label_a,
        color=SERIES_A_FILL,
        edgecolor=SERIES_A_EDGE,
        linewidth=1.35,
        zorder=3,
    )
    bars_b = ax.bar(
        x + width / 2,
        values_b,
        width,
        label=label_b,
        color=SERIES_B_FILL,
        edgecolor=SERIES_B_EDGE,
        linewidth=1.35,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(metric_name, color=TITLE_COLOR, fontweight="bold", pad=12)

    ax.grid(axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    if lower:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, prune=None))

    if lower:
        ymin = min(min(values_a), min(values_b))
        ymax = max(max(values_a), max(values_b))
        span = ymax - ymin
        padding = max(span * 0.22, 0.0015)
        ax.set_ylim(max(0, ymin - 0.12 * padding), ymax + padding)
        add_value_labels_small(ax, bars_a, SERIES_A_TEXT)
        add_value_labels_small(ax, bars_b, SERIES_B_TEXT)
    else:
        ymax = max(max(values_a), max(values_b))
        headroom = max(ymax * 0.12, 6.0)
        ax.set_ylim(0, ymax + headroom)
        add_value_labels(ax, bars_a, SERIES_A_TEXT)
        add_value_labels(ax, bars_b, SERIES_B_TEXT)

    # Legend below the plot so it never overlaps the title
    legend = ax.legend(
        frameon=True,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        framealpha=0.98,
        facecolor="white",
        edgecolor="#E0E0E0",
        fancybox=False,
    )
    for t in legend.get_texts():
        t.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(top=0.90, bottom=0.24, left=0.1, right=0.98)
    save_chart(fig, stem)

grouped(
    "Main Results on SmellNet-Base",
    sensor_only_best_acc1,
    cross_modal_best_acc1,
    "Sensor-only",
    "Cross-modal (GC-MS)",
    "Acc@1 (%)",
    "smellnet_base_best_acc1_barplot",
    figsize=MAIN_RESULTS_FIGSIZE,
)

grouped(
    "Seen Mixtures: Top-1@0.1 by Model and Window",
    seen_metrics[50]["Top-1@0.1"],
    seen_metrics[100]["Top-1@0.1"],
    "w = 50",
    "w = 100",
    "Top-1@0.1 (%)",
    "seen_mixtures_top1_by_window",
)

grouped(
    "Seen Mixtures: Top-K by Model and Window",
    seen_metrics[50]["Top-K (%)"],
    seen_metrics[100]["Top-K (%)"],
    "w = 50",
    "w = 100",
    "Top-K (%)",
    "seen_mixtures_topk_by_window",
)

grouped(
    "Seen Mixtures: MAE by Model and Window",
    seen_metrics[50]["MAE"],
    seen_metrics[100]["MAE"],
    "w = 50",
    "w = 100",
    "MAE (lower is better)",
    "seen_mixtures_mae_by_window",
    lower=True,
)

grouped(
    "Generalization Gap: Top-1@0.1 (Seen vs Unseen, w = 50)",
    seen_metrics[50]["Top-1@0.1"],
    unseen_w50["Top-1@0.1"],
    "Seen (w = 50)",
    "Unseen (w = 50)",
    "Top-1@0.1 (%)",
    "seen_vs_unseen_top1_w50",
)

grouped(
    "Generalization Gap: Top-K (Seen vs Unseen, w = 50)",
    seen_metrics[50]["Top-K (%)"],
    unseen_w50["Top-K (%)"],
    "Seen (w = 50)",
    "Unseen (w = 50)",
    "Top-K (%)",
    "seen_vs_unseen_topk_w50",
)