#!/usr/bin/env python
"""
Generate publication-quality figures for GC-MS food analysis.
Run from gcms_analysis/ or project root. Saves to gcms_analysis/plots/
"""

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PLOTS_DIR = SCRIPT_DIR / "plots"
GCMS_PROCESSED = REPO_ROOT / "gcms_processed"

# Food category mapping for coloring (nuts, spices, fruits, vegetables, herbs)
# Supports both FooDB names (e.g. "Peanut") and CSV names (e.g. "peanuts")
FOOD_CATEGORIES = {
    "Peanut": "Nuts", "peanuts": "Nuts", "Cashew nut": "Nuts", "cashew": "Nuts",
    "Chestnut": "Nuts", "chestnuts": "Nuts", "Pistachio": "Nuts", "pistachios": "Nuts",
    "Almond": "Nuts", "almond": "Nuts", "Hazelnut": "Nuts", "hazelnut": "Nuts",
    "Common walnut": "Nuts", "walnuts": "Nuts", "Pecan nut": "Nuts", "pecans": "Nuts",
    "Brazil nut": "Nuts", "brazil_nut": "Nuts", "Pili nut": "Nuts", "pili_nut": "Nuts",
    "Cumin": "Spices", "cumin": "Spices", "Star anise": "Spices", "star_anise": "Spices",
    "Nutmeg": "Spices", "nutmeg": "Spices", "Cloves": "Spices", "cloves": "Spices",
    "Ginger": "Spices", "ginger": "Spices", "Allspice": "Spices", "allspice": "Spices",
    "Cinnamon": "Spices", "cinnamon": "Spices", "Saffron": "Spices", "saffron": "Spices",
    "White mustard": "Spices", "white_mustard": "Spices",
    "Chervil": "Herbs", "chervil": "Herbs", "Dill": "Herbs", "dill": "Herbs",
    "Angelica": "Herbs", "angelica": "Herbs", "Garlic": "Herbs", "garlic": "Herbs",
    "Chives": "Herbs", "chives": "Herbs", "Mugwort": "Herbs", "mugwort": "Herbs",
    "Roman camomile": "Herbs", "roman_camomile": "Herbs", "Coriander": "Herbs", "coriander": "Herbs",
    "Mexican oregano": "Herbs", "mexican_oregano": "Herbs", "Spearmint": "Herbs", "spearmint": "Herbs",
    "Kiwi": "Fruits", "kiwi": "Fruits", "Pineapple": "Fruits", "pineapple": "Fruits",
    "Banana": "Fruits", "banana": "Fruits", "Lemon": "Fruits", "lemon": "Fruits",
    "Mandarin orange (Clementine, Tangerine)": "Fruits", "mandarin_orange": "Fruits",
    "Strawberry": "Fruits", "strawberry": "Fruits", "Apple": "Fruits", "apple": "Fruits",
    "Mango": "Fruits", "mango": "Fruits", "Peach": "Fruits", "peach": "Fruits",
    "Pear": "Fruits", "pear": "Fruits",
    "Cauliflower": "Vegetables", "cauliflower": "Vegetables", "Brussel sprouts": "Vegetables",
    "brussel_sprouts": "Vegetables", "Broccoli": "Vegetables", "broccoli": "Vegetables",
    "Sweet potato": "Vegetables", "sweet_potato": "Vegetables", "Asparagus": "Vegetables",
    "asparagus": "Vegetables", "Avocado": "Vegetables", "avocado": "Vegetables",
    "Radish": "Vegetables", "radish": "Vegetables", "Garden tomato": "Vegetables",
    "garden_tomato": "Vegetables", "Potato": "Vegetables", "potato": "Vegetables",
    "Common cabbage": "Vegetables", "common_cabbage": "Vegetables", "Turnip": "Vegetables",
    "turnip": "Vegetables",
}


def _slugify(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")
    return name or "plot"


def _save(name: str, fmt: str = "png"):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base = _slugify(name)
    fpath = PLOTS_DIR / f"{base}.{fmt}"
    i = 1
    while fpath.exists():
        fpath = PLOTS_DIR / f"{base}_{i}.{fmt}"
        i += 1
    plt.savefig(fpath, bbox_inches="tight")
    print(f"Saved: {fpath}")


def load_gcms_data():
    """Load food-level GC-MS vectors from npz or CSV."""
    npz_path = GCMS_PROCESSED / "gcms_food_vectors.npz"
    csv_path = GCMS_PROCESSED / "gcms_food_vectors.csv"

    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        X = data["vectors"]
        foods = np.array([str(f) for f in data["food_labels"]])
        mz_min = float(data.get("mz_min", 40))
        mz_max = float(data.get("mz_max", 400))
        bin_size = float(data.get("bin_size", 1))
        return X, foods, mz_min, mz_max, bin_size

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        food_col = "food_name" if "food_name" in df.columns else df.columns[0]
        foods = df[food_col].values
        feat_cols = [c for c in df.columns if c != food_col and c.startswith("f") and c[1:].isdigit()]
        feat_cols = sorted(feat_cols, key=lambda x: int(x[1:]))
        X = df[feat_cols].values.astype(np.float32)
        n_feat = X.shape[1]
        mz_min, mz_max, bin_size = 40.0, 40.0 + n_feat, 1.0
        return X, foods, mz_min, mz_max, bin_size

    raise FileNotFoundError(
        f"Neither {npz_path} nor {csv_path} found. "
        "Run gcms_analysis/analysis.py to build gcms_processed/, or place gcms_food_vectors.csv under gcms_processed/."
    )


def get_category(food: str) -> str:
    f = str(food).strip()
    if f in FOOD_CATEGORIES:
        return FOOD_CATEGORIES[f]
    low = f.lower().replace(" ", "_")
    return FOOD_CATEGORIES.get(low, "Other")


# -----------------------------------------------------------------------------
# Figure 1: PCA 2D projection of foods in GC-MS space
# -----------------------------------------------------------------------------

def fig_pca(X, foods, mz_min, mz_max, bin_size):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X_scaled)

    categories = [get_category(f) for f in foods]
    unique_cats = sorted(set(categories), key=lambda x: (x == "Other", x))
    cmap = plt.cm.tab10
    cat_to_color = {c: cmap(i % 10) for i, c in enumerate(unique_cats)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for cat in unique_cats:
        mask = np.array(categories) == cat
        ax.scatter(Z[mask, 0], Z[mask, 1], label=cat, alpha=0.8, s=60, c=[cat_to_color[cat]])

    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("Foods in GC-MS fingerprint space (PCA)", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    _save("fig1_pca_gcms_foods")


# -----------------------------------------------------------------------------
# Figure 2: Contrasting GC-MS fingerprints (two most different ingredients)
# -----------------------------------------------------------------------------

def _two_most_different_foods(X, foods):
    """Return indices (i, j) of the two food vectors with largest Euclidean distance."""
    n = len(foods)
    if n < 2:
        return (0, 0)
    best_i, best_j, best_d = 0, 1, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            if d > best_d:
                best_d, best_i, best_j = d, i, j
    return (best_i, best_j)


def fig_fingerprint_examples(X, foods, mz_min, bin_size, food_names=None):
    if food_names is None:
        # Use the two foods with most different GC-MS fingerprints
        i, j = _two_most_different_foods(X, foods)
        idxs = [i, j]
        print(f"  Two most different ingredients (fingerprint figure): {foods[i]} and {foods[j]}")
    else:
        idxs = []
        for fn in food_names:
            fn_norm = str(fn).lower().replace(" ", "_")
            for i, f in enumerate(foods):
                f_str = str(f).lower().replace(" ", "_")
                if f_str == fn_norm or fn_norm in f_str or str(f).lower() == str(fn).lower():
                    idxs.append(i)
                    break
        if not idxs:
            idxs = [0, len(foods) // 4, len(foods) // 2, 3 * len(foods) // 4][:2]

    n_plots = len(idxs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 1.8 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    mz_axis = mz_min + np.arange(X.shape[1]) * bin_size

    for ax, i in zip(axes, idxs):
        ax.fill_between(mz_axis, 0, X[i], alpha=0.6)
        ax.plot(mz_axis, X[i], linewidth=0.8)
        ax.set_ylabel("Intensity")
        ax.set_ylim(0, None)
        if n_plots > 1:
            ax.set_title(str(foods[i]))
    axes[-1].set_xlabel("m/z")

    # Single title only to avoid overlap
    if food_names is not None and len(idxs) == 1:
        fig.suptitle(f"{foods[idxs[0]]} – GC-MS fingerprint (binned)", y=1.02, fontweight="bold")
    elif food_names is None and len(idxs) == 2:
        fig.suptitle("Contrasting GC-MS fingerprints (two most different ingredients)", y=1.01, fontweight="bold")
    else:
        fig.suptitle("Example GC-MS fingerprints (binned)", y=1.01, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_name = "fig4_example_fingerprints"
    if food_names is not None and len(idxs) > 0:
        parts = [_slugify(str(foods[i])) for i in idxs]
        save_name = "fig4_fingerprints_" + "_".join(parts)
    _save(save_name)


# -----------------------------------------------------------------------------
# Figure 5: Coverage / data availability summary (compounds & spectra per ingredient)
# -----------------------------------------------------------------------------

def fig_coverage_summary():
    """Bar chart: volatile compounds with EI spectra per ingredient, and spectra per ingredient."""
    parquet_path = GCMS_PROCESSED / "gcms_food_spectra.parquet"
    if not parquet_path.exists():
        print("Skipping coverage summary: parquet not found")
        return
    df = pd.read_parquet(parquet_path)
    # Per food: distinct compounds (fdb_id) and number of spectra
    n_compounds = df.groupby("food")["fdb_id"].nunique()
    n_spectra = df.groupby("food").size()
    # Align and sort by total spectra (descending) for readability
    coverage = pd.DataFrame({"compounds": n_compounds, "spectra": n_spectra}).fillna(0).astype(int)
    coverage = coverage.sort_values("spectra", ascending=True)
    foods = coverage.index.tolist()
    n = len(foods)

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.28)))
    x = np.arange(n)
    w = 0.35
    bars1 = ax.barh(x - w / 2, coverage["compounds"], height=w, label="Compounds (with EI spectra)", color="steelblue", alpha=0.85)
    bars2 = ax.barh(x + w / 2, coverage["spectra"], height=w, label="Spectra", color="coral", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(foods, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_ylabel("Ingredient")
    ax.set_title("FooDB coverage: compounds and GC-MS spectra per ingredient", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, None)
    _save("fig5_coverage_summary")


# -----------------------------------------------------------------------------
# Figure 6: m/z coverage justification (CDF of intensity in 40–500 range)
# -----------------------------------------------------------------------------

def fig_mz_coverage_cdf(mz_cut_low=40, mz_cut_high=500):
    """Cumulative intensity vs m/z upper bound; vertical lines at 40 and 500."""
    npz_path = GCMS_PROCESSED / "gcms_food_spectra_vectors.npz"
    if not npz_path.exists():
        print("Skipping m/z CDF: spectrum-level npz not found")
        return
    data = np.load(npz_path, allow_pickle=True)
    vectors = data["vectors"]  # (N_spectra, D)
    mz_min = float(data["mz_min"])
    mz_max = float(data["mz_max"])
    bin_size = float(data["bin_size"])
    D = vectors.shape[1]
    mz_axis = mz_min + np.arange(D) * bin_size

    # Per spectrum: CDF (cumulative fraction of total intensity)
    totals = vectors.sum(axis=1)
    totals[totals == 0] = 1
    cdfs = np.cumsum(vectors, axis=1) / totals[:, np.newaxis]
    mean_cdf = np.mean(cdfs, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mz_axis, mean_cdf, color="steelblue", linewidth=2, label="Cumulative fraction of intensity")
    ax.axvline(mz_cut_low, color="gray", linestyle="--", linewidth=1.5, label=f"Lower bound ({mz_cut_low} m/z)")
    ax.axvline(mz_cut_high, color="gray", linestyle=":", linewidth=1.5, label=f"Upper bound ({mz_cut_high} m/z)")
    ax.set_xlabel("m/z (upper bound of range)")
    ax.set_ylabel("Fraction of total intensity captured")
    ax.set_title("m/z range justification: cumulative intensity (mean over spectra)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(mz_min, mz_max)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    _save("fig6_mz_coverage_cdf")


# -----------------------------------------------------------------------------
# Figure 7: Spectra count per food (requires spectrum-level parquet)
# -----------------------------------------------------------------------------

def fig_spectra_counts():
    parquet_path = GCMS_PROCESSED / "gcms_food_spectra.parquet"
    if not parquet_path.exists():
        print("Skipping spectra counts: parquet not found")
        return
    df = pd.read_parquet(parquet_path)
    counts = df.groupby("food").size().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(counts) * 0.25)))
    ax.barh(range(len(counts)), counts.values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("Number of GC-MS spectra")
    ax.set_title("Experimental GC-MS spectra per food", fontweight="bold")
    _save("fig7_spectra_counts_per_food")

def _resolve_food_index(foods, query: str) -> int:
    """
    Resolve a user-provided food name to an index in `foods`.
    Supports exact, case-insensitive, and substring match.
    """
    q = str(query).strip()
    if not q:
        raise ValueError("Empty ingredient name")

    foods_str = [str(f) for f in foods]

    # exact match
    for i, f in enumerate(foods_str):
        if f == q:
            return i

    # case-insensitive exact match
    q_low = q.lower()
    for i, f in enumerate(foods_str):
        if f.lower() == q_low:
            return i

    # normalized exact (spaces <-> underscores)
    q_norm = q_low.replace(" ", "_")
    for i, f in enumerate(foods_str):
        f_norm = f.lower().replace(" ", "_")
        if f_norm == q_norm:
            return i

    # substring match
    matches = []
    for i, f in enumerate(foods_str):
        f_low = f.lower()
        f_norm = f_low.replace(" ", "_")
        if q_low in f_low or q_norm in f_norm:
            matches.append((i, f))

    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous ingredient '{query}'. Matches: {[m[1] for m in matches[:10]]}"
            + (" ..." if len(matches) > 10 else "")
        )

    raise ValueError(f"Ingredient not found: '{query}'")


def _l1_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector to sum to 1 (treat as a distribution)."""
    v = np.asarray(v, dtype=np.float32)
    s = float(v.sum())
    if s <= eps:
        return v.copy()
    return v / s


def fig_fingerprint_difference(X, foods, mz_min, bin_size, food_a, food_b, normalize=True):
    """
    Plot two GC-MS fingerprints and their signed difference (A - B).
    If normalize=True, compare as distributions (sum=1), which usually makes
    differences easier to see.
    """
    ia = _resolve_food_index(foods, food_a)
    ib = _resolve_food_index(foods, food_b)

    name_a = str(foods[ia])
    name_b = str(foods[ib])

    va = np.asarray(X[ia], dtype=np.float32)
    vb = np.asarray(X[ib], dtype=np.float32)

    if normalize:
        va_plot = _l1_normalize(va)
        vb_plot = _l1_normalize(vb)
        ylab = "Normalized intensity (sum=1)"
        mode_tag = "normalized"
        title_suffix = " (distribution-normalized)"
    else:
        va_plot = va
        vb_plot = vb
        ylab = "Binned intensity"
        mode_tag = "raw"
        title_suffix = ""

    diff = va_plot - vb_plot
    mz_axis = mz_min + np.arange(X.shape[1]) * bin_size

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4]}
    )
    ax_top, ax_diff = axes

    # Top panel: both fingerprints
    ax_top.plot(mz_axis, va_plot, linewidth=1.1, label=name_a)
    ax_top.plot(mz_axis, vb_plot, linewidth=1.1, label=name_b)
    ax_top.fill_between(mz_axis, 0, va_plot, alpha=0.15)
    ax_top.fill_between(mz_axis, 0, vb_plot, alpha=0.15)
    ax_top.set_ylabel(ylab)
    ax_top.set_title(f"GC-MS fingerprint comparison: {name_a} vs {name_b}{title_suffix}", fontweight="bold")
    ax_top.legend(loc="upper right", framealpha=0.9)
    ax_top.grid(True, alpha=0.2)

    # Bottom panel: signed difference A - B
    ax_diff.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_diff.plot(mz_axis, diff, linewidth=1.0, label=f"{name_a} - {name_b}")
    ax_diff.fill_between(mz_axis, 0, diff, where=(diff >= 0), alpha=0.25, interpolate=True)
    ax_diff.fill_between(mz_axis, 0, diff, where=(diff < 0), alpha=0.25, interpolate=True)
    ax_diff.set_xlabel("m/z")
    ax_diff.set_ylabel("Difference")
    ax_diff.grid(True, alpha=0.2)

    ax_diff.text(
        0.99, 0.04,
        f"Positive = {name_a} stronger\nNegative = {name_b} stronger",
        transform=ax_diff.transAxes,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", alpha=0.15)
    )

    plt.tight_layout()
    _save(f"fig4_diff_{name_a}_minus_{name_b}_{mode_tag}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("Loading GC-MS data...")
    X, foods, mz_min, mz_max, bin_size = load_gcms_data()
    print(f"  {len(foods)} foods, {X.shape[1]} m/z bins")

    print("\nGenerating figures...")
    fig_pca(X, foods, mz_min, mz_max, bin_size)
    fig_heatmap(X, foods, mz_min, bin_size)
    fig_dendrogram(X, foods)
    fig_fingerprint_examples(X, foods, mz_min, bin_size)
    fig_coverage_summary()
    fig_mz_coverage_cdf()
    fig_spectra_counts()

    print("\nDone. Figures saved to:", PLOTS_DIR)


if __name__ == "__main__":
    args = [a.strip() for a in sys.argv[1:] if str(a).strip()]

    # Difference mode:
    #   python gcms_paper_figures.py --diff garlic strawberry
    #   python gcms_paper_figures.py --diff garlic strawberry --raw-diff
    if "--diff" in args:
        try:
            k = args.index("--diff")
            food_a = args[k + 1]
            food_b = args[k + 2]
        except Exception:
            print("Usage: python gcms_paper_figures.py --diff <ingredient_a> <ingredient_b> [--raw-diff]")
            sys.exit(1)

        raw_diff = ("--raw-diff" in args)

        print("Loading GC-MS data...")
        X, foods, mz_min, mz_max, bin_size = load_gcms_data()
        print(f"  Generating difference plot: {food_a} - {food_b} "
              f"({'raw' if raw_diff else 'distribution-normalized'})")

        try:
            fig_fingerprint_difference(
                X, foods, mz_min, bin_size,
                food_a=food_a,
                food_b=food_b,
                normalize=not raw_diff,
            )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print("Done. Figure saved to:", PLOTS_DIR)
        sys.exit(0)

    # Existing fingerprint mode (ignore future flags if you add more later)
    non_flag_args = [a for a in args if not a.startswith("--")]
    if non_flag_args:
        print("Loading GC-MS data...")
        X, foods, mz_min, mz_max, bin_size = load_gcms_data()
        print(f"  Generating fingerprint(s) for: {', '.join(non_flag_args)}")
        fig_fingerprint_examples(X, foods, mz_min, bin_size, food_names=non_flag_args)
        print("Done. Figure saved to:", PLOTS_DIR)
        sys.exit(0)

    main()