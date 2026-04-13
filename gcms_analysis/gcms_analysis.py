#!/usr/bin/env python

import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from food import food_ids as FOOD_QUERIES
import numpy as np
from adjustText import adjust_text

# ---------------------------------------------------------------------
# Plot saving helper
# ---------------------------------------------------------------------

_plot_save_dir = "plots"
os.makedirs(_plot_save_dir, exist_ok=True)

def _slugify(name: str) -> str:
    """Turn arbitrary text into a safe file-name slug."""
    name = name.strip()
    # collapse non-alphanumerics to underscores
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = name.strip("_")
    return name or "plot"

def _save_plot(name: str):
    """
    Save current figure as plots/<name>.png, avoiding overwrites
    by appending _1, _2, ...
    """
    base = _slugify(name)
    fname = os.path.join(_plot_save_dir, base + ".png")

    i = 1
    while os.path.exists(fname):
        fname = os.path.join(_plot_save_dir, f"{base}_{i}.png")
        i += 1

    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved plot to {fname}")

# coding: utf-8

# ---------------------------------------------------------------------
# FooDB JSON + experimental GC-MS metadata
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
# FooDB extracts and experimental spectra live under repo-root gcms_processed/ (see README).
JSON_DIR = _REPO_ROOT / "gcms_processed" / "foodb_2020_04_07_json"

CONTENT_JSON = JSON_DIR / "Content.json"
COMPOUND_JSON = JSON_DIR / "Compound.json"
FOOD_JSON = JSON_DIR / "Food.json"

CMS_DIR = _REPO_ROOT / "gcms_processed" / "foodb_experimental_cms_spectra"

def load_json_records(path: Path):
    """
    FooDB JSON files may be:
      - a single big JSON array, or
      - newline-delimited JSON (one object per line).
    Return a list of dicts.
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return []
        f.seek(0)
        if first == "[":
            return json.load(f)
        records = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

# ---------------------------------------------------------------------
# Build compound, food maps
# ---------------------------------------------------------------------

def build_internal_to_fdb(compound_path: Path):
    compounds = load_json_records(compound_path)
    mapping = {}
    for c in compounds:
        try:
            cid = int(c["id"])
        except Exception:
            continue
        public_id = str(c.get("public_id") or "").strip()
        if public_id.startswith("FDB"):
            mapping[cid] = public_id
    return mapping

def build_food_public_to_internal(food_path: Path):
    foods = load_json_records(food_path)
    mapping = {}
    for f in foods:
        try:
            fid = int(f["id"])
        except Exception:
            continue
        public_id = str(f.get("public_id") or "").strip()
        if public_id.startswith("FOOD"):
            mapping[public_id] = fid
    return mapping

# ---------------------------------------------------------------------
# Map a single food query -> set of FDB IDs
# ---------------------------------------------------------------------

def get_compounds_for_food(query: str,
                           contents,
                           internal_to_fdb,
                           food_public_to_internal):
    """
    query:
        - if starts with 'FOOD', it's treated as food public_id (FOODxxxxx)
        - otherwise it's treated as a substring to search in
          orig_food_common_name / orig_food_scientific_name.

    contents: list of Content.json rows
    internal_to_fdb: {internal_compound_id -> FDBxxxxx}
    food_public_to_internal: {FOODxxxxx -> internal_food_id}
    """
    internal_comp_ids = set()

    # Case 1: FOOD public ID (e.g. 'FOOD00083')
    if query.upper().startswith("FOOD"):
        internal_food_id = food_public_to_internal.get(query.upper())
        if internal_food_id is None:
            return set()
        for row in contents:
            if str(row.get("source_type", "")).strip().lower() != "compound":
                continue
            try:
                if int(row.get("food_id", -1)) == internal_food_id:
                    internal_comp_ids.add(int(row["source_id"]))
            except Exception:
                continue

    # Case 2: name substring (e.g. 'strawberry')
    else:
        q = query.lower()
        for row in contents:
            if str(row.get("source_type", "")).strip().lower() != "compound":
                continue
            common = (row.get("orig_food_common_name") or "").lower()
            sci = (row.get("orig_food_scientific_name") or "").lower()
            if q in common or q in sci:
                try:
                    internal_comp_ids.add(int(row["source_id"]))
                except Exception:
                    continue

    # Map internal compound IDs -> FDB IDs
    return {internal_to_fdb[i] for i in internal_comp_ids if i in internal_to_fdb}

# ---------------------------------------------------------------------
# Parse experimental C-MS spectra XML
# ---------------------------------------------------------------------

def parse_cms_xml(path: Path):
    root = ET.parse(path).getroot()

    def txt(tag):
        el = root.find(tag)
        return el.text if el is not None else None

    meta = {
        "spectrum_id": txt("id"),
        "database_id": txt("database-id"),        # FDBxxxxx
        "structure_id": txt("structure-id"),
        "instrument_type": txt("instrument-type"),
        "chromatography_type": txt("chromatography-type"),
        "ionization_mode": txt("ionization-mode"),
        "splash": txt("splash-key"),
        "predicted": txt("predicted"),
        "peak_counter": txt("peak-counter"),
        "path": str(path),
    }

    mz_vals, int_vals = [], []
    peaks = root.find("c-ms-peaks")
    if peaks is not None:
        for p in peaks.findall("c-ms-peak"):
            try:
                mz_vals.append(float(p.findtext("mass-charge")))
                int_vals.append(float(p.findtext("intensity")))
            except Exception:
                continue
    return meta, mz_vals, int_vals

def build_compound_to_spectra(cms_dir: Path):
    compound_to_spectra = defaultdict(list)
    xml_files = list(cms_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} C-MS XML files")

    for xml_path in xml_files:
        meta, mz, inten = parse_cms_xml(xml_path)

        dbid = meta.get("database_id")
        if not dbid or not dbid.startswith("FDB"):
            continue
        if meta.get("chromatography_type") != "GC":
            continue
        pred = (meta.get("predicted") or "").strip().lower()
        if pred == "true":
            continue

        spec = {**meta, "mz": mz, "intensity": inten}
        compound_to_spectra[dbid].append(spec)

    print("Compounds with at least one GC experimental spectrum:",
          len(compound_to_spectra))
    return compound_to_spectra

# ---------------------------------------------------------------------
# Combine: food -> spectra
# ---------------------------------------------------------------------

def dedupe_spectra(spec_list):
    seen = {}
    for s in spec_list:
        key = (s.get("database_id"), s.get("splash"), s.get("instrument_type"))
        if key not in seen:
            seen[key] = s
    return list(seen.values())

def build_food_to_spectra(food_queries,
                          contents,
                          internal_to_fdb,
                          food_public_to_internal,
                          compound_to_spectra):
    food_to_spectra = {}
    for label, query in food_queries.items():
        fdb_ids = get_compounds_for_food(query, contents,
                                         internal_to_fdb,
                                         food_public_to_internal)
        specs = []
        for fdb in fdb_ids:
            specs.extend(compound_to_spectra.get(fdb, []))
        specs = dedupe_spectra(specs)
        food_to_spectra[label] = specs
        print(f"{label}: {len(fdb_ids)} compounds, {len(specs)} spectra (query='{query}')")
    return food_to_spectra


# ------------------------
# 1. Spectrum binning
# ------------------------

def bin_spectrum(mz, inten, mz_min=40.0, mz_max=400.0, bin_size=1.0):
    """
    Convert lists of (mz, intensity) into a fixed-length vector by binning.

    mz: list/array of m/z values
    inten: list/array of intensities (relative; any scale)
    """
    n_bins = int((mz_max - mz_min) / bin_size)
    v = np.zeros(n_bins, dtype=np.float32)

    if mz is None or len(mz) == 0:
        return v

    mz = np.asarray(mz, dtype=np.float32)
    inten = np.asarray(inten, dtype=np.float32)

    # Normalize intensities so max = 1
    max_i = inten.max()
    if max_i > 0:
        inten = inten / max_i
    else:
        return v

    for m, i in zip(mz, inten):
        if m < mz_min or m >= mz_max:
            continue
        idx = int((m - mz_min) // bin_size)
        if 0 <= idx < n_bins:
            # use max or sum; max is often nice for EI spectra
            v[idx] = max(v[idx], i)

    return v


# ------------------------
# 2. Build spectrum-level dataset from food_to_spectra
# ------------------------

def build_spectrum_level_datasets(food_to_spectra,
                                  out_spectra_parquet="gcms_food_spectra.parquet",
                                  out_spectra_npz="gcms_food_spectra_vectors.npz",
                                  mz_min=40.0,
                                  mz_max=400.0,
                                  bin_size=1.0):
    """
    Inputs:
        food_to_spectra: dict[food_name] -> list of spectrum dicts.
                         Each spectrum dict must have:
                            - 'mz': list[float]
                            - 'intensity': list[float]
                            - 'database_id' (FDBxxxxx, optional but recommended)
                            - 'spectrum_id' (optional; used in spec_id)
                            - 'instrument_type', 'chromatography_type',
                              'ionization_mode', 'splash' (optional)

    Outputs:
        - Parquet with per-spectrum metadata
        - NPZ with:
            vectors: (N_spectra, D)
            spec_ids: (N_spectra,)
            food_labels: (N_spectra,)
            fdb_ids: (N_spectra,)
            mz_min, mz_max, bin_size
    """
    rows = []
    vectors = []
    spec_ids = []
    food_labels = []
    fdb_ids = []

    for food_name, specs in food_to_spectra.items():
        for s in specs:
            mz = s.get("mz") or []
            inten = s.get("intensity") or []

            if len(mz) == 0 or len(inten) == 0:
                continue

            fdb_id = s.get("database_id", None)
            spectrum_id = s.get("spectrum_id", None)

            # build a stable spec_id
            if fdb_id and spectrum_id:
                spec_id = f"{food_name}__{fdb_id}__{spectrum_id}"
            elif fdb_id:
                spec_id = f"{food_name}__{fdb_id}"
            else:
                # fallback
                spec_id = f"{food_name}__{len(spec_ids)}"

            vec = bin_spectrum(mz, inten, mz_min=mz_min, mz_max=mz_max, bin_size=bin_size)

            spec_ids.append(spec_id)
            food_labels.append(food_name)
            fdb_ids.append(fdb_id)
            vectors.append(vec)

            rows.append({
                "spec_id": spec_id,
                "food": food_name,
                "fdb_id": fdb_id,
                "spectrum_id": spectrum_id,
                "instrument_type": s.get("instrument_type"),
                "chrom_type": s.get("chromatography_type"),
                "ion_mode": s.get("ionization_mode"),
                "splash": s.get("splash"),
                "num_peaks": len(mz),
            })

    if not rows:
        raise ValueError("No spectra found to store. Check food_to_spectra contents.")

    df = pd.DataFrame(rows)
    df.to_parquet(out_spectra_parquet, index=False)

    X = np.stack(vectors, axis=0)

    np.savez_compressed(
        out_spectra_npz,
        vectors=X,
        spec_ids=np.array(spec_ids, dtype=object),
        food_labels=np.array(food_labels, dtype=object),
        fdb_ids=np.array(fdb_ids, dtype=object),
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
    )

    print(f"[spectrum-level] Saved {len(df)} spectra "
          f"with dimension {X.shape[1]} to:")
    print("  -", out_spectra_parquet)
    print("  -", out_spectra_npz)


# ------------------------
# 3. Build food-level vectors (aggregate over spectra)
# ------------------------

def build_food_level_dataset(
    spectra_npz="gcms_food_spectra_vectors.npz",
    out_food_npz="gcms_food_vectors.npz",
    agg="mean",
):
    """
    Aggregate spectrum vectors per food label.

    Inputs:
        spectra_npz: NPZ file from build_spectrum_level_datasets
        agg: 'mean' or 'max'

    Outputs:
        out_food_npz: NPZ with
            food_labels: (N_foods,)
            vectors: (N_foods, D)
            (plus mz_min, mz_max, bin_size copied through)
    """
    data = np.load(spectra_npz, allow_pickle=True)
    X = data["vectors"]              # (N_spectra, D)
    food_labels = data["food_labels"]  # (N_spectra,)

    # group indices by food
    idxs_by_food = defaultdict(list)
    for i, food in enumerate(food_labels):
        idxs_by_food[food].append(i)

    food_names = []
    food_vecs = []

    for food, idxs in idxs_by_food.items():
        V = X[idxs]  # shape (n_spectra_for_food, D)
        if agg == "mean":
            vec = V.mean(axis=0)
        elif agg == "max":
            vec = V.max(axis=0)
        else:
            raise ValueError(f"Unknown agg: {agg}")
        food_names.append(food)
        food_vecs.append(vec)

    F = np.stack(food_vecs, axis=0)

    # carry over binning metadata
    meta_keys = [k for k in ("mz_min", "mz_max", "bin_size") if k in data]

    np.savez_compressed(
        out_food_npz,
        food_labels=np.array(food_names, dtype=object),
        vectors=F,
        **{k: data[k] for k in meta_keys},
    )

    print(f"[food-level] Saved {len(food_names)} foods with dimension {F.shape[1]} to:")
    print("  -", out_food_npz)


# ------------------------
# 4. Convenience wrapper
# ------------------------

def build_all_gcms_datasets(food_to_spectra,
                            out_dir="gcms_processed",
                            mz_min=40.0,
                            mz_max=400.0,
                            bin_size=1.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spectra_parquet = out_dir / "gcms_food_spectra.parquet"
    spectra_npz = out_dir / "gcms_food_spectra_vectors.npz"
    food_npz = out_dir / "gcms_food_vectors.npz"

    build_spectrum_level_datasets(
        food_to_spectra,
        out_spectra_parquet=str(spectra_parquet),
        out_spectra_npz=str(spectra_npz),
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
    )

    build_food_level_dataset(
        spectra_npz=str(spectra_npz),
        out_food_npz=str(food_npz),
        agg="mean",
    )

    print("Done.")


# ---------------------------------------------------------------------
# Build maps and food_to_spectra
# ---------------------------------------------------------------------

print("Loading JSON tables...")
contents = load_json_records(CONTENT_JSON)
internal_to_fdb = build_internal_to_fdb(COMPOUND_JSON)
food_public_to_internal = build_food_public_to_internal(FOOD_JSON)
print("  Content rows:", len(contents))
print("  internal_to_fdb:", len(internal_to_fdb),
      "  food_public_to_internal:", len(food_public_to_internal))

print("\nParsing experimental C-MS spectra...")
compound_to_spectra = build_compound_to_spectra(CMS_DIR)

print("\nBuilding food → spectra map...")
food_to_spectra = build_food_to_spectra(
    FOOD_QUERIES,
    contents,
    internal_to_fdb,
    food_public_to_internal,
    compound_to_spectra,
)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_raw_spectrum(spectrum, title=None):
    mz = np.array(spectrum["mz"], dtype=float)
    inten = np.array(spectrum["intensity"], dtype=float)

    # normalize to 0–1 so different spectra are comparable
    if inten.max() > 0:
        inten = inten / inten.max()

    plt.figure(figsize=(8, 4))
    # stick plot style
    for m, i in zip(mz, inten):
        plt.vlines(m, 0, i)
    plt.xlabel("m/z")
    plt.ylabel("Relative intensity")

    if title:
        plt.title(title)
        save_name = title
    else:
        dbid = spectrum.get("database_id", "unknown")
        save_name = f"raw_spectrum_{dbid}"

    plt.tight_layout()
    _save_plot(save_name)


def plot_food_spectra_overlay(food_name, food_to_spectra, max_specs=5):
    plt.figure(figsize=(8, 4))
    specs = food_to_spectra[food_name][:max_specs]

    for idx, s in enumerate(specs):
        mz = np.array(s["mz"], dtype=float)
        inten = np.array(s["intensity"], dtype=float)
        if inten.max() > 0:
            inten = inten / inten.max()
        # offset each spectrum slightly so they don't sit exactly on top
        offset = idx * 0.1
        for m, i in zip(mz, inten):
            plt.vlines(m, offset, offset + i)

    plt.xlabel("m/z")
    plt.ylabel("Relative intensity (offset per spectrum)")
    plt.title(f"{food_name}: first {len(specs)} spectra")
    plt.tight_layout()

    save_name = f"{food_name}_overlay_first_{len(specs)}_spectra"
    _save_plot(save_name)


# Example: first spectrum for Strawberry
food = "Apple"
spec = food_to_spectra[food][0]    # first spectrum for that food
plot_raw_spectrum(spec, title=f"{food} – {spec['database_id']}")

food = "Star anise"
spec = food_to_spectra[food][0]    # first spectrum for that food
plot_raw_spectrum(spec, title=f"{food} – {spec['database_id']}")


# # ---------------------------------------------------------------------
# # Load binned food-level vectors and plot
# # ---------------------------------------------------------------------

data = np.load(_REPO_ROOT / "gcms_processed" / "gcms_food_vectors.npz", allow_pickle=True)
X = data["vectors"]          # shape (N_foods, D)
foods = data["food_labels"]  # shape (N_foods,)
mz_min = float(data["mz_min"])
mz_max = float(data["mz_max"])
bin_size = float(data["bin_size"])

def plot_food_binned_vector(food_name):
    # find food index
    idx = np.where(foods == food_name)[0]
    if len(idx) == 0:
        print("Food not found:", food_name)
        return
    idx = idx[0]

    v = X[idx]   # (D,)
    D = len(v)
    mz_axis = mz_min + np.arange(D) * bin_size

    plt.figure(figsize=(8, 4))
    plt.plot(mz_axis, v)
    plt.xlabel("m/z (binned)")
    plt.ylabel("Binned intensity")
    plt.title(f"{food_name} – GC-MS fingerprint")
    plt.tight_layout()

    save_name = f"{food_name}_gcms_fingerprint"
    _save_plot(save_name)

plot_food_binned_vector("Star anise")
plot_food_binned_vector("Apple")


# def plot_food_heatmap(X, foods, mz_min, bin_size, max_foods=30):
#     # take first N foods just to keep the plot manageable
#     X_sub = X[:max_foods]
#     foods_sub = foods[:max_foods]

#     plt.figure(figsize=(10, 0.3 * len(foods_sub) + 2))
#     plt.imshow(X_sub, aspect="auto", origin="lower")
#     plt.colorbar(label="Binned intensity")
#     plt.yticks(range(len(foods_sub)), foods_sub)
#     D = X_sub.shape[1]
#     plt.xticks(
#         [0, D // 4, D // 2, 3 * D // 4, D - 1],
#         [int(mz_min + i * bin_size) for i in [0, D // 4, D // 2, 3 * D // 4, D - 1]],
#         rotation=45,
#     )
#     plt.xlabel("m/z")
#     plt.title("GC-MS fingerprints across foods")
#     plt.tight_layout()

#     save_name = f"gcms_heatmap_{len(foods_sub)}_foods"
#     _save_plot(save_name)

# plot_food_heatmap(X, foods, mz_min, bin_size, max_foods=25)


# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# Z = pca.fit_transform(X)   # X from gcms_food_vectors.npz

# plt.figure(figsize=(8, 8))
# plt.scatter(Z[:, 0], Z[:, 1], s=20, alpha=0.5)

# texts = []
# for x, y, label in zip(Z[:, 0], Z[:, 1], foods):
#     texts.append(plt.text(x, y, label, fontsize=8))
# adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.5))

# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Foods in GC-MS space (PCA)")
# plt.tight_layout()
# _save_plot("foods_gcms_pca")
# plt.show()
