#!/usr/bin/env python3
# analyze_runs.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel as _paired_ttest
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# --------------------------- IO helpers ---------------------------

def find_jsonl_files(root: Path) -> list[Path]:
    """
    Find all results.jsonl files under root (both nested and flat).
    """
    files = list(root.rglob("results.jsonl"))
    files += list(root.glob("*.jsonl"))  # in case you kept a flat file at root
    # de-dup
    uniq, seen = [], set()
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def flatten_record(rec: dict) -> dict:
    """
    Convert one JSON line (dict) into a flat row suitable for a DataFrame.
    Pulls common metrics up and keeps 'per_category' dict for later expansion.
    """
    out = {k: v for k, v in rec.items() if k != "results"}
    res = rec.get("results", {}) or {}

    # Normalize common fields
    out["model"]        = rec.get("model")
    out["contrastive"]  = bool(rec.get("contrastive", False))
    out["gradient"]     = rec.get("gradient")
    out["window"]       = rec.get("window_size", rec.get("window"))
    out["seed"]         = rec.get("seed")
    out["lr"]           = rec.get("lr")
    out["batch"]        = rec.get("batch_size")
    out["epochs"]       = rec.get("epochs")

    # Metrics
    out["acc1"] = _to_float(res.get("acc@1"))
    out["acc5"] = _to_float(res.get("acc@5"))
    out["precision_macro"] = _to_float(res.get("precision_macro"))
    out["recall_macro"]    = _to_float(res.get("recall_macro"))
    out["f1_macro"]        = _to_float(res.get("f1_macro"))

    # Keep for category expansion
    out["per_category"] = res.get("per_category")

    return out


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for jf in find_jsonl_files(root):
        try:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        row = flatten_record(rec)
                        row["_source_file"] = str(jf)
                        row["run_name"] = rec.get("run_name")
                        row["timestamp"] = rec.get("timestamp")
                        rows.append(row)
                    except Exception:
                        # skip malformed json line
                        pass
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No runs found under {root}")
    # Force numeric where applicable
    for c in ("acc1","acc5","precision_macro","recall_macro","f1_macro","gradient","window","seed","lr","batch","epochs"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "contrastive" in df.columns:
        df["contrastive"] = df["contrastive"].astype(bool)
    return df


# --------------------------- selection & pairing ---------------------------

def best_by_model_classification(df: pd.DataFrame, metric: str = "acc1") -> pd.DataFrame:
    """
    Pick the single best non-contrastive run per model using `metric` (ties -> acc5).
    """
    d = df[df["contrastive"] == False].copy()
    if d.empty:
        return d
    secondary = "acc5" if metric != "acc5" else "acc1"
    d = d.sort_values(["model", metric, secondary], ascending=[True, False, False])
    keep = [
        "model","contrastive","acc1","acc5","precision_macro","recall_macro","f1_macro",
        "gradient","window","seed","lr","batch","epochs",
        "run_name","timestamp","per_category","_source_file"
    ]
    keep = [c for c in keep if c in d.columns]
    return d.groupby("model", as_index=False).first()[keep]


def top_overall(df: pd.DataFrame, n: int = 20, metric: str = "acc1") -> pd.DataFrame:
    secondary = "acc5" if metric != "acc5" else "acc1"
    cols = [
        "model","contrastive","acc1","acc5","precision_macro","recall_macro","f1_macro",
        "gradient","window","seed","lr","batch","epochs",
        "run_name","timestamp","_source_file"
    ]
    cols = [c for c in cols if c in df.columns]
    return df.sort_values([metric, secondary], ascending=[False, False])[cols].head(n)


def paired_contrastive(df: pd.DataFrame, metric: str = "acc1") -> pd.DataFrame:
    """
    Build matched pairs for (model, gradient, window, seed).
    If multiple runs exist on a side, keep the one with best `metric`.
    """
    keys = ["model","gradient","window","seed"]
    for col in [metric]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    base_cols = keys + [
        "acc1","acc5","precision_macro","recall_macro","f1_macro","run_name","per_category"
    ]

    noCL = (df[df["contrastive"] == False]
            .sort_values(keys + [metric], ascending=[True, True, True, True, False])
            .drop_duplicates(subset=keys, keep="first")[base_cols])

    CL = (df[df["contrastive"] == True]
          .sort_values(keys + [metric], ascending=[True, True, True, True, False])
          .drop_duplicates(subset=keys, keep="first")[base_cols])

    paired = noCL.merge(CL, on=keys, suffixes=("_noCL", "_CL"))
    if paired.empty:
        return paired

    paired["delta_acc1"] = paired["acc1_CL"] - paired["acc1_noCL"]
    paired["delta_acc5"] = paired["acc5_CL"] - paired["acc5_noCL"]
    # optional macro deltas
    if "f1_macro_noCL" in paired and "f1_macro_CL" in paired:
        paired["delta_f1"] = paired["f1_macro_CL"] - paired["f1_macro_noCL"]

    return paired


# --------------------------- per-category expansion ---------------------------

def expand_per_category(df_rows: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Expand 'per_category' dict (if present) to long rows:
    columns: model, contrastive, gradient, window, seed, category, {prefix}n, {prefix}acc1, {prefix}acc5, run_name
    """
    out = []
    for _, r in df_rows.iterrows():
        pc = r.get("per_category") or {}
        if not isinstance(pc, dict):
            continue
        for cat, vals in pc.items():
            out.append({
                "run_name": r.get("run_name"),
                "model": r.get("model"),
                "contrastive": bool(r.get("contrastive", False)),
                "gradient": r.get("gradient"),
                "window": r.get("window"),
                "seed": r.get("seed"),
                "category": cat,
                f"{prefix}n": vals.get("n"),
                f"{prefix}acc1": vals.get("acc@1"),
                f"{prefix}acc5": vals.get("acc@5"),
            })
    return pd.DataFrame(out)


def expand_pair_per_category(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Long per-category comparison for each matched pair:
    columns: model, gradient, window, seed, category,
             n_noCL, acc1_noCL, acc5_noCL, n_CL, acc1_CL, acc5_CL,
             delta_acc1, delta_acc5, run_name_noCL, run_name_CL
    """
    rows = []
    for _, r in pairs.iterrows():
        pc_no = r.get("per_category_noCL") or {}
        pc_cl = r.get("per_category_CL") or {}
        cats = sorted(set(pc_no.keys()) | set(pc_cl.keys()))
        for cat in cats:
            a_no = pc_no.get(cat, {})
            a_cl = pc_cl.get(cat, {})
            acc1_no = a_no.get("acc@1")
            acc1_cl = a_cl.get("acc@1")
            acc5_no = a_no.get("acc@5")
            acc5_cl = a_cl.get("acc@5")
            rows.append({
                "model": r["model"], "gradient": r["gradient"], "window": r["window"], "seed": r["seed"],
                "category": cat,
                "n_noCL": a_no.get("n"), "acc1_noCL": acc1_no, "acc5_noCL": acc5_no,
                "n_CL": a_cl.get("n"), "acc1_CL": acc1_cl, "acc5_CL": acc5_cl,
                "delta_acc1": (acc1_cl - acc1_no) if (acc1_no is not None and acc1_cl is not None) else None,
                "delta_acc5": (acc5_cl - acc5_no) if (acc5_no is not None and acc5_cl is not None) else None,
                "run_name_noCL": r["run_name_noCL"], "run_name_CL": r["run_name_CL"],
            })
    return pd.DataFrame(rows)


def best_by_model_and_gradient(df: pd.DataFrame, metric: str = "acc1") -> pd.DataFrame:
    """
    Pick the single best run per (model, gradient) using `metric`.
    Ties are broken by a secondary metric (acc5 unless acc5 is the primary).
    Returns one row per (model, gradient).
    """
    if df.empty:
        return df

    # Ensure gradient is numeric and not NaN
    if "gradient" in df.columns:
        df = df.copy()
        df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
        df = df.dropna(subset=["gradient"])

    secondary = "acc5" if metric != "acc5" else "acc1"

    # Sort so .drop_duplicates keeps the best row per (model, gradient)
    order_cols = ["model", "gradient", metric, secondary]
    existing_order_cols = [c for c in order_cols if c in df.columns]
    d = df.sort_values(existing_order_cols, ascending=[True, True, False, False][:len(existing_order_cols)])

    keep = [
        "model","gradient","contrastive","acc1","acc5","precision_macro","recall_macro","f1_macro",
        "window","seed","lr","batch","epochs","run_name","timestamp","per_category","_source_file"
    ]
    keep = [c for c in keep if c in d.columns]

    return d.drop_duplicates(subset=["model","gradient"], keep="first")[keep]


# --------------------------- CLI / main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate SmellNet experiment logs.")
    ap.add_argument("--log-dir", required=True, help="Root folder containing per-run results.jsonl")
    ap.add_argument("--out", default=None, help="Output folder (default: same as --log-dir)")
    ap.add_argument("--select-metric", default="acc1",
                    choices=["acc1","acc5","precision_macro","recall_macro","f1_macro"],
                    help="Metric used to pick best runs (default: acc1)")
    ap.add_argument("--top-n", type=int, default=20, help="How many runs to list in topN_overall.csv")
    # Filters (optional)
    ap.add_argument("--models", nargs="*", default=None, help="Restrict to these models")
    ap.add_argument("--gradients", nargs="*", type=int, default=None, help="Restrict to these gradient values")
    ap.add_argument("--windows", nargs="*", type=int, default=None, help="Restrict to these window sizes")
    ap.add_argument("--contrastive", default="both", choices=["both","off","on"],
                    help="Filter by contrastive mode (default: both)")
    ap.add_argument("--ttest", action="store_true", help="Compute paired t-tests for CL deltas if SciPy is available")
    args = ap.parse_args()

    root = Path(args.log_dir).resolve()
    outdir = Path(args.out).resolve() if args.out else root / "summary"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_runs(root)

    # Apply filters
    if args.models:
        df = df[df["model"].isin(args.models)]
    if args.gradients:
        df = df[df["gradient"].isin(args.gradients)]
    if args.windows:
        df = df[df["window"].isin(args.windows)]
    if args.contrastive != "both":
        want = (args.contrastive == "on")
        df = df[df["contrastive"] == want]

    # Save all flat
    df.to_csv(outdir / "all_runs_flat.csv", index=False)

    # Best non-contrastive per model
    df_best_cls = best_by_model_classification(df, metric=args.select_metric)
    if not df_best_cls.empty:
        df_best_cls.to_csv(outdir / "best_by_model_classification.csv", index=False)

    # Top-N overall
    topN = top_overall(df, n=args.top_n, metric=args.select_metric)
    topN.to_csv(outdir / "topN_overall.csv", index=False)

    # Paired CL vs no-CL
    df_pairs = paired_contrastive(df, metric=args.select_metric)
    if not df_pairs.empty:
        df_pairs.to_csv(outdir / "contrastive_pairs.csv", index=False)

        # Per-model summary of deltas
        summary = (df_pairs.groupby("model")
                   .agg(n_pairs=("model","size"),
                        mean_acc1_noCL=("acc1_noCL","mean"),
                        mean_acc1_CL=("acc1_CL","mean"),
                        mean_delta_acc1=("delta_acc1","mean"),
                        mean_acc5_noCL=("acc5_noCL","mean"),
                        mean_acc5_CL=("acc5_CL","mean"),
                        mean_delta_acc5=("delta_acc5","mean"))
                   .reset_index())

        # Optional paired t-tests across seeds if requested & SciPy present
        if args.ttest and _HAVE_SCIPY:
            pvals = []
            for m, g in summary["model"].items():
                pass
            # compute per model p-values on acc1 deltas
            pmap = {}
            for m, g in df_pairs.groupby("model"):
                try:
                    a = g["acc1_noCL"].to_numpy(dtype=float)
                    b = g["acc1_CL"].to_numpy(dtype=float)
                    if len(a) >= 2 and len(a) == len(b):
                        stat = _paired_ttest(a, b, alternative="two-sided")
                        pmap[m] = float(stat.pvalue)
                except Exception:
                    pmap[m] = np.nan
            summary["pvalue_acc1"] = summary["model"].map(pmap)
        summary.to_csv(outdir / "contrastive_pairs_summary.csv", index=False)

    # Per-category exports
    # (a) For best classification runs (non-contrastive)
    if not df_best_cls.empty:
        pc_best_long = expand_per_category(df_best_cls)
        if not pc_best_long.empty:
            pc_best_long.to_csv(outdir / "per_category_best_classification.csv", index=False)
            # wide pivot of acc1
            pc_best_wide = (pc_best_long
                            .pivot_table(index="model", columns="category", values="acc1", aggfunc="first")
                            .sort_index(axis=1))
            pc_best_wide.to_csv(outdir / "per_category_best_classification_wide_acc1.csv")

    # (b) For paired CL vs no-CL (same (model, gradient, window, seed))
    if not df_pairs.empty:
        pc_pairs_long = expand_pair_per_category(df_pairs)
        if not pc_pairs_long.empty:
            pc_pairs_long.to_csv(outdir / "per_category_paired_contrastive.csv", index=False)
            # Average Δacc1 across seeds -> wide pivot
            pc_pairs_wide = (pc_pairs_long
                             .groupby(["model","category"], as_index=False)["delta_acc1"].mean()
                             .pivot(index="model", columns="category", values="delta_acc1")
                             .sort_index(axis=1))
            pc_pairs_wide.to_csv(outdir / "per_category_paired_contrastive_wide_delta_acc1.csv")

    print("Saved outputs to:", outdir.as_posix())
    print(" - all_runs_flat.csv")
    if not df_best_cls.empty:
        print(" - best_by_model_classification.csv")
        print(" - per_category_best_classification.csv")
        print(" - per_category_best_classification_wide_acc1.csv")
    print(" - topN_overall.csv")
    if not df_pairs.empty:
        print(" - contrastive_pairs.csv")
        print(" - contrastive_pairs_summary.csv")
        print(" - per_category_paired_contrastive.csv")
        print(" - per_category_paired_contrastive_wide_delta_acc1.csv")
    if args.ttest and not _HAVE_SCIPY:
        print("Note: --ttest requested but SciPy not available; skipped p-values.")

    # Best per model overall (considers both CL and non-CL)
    secondary = "acc5" if args.select_metric != "acc5" else "acc1"
    cols = [
        "model","contrastive","acc1","acc5","precision_macro","recall_macro","f1_macro",
        "gradient","window","seed","lr","batch","epochs","run_name","timestamp","per_category","_source_file"
    ]
    cols = [c for c in cols if c in df.columns]
    df_best_overall = (df.sort_values(["model", args.select_metric, secondary],
                                    ascending=[True, False, False])
                        .groupby("model", as_index=False)
                        .first()[cols])
    df_best_overall.to_csv(outdir / "best_by_model_overall.csv", index=False)

    # (optional) per-category for those overall winners
    pc_best_overall = expand_per_category(df_best_overall)
    if not pc_best_overall.empty:
        pc_best_overall.to_csv(outdir / "per_category_best_overall.csv", index=False)
        pc_best_overall.pivot_table(index="model", columns="category", values="acc1", aggfunc="first") \
                    .to_csv(outdir / "per_category_best_overall_wide_acc1.csv")
        
    
    # Best run per (model, gradient) pair
    df_best_mg = best_by_model_and_gradient(df, metric=args.select_metric)
    if not df_best_mg.empty:
        df_best_mg.to_csv(outdir / "best_by_model_and_gradient.csv", index=False)
        print(" - best_by_model_and_gradient.csv")



if __name__ == "__main__":
    main()
