#!/usr/bin/env python3
# analyze_run_mixture.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

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
    Convert one JSON line (dict) from a *mixture* run into a flat row suitable for a DataFrame.

    Example source schema (shortened):
    {
        "timestamp": "...",
        "run_name": "mix-cnn-g0-w100-bs64-lr0.001-seed42-...",
        "model": "cnn",
        "gradient": 0,
        "window_size": 100,
        "epochs": 90,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 42,
        "dataset": {"train_windows": 11011, "test_windows": 2717, "T": 100, "C": 5, "classes": 12},
        "results": {
            "kl": 0.8483,
            "mae": 0.0657,
            "acc@0.1": 0.3018,
            "acc@0.2": 0.5322,
            "dynTopK%": 73.1468,
            "presence": {"f1": 0.7369, "precision": 0.7467, "recall": 0.7274}
        }
    }
    """
    out = {}
    out["timestamp"] = rec.get("timestamp")
    out["run_name"]  = rec.get("run_name")
    out["_source_file"] = None  # filled by loader

    # Hyperparams / config
    out["model"]    = rec.get("model")
    out["gradient"] = rec.get("gradient")
    out["window"]   = rec.get("window_size", rec.get("window"))
    out["epochs"]   = rec.get("epochs")
    out["batch"]    = rec.get("batch_size")
    out["lr"]       = rec.get("lr")
    out["seed"]     = rec.get("seed")
    out["device"]   = rec.get("device")
    out["fft"]      = bool(rec.get("fft", False))
    out["fft_cutoff"] = rec.get("fft_cutoff")
    out["sampling_rate"] = rec.get("sampling_rate")
    out["stride"]   = rec.get("stride")
    out["dtype"]    = rec.get("dtype")

    # Dataset details (prefixed with ds_ to avoid collisions)
    ds = rec.get("dataset") or {}
    if isinstance(ds, dict):
        out["ds_train_windows"] = ds.get("train_windows")
        out["ds_test_windows"]  = ds.get("test_windows")
        out["ds_T"]             = ds.get("T")
        out["ds_C"]             = ds.get("C")
        out["ds_classes"]       = ds.get("classes")

    # Results / metrics (mixture-specific)
    res = rec.get("results") or {}
    out["kl"]  = _to_float(res.get("kl"))
    out["mae"] = _to_float(res.get("mae"))
    # accuracy thresholds
    out["acc@0.1"] = _to_float(res.get("acc@0.1"))
    out["acc@0.2"] = _to_float(res.get("acc@0.2"))
    out["dynTopK%"] = _to_float(res.get("dynTopK%"))

    # presence block
    pres = res.get("presence") or {}
    if isinstance(pres, dict):
        out["presence_f1"]        = _to_float(pres.get("f1"))
        out["presence_precision"] = _to_float(pres.get("precision"))
        out["presence_recall"]    = _to_float(pres.get("recall"))
    else:
        out["presence_f1"] = out["presence_precision"] = out["presence_recall"] = np.nan

    # Keep raw blobs for debugging (optional)
    # out["raw_results"] = res

    return out


def load_runs(root: Path) -> pd.DataFrame:
    rows = []
    for jf in find_jsonl_files(root):
        try:
            with jf.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):  # <-- track order
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    row = flatten_record(rec)
                    row["_source_file"] = str(jf)
                    row["_lineno"] = lineno               # <-- keep file order
                    rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No runs found under {root}")

    # --- NEW: prefer the *last* row written for a given run_name ---
    # If run_name is missing for any row, we skip dedup on those.
    if "run_name" in df.columns:
        df = (df
              .sort_values(["_source_file", "run_name", "_lineno"])
              .drop_duplicates(subset=["run_name"], keep="last")
             )
    # ---------------------------------------------------------------

    # Cast numerics (unchanged)
    for c in [
        "gradient","window","epochs","batch","lr","seed","fft_cutoff","sampling_rate","stride",
        "ds_train_windows","ds_test_windows","ds_T","ds_C","ds_classes",
        "kl","mae","acc@0.1","acc@0.2","dynTopK%",
        "presence_f1","presence_precision","presence_recall"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "fft" in df.columns:
        df["fft"] = df["fft"].astype(bool)

    return df


# --------------------------- selection helpers ---------------------------

# Whether higher is better for each metric
_HIGHER_IS_BETTER = {
    "kl": False,
    "mae": False,
    "acc@0.1": True,
    "acc@0.2": True,
    "dynTopK%": True,
    "presence_f1": True,
    "presence_precision": True,
    "presence_recall": True,
}

def _sort_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort dataframe by the chosen metric with correct ascending/descending direction.
    Tie-breakers: prefer presence_f1 then acc@0.2 (if available).
    """
    ascending = not _HIGHER_IS_BETTER.get(metric, True)
    tie1 = "presence_f1" if metric != "presence_f1" and "presence_f1" in df.columns else None
    tie2 = "acc@0.2" if metric != "acc@0.2" and "acc@0.2" in df.columns else None
    sort_cols = [metric] + [c for c in (tie1, tie2) if c]
    sort_asc  = [ascending] + [not _HIGHER_IS_BETTER.get(c, True) for c in (tie1, tie2) if c]
    return df.sort_values(sort_cols, ascending=sort_asc)


def best_by_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pick the single best run per model using `metric`."""
    if df.empty:
        return df
    d = _sort_by_metric(df.copy(), metric)
    keep = [
        "model","gradient","window","seed","lr","batch","epochs",
        "kl","mae","acc@0.1","acc@0.2","dynTopK%","presence_f1","presence_precision","presence_recall",
        "run_name","timestamp","_source_file"
    ]
    keep = [c for c in keep if c in d.columns]
    return d.groupby("model", as_index=False).first()[keep]


def best_by_model_and_gradient(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pick the best run per (model, gradient)."""
    if df.empty:
        return df
    d = _sort_by_metric(df.copy(), metric)
    keep = [
        "model","gradient","window","seed","lr","batch","epochs",
        "kl","mae","acc@0.1","acc@0.2","dynTopK%","presence_f1","presence_precision","presence_recall",
        "run_name","timestamp","_source_file"
    ]
    keep = [c for c in keep if c in d.columns]
    return d.drop_duplicates(subset=["model","gradient"], keep="first")[keep]


def top_overall(df: pd.DataFrame, n: int, metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = _sort_by_metric(df.copy(), metric)
    cols = [
        "model","gradient","window","seed","lr","batch","epochs",
        "kl","mae","acc@0.1","acc@0.2","dynTopK%","presence_f1","presence_precision","presence_recall",
        "run_name","timestamp","_source_file"
    ]
    cols = [c for c in cols if c in d.columns]
    return d[cols].head(n)


def summarize_by(df: pd.DataFrame, by_cols: list[str], metric: str) -> pd.DataFrame:
    """Mean and std of the selected metric grouped by columns (e.g., model/window)."""
    if df.empty:
        return df
    if metric not in df.columns:
        return pd.DataFrame()
    g = (df.groupby(by_cols)[metric]
           .agg(mean="mean", std="std", n="size")
           .reset_index())
    return g


# --------------------------- CLI / main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate *mixture* experiment logs (results.jsonl).")
    ap.add_argument("--log-dir", required=True, help="Root folder containing per-run results.jsonl")
    ap.add_argument("--out", default=None, help="Output folder (default: <log-dir>/summary_mixture)")
    ap.add_argument("--select-metric", default="acc@0.2",
                    choices=list(_HIGHER_IS_BETTER.keys()),
                    help="Metric used to pick best runs (default: acc@0.2)")
    ap.add_argument("--top-n", type=int, default=20, help="How many runs to list in topN_overall.csv")

    # Optional filters
    ap.add_argument("--models", nargs="*", default=None, help="Restrict to these models")
    ap.add_argument("--gradients", nargs="*", type=int, default=None, help="Restrict to these gradient values")
    ap.add_argument("--windows", nargs="*", type=int, default=None, help="Restrict to these window sizes")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="Restrict to these seeds")

    # Optional group summaries
    ap.add_argument("--summarize", nargs="*", default=None,
                    help="Group-by columns for summary (e.g., --summarize model window)")

    args = ap.parse_args()

    root = Path(args.log_dir).resolve()
    outdir = Path(args.out).resolve() if args.out else root / "summary_mixture"
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
    if args.seeds:
        df = df[df["seed"].isin(args.seeds)]

    # Save all flat
    df.to_csv(outdir / "all_runs_flat.csv", index=False)

    # Best per model
    df_best_model = best_by_model(df, metric=args.select_metric)
    if not df_best_model.empty:
        df_best_model.to_csv(outdir / "best_by_model.csv", index=False)

    # Best per (model, gradient)
    df_best_mg = best_by_model_and_gradient(df, metric=args.select_metric)
    if not df_best_mg.empty:
        df_best_mg.to_csv(outdir / "best_by_model_and_gradient.csv", index=False)

    # Top-N overall
    topN = top_overall(df, n=args.top_n, metric=args.select_metric)
    if not topN.empty:
        topN.to_csv(outdir / "topN_overall.csv", index=False)

    # Optional summaries
    if args.summarize:
        by_cols = [c for c in args.summarize if c in df.columns]
        if by_cols and args.select_metric in df.columns:
            summary = summarize_by(df, by_cols, args.select_metric)
            if not summary.empty:
                # create a friendly filename like summary_by_model_window.csv
                fname = "summary_by_" + "_".join(by_cols) + ".csv"
                summary.to_csv(outdir / fname, index=False)

    print("Saved outputs to:", outdir.as_posix())
    print(" - all_runs_flat.csv")
    if not df_best_model.empty:
        print(" - best_by_model.csv")
    if not df_best_mg.empty:
        print(" - best_by_model_and_gradient.csv")
    if not topN.empty:
        print(" - topN_overall.csv")
    if args.summarize:
        print(" - summary_by_*.csv (if any)")


if __name__ == "__main__":
    main()
