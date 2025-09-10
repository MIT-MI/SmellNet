#!/usr/bin/env python3
# pick_best_from_existing_logs.py
# Use ALREADY-COMPUTED evaluation logs and per-class CSVs to:
#   1) Parse PURE and MIX metrics (incl. the chosen thresh shown in logs).
#   2) Select the better subset per model using a metric (default Top-2 @0.2).
#   3) Emit a summary CSV and Overleaf-ready LaTeX table.
#   4) (Optional) Emit per-ingredient LaTeX tables for the selected subset(s).
#
# No evaluation is run; this only reads .log and *_perclass_TEST.csv files.
#
# Example:
#   python pick_best_from_existing_logs.py \
#     --logs-root /home/dewei/workspace/SmellNet/logs_eval/20250909_222436 \
#     --out-dir overleaf_tables_best \
#     --select-metric top2 \
#     --perclass-tex 1 \
#     --class-names /home/dewei/workspace/SmellNet/classes.txt
#
import argparse, re, json
from pathlib import Path
import pandas as pd

LINE_RE = re.compile(
    r'\[(?P<split>VAL|TEST)\s+thresh=(?P<th>[\d.]+)\]\s+'
    r'KL=(?P<kl>[\d.NaNnan]+)\s+'
    r'MAE=(?P<mae>[\d.]+)\s+'
    r'@0\.1=(?P<top1>[\d.]+)\s+'
    r'@0\.2=(?P<top2>[\d.]+)\s+'
    r'dynTopK=(?P<dyn>[\d.]+)%\s+'
    r'Pres\(F1/Prec/Rec\)=(?P<f1>[\d.NaNnan]+)/(?P<prec>[\d.NaNnan]+)/(?P<rec>[\d.NaNnan]+)'
)

def infer_subset_from_name(name: str) -> str:
    if name.endswith("_PURE.log"): return "PURE"
    if name.endswith("_MIX.log"):  return "MIX"
    return "ALL"

def strip_subset(stem: str) -> str:
    if stem.endswith("_PURE"): return stem[:-5]
    if stem.endswith("_MIX"):  return stem[:-4]
    return stem

def infer_arch(stem: str) -> str:
    s = stem.lower()
    if "transformer" in s: return "Transformer"
    if "lstm" in s: return "LSTM"
    return "TCN"

def std_from_dir(label: str) -> str:
    return "No" if label.lower() in {"no1","nostd","no-std","no_std"} else "Yes"

def parse_test_line(log_path: Path) -> dict:
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if m and m.group("split") == "TEST":
            last = m
    if not last:
        # fallback to any last matching line
        for line in reversed(log_path.read_text(errors="ignore").splitlines()):
            m = LINE_RE.search(line)
            if m: last = m; break
    if not last: return {}
    g = last.groupdict()
    for k in ["th","kl","mae","top1","top2","dyn","f1","prec","rec"]:
        try: g[k] = float(g[k])
        except Exception: g[k] = float("nan")
    return g

def find_perclass_csv(log_path: Path):
    base = log_path.with_suffix("")
    test_csv = Path(f"{base}_perclass_TEST.csv")
    val_csv  = Path(f"{base}_perclass_VAL.csv")
    return test_csv if test_csv.exists() else (val_csv if val_csv.exists() else None)

def load_class_names(path):
    if not path: return None
    p = Path(path)
    try:
        if p.suffix.lower()==".json":
            return json.loads(p.read_text())
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    except Exception:
        return None

def perclass_tex(df: pd.DataFrame, title="Per-ingredient (TEST)", label="tab:perclass"):
    cols = ["ingredient","mae","mse","nonzero_gts"]
    thr_col = next((c for c in df.columns if c.startswith("within_")), None)
    if thr_col: cols.append(thr_col)
    df = df[[c for c in cols if c in df.columns]].copy()

    headers = {"ingredient":"Ingredient","mae":"MAE$\\downarrow$","mse":"MSE$\\downarrow$","nonzero_gts":"Non-zero GTs"}
    if thr_col: headers[thr_col] = thr_col.replace("within_","Within ").replace("_",".")
    df.rename(columns=headers, inplace=True)

    for c in df.columns:
        if c in {"Ingredient","Non-zero GTs"}: continue
        if c.startswith("Within"):
            df[c] = df[c].apply(lambda v: "" if pd.isna(v) else f"{100*float(v):.1f}\\%")
        else:
            df[c] = df[c].apply(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return df.to_latex(index=False, escape=False, caption=title, label=label)

def make_overview_table(rows: pd.DataFrame, caption, label):
    # Overview for best subset per model
    cols = [
        ("arch","Model"),("std","Std."),("selected_subset","Subset"),
        ("test_kl","KL$\\downarrow$"),("test_mae","MAE$\\downarrow$"),
        ("test_top1","Top-1 @0.1$\\uparrow$"),("test_top2","Top-2 @0.2$\\uparrow$"),
        ("test_dyn","Dyn Top-$K$ $\\uparrow$ (\\%)"),
        ("test_f1","F1$\\uparrow$"),("test_thresh","Thresh")
    ]
    show = {}
    for k, pretty in cols:
        if k in rows.columns:
            rnd = 2 if k in {"test_dyn","test_thresh"} else 3
            if rows[k].dtype.kind in "biufc":
                show[pretty] = rows[k].astype(float).round(rnd)
            else:
                show[pretty] = rows[k].astype(str)
    df = pd.DataFrame(show)
    order = [p for _, p in cols if p in df.columns]
    df = df[order]
    return df.to_latex(index=False, escape=False, caption=caption, label=label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True, help="Folder containing subdirs (e.g., logs_eval/20250909_222436)")
    ap.add_argument("--out-dir", default="overleaf_tables_best")
    ap.add_argument("--select-metric", choices=["top2","top1","dyn","f1","prec","rec","mae","kl"], default="top2",
                    help="Metric used to choose better subset (PURE vs MIX) per model")
    ap.add_argument("--perclass-tex", type=int, default=1, help="Write per-ingredient LaTeX for selected subset(s)")
    ap.add_argument("--class-names", type=str, default=None, help="Optional class names file if CSV lacks 'ingredient'")
    args = ap.parse_args()

    root = Path(args.logs_root)
    out  = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Collect rows for PURE/MIX logs
    records = []
    for log in sorted(root.rglob("*.log")):
        subset = infer_subset_from_name(log.name)
        if subset not in {"PURE","MIX"}: 
            continue
        label_dir = log.parent.name  # e.g., no1 / yes1
        stem = log.stem              # ..._PURE or ..._MIX
        weight_base = strip_subset(stem)
        arch = infer_arch(stem)
        test_metrics = parse_test_line(log)
        pc_csv = find_perclass_csv(log)
        rec = {
            "dir": label_dir,
            "weight_base": weight_base,
            "subset": subset,
            "arch": arch,
            "std": std_from_dir(label_dir),
            "test_thresh": test_metrics.get("th", float("nan")),
            "test_kl": test_metrics.get("kl", float("nan")),
            "test_mae": test_metrics.get("mae", float("nan")),
            "test_top1": test_metrics.get("top1", float("nan")),
            "test_top2": test_metrics.get("top2", float("nan")),
            "test_dyn": test_metrics.get("dyn", float("nan")),
            "test_f1": test_metrics.get("f1", float("nan")),
            "test_prec": test_metrics.get("prec", float("nan")),
            "test_rec": test_metrics.get("rec", float("nan")),
            "log_path": str(log),
            "perclass_test_csv": str(pc_csv) if pc_csv else "",
        }
        records.append(rec)

    if not records:
        print(f"[ERROR] No *.log files found under {root}")
        return

    df = pd.DataFrame(records)
    df.sort_values(["dir","arch","weight_base","subset"], inplace=True)
    df.to_csv(out / "all_log_metrics.csv", index=False)
    print(f"[OK] wrote {out/'all_log_metrics.csv'} ({len(df)} rows)")

    # Choose best subset per (dir, weight_base)
    selected = []
    metric_key = {"top2":"test_top2","top1":"test_top1","dyn":"test_dyn","f1":"test_f1","prec":"test_prec","rec":"test_rec","mae":"test_mae","kl":"test_kl"}[args.select_metric]
    for (label, wbase), grp in df.groupby(["dir","weight_base"]):
        if grp.empty: continue
        # prefer higher except mae/kl
        asc = args.select_metric in {"mae","kl"}
        gsort = grp.sort_values(metric_key, ascending=asc)
        best = gsort.iloc[-1]  # last is best when ascending=False; swap if asc
        if asc: best = gsort.iloc[0]
        selected.append(best)
    best_df = pd.DataFrame(selected)
    best_df.to_csv(out / "best_subset_per_model.csv", index=False)
    print(f"[OK] wrote {out/'best_subset_per_model.csv'} ({len(best_df)} models)")

    # Overview LaTeX
    tex = make_overview_table(best_df, caption=f"Best subset per model chosen by TEST {args.select_metric.upper()}.", label="tab:best_subset_overview")
    (out / "best_subset_overview.tex").write_text(tex)
    print(f"[OK] wrote {out/'best_subset_overview.tex'}")

    # Per-class LaTeX for each selected model (uses existing CSV)
    if args.perclass_tex:
        cls_names = load_class_names(args.class_names)
        for _, row in best_df.iterrows():
            pc_path = row.get("perclass_test_csv","")
            if not pc_path:
                print(f"[WARN] Missing per-class CSV for {row['weight_base']} ({row['subset']})")
                continue
            p = Path(pc_path)
            if not p.exists():
                print(f"[WARN] Per-class CSV not found on disk: {p}")
                continue
            df_pc = pd.read_csv(p)
            if "ingredient" not in df_pc.columns and cls_names is not None:
                df_pc = df_pc.copy()
                df_pc["ingredient"] = cls_names[:len(df_pc)]
            title = f"Per-ingredient ({row['subset']}, TEST) — {row['arch']} ({row['std']})"
            label = f"tab:perclass_{row['arch'].lower()}_{row['dir'].lower()}"
            tex_pc = perclass_tex(df_pc, title=title, label=label)
            out_tex = out / f"perclass_{row['arch'].lower()}_{row['dir'].lower()}.tex"
            out_tex.write_text(tex_pc)
            print(f"[OK] wrote {out_tex}")

if __name__ == "__main__":
    main()
