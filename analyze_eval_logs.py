#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd

VAL_RE  = re.compile(r'^\[VAL thresh=(?P<th>[\d\.]+)\]\s+KL=(?P<kl>[\d\.]+)\s+MAE=(?P<mae>[\d\.]+)\s+@0\.1=(?P<t1>[\d\.]+)\s+@0\.2=(?P<t2>[\d\.]+)\s+dynTopK=(?P<dyn>[\d\.]+)%\s+Pres\(F1/Prec/Rec\)=(?P<f1>[\d\.nan]+)/(?P<prec>[\d\.nan]+)/(?P<rec>[\d\.nan]+)')
TEST_RE = re.compile(r'^\[TEST thresh=(?P<th>[\d\.]+)\]\s+KL=(?P<kl>[\d\.]+)\s+MAE=(?P<mae>[\d\.]+)\s+@0\.1=(?P<t1>[\d\.]+)\s+@0\.2=(?P<t2>[\d\.]+)\s+dynTopK=(?P<dyn>[\d\.]+)%\s+Pres\(F1/Prec/Rec\)=(?P<f1>[\d\.nan]+)/(?P<prec>[\d\.nan]+)/(?P<rec>[\d\.nan]+)')

def infer_arch(name:str)->str:
    n = name.lower()
    if "transformer" in n: return "Transformer"
    if "lstm" in n: return "LSTM"
    return "TCN"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs_root", type=str, help="logs_eval/<timestamp> root OR any dir containing *_PURE.log / *_MIX.log")
    ap.add_argument("--out", type=str, default="analysis_pure_mix", help="output directory")
    ap.add_argument("--dir-label", type=str, default=None, help="override directory column with a friendly label")
    args = ap.parse_args()

    root = Path(args.logs_root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    logs = list(root.rglob("*.log"))
    if not logs:
        print(f"[WARN] no .log in {root}"); return

    rows = []
    for lf in logs:
        subset = "PURE" if lf.name.endswith("_PURE.log") else ("MIX" if lf.name.endswith("_MIX.log") else None)
        arch = infer_arch(lf.name)
        with lf.open("r", errors="ignore") as f:
            val = test = None; th_v = th_t = None
            for line in f:
                line = line.strip()
                m = VAL_RE.match(line)
                if m:
                    d = m.groupdict()
                    val = d; th_v = float(d["th"])
                m2 = TEST_RE.match(line)
                if m2:
                    d = m2.groupdict()
                    test = d; th_t = float(d["th"])
        if subset is None or (val is None and test is None): 
            continue
        rows.append({
            "dir": lf.parent.name if args.dir_label is None else args.dir_label,
            "arch": arch,
            "subset": subset,
            "val_thresh": float(val["th"]) if val else None,
            "val_kl": float(val["kl"]) if val else None,
            "val_mae": float(val["mae"]) if val else None,
            "val_top1": float(val["t1"]) if val else None,
            "val_top2": float(val["t2"]) if val else None,
            "val_dyn": float(val["dyn"]) if val else None,
            "val_f1": float("nan") if not val or val["f1"]=="nan" else float(val["f1"]),
            "val_prec": float("nan") if not val or val["prec"]=="nan" else float(val["prec"]),
            "val_rec": float("nan") if not val or val["rec"]=="nan" else float(val["rec"]),
            "test_thresh": float(test["th"]) if test else None,
            "test_kl": float(test["kl"]) if test else None,
            "test_mae": float(test["mae"]) if test else None,
            "test_top1": float(test["t1"]) if test else None,
            "test_top2": float(test["t2"]) if test else None,
            "test_dyn": float(test["dyn"]) if test else None,
            "test_f1": float("nan") if not test or test["f1"]=="nan" else float(test["f1"]),
            "test_prec": float("nan") if not test or test["prec"]=="nan" else float(test["prec"]),
            "test_rec": float("nan") if not test or test["rec"]=="nan" else float(test["rec"]),
            "log_path": str(lf),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "eval_runs.csv", index=False)

    # Best per (dir, subset, arch) by val_top2
    best_rows = []
    for (d, s, a), g in df.groupby(["dir","subset","arch"]):
        g2 = g.dropna(subset=["val_top2"]).copy()
        if g2.empty: continue
        g2 = g2.sort_values(by=["val_top2","val_kl","val_dyn"], ascending=[False, True, False])
        best_rows.append(g2.iloc[0])
    best = pd.DataFrame(best_rows).reset_index(drop=True)
    best.to_csv(out / "best_eval_runs.csv", index=False)

    # Produce two TEST tables: PURE and MIX, each with one row per (dir, arch)
    def table_for(subset):
        dfx = best[best["subset"]==subset].copy()
        if dfx.empty: return ""
        dfx["Std."] = dfx["dir"].map(lambda x: "No" if str(x).lower() in {"nostd","no1","raw","ns"} else "Yes")
        tbl = pd.DataFrame({
            "Model": dfx["arch"],
            "Std.": dfx["Std."],
            "KL$\\downarrow$": dfx["test_kl"].round(3),
            "MAE$\\downarrow$": dfx["test_mae"].round(3),
            "Top-1 @0.1$\\uparrow$": dfx["test_top1"].round(3),
            "Top-2 @0.2$\\uparrow$": dfx["test_top2"].round(3),
            "Dyn Top-$K$ $\\uparrow$ (\\%)": dfx["test_dyn"].round(2),
            "F1$\\uparrow$": dfx["test_f1"].round(3),
            "Prec": dfx["test_prec"].round(3),
            "Rec": dfx["test_rec"].round(3),
            "Thresh": dfx["test_thresh"].round(2),
        })
        return tbl.to_latex(index=False, escape=False,
            caption=f"Test metrics on {subset.lower()}-only subset (best by validation Top-2 @0.2 per (dir, arch)).",
            label=f"tab:test_{subset.lower()}_only")
    tex_pure = table_for("PURE")
    tex_mix  = table_for("MIX")
    if tex_pure: (out / "table_test_pure_only.tex").write_text(tex_pure)
    if tex_mix:  (out / "table_test_mix_only.tex").write_text(tex_mix)

    print(f"[OK] Wrote {(out/'eval_runs.csv')}, {(out/'best_eval_runs.csv')} and any subset tables.")
if __name__ == "__main__":
    main()
