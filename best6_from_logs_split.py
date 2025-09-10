#!/usr/bin/env python3
# best6_from_logs_split.py
# Read EXISTING evaluation logs and per-class CSVs and produce:
#   (A) BEST per (architecture × standardization) for PURE only  → 6 rows
#   (B) BEST per (architecture × standardization) for MIX only   → 6 rows
#   (C) BEST-of-six winner for PURE and for MIX (2 rows total)
#   (D) Copy the per-class CSVs of those two winners into the output folder
#
# Nothing is re-evaluated.
#
# Example:
#   python best6_from_logs_split.py \
#     --logs-root /home/dewei/workspace/SmellNet/logs_eval/20250909_222436 \
#     --out-dir   best6_by_subset \
#     --select-metric top2
#
import argparse, re
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

def best_by_arch_std(df_subset: pd.DataFrame, metric_col: str, higher_is_better: bool) -> pd.DataFrame:
    rows = []
    for (arch, std), grp in df_subset.groupby(["arch","std"]):
        g = grp.dropna(subset=[metric_col]).copy()
        if g.empty: 
            continue
        idx = g[metric_col].idxmax() if higher_is_better else g[metric_col].idxmin()
        rows.append(df_subset.loc[idx])
    out = pd.DataFrame(rows).sort_values(["arch","std"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True, help="Folder containing subdirs with *.log and *_perclass_TEST.csv")
    ap.add_argument("--out-dir",   default="best6_by_subset")
    ap.add_argument("--select-metric", choices=["top2","top1","dyn","f1","prec","rec","mae","kl"], default="top2",
                    help="Metric used to choose winners per (arch × std) and overall per subset")
    args = ap.parse_args()

    root = Path(args.logs_root)
    out  = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Parse all logs
    rows = []
    for log in sorted(root.rglob("*.log")):
        subset = infer_subset_from_name(log.name)
        if subset not in {"PURE","MIX"}: 
            continue
        label_dir = log.parent.name
        stem = log.stem
        weight_base = strip_subset(stem)
        arch = infer_arch(stem)
        tm = parse_test_line(log)
        rows.append({
            "dir": label_dir,
            "std": std_from_dir(label_dir),
            "arch": arch,
            "subset": subset,
            "weight_base": weight_base,
            "test_thresh": tm.get("th", float("nan")),
            "test_kl":     tm.get("kl", float("nan")),
            "test_mae":    tm.get("mae", float("nan")),
            "test_top1":   tm.get("top1", float("nan")),
            "test_top2":   tm.get("top2", float("nan")),
            "test_dyn":    tm.get("dyn", float("nan")),
            "test_f1":     tm.get("f1", float("nan")),
            "test_prec":   tm.get("prec", float("nan")),
            "test_rec":    tm.get("rec", float("nan")),
            "log_path":    str(log),
            "perclass_csv": str(find_perclass_csv(log) or ""),
        })
    if not rows:
        raise SystemExit(f"[ERROR] No logs found under {root}")

    df_all = pd.DataFrame(rows)
    df_all.to_csv(out / "all_parsed_logs.csv", index=False)

    metric_col = {
        "top2":"test_top2","top1":"test_top1","dyn":"test_dyn","f1":"test_f1",
        "prec":"test_prec","rec":"test_rec","mae":"test_mae","kl":"test_kl"
    }[args.select_metric]
    higher_is_better = args.select_metric not in {"mae","kl"}

    # Split by subset
    df_pure = df_all[df_all["subset"] == "PURE"].copy()
    df_mix  = df_all[df_all["subset"] == "MIX"].copy()

    best6_pure = best_by_arch_std(df_pure, metric_col, higher_is_better)
    best6_mix  = best_by_arch_std(df_mix,  metric_col, higher_is_better)

    best6_pure.to_csv(out / "best6_PURE.csv", index=False)
    best6_mix.to_csv( out / "best6_MIX.csv",  index=False)

    # Winners (best-of-six) for each subset
    def pick_best(df6: pd.DataFrame) -> pd.Series:
        if df6.empty: return None
        idx = df6[metric_col].idxmax() if higher_is_better else df6[metric_col].idxmin()
        return df6.loc[idx]

    win_pure = pick_best(best6_pure)
    win_mix  = pick_best(best6_mix)

    # Save winners and per-class CSVs
    if win_pure is not None:
        (out / "winner_PURE_row.csv").write_text(win_pure.to_csv(index=False))
        pc_pure = win_pure.get("perclass_csv", "")
        if pc_pure and Path(pc_pure).exists():
            pd.read_csv(pc_pure).to_csv(out / "winner_PURE_perclass.csv", index=False)
        else:
            print(f"[WARN] Missing per-class CSV for PURE winner: {pc_pure}")
    else:
        print("[WARN] No PURE winners (check logs).")

    if win_mix is not None:
        (out / "winner_MIX_row.csv").write_text(win_mix.to_csv(index=False))
        pc_mix = win_mix.get("perclass_csv", "")
        if pc_mix and Path(pc_mix).exists():
            pd.read_csv(pc_mix).to_csv(out / "winner_MIX_perclass.csv", index=False)
        else:
            print(f"[WARN] Missing per-class CSV for MIX winner: {pc_mix}")
    else:
        print("[WARN] No MIX winners (check logs).")

    # Also provide a combined 12-row CSV with a 'subset' column
    best12 = pd.concat([best6_pure.assign(subset_report="PURE"),
                        best6_mix.assign(subset_report="MIX")], ignore_index=True)
    best12.to_csv(out / "best12_by_arch_std_with_subset.csv", index=False)

    print(f"[OK] Done. Outputs in: {out}")

if __name__ == "__main__":
    main()
