#!/usr/bin/env python3
# best6_from_logs.py
# Parse existing evaluation logs and per-class CSVs and produce:
#  (1) A CSV with the BEST performance for each (architecture × standardization) = 6 rows.
#  (2) A per-class CSV for the single BEST of those six rows (copied to the output dir).
#
# Nothing is re-evaluated. We only read *.log and *_perclass_TEST.csv files.
#
# Example:
#   python best6_from_logs.py \
#     --logs-root /home/dewei/workspace/SmellNet/logs_eval/20250909_222436 \
#     --out-dir   best6_summary \
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", required=True, help="Folder containing subdirs with *.log and *_perclass_TEST.csv")
    ap.add_argument("--out-dir",   default="best6_summary")
    ap.add_argument("--select-metric", choices=["top2","top1","dyn","f1","prec","rec","mae","kl"], default="top2",
                    help="Metric used to choose the best row per (arch × std) and overall best-of-six")
    args = ap.parse_args()

    root = Path(args.logs_root)
    out  = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Parse all logs -> df_all
    rows = []
    for log in sorted(root.rglob("*.log")):
        subset = infer_subset_from_name(log.name)
        if subset not in {"PURE","MIX"}: continue
        label_dir = log.parent.name              # no1/yes1
        stem = log.stem                          # weight_base + _PURE/_MIX
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

    # 2) For each (arch × std), pick the best row across ANY weight/subset
    metric_col = {
        "top2":"test_top2","top1":"test_top1","dyn":"test_dyn","f1":"test_f1",
        "prec":"test_prec","rec":"test_rec","mae":"test_mae","kl":"test_kl"
    }[args.select_metric]
    higher_is_better = args.select_metric not in {"mae","kl"}

    best_rows = []
    for (arch, std), grp in df_all.groupby(["arch","std"]):
        g = grp.dropna(subset=[metric_col]).copy()
        if g.empty: continue
        if higher_is_better:
            idx = g[metric_col].idxmax()
        else:
            idx = g[metric_col].idxmin()
        best_rows.append(df_all.loc[idx])

    best6 = pd.DataFrame(best_rows).sort_values(["arch","std"])
    best6.to_csv(out / "best6_by_arch_std.csv", index=False)

    # 3) Pick the overall best among those six
    if best6.empty:
        raise SystemExit("[ERROR] No best rows computed (check logs format).")
    if higher_is_better:
        best_idx = best6[metric_col].idxmax()
    else:
        best_idx = best6[metric_col].idxmin()
    best_one = best6.loc[best_idx]
    (out / "best_overall_row.csv").write_text(best_one.to_csv(index=False))

    # 4) Copy/store the per-class CSV for that best row (if found)
    pc_path = best_one.get("perclass_csv", "")
    if pc_path and Path(pc_path).exists():
        df_pc = pd.read_csv(pc_path)
        df_pc.to_csv(out / "best_overall_perclass.csv", index=False)
        print(f"[OK] Wrote per-class CSV for best model → {out/'best_overall_perclass.csv'}")
    else:
        print(f"[WARN] No per-class CSV found for best row: {pc_path}")

    print(f"[OK] Done. Summary CSVs in {out}")

if __name__ == "__main__":
    main()
