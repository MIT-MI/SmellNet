#!/usr/bin/env python3
# (See previous message for full description)
import re, sys, argparse
from pathlib import Path
import pandas as pd

RE_CMD = re.compile(r'^\s*(python .*?train_dist_model\.py.*?)\s*$')
RE_VAL = re.compile(r'^\[VAL thresh=(?P<thresh>[\d.]+)\] KL=(?P<kl>[\d.]+) MAE=(?P<mae>[\d.]+) @0\.1=(?P<top1>[\d.]+) @0\.2=(?P<top2>[\d.]+) dynTopK=(?P<dyn>[\d.]+)% Pres\(F1/Prec/Rec\)=(?P<f1>[\d.]+)\/(?P<prec>[\d.]+)\/(?P<rec>[\d.]+)')
RE_TEST= re.compile(r'^\[TEST thresh=(?P<thresh>[\d.]+)\] KL=(?P<kl>[\d.]+) MAE=(?P<mae>[\d.]+) @0\.1=(?P<top1>[\d.]+) @0\.2=(?P<top2>[\d.]+) dynTopK=(?P<dyn>[\d.]+)% Pres\(F1/Prec/Rec\)=(?P<f1>[\d.]+)\/(?P<prec>[\d.]+)\/(?P<rec>[\d.]+)')
RE_BENCH1 = re.compile(r'^\[BENCH\] Params:\s*(?P<params>[\d,]+)\s*\(~(?P<model_mb>[\d.]+)\s*MB\)')
RE_BENCH2 = re.compile(r'^\[BENCH\] Batch=(?P<batch>\d+)\s*latency:\s*(?P<lat_ms>[\d.]+)\s*ms(?:\s*\|\s*peak GPU mem:\s*(?P<gpu_mb>[\d.]+)\s*MB)?')
RE_BENCH3 = re.compile(r'^\[BENCH\] Batch=1\s*latency:\s*(?P<lat1_ms>[\d.]+)\s*ms(?:\s*\|\s*peak GPU mem:\s*(?P<gpu1_mb>[\d.]+)\s*MB)?')

def parse_cmd(cmd_path: Path):
    txt = cmd_path.read_text(encoding="utf-8", errors="ignore") if cmd_path.exists() else ""
    m = RE_CMD.search(txt)
    cmd = m.group(1) if m else txt.strip()
    def get(flag):
        pat = re.compile(rf'{flag}\s+([^\s]+)')
        mm = pat.search(cmd)
        return mm.group(1) if mm else ''
    return {
        "arch": get("--arch"),
        "no_standardize": "true" if "--no-standardize" in cmd else "false",
        "seed": get("--seed"),
        "lr": get("--lr"),
        "weight_decay": get("--weight-decay"),
        "alpha": get("--alpha"),
        "beta": get("--beta"),
        "lag": get("--lag"),
        "batch": get("--batch-size"),
        "epochs": get("--epochs"),
        "cmd": cmd,
    }

def parse_log(log_path: Path):
    vals = {
        "val_thresh":"","val_kl":"","val_mae":"","val_top1":"","val_top2":"","val_dyn":"","val_f1":"","val_prec":"","val_rec":"",
        "test_thresh":"","test_kl":"","test_mae":"","test_top1":"","test_top2":"","test_dyn":"","test_f1":"","test_prec":"","test_rec":"",
        "bench_params":"","bench_model_mb":"","bench_batch":"","bench_lat_ms":"","bench_gpu_mb":"","bench_lat1_ms":"","bench_gpu1_mb":"",
    }
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        m = RE_VAL.match(s)
        if m: vals.update({f"val_{k}": m.group(k) for k in ["thresh","kl","mae","top1","top2","dyn","f1","prec","rec"]})
        m = RE_TEST.match(s)
        if m: vals.update({f"test_{k}": m.group(k) for k in ["thresh","kl","mae","top1","top2","dyn","f1","prec","rec"]})
        m = RE_BENCH1.match(s)
        if m: vals.update({"bench_params": m.group("params"), "bench_model_mb": m.group("model_mb")})
        m = RE_BENCH2.match(s)
        if m: vals.update({"bench_batch": m.group("batch"), "bench_lat_ms": m.group("lat_ms"), "bench_gpu_mb": (m.group("gpu_mb") or "")})
        m = RE_BENCH3.match(s)
        if m: vals.update({"bench_lat1_ms": m.group("lat1_ms"), "bench_gpu1_mb": (m.group("gpu1_mb") or "")})
    return vals

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def select_best(df):
    df2 = df.copy()
    df2 = coerce_numeric(df2, ["val_top2","val_kl","val_dyn"])
    df2["rank_key"] = list(zip(-df2["val_top2"].fillna(-1e9),
                               df2["val_kl"].fillna(1e9),
                               -df2["val_dyn"].fillna(-1e9)))
    best_rows = []
    for (label, arch), g in df2.groupby(["label","arch"]):
        idx = g["rank_key"].idxmin()
        best_rows.append(df.loc[idx])
    return pd.DataFrame(best_rows).reset_index(drop=True)

def fmt_model_name(arch):
    return {"lstm":"LSTM","transformer":"Transformer","tcn":"TCN"}.get(arch, arch.upper())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_root", help="Path to logs_all/<timestamp>")
    ap.add_argument("--out", default="analysis_out", help="Output directory")
    args = ap.parse_args()

    root = Path(args.log_root)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in root.rglob("*.log"):
        label = log.parent.name  # std or nostd
        cmd_fields = parse_cmd(log.with_suffix(".cmd"))
        metrics = parse_log(log)
        rows.append({"label": label, **cmd_fields, **metrics, "logfile": str(log)})
    if not rows:
        print(f"No logs found under {root}", file=sys.stderr); sys.exit(2)

    import pandas as pd
    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if any(k in c for k in ["val_","test_","bench_"])]
    df = coerce_numeric(df, num_cols)

    # Save all
    all_csv = outd / "all_runs.csv"
    df.to_csv(all_csv, index=False)

    # Best per (label, arch)
    best = select_best(df)
    keep = ["label","arch","no_standardize","seed","lr","weight_decay","alpha","beta","lag","batch","epochs",
            "val_thresh","val_kl","val_mae","val_top1","val_top2","val_dyn","val_f1","val_prec","val_rec",
            "test_thresh","test_kl","test_mae","test_top1","test_top2","test_dyn","test_f1","test_prec","test_rec",
            "bench_params","bench_model_mb","bench_batch","bench_lat_ms","bench_gpu_mb","bench_lat1_ms","bench_gpu1_mb",
            "logfile"]
    best_csv = outd / "best_runs.csv"
    best[keep].to_csv(best_csv, index=False)

    def to_tex_table(tbl, caption, label):
        return tbl.to_latex(index=False, escape=False, caption=caption, label=label)

    # TEST table
    tbl = pd.DataFrame({
        "Model": [fmt_model_name(a) for a in best["arch"]],
        "Std.": ["Yes" if x=="false" else "No" for x in best["no_standardize"]],
        "KL↓": best["test_kl"].round(3),
        "MAE↓": best["test_mae"].round(3),
        "Top-1 @0.1↑": best["test_top1"].round(3),
        "Top-2 @0.2↑": best["test_top2"].round(3),
        "Dyn Top-K↑ (%)": best["test_dyn"].round(2),
        "Presence F1↑": best["test_f1"].round(3),
        "Precision": best["test_prec"].round(3),
        "Recall": best["test_rec"].round(3),
        "Thresh": best["test_thresh"].round(2),
    })
    tbl["StdOrder"] = tbl["Std."].map({"Yes":0,"No":1})
    tbl["ModelOrder"] = tbl["Model"].map({"TCN":0,"LSTM":1,"Transformer":2})
    tbl = tbl.sort_values(["ModelOrder","StdOrder"]).drop(columns=["StdOrder","ModelOrder"])
    main_tex = to_tex_table(tbl, caption="Test metrics for best configurations per architecture (selected by validation Top-2 @0.2).", label="tab:test_main")
    (outd / "table_test_main.tex").write_text(main_tex)

    # VAL table
    val_tbl = pd.DataFrame({
        "Model": [fmt_model_name(a) for a in best["arch"]],
        "Std.": ["Yes" if x=="false" else "No" for x in best["no_standardize"]],
        "KL↓": best["val_kl"].round(3),
        "MAE↓": best["val_mae"].round(3),
        "Top-1 @0.1↑": best["val_top1"].round(3),
        "Top-2 @0.2↑": best["val_top2"].round(3),
        "Dyn Top-K↑ (%)": best["val_dyn"].round(2),
        "Presence F1↑": best["val_f1"].round(3),
        "Precision": best["val_prec"].round(3),
        "Recall": best["val_rec"].round(3),
        "Thresh": best["val_thresh"].round(2),
    })
    val_tbl["StdOrder"] = val_tbl["Std."].map({"Yes":0,"No":1})
    val_tbl["ModelOrder"] = val_tbl["Model"].map({"TCN":0,"LSTM":1,"Transformer":2})
    val_tbl = val_tbl.sort_values(["ModelOrder","StdOrder"]).drop(columns=["StdOrder","ModelOrder"])
    val_tex = to_tex_table(val_tbl, caption="Validation metrics (selection based on Top-2 @0.2).", label="tab:val_main")
    (outd / "table_val_main.tex").write_text(val_tex)

    # Costs table if present
    has_bench = best["bench_params"].notna().any() and (best["bench_params"] != 0).any()
    if has_bench:
        cost_tbl = pd.DataFrame({
            "Model": [fmt_model_name(a) for a in best["arch"]],
            "Std.": ["Yes" if x=="false" else "No" for x in best["no_standardize"]],
            "#Params": best["bench_params"],
            "Model MB": best["bench_model_mb"].round(2),
            "Batch": best["bench_batch"].fillna("").astype(str),
            "Batch Lat. (ms)": best["bench_lat_ms"].round(2),
            "GPU MB (peak)": best["bench_gpu_mb"].round(1),
            "B=1 Lat. (ms)": best["bench_lat1_ms"].round(2),
            "B=1 GPU MB": best["bench_gpu1_mb"].round(1),
        })
        cost_tbl["StdOrder"] = cost_tbl["Std."].map({"Yes":0,"No":1})
        cost_tbl["ModelOrder"] = cost_tbl["Model"].map({"TCN":0,"LSTM":1,"Transformer":2})
        cost_tbl = cost_tbl.sort_values(["ModelOrder","StdOrder"]).drop(columns=["StdOrder","ModelOrder"])
        cost_tex = cost_tbl.to_latex(index=False, escape=False, caption="Model cost metrics (if available).", label="tab:model_costs")
        (outd / "table_costs.tex").write_text(cost_tex)

    # Std vs NoStd deltas
    try:
        pvt = best.copy()
        pvt["StdFlag"] = pvt["no_standardize"].map({"false":"Std","true":"NoStd"})
        pvt["Model"] = [fmt_model_name(a) for a in pvt["arch"]]
        keep_cols = ["Model","StdFlag","test_top2","test_f1","test_kl","test_mae","test_dyn"]
        pvt = pvt[keep_cols]
        delta_rows = []
        for model, g in pvt.groupby("Model"):
            if set(g["StdFlag"]) == {"Std","NoStd"}:
                std_row = g[g["StdFlag"]=="Std"].iloc[0]
                ns_row  = g[g["StdFlag"]=="NoStd"].iloc[0]
                delta_rows.append({
                    "Model": model,
                    "ΔTop-2 (Std - NoStd)": round((std_row["test_top2"] - ns_row["test_top2"]), 3),
                    "ΔF1 (Std - NoStd)": round((std_row["test_f1"] - ns_row["test_f1"]), 3),
                    "ΔKL (Std - NoStd)": round((std_row["test_kl"] - ns_row["test_kl"]), 3),
                    "ΔMAE (Std - NoStd)": round((std_row["test_mae"] - ns_row["test_mae"]), 3),
                    "ΔDynTopK (Std - NoStd)": round((std_row["test_dyn"] - ns_row["test_dyn"]), 2),
                })
        if delta_rows:
            dtbl = pd.DataFrame(delta_rows)
            delta_tex = dtbl.to_latex(index=False, escape=False, caption="Effect of standardization (Std minus No-Std). Higher is better for Top-2/F1/Dyn; lower is better for KL/MAE.", label="tab:std_vs_nostd")
            (outd / "std_vs_nostd.tex").write_text(delta_tex)
    except Exception:
        pass

    # README
    (outd / "README.txt").write_text(
        "Generated files:\n"
        f"- {outd/'all_runs.csv'}\n- {outd/'best_runs.csv'}\n"
        f"- {outd/'table_test_main.tex'}\n- {outd/'table_val_main.tex'}\n"
        f"- {outd/'table_costs.tex'} (if BENCH present)\n- {outd/'std_vs_nostd.tex'} (if both present)\n\n"
        "Selection criterion: highest validation Top-2 @0.2; ties → lower val KL, then higher val DynTopK.\n"
        "Recommendation: use table_test_main.tex in the main paper, costs/deltas in the appendix.\n"
    )

    print("[OK] Wrote analysis outputs to", outd)

if __name__ == "__main__":
    main()
