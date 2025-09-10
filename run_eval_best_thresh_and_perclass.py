#!/usr/bin/env python3
# run_eval_best_thresh_and_perclass.py
# Evaluate each saved model (standardized and non-standardized) on PURE and MIX,
# let the evaluator pick the best presence threshold (to maximize @0.2) per subset,
# then keep per-ingredient accuracy for the better subset (by a chosen metric).
#
# Outputs:
#   logs_eval_best/<STAMP>/<dir>/*.cmd           # exact command lines
#   logs_eval_best/<STAMP>/<dir>/*_{PURE|MIX}.log
#   logs_eval_best/<STAMP>/<dir>/*_{PURE|MIX}_perclass_{VAL,TEST}.csv (keeps only selected subset unless --both-perclass 1)
#   logs_eval_best/<STAMP>/best_thresholds_and_perclass_summary.csv
#
# Example:
#   python run_eval_best_thresh_and_perclass.py \
#     --models-root /home/dewei/workspace/SmellNet/models_all \
#     --train-dir  /home/dewei/workspace/SmellNet/chi_paper_data/training_new \
#     --test-dir   /home/dewei/workspace/SmellNet/chi_paper_data/test_seen \
#     --out-root   logs_eval_best \
#     --class-names /home/dewei/workspace/SmellNet/classes.txt \
#     --select-metric top2 \
#     --no-temp \
#     --both-perclass 0
#
import argparse, re, shlex, subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime

LINE_RE = re.compile(
    r'\[(?P<split>VAL|TEST)\s+thresh=(?P<th>[\d.]+)\]\s+'
    r'KL=(?P<kl>[\d.NaNnan]+)\s+'
    r'MAE=(?P<mae>[\d.]+)\s+'
    r'@0\.1=(?P<top1>[\d.]+)\s+'
    r'@0\.2=(?P<top2>[\d.]+)\s+'
    r'dynTopK=(?P<dyn>[\d.]+)%\s+'
    r'Pres\(F1/Prec/Rec\)=(?P<f1>[\d.NaNnan]+)/(?P<prec>[\d.NaNnan]+)/(?P<rec>[\d.NaNnan]+)'
)

def infer_arch_from_name(name: str) -> str:
    n = name.lower()
    if 'transformer' in n: return 'transformer'
    if 'lstm' in n: return 'lstm'
    return 'tcn'

def infer_lag_from_name(name: str, default=0) -> int:
    m = re.search(r'lag(\d+)', name)
    return int(m.group(1)) if m else int(default)

def run_eval(weights_path: Path, subset: str, out_dir: Path, args, label_dir: str):
    """Run evaluator for a given weight/subset. Returns (log_file, perclass_prefix)."""
    arch = infer_arch_from_name(weights_path.name)
    lag  = infer_lag_from_name(weights_path.name, default=args.lag_default)
    subset_flag = '--eval-pure-only' if subset == 'PURE' else '--eval-mixture-only'
    stdflag = '--no-standardize' if label_dir in {s.strip() for s in args.nostd_dirs.split(',') if s.strip()} else ''
    base = weights_path.stem
    prefix = out_dir / f"{base}_{subset}_perclass"

    cmd = [
        args.python_bin, args.eval_path,
        '--weights', str(weights_path),
        '--train-dir', args.train_dir, '--test-dir', args.test_dir,
        '--arch', arch, '--batch-size', str(args.batch_size), '--max-len', str(args.max_len),
        '--lag', str(lag),
        subset_flag, '--pure-eps', str(args.pure_eps),
        '--per-class', '--per-class-save', str(prefix),
        '--thr-acc', str(args.thr_acc),
    ]
    # Threshold/temperature selection hints (use only if your evaluator supports them)
    if args.calibrate_on: cmd += ['--calibrate-on', args.calibrate_on]
    if args.thresh_on:    cmd += ['--thresh-on',    args.thresh_on]
    if args.class_names:  cmd += ['--class-names', args.class_names]
    if args.no_temp:      cmd.append('--no-temp')
    if stdflag:           cmd.append(stdflag)

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd_file = out_dir / f"{base}_{subset}.cmd"
    log_file = out_dir / f"{base}_{subset}.log"
    cmd_file.write_text(' '.join(shlex.quote(c) for c in cmd) + '\n')
    print('[RUN]', ' '.join(shlex.quote(c) for c in cmd))
    with log_file.open('w') as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(f"[ERROR] Eval failed for {weights_path} ({subset}). See {log_file}")
    return log_file, prefix

def parse_test_line(log_file: Path):
    """Return dict with metrics from the last TEST line in the log."""
    txt = log_file.read_text(errors='ignore').splitlines()
    last = None
    for line in txt:
        m = LINE_RE.search(line)
        if m and m.group('split') == 'TEST':
            last = m
    if not last:
        # Fallback: pick any last matching line
        for line in reversed(txt):
            m = LINE_RE.search(line)
            if m: last = m; break
    if not last:
        return {}
    g = last.groupdict()
    for k in ['th','kl','mae','top1','top2','dyn','f1','prec','rec']:
        try: g[k] = float(g[k])
        except Exception: g[k] = float('nan')
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-root', type=str, default='models_all')
    ap.add_argument('--dirs', type=str, default='no1,yes1', help='Comma-separated subdirs under models_root to scan')
    ap.add_argument('--train-dir', type=str, required=True)
    ap.add_argument('--test-dir',  type=str, required=True)
    ap.add_argument('--out-root',  type=str, default='logs_eval_best')
    ap.add_argument('--eval-path', type=str, default='models/eval_saved_models.py')
    ap.add_argument('--python-bin', type=str, default='python')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=600)
    ap.add_argument('--lag-default', type=int, default=0)
    ap.add_argument('--thr-acc', type=float, default=0.2, help="Threshold for 'Within 0.2' per-class metric")
    ap.add_argument('--pure-eps', type=float, default=1e-6)
    ap.add_argument('--nostd-dirs', type=str, default='no1', help='Dirs to pass --no-standardize')
    ap.add_argument('--class-names', type=str, default=None)
    ap.add_argument('--no-temp', action='store_true')
    # If your evaluator supports these, they enable per-subset threshold selection & calibration
    ap.add_argument('--calibrate-on', type=str, default='subset', help="Use 'subset' or 'all' (leave blank to skip flag)")
    ap.add_argument('--thresh-on',    type=str, default='subset', help="Use 'subset' or 'all' (leave blank to skip flag)")
    ap.add_argument('--select-metric', choices=['top2','top1','dyn','f1','prec','rec','mae','kl'], default='top2',
                    help='Metric for choosing better subset (PURE vs MIX)')
    ap.add_argument('--both-perclass', type=int, default=0, help='Keep per-class CSVs for both subsets')
    args = ap.parse_args()

    models_root = Path(args.models_root)
    dirs = [d.strip() for d in args.dirs.split(',') if d.strip()]
    stamp_dir = Path(args.out_root) / datetime.now().strftime('%Y%m%d_%H%M%S')
    stamp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for d in dirs:
        d_root = models_root / d
        if not d_root.exists():
            print(f"[WARN] Missing directory: {d_root}; skipping")
            continue
        out_dir = stamp_dir / d
        out_dir.mkdir(parents=True, exist_ok=True)

        weights = sorted(d_root.glob('*.pt'))
        if not weights:
            print(f"[WARN] No weights found under {d_root}")
            continue

        for w in weights:
            # Evaluate on PURE and MIX (each run lets evaluator pick best thresh for that subset)
            pure_log, pure_prefix = run_eval(w, 'PURE', out_dir, args, d)
            mix_log,  mix_prefix  = run_eval(w, 'MIX',  out_dir, args, d)

            pm = parse_test_line(pure_log)
            mm = parse_test_line(mix_log)

            # Decide which subset is 'best' for this model
            metric_map = {'top2':'top2','top1':'top1','dyn':'dyn','f1':'f1','prec':'prec','rec':'rec','mae':'mae','kl':'kl'}
            key = metric_map[args.select_metric]
            def score(m):
                if not m: return float('-inf')
                v = m.get(key, float('nan'))
                if args.select_metric in {'mae','kl'}: return -v  # lower is better
                return v
            s_pure, s_mix = score(pm), score(mm)
            best_subset = 'PURE' if s_pure >= s_mix else 'MIX'
            best_prefix = pure_prefix if best_subset=='PURE' else mix_prefix

            summary_rows.append({
                'dir': d,
                'weight': w.stem,
                'arch': infer_arch_from_name(w.name),
                'pure_thresh': pm.get('th', float('nan')),
                'pure_top2': pm.get('top2', float('nan')),
                'pure_top1': pm.get('top1', float('nan')),
                'pure_dyn': pm.get('dyn', float('nan')),
                'pure_f1': pm.get('f1', float('nan')),
                'pure_mae': pm.get('mae', float('nan')),
                'mix_thresh': mm.get('th', float('nan')),
                'mix_top2': mm.get('top2', float('nan')),
                'mix_top1': mm.get('top1', float('nan')),
                'mix_dyn': mm.get('dyn', float('nan')),
                'mix_f1': mm.get('f1', float('nan')),
                'mix_mae': mm.get('mae', float('nan')),
                'selected_subset': best_subset,
                'selected_perclass_csv': f"{best_prefix}_TEST.csv",
            })

            # Optionally clean up per-class CSVs for non-selected subset
            if not args.both_perclass:
                other = mix_prefix if best_subset=='PURE' else pure_prefix
                for suffix in ['_VAL.csv','_TEST.csv']:
                    p = Path(f"{other}{suffix}")
                    if p.exists():
                        try: p.unlink()
                        except Exception: pass

    df = pd.DataFrame(summary_rows)
    out_csv = stamp_dir / 'best_thresholds_and_perclass_summary.csv'
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote summary: {out_csv}")
    print(f"[OK] All logs & per-class CSVs under: {stamp_dir}")

if __name__ == '__main__':
    main()