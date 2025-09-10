#!/usr/bin/env python3
# Evaluate a saved SmellNet model (TCN / LSTM / Transformer) without training.
# Supports evaluating on PURE-only or MIXTURE-only subsets of val/test.
#
# Now supports:
#   --per-class-save <prefix>  -> writes <prefix>_VAL.csv and <prefix>_TEST.csv
#   --no-temp                  -> skip temperature scaling
#   --calibrate-on all|subset  -> where to fit temp (default: subset)
#   --thresh-on    all|subset  -> where to sweep presence threshold (default: subset)
import argparse, random, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import json, csv, os

# --------- Try to import project modules (fallbacks included) ---------
try:
    from load_data import load_smell_recognition_data
except Exception:
    load_smell_recognition_data = None

# Backbones
SmellTemporalCNN = None
try:
    from run_chi_model import SmellTemporalCNN as _SmellTemporalCNN
    SmellTemporalCNN = _SmellTemporalCNN
except Exception:
    pass

try:
    from extra_models import SmellBiLSTM, SmellTransformer
except Exception:
    SmellBiLSTM = None
    SmellTransformer = None

# --------------------- Utilities mirrored from training ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.float32)
    if label.sum() > 1.1:
        label = label / 100.0
    s = label.sum()
    if s <= 0:
        label = np.ones_like(label) / len(label)
    else:
        label = label / s
    return label

def lag_difference(x: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0: return x
    T, F = x.shape
    if T <= lag:
        raise ValueError(f"Sequence length {T} too short for lag={lag}")
    return x[lag:, :] - x[:-lag, :]

class DistDataset(Dataset):
    def __init__(self, pairs: List[Tuple], max_len: int, scaler: Optional[StandardScaler],
                 num_classes: int, lag: int = 0):
        X, Y = [], []
        feat_dim = None
        for df, label in pairs:
            x = df.values.astype(np.float32)
            if feat_dim is None:
                feat_dim = x.shape[1]
            elif x.shape[1] != feat_dim:
                raise ValueError("All samples must share the same feature dimension.")
            if lag > 0:
                x = lag_difference(x, lag)
            if scaler is not None:
                x = scaler.transform(x)
            if x.shape[0] >= max_len:
                x = x[:max_len, :]
            else:
                pad = np.zeros((max_len - x.shape[0], x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
            X.append(np.expand_dims(x, axis=0))
            Y.append(normalize_label(np.asarray(label, dtype=np.float32)))
        self.X = torch.from_numpy(np.stack(X))
        self.Y = torch.from_numpy(np.stack(Y))

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def fit_global_scaler(pairs: List[Tuple], lag: int, max_len: int) -> StandardScaler:
    rows = []
    for df, _ in pairs:
        x = df.values.astype(np.float32)
        if lag > 0:
            if x.shape[0] <= lag: continue
            x = x[lag:, :] - x[:-lag, :]
        x = x[:max_len, :]
        rows.append(x)
    X = np.concatenate(rows, axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

@dataclass
class EvalOut:
    kl: float
    mae: float
    thr01: float
    thr02: float
    dyn_topk: float
    presence_f1: Optional[float] = None
    presence_precision: Optional[float] = None
    presence_recall: Optional[float] = None

def thr_acc_nonzero(pred: torch.Tensor, tgt: torch.Tensor, th: float) -> float:
    diff = (pred - tgt).abs()
    accs = []
    for b in range(pred.size(0)):
        mask = tgt[b] > 0
        if mask.any():
            accs.append((diff[b][mask] < th).float().mean().item())
    return float(np.mean(accs)) if accs else 0.0

def dyn_topk(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    B, C = pred.shape
    hits = total = 0
    for b in range(B):
        true_idx = (tgt[b] > 0).nonzero(as_tuple=True)[0]
        P = true_idx.numel()
        if P == 0: continue
        k = min(P, C)
        top_idx = torch.topk(pred[b], k=k, dim=0).indices
        hits += torch.isin(true_idx, top_idx).sum().item()
        total += P
    return 100.0 * hits / max(total, 1)


def _load_class_names(n_classes: int, path: Optional[str]):
    if path is None:
        return [f"class_{i}" for i in range(n_classes)]
    try:
        if path.lower().endswith(".json"):
            with open(path, "r") as f:
                names = json.load(f)
        else:
            with open(path, "r") as f:
                names = [ln.strip() for ln in f if ln.strip()]
        if len(names) != n_classes:
            print(f"[WARN] {path} has {len(names)} names but model has {n_classes} classes. Using indices.")
            return [f"class_{i}" for i in range(n_classes)]
        return names
    except Exception as e:
        print(f"[WARN] Failed to read class names from {path}: {e}. Using indices.")
        return [f"class_{i}" for i in range(n_classes)]

@torch.no_grad()
def _collect_probs_and_targets(model, loader, device, temp_scaler=None, has_presence_head=True):
    model.eval()
    all_pred, all_tgt = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        if has_presence_head:
            logits, _ = model(x)
        else:
            logits = model(x)
        if temp_scaler is not None:
            logits = temp_scaler(logits)
        probs = torch.softmax(logits, dim=1)
        all_pred.append(probs.cpu())
        all_tgt.append(y.cpu())
    return torch.cat(all_pred, 0), torch.cat(all_tgt, 0)

def _per_class_metrics(pred: torch.Tensor, tgt: torch.Tensor, eps: float, thr: float):
    B, C = pred.shape
    out = []
    with torch.no_grad():
        for c in range(C):
            gt = tgt[:, c]
            pd = pred[:, c]
            mask = gt > eps
            nz = int(mask.sum().item())
            if nz == 0:
                out.append({"mae": float("nan"), "mse": float("nan"), "nonzero": 0, "within": float("nan")})
                continue
            diff = (pd[mask] - gt[mask]).abs()
            mae = float(diff.mean().item())
            mse = float(((pd[mask] - gt[mask]) ** 2).mean().item())
            within = float((diff < thr).float().mean().item())
            out.append({"mae": mae, "mse": mse, "nonzero": nz, "within": within})
    return out

def _print_per_class_table(title: str, names: list[str], stats: list[dict], thr: float):
    bar = "-" * 90
    print(bar)
    print(title)
    print(bar)
    print(f"{'Ingredient':<16} | {'MAE':>7} | {'MSE':>10} | {'Non-zero GTs':>12} | Within {thr:.1f}")
    print("-" * 90)
    for name, st in zip(names, stats):
        if st["nonzero"] == 0:
            print(f"{name:<16} | {'nan':>7} | {'nan':>10} | {0:>12} | {'N/A':>10}")
        else:
            within_pct = 100.0 * st["within"]
            k_in = int(round(st["within"] * st["nonzero"]))
            print(f"{name:<16} | {st['mae']:>7.4f} | {st['mse']:>10.6f} | {st['nonzero']:>12} | "
                  f"{k_in}/{st['nonzero']} ({within_pct:.1f}%)")

def _save_per_class_csv(path: str, names: list[str], stats: list[dict], thr: float):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    import math as _m
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ingredient", "mae", "mse", "nonzero_gts", f"within_{thr}"])
        for name, st in zip(names, stats):
            mae = "" if _m.isnan(st["mae"]) else f"{st['mae']:.6f}"
            mse = "" if _m.isnan(st["mse"]) else f"{st['mse']:.6f}"
            within = "" if _m.isnan(st["within"]) else f"{st['within']:.4f}"
            w.writerow([name, mae, mse, st["nonzero"], within])

class TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.t = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.t.clamp_min(1e-3)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             temp_scaler: Optional[TempScaler] = None, has_presence_head: bool = True,
             present_thresh: float = 0.5) -> EvalOut:
    model.eval()
    kls, maes = [], []
    all_pred, all_tgt = [], []
    pres_tp = pres_fp = pres_fn = 0.0

    for x, y in loader:
        x = x.to(device); y = y.to(device)
        if has_presence_head:
            logits, presence_logits = model(x)
        else:
            logits = model(x)
            presence_logits = None
        if temp_scaler is not None:
            logits = temp_scaler(logits)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        kl  = F.kl_div(log_probs, y, reduction="batchmean")
        mae = (probs - y).abs().mean()
        kls.append(kl.item()); maes.append(mae.item())
        all_pred.append(probs.cpu()); all_tgt.append(y.cpu())

        if has_presence_head and presence_logits is not None:
            present_tgt = (y > 0).float()
            present_pred = (torch.sigmoid(presence_logits) > present_thresh).float()
            tp = (present_pred * present_tgt).sum().item()
            fp = (present_pred * (1 - present_tgt)).sum().item()
            fn = ((1 - present_pred) * present_tgt).sum().item()
            pres_tp += tp; pres_fp += fp; pres_fn += fn

    pred = torch.cat(all_pred, 0)
    tgt  = torch.cat(all_tgt, 0)

    metrics = EvalOut(
        kl=float(np.mean(kls)),
        mae=float(np.mean(maes)),
        thr01=thr_acc_nonzero(pred, tgt, 0.1),
        thr02=thr_acc_nonzero(pred, tgt, 0.2),
        dyn_topk=dyn_topk(pred, tgt),
    )
    if has_presence_head:
        precision = pres_tp / max(pres_tp + pres_fp, 1e-8)
        recall    = pres_tp / max(pres_tp + pres_fn, 1e-8)
        f1        = 2*precision*recall / max(precision+recall, 1e-8)
        metrics.presence_precision = precision
        metrics.presence_recall    = recall
        metrics.presence_f1        = f1
    return metrics

def fit_temperature(model, val_loader, device, steps=150, lr=0.01, has_presence_head=True):
    model.eval()
    scaler = TempScaler().to(device)
    scaler.train()
    opt = torch.optim.Adam(scaler.parameters(), lr=lr)
    for _ in range(steps):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                if has_presence_head:
                    logits, _ = model(x)
                else:
                    logits = model(x)
            logits = scaler(logits)
            probs  = torch.softmax(logits, dim=1)
            present = (y > 0).float()
            abs_err = (probs - y).abs()
            loss = (abs_err * present).sum() / present.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return scaler

@torch.no_grad()
def sweep_presence_thresh(model, loader, device, temp_scaler=None, has_presence_head=True,
                          candidates=(0.35,0.40,0.45,0.50,0.55,0.60,0.65)):
    best_t, best_thr02 = candidates[0], -1.0
    for t in candidates:
        out = evaluate(model, loader, device, temp_scaler=temp_scaler,
                       has_presence_head=has_presence_head, present_thresh=t)
        if out.thr02 > best_thr02:
            best_thr02, best_t = out.thr02, t
    return best_t, best_thr02

# --------------------- Subset helpers ---------------------
def pure_indices_from_dataset(ds, eps: float = 1e-6):
    y = ds.Y if isinstance(ds.Y, torch.Tensor) else torch.tensor(ds.Y)
    idx = (y > eps).sum(dim=1).eq(1).nonzero(as_tuple=True)[0].cpu().tolist()
    return idx

def mixture_indices_from_dataset(ds, eps: float = 1e-6):
    y = ds.Y if isinstance(ds.Y, torch.Tensor) else torch.tensor(ds.Y)
    idx = (y > eps).sum(dim=1).ge(2).nonzero(as_tuple=True)[0].cpu().tolist()
    return idx

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    ap.add_argument("--arch", choices=["tcn","lstm","transformer"], required=False, help="Backbone arch; inferred from filename if omitted")
    ap.add_argument("--train-dir", type=str, required=True)
    ap.add_argument("--test-dir", type=str, required=False, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # subset selection
    ap.add_argument("--eval-pure-only", action="store_true")
    ap.add_argument("--eval-mixture-only", action="store_true")
    ap.add_argument("--pure-eps", type=float, default=1e-6)
    # per-class
    ap.add_argument("--per-class", action="store_true",
                help="Print per-ingredient analysis (MAE, MSE, non-zero count, within-threshold) for VAL/TEST.")
    ap.add_argument("--thr-acc", type=float, default=0.2,
                    help="Absolute error tolerance for 'Within x' accuracy (default: 0.2).")
    ap.add_argument("--class-names", type=str, default=None,
                help="Optional path to class names (JSON list or TXT: one name per line).")
    ap.add_argument("--per-class-save", type=str, default=None,
                help="Prefix to save per-class CSVs as <prefix>_VAL.csv and <prefix>_TEST.csv")
    # calibration controls
    ap.add_argument("--no-temp", action="store_true",
                help="Skip temperature scaling.")
    ap.add_argument("--calibrate-on", choices=["subset","all"], default="subset",
                help="Fit temperature on validation 'subset' (PURE/MIX) or 'all' validation data.")
    ap.add_argument("--thresh-on", choices=["subset","all"], default="subset",
                help="Sweep presence threshold on 'subset' or 'all' validation data.")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_smell_recognition_data is None:
        raise RuntimeError("Could not import load_smell_recognition_data.")

    # Load data pairs
    train_pairs = load_smell_recognition_data(args.train_dir)
    test_pairs  = load_smell_recognition_data(args.test_dir) if args.test_dir else None

    # Standardization (fit on train only)
    scaler = None
    if not args.no_standardize:
        scaler = fit_global_scaler(train_pairs, lag=args.lag, max_len=args.max_len)

    # Build DS/Loaders
    n = len(train_pairs)
    val_size = max(1, int(n * 0.15))
    idx = np.arange(n); rng = np.random.default_rng(seed=args.seed); rng.shuffle(idx)
    val_idx = set(idx[:val_size].tolist())
    train_list = [train_pairs[i] for i in range(n) if i not in val_idx]
    val_list   = [train_pairs[i] for i in range(n) if i in val_idx]

    num_classes = len(train_pairs[0][1])
    val_ds   = DistDataset(val_list,   args.max_len, scaler, num_classes=num_classes, lag=args.lag)
    test_ds  = DistDataset(test_pairs, args.max_len, scaler, num_classes=num_classes, lag=args.lag) if test_pairs else None

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False) if test_ds else None

    # Build model
    in_ch = val_ds.X.shape[-1]
    arch = args.arch
    if arch is None:
        wname = args.weights.lower()
        if "transformer" in wname: arch = "transformer"
        elif "lstm" in wname: arch = "lstm"
        else: arch = "tcn"

    if arch == "tcn":
        ModelClass = SmellTemporalCNN
        assert ModelClass is not None, "Could not import SmellTemporalCNN from run_chi_model.py"
        model = ModelClass(in_ch=in_ch, num_classes=num_classes)
    elif arch == "lstm":
        assert SmellBiLSTM is not None, "extra_models.SmellBiLSTM not found"
        model = SmellBiLSTM(in_ch=in_ch, num_classes=num_classes)
    elif arch == "transformer":
        assert SmellTransformer is not None, "extra_models.SmellTransformer not found"
        model = SmellTransformer(in_ch=in_ch, num_classes=num_classes)
    else:
        raise ValueError("Unknown arch")

    # Load weights
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    # Subset selection on VAL/TEST
    eval_val_loader  = val_loader
    eval_test_loader = test_loader

    if args.eval_pure_only and args.eval_mixture_only:
        raise ValueError("Choose only one of --eval-pure-only or --eval-mixture-only")
    if args.eval_pure_only:
        vidx = pure_indices_from_dataset(val_ds, eps=args.pure_eps)
        tidx = pure_indices_from_dataset(test_ds, eps=args.pure_eps) if test_ds is not None else []
        subset_name = "PURE"
    elif args.eval_mixture_only:
        vidx = mixture_indices_from_dataset(val_ds, eps=args.pure_eps)
        tidx = mixture_indices_from_dataset(test_ds, eps=args.pure_eps) if test_ds is not None else []
        subset_name = "MIX"
    else:
        vidx, tidx, subset_name = None, None, "ALL"

    print(f"[SUBSET] {subset_name} (eps={args.pure_eps:g})")
    if vidx is not None:
        print(f"[INFO] VAL {len(val_ds)} -> {len(vidx)} samples")
        if len(vidx) > 0:
            eval_val_loader = DataLoader(Subset(val_ds, vidx), batch_size=args.batch_size, shuffle=False)
    if eval_test_loader is not None and tidx is not None:
        print(f"[INFO] TEST {len(test_ds)} -> {len(tidx)} samples")
        if len(tidx) > 0:
            eval_test_loader = DataLoader(Subset(test_ds, tidx), batch_size=args.batch_size, shuffle=False)

    # Calibration / thresholds
    has_presence_head = hasattr(model, "presence_head")
    calib_loader  = eval_val_loader if args.calibrate_on == "subset" else val_loader
    thresh_loader = eval_val_loader if args.thresh_on    == "subset" else val_loader

    temp_scaler = None
    if not args.no_temp:
        temp_scaler = fit_temperature(model, calib_loader, device, has_presence_head=has_presence_head)
    best_t, _ = sweep_presence_thresh(model, thresh_loader, device, temp_scaler=temp_scaler, has_presence_head=has_presence_head)

    # Evaluate
    val_out = evaluate(model, eval_val_loader, device, temp_scaler=temp_scaler, has_presence_head=has_presence_head, present_thresh=best_t)
    print(f"[VAL thresh={best_t:.2f}] KL={val_out.kl:.4f} MAE={val_out.mae:.4f} @0.1={val_out.thr01:.3f} @0.2={val_out.thr02:.3f} "
          f"dynTopK={val_out.dyn_topk:.2f}% Pres(F1/Prec/Rec)={val_out.presence_f1 if val_out.presence_f1 is not None else float('nan'):.3f}/"
          f"{val_out.presence_precision if val_out.presence_precision is not None else float('nan'):.3f}/"
          f"{val_out.presence_recall if val_out.presence_recall is not None else float('nan'):.3f}")

    if eval_test_loader is not None:
        test_out = evaluate(model, eval_test_loader, device, temp_scaler=temp_scaler, has_presence_head=has_presence_head, present_thresh=best_t)
        print(f"[TEST thresh={best_t:.2f}] KL={test_out.kl:.4f} MAE={test_out.mae:.4f} @0.1={test_out.thr01:.3f} @0.2={test_out.thr02:.3f} "
              f"dynTopK={test_out.dyn_topk:.2f}% Pres(F1/Prec/Rec)={test_out.presence_f1 if test_out.presence_f1 is not None else float('nan'):.3f}/"
              f"{test_out.presence_precision if test_out.presence_precision is not None else float('nan'):.3f}/"
              f"{test_out.presence_recall if test_out.presence_recall is not None else float('nan'):.3f}")

    # Per-class analysis
    if args.per_class:
        names = _load_class_names(num_classes, args.class_names)

        # VAL
        val_pred, val_tgt = _collect_probs_and_targets(model, eval_val_loader, device,
                                                       temp_scaler=temp_scaler,
                                                       has_presence_head=has_presence_head)
        val_stats = _per_class_metrics(val_pred, val_tgt, eps=args.pure_eps, thr=args.thr_acc)
        _print_per_class_table("PER-INGREDIENT ANALYSIS (VAL)", names, val_stats, args.thr_acc)
        if args.per_class_save:
            _save_per_class_csv(args.per_class_save + "_VAL.csv", names, val_stats, args.thr_acc)
        else:
            os.makedirs("analysis_eval", exist_ok=True)
            _save_per_class_csv("analysis_eval/per_class_val.csv", names, val_stats, args.thr_acc)

        # TEST
        if eval_test_loader is not None:
            test_pred, test_tgt = _collect_probs_and_targets(model, eval_test_loader, device,
                                                             temp_scaler=temp_scaler,
                                                             has_presence_head=has_presence_head)
            test_stats = _per_class_metrics(test_pred, test_tgt, eps=args.pure_eps, thr=args.thr_acc)
            _print_per_class_table("PER-INGREDIENT ANALYSIS (TEST)", names, test_stats, args.thr_acc)
            if args.per_class_save:
                _save_per_class_csv(args.per_class_save + "_TEST.csv", names, test_stats, args.thr_acc)
            else:
                os.makedirs("analysis_eval", exist_ok=True)
                _save_per_class_csv("analysis_eval/per_class_test.csv", names, test_stats, args.thr_acc)

if __name__ == "__main__":
    main()
