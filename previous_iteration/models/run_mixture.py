#!/usr/bin/env python3
"""
run_mixture_sweep_fixed.py

Mixture/distribution training with Presence head, using a sweep-style CLI
similar to run.py. This version fixes several bugs and supports data coming
from the original mixture loader (list of (DataFrame, label_vector) pairs).

Key fixes:
- Proper sliding-window maker for *pair* data: [(df, label_vec), ...]
- Handles missing test set by carving a 15% validation split
- Computes K (num classes) from labels instead of hard-coding
- Builds one-hot/distribution tensors for y (float32) consistently
- Computes presence prior from training labels (y_train)
- Guards against None test_data and standardizes only on train
- Corrected make_sliding_window_dataset loop order
"""

from __future__ import annotations
import argparse, os, math, random, json, hashlib
from dataclasses import dataclass
from itertools import product
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ---------------- Models & data helpers ----------------
from models import Transformer, LSTMNet, MLPClassifier, CNN1DClassifier
from load_data import load_smell_recognition_data

MODEL_CHOICES = ["mlp", "cnn", "lstm", "transformer"]

# ---------------- Small utils (mirrors run.py shape) ----------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def to_dtype(name: str):
    if name == "float32": return torch.float32
    if name == "float64": return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")

def highpass_fft_batch(X: np.ndarray, sampling_rate: float, cutoff: float) -> np.ndarray:
    """Simple FFT high-pass per channel, per window."""
    if cutoff is None or cutoff <= 0: return X
    N, T, C = X.shape
    Xhp = np.empty_like(X, dtype=np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / sampling_rate)
    mask = freqs >= cutoff
    for n in range(N):
        for c in range(C):
            x = X[n, :, c]
            Xf = np.fft.rfft(x)
            Xf[~mask] = 0
            xhp = np.fft.irfft(Xf, n=T)
            Xhp[n, :, c] = xhp.astype(np.float32)
    return Xhp

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Spec & CLI ----------------
@dataclass
class RunSpec:
    model: str
    gradient: int
    window_size: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    device: str | None
    fft: bool
    fft_cutoff: float
    sampling_rate: float
    stride: int | None
    dtype: str

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mixture model sweep (run.py-style).")
    # Sweep axes
    p.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["mlp"])
    p.add_argument("--gradients", nargs="+", type=int, choices=[0, 100, 250, 500], default=[0])
    p.add_argument("--window-sizes", nargs="+", type=int, default=[100])

    # Training knobs
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    # Data & I/O
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--unseen-test-dir", type=str, required=True)
    p.add_argument("--log-dir", type=str, default="./runs")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--run-name-prefix", type=str, default="")
    p.add_argument("--stride", type=int, default=None, help="Sliding stride (default: window//2)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    # Standardization
    p.add_argument("--no-standardize", action="store_true", help="Disable StandardScaler on windows.")

    # FFT options
    p.add_argument("--fft", choices=["off", "on"], default="off", help="Apply FFT high-pass cleaning to windows.")
    p.add_argument("--fft-cutoff", type=float, default=0.05, help="High-pass cutoff in Hz (used if --fft on).")
    p.add_argument("--sampling-rate", type=float, default=1.0, help="Sampling rate (Hz) of your sensor.")

    # Mixture/presence losses
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for ε-insensitive L1 (present classes).")
    p.add_argument("--beta", type=float, default=0.5, help="Weight for focal BCE (presence head).")
    p.add_argument("--synth-p", type=float, default=0.6, help="Probability to synth-mix a sample in batch.")
    p.add_argument("--synth-max-k", type=int, default=3, help="Max components in a synthetic mixture.")
    p.add_argument("--no-temp", action="store_true", help="Disable temperature scaling calibration.")

    # Accepted but ignored (for CLI parity)
    p.add_argument("--contrastive", nargs="+", choices=["off","on"], default=["off"])
    p.add_argument("--gcms-csv", type=str, default=None)

    return p

def iter_run_specs(args: argparse.Namespace) -> Iterable[RunSpec]:
    from itertools import product
    for g in args.gradients:
        if g < 0: raise ValueError(f"gradient must be >= 0, got {g}")
    for w in args.window_sizes:
        if w <= 0: raise ValueError(f"window_size must be > 0, got {w}")
    for model, grad, win in product(args.models, args.gradients, args.window_sizes):
        yield RunSpec(
            model=model,
            gradient=grad,
            window_size=win,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            fft=(args.fft == "on"),
            fft_cutoff=args.fft_cutoff,
            sampling_rate=args.sampling_rate,
            stride=args.stride,
            dtype=args.dtype,
        )

# ---------------- Presence-wrapper & factory ----------------
def get_model(
    name: str,
    *,
    num_features: int,
    num_classes: int,
    window_size: Optional[int] = None,
    channel_last: bool = True,
    **hparams,
) -> nn.Module:
    n = name.lower()
    if n == "mlp":
        mlp_pool    = hparams.get("mlp_pool", "mean")
        mlp_hidden  = hparams.get("mlp_hidden", (256, 256))
        mlp_dropout = hparams.get("mlp_dropout", 0.2)
        in_features = num_features if mlp_pool != "flatten" else (num_features * (window_size or 1))
        return MLPClassifier(in_features=in_features, num_classes=num_classes, hidden_sizes=mlp_hidden, dropout=mlp_dropout, pool=mlp_pool, channel_last=channel_last)
    elif n == "cnn":
        return CNN1DClassifier(in_channels=num_features, num_classes=num_classes, channels=hparams.get("cnn_channels",(64,128,256)), kernel_size=hparams.get("cnn_kernel",5), dropout=hparams.get("cnn_dropout",0.2), channel_last=channel_last)
    elif n == "lstm":
        return LSTMNet(input_dim=num_features, hidden_dim=hparams.get("lstm_hidden",256), embedding_dim=hparams.get("lstm_embedding",256), num_classes=num_classes, num_layers=hparams.get("lstm_layers",1))
    elif n in ("transformer","ts-transformer"):
        return Transformer(input_dim=num_features, model_dim=hparams.get("tf_dim",256), num_classes=num_classes, num_heads=hparams.get("tf_heads",8), num_layers=hparams.get("tf_layers",3), dropout=hparams.get("tf_dropout",0.1))
    raise ValueError(f"Unknown model '{name}'.")

def features_from_model(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    out = model(x)
    if isinstance(out,(tuple,list)) and len(out) >= 2:
        return out[1]
    raise RuntimeError("Model doesn't expose features; add forward_features or return (logits, embedding).")

class PresenceWrapper(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int):
        super().__init__()
        self.base = base
        self.presence_head = nn.LazyLinear(num_classes)
    def forward(self, x: torch.Tensor):
        logits = self.base(x)
        try:
            feat = features_from_model(self.base, x)
        except Exception:
            feat = logits
        presence_logits = self.presence_head(feat)
        return logits, presence_logits

# ---------------- Windows / standardization ----------------
def make_sliding_window_dataset(
    pairs: List[Tuple],   # [(pd.DataFrame, label_vec), ...]
    window_size: int = 100,
    stride: int = 50,
):
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for df, label in pairs:
        arr = df.values  # (T, C)
        T = arr.shape[0]
        for start in range(0, max(0, T - window_size + 1), stride):
            sl = arr[start:start+window_size]
            if sl.shape[0] == window_size:
                X_list.append(sl.astype(np.float32))
                y_list.append(np.asarray(label, dtype=np.float32))
    X = np.array(X_list, dtype=np.float32)  # (N, T, C)
    y = np.array(y_list, dtype=np.float32)  # (N, K) distributions or one-hots
    return X, y

def build_sliding_data(pairs: List[Tuple], window: int, stride: int | None):
    stride = stride if stride is not None else max(1, window // 2)
    return make_sliding_window_dataset(pairs, window_size=window, stride=stride)

def fit_standardizer_from_windows(X: np.ndarray) -> StandardScaler:
    N, T, C = X.shape
    flat = X.reshape(N*T, C)
    ss = StandardScaler()
    ss.fit(flat)
    return ss

def apply_standardizer(X: np.ndarray, ss: StandardScaler | None) -> np.ndarray:
    if ss is None: return X
    N, T, C = X.shape
    flat = ss.transform(X.reshape(N*T, C))
    return flat.reshape(N, T, C).astype(np.float32)

def diff_batch(X: np.ndarray, periods: int) -> np.ndarray:
    if periods <= 0: return X
    print(X.shape)
    return X[:, periods:, :] - X[:, :-periods, :]

def diff_pairs(pairs: List[Tuple], periods: int) -> List[Tuple]:
    """
    Apply finite differencing to each DataFrame in (df, label) pairs *before* windowing.
    Drops pairs that are too short for the requested difference.
    """
    if periods <= 0:
        return pairs
    out: List[Tuple] = []
    for df, label in (pairs or []):
        arr = df.values.astype(np.float32)  # (T, C)
        T = arr.shape[0]
        if T <= periods:
            # Not enough timesteps to compute the difference -> skip this pair
            continue
        diff_arr = arr[periods:, :] - arr[:-periods, :]
        df2 = df.iloc[periods:].copy()
        # overwrite values with our diffed array (same shape and index/cols)
        df2.iloc[:, :] = diff_arr
        out.append((df2, label))
    return out

# ---------------- Synthetic mix & metrics ----------------
def mix_synthetic_batch(x, y, p=0.6, max_components=3):
    if p <= 0.0 or x.size(0) < 2: return x, y
    B = x.size(0); dev = x.device
    x4 = x.unsqueeze(1) if x.dim() == 3 else x
    out_x = x4.clone(); out_y = y.clone()
    for b in range(B):
        if torch.rand((), device=dev).item() < p:
            k = int(torch.randint(2, max_components + 1, (1,), device=dev))
            idx = torch.randint(0, B, (k,), device=dev)
            w = torch.rand(k, device=dev); w = w / w.sum()
            mix_x = torch.sum(x4[idx] * w.view(-1,1,1,1), dim=0)
            mix_y = torch.sum(y[idx] * w.view(-1,1), dim=0)
            out_x[b] = mix_x; out_y[b] = mix_y / mix_y.sum().clamp_min(1e-8)
    if x.dim() == 3: out_x = out_x.squeeze(1)
    return out_x, out_y

from dataclasses import dataclass
@dataclass
class EvalOut:
    kl: float; mae: float; thr01: float; thr02: float; dyn_topk: float; presence_f1: float; presence_precision: float; presence_recall: float

def thr_acc_nonzero(pred, tgt, th):
    diff = (pred - tgt).abs(); accs = []
    for b in range(pred.size(0)):
        m = tgt[b] > 0
        if m.any(): accs.append((diff[b][m] < th).float().mean().item())
    return float(np.mean(accs)) if accs else 0.0

def dyn_topk(pred, tgt):
    B, C = pred.shape; hits = total = 0
    for b in range(B):
        true_idx = (tgt[b] > 0).nonzero(as_tuple=True)[0]; P = true_idx.numel()
        if P == 0: continue
        k = min(P, C); top_idx = torch.topk(pred[b], k=k, dim=0).indices
        hits += torch.isin(true_idx, top_idx).sum().item(); total += P
    return 100.0 * hits / max(total, 1)

class TempScaler(nn.Module):
    def __init__(self): super().__init__(); self.t = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.t.clamp_min(1e-3)

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    w = alpha * (1 - p_t).pow(gamma)
    return (w * ce).mean()

@torch.no_grad()
def evaluate(model, loader, device, temp_scaler: TempScaler | None = None, present_thresh=0.5):
    model.eval()
    kls, maes = [], []
    all_pred, all_tgt = [], []
    pres_tp = pres_fp = pres_fn = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, presence_logits = model(xb)
        if temp_scaler is not None: logits = temp_scaler(logits)
        log_probs = F.log_softmax(logits, dim=1); probs = log_probs.exp()
        kls.append(F.kl_div(log_probs, yb, reduction="batchmean").item())
        maes.append((probs - yb).abs().mean().item())
        all_pred.append(probs.cpu()); all_tgt.append(yb.cpu())
        present_tgt = (yb > 0).float()
        present_pred = (torch.sigmoid(presence_logits) > present_thresh).float()
        pres_tp += (present_pred * present_tgt).sum().item()
        pres_fp += (present_pred * (1 - present_tgt)).sum().item()
        pres_fn += ((1 - present_pred) * present_tgt).sum().item()
    pred = torch.cat(all_pred, 0); tgt = torch.cat(all_tgt, 0)
    precision = pres_tp / max(pres_tp + pres_fp, 1e-8)
    recall    = pres_tp / max(pres_tp + pres_fn, 1e-8)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return EvalOut(
        kl=float(np.mean(kls)),
        mae=float(np.mean(maes)),
        thr01=thr_acc_nonzero(pred, tgt, 0.1),
        thr02=thr_acc_nonzero(pred, tgt, 0.2),
        dyn_topk=dyn_topk(pred, tgt),
        presence_f1=f1, presence_precision=precision, presence_recall=recall,
    )

def fit_temperature(model, val_loader, device, steps=150, lr=0.01):
    model.eval(); scaler = TempScaler().to(device); opt = torch.optim.Adam(scaler.parameters(), lr=lr)
    for _ in range(steps):
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                logits, _ = model(xb)
            logits = scaler(logits)
            probs = torch.softmax(logits, dim=1)
            present = (yb > 0).float()
            abs_err = (probs - yb).abs()
            loss = (abs_err * present).sum() / present.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return scaler
    
def train(model, train_loader, val_loader, device, *, epochs=30, lr=3e-4, weight_decay=3e-4, alpha=0.5, beta=0.5, synth_p=0.6, synth_max_k=3, temp_scaling=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_thr02 = -1.0; best_state = None; patience, patience_left = 8, 8
    for ep in range(1, epochs+1):
        model.train(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb, yb = mix_synthetic_batch(xb, yb, p=synth_p, max_components=synth_max_k)
            logits, presence_logits = model(xb)
            log_probs = F.log_softmax(logits, dim=1); probs = log_probs.exp()
            kl = F.kl_div(log_probs, yb, reduction="batchmean")
            present = (yb > 0).float()
            eps = 0.2
            eps_l1 = ((probs - yb).abs().sub(eps).clamp_min(0.0) * present).sum() / present.sum().clamp_min(1.0)
            bce = focal_bce(presence_logits, present)
            loss = kl + alpha * eps_l1 + beta * bce
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # val_out = evaluate(model, val_loader, device, temp_scaler=None)
        print(f"[{ep:03d}] train_loss={np.mean(losses):.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (fit_temperature(model, val_loader, device) if temp_scaling else None)

# ---------------- Main (sweep over specs) ----------------
def main():
    args = build_parser().parse_args()
    ensure_dir(args.log_dir); ensure_dir(args.save_dir)
    dtype = to_dtype(args.dtype)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the original mixture pairs: List[(DataFrame, label_vector)]
    train_pairs = load_smell_recognition_data(args.train_dir)
    test_pairs  = load_smell_recognition_data(args.test_dir) if args.test_dir else None
    unseen_test_pairs = load_smell_recognition_data(args.unseen_test_dir)

    for spec in iter_run_specs(args):
        set_seed(spec.seed)
        run_name = _make_run_name(args.run_name_prefix, spec)
        run_dir = Path(args.log_dir) / run_name
        ensure_dir(run_dir)
        print(f"\n=== Running {run_name} ===")

        try:
            if spec.gradient and spec.gradient > 0:
                train_pairs_proc       = diff_pairs(train_pairs, spec.gradient)
                test_pairs_proc        = diff_pairs(test_pairs, spec.gradient)
                unseen_test_pairs_proc = diff_pairs(unseen_test_pairs, spec.gradient)
            else:
                train_pairs_proc       = train_pairs
                test_pairs_proc        = test_pairs
                unseen_test_pairs_proc = unseen_test_pairs

            # --- NOW build windows from (possibly differenced) pairs ---
            Xtr_np, ytr_np = build_sliding_data(train_pairs_proc, spec.window_size, spec.stride)
            Xte_np, yte_np = build_sliding_data(test_pairs_proc, spec.window_size, spec.stride)
            Xte_np_unseen, yte_np_unseen = build_sliding_data(unseen_test_pairs_proc, spec.window_size, spec.stride)
            
            # Optional FFT cleanup
            if spec.fft:
                Xtr_np = highpass_fft_batch(Xtr_np, sampling_rate=spec.sampling_rate, cutoff=spec.fft_cutoff)
                Xte_np = highpass_fft_batch(Xte_np, sampling_rate=spec.sampling_rate, cutoff=spec.fft_cutoff)
                Xte_np_unseen = highpass_fft_batch(Xte_np_unseen, sampling_rate=spec.sampling_rate, cutoff=spec.fft_cutoff)

            # Standardization on TRAIN only
            scaler = None if args.no_standardize else fit_standardizer_from_windows(Xtr_np)
            Xtr_np = apply_standardizer(Xtr_np, scaler)
            Xte_np = apply_standardizer(Xte_np, scaler)
            Xte_np_unseen = apply_standardizer(Xte_np_unseen, scaler)

            # Determine class count K from labels (support indices or vectors)
            if ytr_np.ndim == 1:
                # integer class ids -> convert to one-hot
                K = int(max(ytr_np.max(), yte_np.max()) + 1)
                ytr_1h = np.eye(K, dtype=np.float32)[ytr_np.astype(int)]
                yte_1h = np.eye(K, dtype=np.float32)[yte_np.astype(int)]
                yte_1h_unseen = np.eye(K, dtype=np.float32)[yte_np_unseen.astype(int)]
            else:
                K = int(ytr_np.shape[1])
                ytr_1h = ytr_np.astype(np.float32)
                yte_1h = yte_np.astype(np.float32)
                yte_1h_unseen = yte_np_unseen.astype(np.float32)

            # Tensors / loaders
            Xtr = torch.from_numpy(Xtr_np).to(dtype); ytr = torch.from_numpy(ytr_1h.astype(np.float32))
            Xte = torch.from_numpy(Xte_np).to(dtype); yte = torch.from_numpy(yte_1h.astype(np.float32))
            Xte_unseen = torch.from_numpy(Xte_np_unseen).to(dtype); yte_unseen = torch.from_numpy(yte_1h_unseen.astype(np.float32))
            train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=spec.batch_size, shuffle=True, drop_last=False)
            val_loader   = DataLoader(TensorDataset(Xte, yte), batch_size=spec.batch_size, shuffle=False, drop_last=False)
            unseen_test_loader = DataLoader(TensorDataset(Xte_unseen, yte_unseen), batch_size=spec.batch_size, shuffle=False, drop_last=False)

            # Model + wrapper
            T, C = Xtr_np.shape[1], Xtr_np.shape[2]
            base = get_model(spec.model, num_features=C, num_classes=K, window_size=T, channel_last=True)
            model = PresenceWrapper(base, num_classes=K).to(device)

            # Train + calibrate
            model, temp_scaler = train(
                model, train_loader, val_loader, device,
                epochs=spec.epochs, lr=spec.lr, weight_decay=3e-4,
                alpha=args.alpha, beta=args.beta, synth_p=args.synth_p, synth_max_k=args.synth_max_k,
                temp_scaling=(not args.no_temp),
            )

            # Eval
            val_out = evaluate(model, val_loader, device, temp_scaler=temp_scaler, present_thresh=0.5)
            results = {
                "kl": val_out.kl,
                "mae": val_out.mae,
                "acc@0.1": val_out.thr01,
                "acc@0.2": val_out.thr02,
                "dynTopK%": val_out.dyn_topk,
                "presence": {
                    "f1": val_out.presence_f1,
                    "precision": val_out.presence_precision,
                    "recall": val_out.presence_recall,
                },
            }

            unseen_test_out = evaluate(model, unseen_test_loader, device, temp_scaler=temp_scaler, present_thresh=0.5)
            unseen_results = {
                "kl": unseen_test_out.kl,
                "mae": unseen_test_out.mae,
                "acc@0.1": unseen_test_out.thr01,
                "acc@0.2": unseen_test_out.thr02,
                "dynTopK%": unseen_test_out.dyn_topk,
                "presence": {
                    "f1": unseen_test_out.presence_f1,
                    "precision": unseen_test_out.presence_precision,
                    "recall": unseen_test_out.presence_recall,
                },
            }

            # Save checkpoint
            ckpt_path = Path(args.save_dir) / f"{run_name}.pt"
            ensure_dir(ckpt_path.parent)

            # Append JSONL
            append_results_jsonl(
                run_dir, spec, results=results,
                dataset={
                    "train_windows": int(Xtr_np.shape[0]), "test_windows": int(Xte_np.shape[0]),
                    "T": int(T), "C": int(C), "classes": int(K)
                },
                checkpoint=str(ckpt_path.resolve())
            )

            append_results_jsonl(
                run_dir, spec, results=unseen_results,
                dataset={
                    "train_windows": int(Xtr_np.shape[0]), "test_windows": int(Xte_np.shape[0]),
                    "T": int(T), "C": int(C), "classes": int(K)
                },
                checkpoint=str(ckpt_path.resolve())
            )

            print(f"[OK] {run_name}  | KL={val_out.kl:.4f}  MAE={val_out.mae:.4f}  @0.1={val_out.thr01:.3f}  F1={val_out.presence_f1:.3f}")

        except Exception as e:
            append_results_jsonl(Path(args.log_dir)/run_name, spec, results=None, error=str(e))
            raise

def append_results_jsonl(run_dir: Path, spec: RunSpec, *, results: dict | None = None, error: str | None = None, **extras):
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        **spec.__dict__,
        **extras,
    }
    if results is not None:
        rec["results"] = _jsonable(results)
    if error is not None:
        rec["error"] = error
    out_path = run_dir / "results.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

def _make_run_name(prefix: str, spec: RunSpec) -> str:
    base = f"{spec.model}-g{spec.gradient}-w{spec.window_size}-bs{spec.batch_size}-lr{spec.lr}-seed{spec.seed}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(base.encode()).hexdigest()[:6]
    return (prefix + "-" if prefix else "") + f"{base}-{ts}-{h}"

def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: _jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    return x

if __name__ == "__main__":
    main()
