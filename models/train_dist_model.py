#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, math, random, time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

try:
    from load_data import load_smell_recognition_data
except Exception:
    load_smell_recognition_data = None

class FallbackTemporalCNN(nn.Module):
    def __init__(self, in_ch=4, num_classes=12, base=64, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(base, base, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(base, base*2, kernel_size=7, padding=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base*2, base*2, kernel_size=7, padding=6, dilation=3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(base*2, 128, kernel_size=1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1,2)
        else:
            x = x.transpose(1,2)
        h = self.block1(x)
        h = self.block2(h)
        h = self.proj(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.head(h)

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
    if lag <= 0:
        return x
    T, F = x.shape
    if T <= lag:
        raise ValueError(f"Sequence length {T} too short for lag={lag}")
    return x[lag:, :] - x[:-lag, :]

class DistDataset(Dataset):
    def __init__(self, pairs: List[Tuple], max_len: int, scaler: StandardScaler | None,
                 num_classes: int, lag: int = 0):
        X, Y = [], []
        feat_dim = None
        for df, label in pairs:
            x = df.values.astype(np.float32)
            if feat_dim is None:
                feat_dim = x.shape[1]
            elif x.shape[1] != feat_dim:
                raise ValueError("Inconsistent feature dimension.")
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

@dataclass
class EvalOut:
    kl: float
    mae: float
    thr01: float
    thr02: float
    dyn_topk: float

def threshold_accuracy_nonzero(pred_probs: torch.Tensor, targets: torch.Tensor, th: float) -> float:
    """
    Match eval_patched: compute accuracy ONLY over non-zero target positions.
    For each sample b: mean_{i: y_i>0} [ |p_i - y_i| < th ]. Then average over samples.
    """
    B, C = pred_probs.shape
    accs = []
    diff = (pred_probs - targets).abs()
    for b in range(B):
        mask = targets[b] > 0
        if mask.any():
            accs.append((diff[b][mask] < th).float().mean().item())
    return float(np.mean(accs)) if accs else 0.0

def dynamic_topk_accuracy(pred_probs: torch.Tensor, targets: torch.Tensor) -> float:
    B, C = pred_probs.shape
    hits = 0
    total = 0
    for b in range(B):
        true_present = (targets[b] > 0).nonzero(as_tuple=True)[0]
        P = true_present.numel()
        if P == 0:
            continue
        k = min(P, C)
        top_idx = torch.topk(pred_probs[b], k=k, dim=0).indices
        hits += torch.isin(true_present, top_idx).sum().item()
        total += P
    return 100.0 * hits / max(total, 1)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalOut:
    model.eval()
    kls, maes = [], []
    all_pred, all_tgt = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        kl = F.kl_div(log_probs, y, reduction='batchmean')
        mae = (probs - y).abs().mean()
        kls.append(kl.item()); maes.append(mae.item())
        all_pred.append(probs.cpu()); all_tgt.append(y.cpu())
    pred = torch.cat(all_pred, 0)
    tgt  = torch.cat(all_tgt, 0)
    thr01 = threshold_accuracy_nonzero(pred, tgt, 0.1)
    thr02 = threshold_accuracy_nonzero(pred, tgt, 0.2)
    dyn   = dynamic_topk_accuracy(pred, tgt)
    return EvalOut(kl=float(np.mean(kls)), mae=float(np.mean(maes)), thr01=thr01, thr02=thr02, dyn_topk=dyn)

def fit_global_scaler(pairs: List[Tuple], lag: int, max_len: int) -> StandardScaler:
    rows = []
    for df, _ in pairs:
        x = df.values.astype(np.float32)
        if lag > 0:
            if x.shape[0] <= lag:
                continue
            x = x[lag:, :] - x[:-lag, :]
        x = x[:max_len, :]
        rows.append(x)
    X = np.concatenate(rows, axis=0) if rows else None
    if X is None:
        raise RuntimeError("No data to fit scaler.")
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=600)
    p.add_argument("--lag", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--save", type=str, default="dist_model.pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_smell_recognition_data is None:
        raise RuntimeError("Import load_smell_recognition_data failed. Please adjust import.")

    print("[1/4] Loading data...")
    train_pairs = load_smell_recognition_data(args.train_dir)
    test_pairs = load_smell_recognition_data(args.test_dir) if args.test_dir else None

    print("[2/4] Fitting StandardScaler on training set...")
    scaler = fit_global_scaler(train_pairs, lag=args.lag, max_len=args.max_len)

    # split train/val
    n = len(train_pairs)
    val_size = max(1, int(n * args.val_split))
    idx = np.arange(n)
    rng = np.random.default_rng(seed=args.seed)
    rng.shuffle(idx)
    val_idx = set(idx[:val_size].tolist())
    train_list = [train_pairs[i] for i in range(n) if i not in val_idx]
    val_list   = [train_pairs[i] for i in range(n) if i in val_idx]

    num_classes = len(train_pairs[0][1])
    train_ds = DistDataset(train_list, args.max_len, scaler, num_classes=num_classes, lag=args.lag)
    val_ds   = DistDataset(val_list,   args.max_len, scaler, num_classes=num_classes, lag=args.lag)
    test_ds  = DistDataset(test_pairs, args.max_len, scaler, num_classes=num_classes, lag=args.lag) if test_pairs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False) if test_ds else None

    in_ch = train_ds.X.shape[-1]
    try:
        from run_chi_model import SmellTemporalCNN as SmellCNN
        ModelClass = SmellCNN
    except Exception:
        ModelClass = FallbackTemporalCNN
    model = ModelClass(in_ch=in_ch, num_classes=num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_kl = float("inf")
    best_state = None
    patience, patience_left = 7, 7

    print("[3/4] Training...")
    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, y, reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_out = evaluate(model, val_loader, device)
        print(f"[{ep:03d}] train_kl={np.mean(losses):.4f} | val_kl={val_out.kl:.4f} "
              f"| val_mae={val_out.mae:.4f} | val@0.1={val_out.thr01:.3f} | val@0.2={val_out.thr02:.3f} | dynTopK={val_out.dyn_topk:.2f}%")
        if val_out.kl < best_kl - 1e-6:
            best_kl = val_out.kl
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("[4/4] Evaluating...")
    val_out = evaluate(model, val_loader, device)
    print(f"Val: KL={val_out.kl:.4f} MAE={val_out.mae:.4f} @0.1={val_out.thr01:.3f} @0.2={val_out.thr02:.3f} dynTopK={val_out.dyn_topk:.2f}%")
    if test_loader is not None:
        test_out = evaluate(model, test_loader, device)
        print(f"Test: KL={test_out.kl:.4f} MAE={test_out.mae:.4f} @0.1={test_out.thr01:.3f} @0.2={test_out.thr02:.3f} dynTopK={test_out.dyn_topk:.2f}%")

    torch.save(model.state_dict(), args.save)
    print(f"[OK] Saved weights to {args.save}")

if __name__ == "__main__":
    main()
