#!/usr/bin/env python3
"""
SmellNet — single-file training & evaluation script

This integrates:
- Model (SmellCNN)
- Dataset (variable-length time series -> padded tensors)
- Train / Eval loops (with soft cross entropy and dynamic top‑K metric)
- Minimal example usage (with synthetic data as a fallback)

You can replace the `build_pairs_from_your_data()` function with your own loader
that returns a list of (np.ndarray[T, F], np.ndarray[12]) where labels are
percentage-style targets that sum to 100 (we normalize to sum to 1).

Run:
  python smellnet_single_file.py --epochs 10 --batch-size 32
"""

from __future__ import annotations

from load_data import load_smell_recognition_data
import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from models import SmellReproductionLSTMNet

# -----------------------------
# Utilities & Loss
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets (probability distributions).
    Args:
        logits: (B, C) unnormalized scores from the model
        target_probs: (B, C) target distribution that sums to 1
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1).mean()
    return loss


# -----------------------------
# Model
# -----------------------------

class SmellTemporalCNN(nn.Module):
    def __init__(self, in_ch=4, num_classes=12, base=64, dropout=0.1):
        super().__init__()
        # (B, C=4, T)
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
        self.proj   = nn.Conv1d(base*2, 128, kernel_size=1)
        # global pooling over time keeps long-range info
        self.head   = nn.Linear(128, num_classes)
        self.presence_head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 1, T, F) or (B, T, F). Convert to (B, C=F, T)
        if x.dim() == 4:   # (B,1,T,F) -> (B,F,T)
            x = x.squeeze(1).transpose(1,2)
        else:              # (B,T,F) -> (B,F,T)
            x = x.transpose(1,2)

        h = self.block1(x)
        h = self.block2(h)
        h = self.proj(h)                 # (B,128,T)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)  # (B,128)
        logits = self.head(h)
        presence_logits = self.presence_head(h)
        return logits, presence_logits


# -----------------------------
# Dataset
# -----------------------------

class SmellDataset(Dataset):
    """
    Dataset for smell recognition.
    Each item is a [pandas.DataFrame (T,F), label (array-like of length num_classes)].

    - Applies a lag-difference transform: x[t+25] - x[t]
    - Pads/truncates the time dimension to `max_len`
    - Ensures consistent feature dimension across all samples
    - Normalizes labels to a probability distribution (sum=1)
    """

    def __init__(self, pairs, max_len: int, num_classes: int = 12, lag: int = 200):
        """
        Args:
            pairs: list of [DataFrame, label]
            max_len: pad/truncate T to this length (after lag transform)
            num_classes: number of classes (default=12)
            lag: number of steps for differencing (default=25)
        """
        self.X = []
        self.y = []
        self.max_len = max_len
        self.num_classes = num_classes
        self.lag = lag

        feature_count = None
        for df, label in pairs:
            df = df.iloc[:, 1:]  # drop first column
            if not isinstance(df, pd.DataFrame):
                raise TypeError("First element of each pair must be a pandas.DataFrame")

            if feature_count is None:
                feature_count = df.shape[1]
            elif df.shape[1] != feature_count:
                raise ValueError("All samples must have the same number of features (columns).")

            # convert label
            label_arr = np.asarray(label, dtype=np.float32)
            if label_arr.shape[0] != num_classes:
                raise ValueError(f"Label length {label_arr.shape[0]} does not match num_classes={num_classes}")

            # normalize label to distribution
            if label_arr.sum() > 1.1:  # looks like percentages
                label_arr = label_arr / 100.0
            label_arr = label_arr / (label_arr.sum() + 1e-8)

            # lag-difference transform
            x_np = df.values.astype(np.float32)
            T, F = x_np.shape
            if T <= self.lag:
                raise ValueError(f"Sequence length {T} too short for lag={self.lag}")
            x_diff = x_np[self.lag:, :] - x_np[:-self.lag, :]  # (T-lag, F)

            # pad/truncate to max_len
            T2 = x_diff.shape[0]
            if T2 > max_len:
                x_diff = x_diff[:max_len, :]
            else:
                pad_T = max_len - T2
                x_diff = np.pad(x_diff, ((0, pad_T), (0, 0)), mode="constant")

            # add channel dimension (1, T2, F)
            x_diff = np.expand_dims(x_diff, axis=0)

            self.X.append(torch.from_numpy(x_diff))
            self.y.append(torch.from_numpy(label_arr))

        self.X = torch.stack(self.X)  # (N, 1, max_len, F)
        self.y = torch.stack(self.y)  # (N, num_classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# class SmellDataset(Dataset):
#     def __init__(self, pairs, max_len: int, num_classes: int = 12, lag: int = 150, drop_first_col: bool = True):
#         self.X, self.y = [], []
#         self.max_len, self.num_classes, self.lag = max_len, num_classes, lag

#         feature_count = None
#         for df, label in pairs:
#             if drop_first_col:
#                 df = df.iloc[:, 1:]   # drop first column if needed

#             if feature_count is None:
#                 feature_count = df.shape[1]
#             elif df.shape[1] != feature_count:
#                 raise ValueError("All samples must have same feature count.")

#             # labels -> probs
#             label_arr = np.asarray(label, dtype=np.float32)
#             if label_arr.sum() > 1.1:
#                 label_arr = label_arr / 100.0
#             label_arr = label_arr / (label_arr.sum() + 1e-8)

#             # lag-diff transform: x[t+lag] - x[t]
#             x = df.values.astype(np.float32)
#             if x.shape[0] <= self.lag:
#                 raise ValueError(f"Sequence length {x.shape[0]} too short for lag={self.lag}")
#             x = x[self.lag:, :] - x[:-self.lag, :]  # (T-lag, F)

#             # pad/trim to max_len (time dim)
#             T = x.shape[0]
#             if T > max_len:
#                 x = x[:max_len, :]
#             else:
#                 x = np.pad(x, ((0, max_len - T), (0, 0)), mode="constant")

#             self.X.append(torch.from_numpy(x))                # (max_len, F)
#             self.y.append(torch.from_numpy(label_arr))        # (C,)

#         self.X = torch.stack(self.X)  # (N, max_len, F)
#         self.y = torch.stack(self.y)  # (N, C)

#     def __len__(self): return len(self.X)
#     def __getitem__(self, i): return self.X[i], self.y[i]


# -----------------------------
# Metrics & Evaluation
# -----------------------------

@dataclass
class EvalResult:
    loss: float
    quant_error: float
    dynamic_topk_acc: float
    samples: int


@torch.no_grad()
def evaluate_model(test_loader: DataLoader, model: nn.Module, logger: logging.Logger) -> EvalResult:
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_samples = 0
    total_quant_error = 0.0
    topk_total_hits = 0
    topk_total_possible = 0
    bad_sample = 0

    for batch_x, batch_label in test_loader:
        batch_x = batch_x.to(device, dtype=torch.float32)
        batch_label = batch_label.to(device, dtype=torch.float32)

        logits = model(batch_x)
        probs = F.softmax(logits, dim=1)
        
        # quantization style error (optional)
        probs_rounded = torch.round(probs * 10) / 10.0
        quant_error = torch.sum(torch.abs(probs_rounded - batch_label))
        total_quant_error += quant_error.item()

        loss = soft_cross_entropy(probs, batch_label)
        total_loss += loss.item()

        B, C = probs.shape
        for b in range(B):
            true_present = (batch_label[b] > 0).nonzero(as_tuple=True)[0]
            P = true_present.numel()
            if P == 0:
                bad_sample += 1
                continue
            k = min(P, C)
            top_p_preds = torch.topk(probs[b], k=k, dim=0).indices
            hits = torch.isin(true_present, top_p_preds).sum().item()
            topk_total_hits += hits
            topk_total_possible += P
            total_samples += 1

    avg_loss = total_loss / max(total_samples, 1)
    avg_quant_error = total_quant_error / max(total_samples, 1)
    dynamic_topk_acc = 100.0 * topk_total_hits / max(topk_total_possible, 1)

    logger.info(f"Test: Loss = {avg_loss:.4f}, Quant Error = {avg_quant_error:.4f}")
    logger.info(f"Dynamic Top-K Accuracy (per sample): {dynamic_topk_acc:.2f}%")

    return EvalResult(
        loss=avg_loss,
        quant_error=avg_quant_error,
        dynamic_topk_acc=dynamic_topk_acc,
        samples=total_samples,
    )


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=1)
        loss = nn.KLDivLoss(reduction="batchmean")(log_probs, y)  # y sums to 1
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int = 10,
        lr: float = 1e-3,
        logger: logging.Logger | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, opt, device)
        dur = time.time() - t0
        if logger:
            logger.info(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | {dur:.1f}s")
        else:
            print(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} | {dur:.1f}s")

        if val_loader is not None:
            _ = evaluate_model(val_loader, model, logger or logging.getLogger(__name__))

    return model


# -----------------------------
# Example data builder (replace with your own)
# -----------------------------

def build_pairs_from_your_data(n_samples: int = 64, T_min: int = 40, T_max: int = 100, F: int = 20, C: int = 12):
    """
    Generates synthetic pairs for demo purposes. Replace this with a loader
    that returns real pairs: List[(np.ndarray[T, F], np.ndarray[C])].
    """
    pairs = []
    for _ in range(n_samples):
        T = np.random.randint(T_min, T_max + 1)
        x = np.random.randn(T, F).astype(np.float32)
        y = np.random.rand(C).astype(np.float32)
        y = 100.0 * y / (y.sum() + 1e-8)  # pretend percentages
        pairs.append((x, y))
    return pairs


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction for validation split.")
    parser.add_argument("--max-len", type=int, default=96, help="Pad/trim T to this length.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *( [logging.FileHandler(args.log_file)] if args.log_file else [] )
        ],
    )
    logger = logging.getLogger("SmellNet")

    # Build or load your data
    directory_path = "/home/dewei/workspace/SmellNet/chi_paper_data/training_new"
    pairs = load_smell_recognition_data(directory_path)

    # Dataset & split
    dataset = SmellDataset(pairs, max_len=args.max_len)
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if n_val > 0 else None

    # Model
    model = SmellTemporalCNN()

    # Train
    fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, logger=logger)

    # Save weights
    out_path = "smellcnn_weights.pth"
    torch.save(model.state_dict(), out_path)
    logger.info(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()
