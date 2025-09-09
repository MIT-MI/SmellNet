#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple
import time, os, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from extra_models import SmellBiLSTM, SmellTransformer


try:
    from load_data import load_smell_recognition_data
except Exception:
    load_smell_recognition_data = None

SmellTemporalCNN = None
try:
    from run_chi_model import SmellTemporalCNN as _SmellTemporalCNN

    SmellTemporalCNN = _SmellTemporalCNN
except Exception:
    pass


class FallbackTemporalCNN(nn.Module):
    def __init__(self, in_ch=4, num_classes=12, base=64, dropout=0.2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(base, base, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(base, base * 2, kernel_size=7, padding=3, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base * 2, base * 2, kernel_size=7, padding=6, dilation=3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(base * 2, 128, kernel_size=1)
        self.head = nn.Linear(128, num_classes)
        self.presence_head = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1).transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        h = self.block1(x)
        h = self.block2(h)
        h = self.proj(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        logits = self.head(h)
        presence_logits = self.presence_head(h)
        return logits, presence_logits


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
    if lag <= 0:
        return x
    T, F = x.shape
    if T <= lag:
        raise ValueError(f"Sequence length {T} too short for lag={lag}")
    return x[lag:, :] - x[:-lag, :]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    bytes_total = 0
    for p in model.parameters():
        bytes_total += p.numel() * p.element_size()
    return bytes_total / (1024**2)


@torch.no_grad()
def bench_inference(model, x, device, iters=200, warmup=20):
    """
    Returns (lat_ms, peak_mem_mb or None) for a single forward pass at the given batch shape.
    """
    model.eval()
    x = x.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        # warmup
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms
        lat_ms = sum(times) / len(times)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        # CPU timing
        for _ in range(warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0 / iters
        peak_mb = None
    return lat_ms, peak_mb


@torch.no_grad()
def sweep_presence_thresh(
    model,
    loader,
    device,
    temp_scaler: TempScaler | None = None,
    has_presence_head: bool = True,
    candidates=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
):
    best_t, best_thr02 = candidates[0], -1.0
    for t in candidates:
        out = evaluate(
            model,
            loader,
            device,
            temp_scaler=temp_scaler,
            has_presence_head=has_presence_head,
            present_thresh=t,
        )
        if out.thr02 > best_thr02:
            best_thr02, best_t = out.thr02, t
    return best_t, best_thr02


class DistDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple],
        max_len: int,
        scaler: StandardScaler | None,
        num_classes: int,
        lag: int = 0,
    ):
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

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


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
    X = np.concatenate(rows, axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def mix_synthetic_batch(x, y, p=0.6, max_components=3):
    if p <= 0.0 or x.size(0) < 2:
        return x, y
    B = x.size(0)
    device = x.device
    squeeze_back = False
    if x.dim() == 3:
        x = x.unsqueeze(1)
        squeeze_back = True
    out_x = x.clone()
    out_y = y.clone()
    for b in range(B):
        if torch.rand(1, device=device).item() < p:
            K = int(torch.randint(2, max_components + 1, (1,), device=device))
            idx = torch.randint(0, B, (K,), device=device)
            w = torch.rand(K, device=device)
            w = w / w.sum()
            mix_x = torch.sum(x[idx] * w.view(-1, 1, 1, 1), dim=0)
            mix_y = torch.sum(y[idx] * w.view(-1, 1), dim=0)
            out_x[b] = mix_x
            out_y[b] = mix_y / mix_y.sum().clamp_min(1e-8)
    if squeeze_back:
        out_x = out_x.squeeze(1)
    return out_x, out_y


@dataclass
class EvalOut:
    kl: float
    mae: float
    thr01: float
    thr02: float
    dyn_topk: float
    presence_f1: float | None = None
    presence_precision: float | None = None
    presence_recall: float | None = None


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
        if P == 0:
            continue
        k = min(P, C)
        top_idx = torch.topk(pred[b], k=k, dim=0).indices
        hits += torch.isin(true_idx, top_idx).sum().item()
        total += P
    return 100.0 * hits / max(total, 1)


class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.t.clamp_min(1e-3)


@torch.no_grad()
def _masked_l1_for_calib(
    model, loader, device, scaler_module=None, has_presence_head=True
):
    model.eval()
    total, count = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if has_presence_head:
            logits, _ = model(x)
        else:
            logits = model(x)
        if scaler_module is not None:
            logits = scaler_module(logits)
        probs = torch.softmax(logits, dim=1)
        present = (y > 0).float()
        abs_err = (probs - y).abs()
        total += (abs_err * present).sum().item()
        count += present.sum().item()
    return total / max(count, 1.0)


def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    w = alpha * (1 - p_t).pow(gamma)
    return (w * ce).mean()


def fit_temperature(
    model, val_loader, device, steps=150, lr=0.01, has_presence_head=True
):
    model.eval()  # keep dropout/BN off
    scaler = TempScaler().to(device)
    scaler.train()
    opt = torch.optim.Adam(scaler.parameters(), lr=lr)

    for _ in range(steps):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # >>> KEY CHANGE: no graph through the model <<<
            with torch.no_grad():
                if has_presence_head:
                    logits, _ = model(x)
                else:
                    logits = model(x)

            # Only the scaler has gradients
            logits = scaler(logits)
            probs = torch.softmax(logits, dim=1)

            present = (y > 0).float()
            abs_err = (probs - y).abs()
            loss = (abs_err * present).sum() / present.sum().clamp_min(1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return scaler


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temp_scaler: TempScaler | None = None,
    has_presence_head: bool = True,
    present_thresh: float = 0.5,
) -> EvalOut:
    model.eval()
    kls, maes = [], []
    all_pred, all_tgt = [], []
    pres_tp = pres_fp = pres_fn = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if has_presence_head:
            logits, presence_logits = model(x)
        else:
            logits = model(x)
            presence_logits = None
        if temp_scaler is not None:
            logits = temp_scaler(logits)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        kl = F.kl_div(log_probs, y, reduction="batchmean")
        mae = (probs - y).abs().mean()
        kls.append(kl.item())
        maes.append(mae.item())
        all_pred.append(probs.cpu())
        all_tgt.append(y.cpu())

        if has_presence_head and presence_logits is not None:
            present_tgt = (y > 0).float()
            present_pred = (torch.sigmoid(presence_logits) > present_thresh).float()
            # present_pred = (torch.sigmoid(presence_logits) > 0.35).float()
            tp = (present_pred * present_tgt).sum().item()
            fp = (present_pred * (1 - present_tgt)).sum().item()
            fn = ((1 - present_pred) * present_tgt).sum().item()
            pres_tp += tp
            pres_fp += fp
            pres_fn += fn

    pred = torch.cat(all_pred, 0)
    tgt = torch.cat(all_tgt, 0)

    metrics = EvalOut(
        kl=float(np.mean(kls)),
        mae=float(np.mean(maes)),
        thr01=thr_acc_nonzero(pred, tgt, 0.1),
        thr02=thr_acc_nonzero(pred, tgt, 0.2),
        dyn_topk=dyn_topk(pred, tgt),
    )

    if has_presence_head:
        precision = pres_tp / max(pres_tp + pres_fp, 1e-8)
        recall = pres_tp / max(pres_tp + pres_fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        metrics.presence_precision = precision
        metrics.presence_recall = recall
        metrics.presence_f1 = f1

    return metrics


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    lr=3e-4,
    weight_decay=3e-4,
    alpha=0.5,
    beta=0.5,
    synth_p=0.6,
    synth_max_k=3,
    use_temp_scaling=True,
):

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_thr02 = -1.0
    best_state = None
    patience, patience_left = 8, 8

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        # progressive schedule
        phase = ep / max(epochs, 1)
        if phase < 1 / 3:
            curr_p, curr_k = max(0.0, min(1.0, synth_p * 0.5)), max(
                2, min(3, synth_max_k if synth_max_k >= 2 else 2)
            )
        elif phase < 2 / 3:
            curr_p, curr_k = max(0.0, min(1.0, max(synth_p, 0.5))), max(3, synth_max_k)
        else:
            curr_p, curr_k = max(0.0, min(1.0, max(synth_p, 0.7))), max(4, synth_max_k)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = mix_synthetic_batch(
                x, y, p=float(curr_p), max_components=int(curr_k)
            )

            logits, presence_logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()

            kl = F.kl_div(log_probs, y, reduction="batchmean")
            eps = 0.2
            present = (y > 0).float()
            abs_err = (probs - y).abs()
            # masked_l1 = (abs_err * present).sum() / present.sum().clamp_min(1.0)
            # bce = F.binary_cross_entropy_with_logits(presence_logits, present)
            bce = focal_bce(presence_logits, present)

            # new (focus on errors > 0.2):
            eps_l1 = (
                (abs_err - eps).clamp_min(0.0) * present
            ).sum() / present.sum().clamp_min(1.0)

            # loss = kl + alpha * masked_l1 + beta * bce
            loss = kl + alpha * eps_l1 + beta * bce

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_out = evaluate(
            model, val_loader, device, temp_scaler=None, has_presence_head=True
        )
        print(
            f"[{ep:03d}] train_loss={np.mean(losses):.4f} | val_kl={val_out.kl:.4f} "
            f"| val_mae={val_out.mae:.4f} | val@0.1={val_out.thr01:.3f} | val@0.2={val_out.thr02:.3f} "
            f"| dynTopK={val_out.dyn_topk:.2f}% | presF1={val_out.presence_f1 if val_out.presence_f1 is not None else float('nan'):.3f}"
        )

        # keep best by val @0.2
        if val_out.thr02 > best_thr02 + 1e-6:
            best_thr02 = val_out.thr02
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = patience

    if best_state is not None:
        model.load_state_dict(best_state)

    temp_scaler = None
    if use_temp_scaling:
        temp_scaler = fit_temperature(model, val_loader, device, has_presence_head=True)

    return model, temp_scaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=600)
    p.add_argument("--lag", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--synth-p", type=float, default=0.6)
    p.add_argument("--synth-max-k", type=int, default=3)
    p.add_argument("--no-temp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="dist_model.pt")
    p.add_argument("--arch", choices=["tcn", "lstm", "transformer"], default="tcn")
    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable StandardScaler (use raw sensor values).",
    )
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_smell_recognition_data is None:
        raise RuntimeError(
            "Could not import load_smell_recognition_data. Please adjust import."
        )

    print("[1/4] Loading data...")
    train_pairs = load_smell_recognition_data(args.train_dir)
    test_pairs = load_smell_recognition_data(args.test_dir) if args.test_dir else None

    print("[2/4] Fitting StandardScaler on training set...")
    scaler = (
        None
        if args.no_standardize
        else fit_global_scaler(train_pairs, lag=args.lag, max_len=args.max_len)
    )

    n = len(train_pairs)
    val_size = max(1, int(n * args.val_split))
    idx = np.arange(n)
    rng = np.random.default_rng(seed=args.seed)
    rng.shuffle(idx)
    val_idx = set(idx[:val_size].tolist())
    train_list = [train_pairs[i] for i in range(n) if i not in val_idx]
    val_list = [train_pairs[i] for i in range(n) if i in val_idx]

    num_classes = len(train_pairs[0][1])
    train_ds = DistDataset(
        train_list, args.max_len, scaler, num_classes=num_classes, lag=args.lag
    )
    val_ds = DistDataset(
        val_list, args.max_len, scaler, num_classes=num_classes, lag=args.lag
    )
    test_ds = (
        DistDataset(
            test_pairs, args.max_len, scaler, num_classes=num_classes, lag=args.lag
        )
        if test_pairs
        else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        if test_ds
        else None
    )

    in_ch = train_ds.X.shape[-1]

    if args.arch == "tcn":
        ModelClass = (
            SmellTemporalCNN if SmellTemporalCNN is not None else FallbackTemporalCNN
        )
    elif args.arch == "lstm":
        ModelClass = SmellBiLSTM
    elif args.arch == "transformer":
        ModelClass = SmellTransformer
    else:
        raise ValueError(f"Unknown arch {args.arch}")

    model = ModelClass(in_ch=in_ch, num_classes=num_classes)

    with torch.no_grad():
        prevalence = (train_ds.Y.float() > 0).float().mean(dim=0)  # P(y_i>0) per class
        prior_logit = torch.log(prevalence / (1 - prevalence).clamp_min(1e-6))
        if hasattr(model, "presence_head") and hasattr(model.presence_head, "bias"):
            model.presence_head.bias.copy_(prior_logit)

    print("[3/4] Training...")
    model, temp_scaler = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        beta=args.beta,
        synth_p=args.synth_p,
        synth_max_k=args.synth_max_k,
        use_temp_scaling=(not args.no_temp),
    )

    print("[BENCH] Measuring inference cost...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Grab one batch from val_loader
    xb, yb = next(iter(val_loader))
    # Measure batch latency for (B, 1, T, F) and single-sample latency
    lat_b, mem_b = bench_inference(model, xb, device, iters=100, warmup=20)
    lat_1, mem_1 = bench_inference(model, xb[:1], device, iters=200, warmup=30)

    print(f"[BENCH] Params: {count_params(model):,} (~{model_size_mb(model):.2f} MB)")
    print(
        f"[BENCH] Batch={xb.shape[0]} latency: {lat_b:.2f} ms"
        + (f" | peak GPU mem: {mem_b:.1f} MB" if mem_b is not None else "")
    )
    print(
        f"[BENCH] Batch=1 latency: {lat_1:.2f} ms"
        + (f" | peak GPU mem: {mem_1:.1f} MB" if mem_1 is not None else "")
    )

    print("[4/4] Evaluating...")
    # Pick presence threshold that maximizes val @0.2
    best_t, best_val_thr02 = sweep_presence_thresh(
        model, val_loader, device, temp_scaler=temp_scaler, has_presence_head=True
    )

    val_out = evaluate(
        model,
        val_loader,
        device,
        temp_scaler=temp_scaler,
        has_presence_head=True,
        present_thresh=best_t,
    )
    print(
        f"[VAL thresh={best_t:.2f}] KL={val_out.kl:.4f} MAE={val_out.mae:.4f} @0.1={val_out.thr01:.3f} @0.2={val_out.thr02:.3f} "
        f"dynTopK={val_out.dyn_topk:.2f}% Pres(F1/Prec/Rec)={val_out.presence_f1 if val_out.presence_f1 is not None else float('nan'):.3f}/"
        f"{val_out.presence_precision if val_out.presence_precision is not None else float('nan'):.3f}/"
        f"{val_out.presence_recall if val_out.presence_recall is not None else float('nan'):.3f}"
    )

    if test_loader is not None:
        test_out = evaluate(
            model,
            test_loader,
            device,
            temp_scaler=temp_scaler,
            has_presence_head=True,
            present_thresh=best_t,
        )
        print(
            f"[TEST thresh={best_t:.2f}] KL={test_out.kl:.4f} MAE={test_out.mae:.4f} @0.1={test_out.thr01:.3f} @0.2={test_out.thr02:.3f} "
            f"dynTopK={test_out.dyn_topk:.2f}% Pres(F1/Prec/Rec)={test_out.presence_f1 if test_out.presence_f1 is not None else float('nan'):.3f}/"
            f"{test_out.presence_precision if test_out.presence_precision is not None else float('nan'):.3f}/"
            f"{test_out.presence_recall if test_out.presence_recall is not None else float('nan'):.3f}"
        )

        torch.save(model.state_dict(), args.save)
        print(f"[OK] Saved weights to {args.save}")


if __name__ == "__main__":
    main()
