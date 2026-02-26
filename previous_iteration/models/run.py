# run.py
from __future__ import annotations
import argparse, os, random, math
from dataclasses import dataclass
from itertools import product
import csv, json, hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


from models import Transformer, GCMSMLPEncoder, LSTMNet, MLPClassifier, CNN1DClassifier

# Data helpers
from load_data import (
    load_sensor_data,
    load_gcms_data,
    make_sliding_window_dataset,
    diff_data_like,
    create_pair_data,
    highpass_fft_batch,
)

from train import train, contrastive_train

from evaluate import evaluate, evaluate_contrastive

from dataset import PairedDataset, UniqueGCMSampler

from utils import ingredient_to_category

# ----------------------- CLI -----------------------
MODEL_CHOICES = ["mlp", "cnn", "lstm", "transformer"]

@dataclass(frozen=True)
class RunSpec:
    model: str                 # one of MODEL_CHOICES
    contrastive: bool          # True = contrastive learning
    gradient: int              # 0, 25, 50
    window_size: int           # 50, 100, 500
    epochs: int
    batch_size: int
    lr: float
    seed: int
    device: str | None         # 'cuda', 'cpu', or None
    fft: bool
    fft_cutoff: float
    sampling_rate: float

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run model/contrastive/gradient/window sweeps.")

    # Sweep axes
    p.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=["mlp"])
    p.add_argument("--contrastive", nargs="+", choices=["off", "on"], default=["off"])
    p.add_argument("--gradients", nargs="+", type=int, choices=[0, 25, 50], default=[0])
    p.add_argument("--window-sizes", nargs="+", type=int, choices=[50, 100, 500], default=[100])

    # Training knobs
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None])

    # Paths
    p.add_argument("--train-dir", type=str, required=True, help="Training CSV folders (sensor).")
    p.add_argument("--test-dir", type=str, required=True, help="Testing CSV folders (sensor).")
    p.add_argument("--real-test-dir", type=str, required=True, help="real-time test folder.")
    p.add_argument("--gcms-csv", type=str, default=True, help="GCMS CSV (needed for contrastive).")
    p.add_argument("--no-standardize", action="store_true", help="Disable StandardScaler on (N,T,C) windows (train-only fit).")

    # Misc
    p.add_argument("--run-name-prefix", type=str, default="")
    p.add_argument("--log-dir", type=str, default="./runs")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--stride", type=int, default=None, help="Sliding stride (default: window//2)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    p.add_argument("--fft", choices=["off", "on"], default="off",
               help="Apply FFT high-pass cleaning to windows.")
    p.add_argument("--fft-cutoff", type=float, default=0.05,
                help="High-pass cutoff in Hz (used if --fft on).")
    p.add_argument("--sampling-rate", type=float, default=1.0,
                help="Sampling rate (Hz) of your sensor.")

    return p

def iter_run_specs(args: argparse.Namespace) -> Iterable[RunSpec]:
    for g in args.gradients:
        if g < 0: raise ValueError(f"gradient must be >= 0, got {g}")
    for w in args.window_sizes:
        if w <= 0: raise ValueError(f"window_size must be > 0, got {w}")

    for model, cstr, grad, win in product(args.models, args.contrastive, args.gradients, args.window_sizes):
        yield RunSpec(
            model=model,
            contrastive=(cstr == "on"),
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
        )



# ----------------------- Model factory -----------------------
def get_model(
    name: str,
    *,
    num_features: int,    # C
    num_classes: int,     # K (ignored for contrastive encoders)
    window_size: Optional[int] = None,
    channel_last: bool = True,
    **hparams,
) -> nn.Module:
    n = name.lower()
    if n == "mlp":
        if MLPClassifier is None:
            raise RuntimeError("MLPClassifier not found in models.py. Please add it or choose another model.")
        mlp_pool    = hparams.get("mlp_pool", "mean")  # 'mean'|'max'|'flatten'
        mlp_hidden  = hparams.get("mlp_hidden", (256, 256))
        mlp_dropout = hparams.get("mlp_dropout", 0.2)
        if mlp_pool == "flatten":
            assert window_size is not None, "MLP(flatten) requires window_size"
            in_features = num_features * window_size
        else:
            in_features = num_features
        return MLPClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=mlp_hidden,
            dropout=mlp_dropout,
            pool=mlp_pool,
            channel_last=channel_last,
        )

    elif n == "cnn":
        if CNN1DClassifier is None:
            raise RuntimeError("CNN1DClassifier not found in models.py. Please add it or choose another model.")
        return CNN1DClassifier(
            in_channels=num_features,
            num_classes=num_classes,
            channels=hparams.get("cnn_channels", (64, 128, 256)),
            kernel_size=hparams.get("cnn_kernel", 5),
            dropout=hparams.get("cnn_dropout", 0.2),
            channel_last=channel_last,
        )

    elif n == "lstm":
        return LSTMNet(
            input_dim=num_features,
            hidden_dim=hparams.get("lstm_hidden", 256),
            embedding_dim=hparams.get("lstm_embedding", 256),
            num_classes=num_classes,
            num_layers=hparams.get("lstm_layers", 1) if "lstm_layers" in hparams else 1,
            # bidirectional/pool args are supported in your newer LSTM; older one will ignore them.
        )

    elif n in ("transformer", "ts-transformer"):
        return Transformer(
            input_dim=num_features,
            model_dim=hparams.get("tf_dim", 256),
            num_classes=num_classes,
            num_heads=hparams.get("tf_heads", 8),
            num_layers=hparams.get("tf_layers", 3),
            dropout=hparams.get("tf_dropout", 0.1),
        )

    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_CHOICES)}")

def get_gcms_encoder(in_features: int, embedding_dim: int = 256) -> nn.Module:
    if GCMSMLPEncoder is not None:
        return GCMSMLPEncoder(in_features=in_features, embedding_dim=embedding_dim, l2_normalize=False)
    # Minimal fallback encoder
    return nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, embedding_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(embedding_dim, embedding_dim),
    )

# ----------------------- Utils -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(force: Optional[str]) -> torch.device:
    if force == "cpu": return torch.device("cpu")
    if force == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_dtype(s: str) -> torch.dtype:
    return torch.float64 if s == "float64" else torch.float32

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

class SpecCSV:
    """One CSV per run/spec: <log-dir>/<run_name>/results.csv"""
    def __init__(self, run_dir: Path, spec):
        self.path = run_dir / "results.csv"
        self.header = [
            "timestamp","model","contrastive","gradient","window",
            "epochs","batch","lr","seed","stage","epoch","acc@1","acc@5","loss","extra"
        ]
        self.fixed = [
            spec.model, int(spec.contrastive), spec.gradient, spec.window_size,
            spec.epochs, spec.batch_size, spec.lr, spec.seed
        ]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.header)

    def write(self, *, stage: str, epoch: int | None = None,
              acc1: float | None = None, acc5: float | None = None,
              loss: float | None = None, extra: str = ""):
        row = [
            datetime.now().isoformat(timespec="seconds"),
            *self.fixed,
            stage,
            (epoch if epoch is not None else ""),
            ("" if acc1 is None else float(acc1)),
            ("" if acc5 is None else float(acc5)),
            ("" if loss is None else float(loss)),
            extra,
        ]
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

def features_from_model(model: nn.Module, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return model.forward_features(x, lengths=lengths)
    out = model(x)  # try tuple (logits, embedding)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        return out[1]
    raise RuntimeError("This model does not expose features. Add forward_features or return (logits, embedding).")

def _spec_to_dict(spec) -> dict:
    return {k: getattr(spec, k) for k in spec.__dataclass_fields__.keys()}

def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict):       return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    return x

def append_results_jsonl(run_dir: Path, spec, *, results: dict | None = None, error: str | None = None, **extras):
    """
    Appends one JSON record to <run_dir>/results.jsonl.
    Put your metrics in `results` (e.g., {'acc@1':..., 'acc@5':..., 'per_category':...}).
    Any extra keyword args are added at top-level (e.g., dataset sizes, paths).
    """
    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        **_spec_to_dict(spec),
        **extras,
    }
    if results is not None:
        rec["results"] = _jsonable(results)
    if error is not None:
        rec["error"] = error

    out_path = run_dir / "results.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

def _make_run_name(prefix: str, spec) -> str:
    base = f"{spec.model}-c{int(spec.contrastive)}-g{spec.gradient}-w{spec.window_size}-bs{spec.batch_size}-lr{spec.lr}-seed{spec.seed}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha1(base.encode()).hexdigest()[:6]
    return (prefix + "-" if prefix else "") + f"{base}-{ts}-{h}"

def fit_standardizer_from_windows(X: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on *train* windows only.
    X: (N, T, C) float array
    Returns a fitted scaler that standardizes per feature (across all time steps).
    """
    assert X.ndim == 3, "expected (N,T,C) array"
    N, T, C = X.shape
    flat = X.reshape(N * T, C)
    ss = StandardScaler()
    ss.fit(flat)
    return ss

def apply_standardizer(X: np.ndarray, ss: StandardScaler | None) -> np.ndarray:
    """
    Apply a fitted StandardScaler to (N,T,C).
    Returns float32 like the mixture script.
    """
    if ss is None:
        return X
    assert X.ndim == 3, "expected (N,T,C) array"
    N, T, C = X.shape
    flat = ss.transform(X.reshape(N * T, C))
    return flat.reshape(N, T, C).astype(np.float32, copy=False)



# ----------------------- Data prep -----------------------
def build_sliding_data(
    sensor_data_dict: Dict[str, list],    # {label: [pd.DataFrame, ...]}
    le: LabelEncoder,
    window: int,
    stride: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    stride = stride if stride is not None else max(1, window // 2)
    X, y = make_sliding_window_dataset(
        sensor_data_dict, le, window_size=window, stride=stride
    )
    return X, y

# ----------------------- Main orchestration -----------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.log_dir); ensure_dir(args.save_dir)
    dtype = to_dtype(args.dtype)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)

    for spec in iter_run_specs(args):
        print(f"\n==== Run: model={spec.model} | contrastive={spec.contrastive} | grad={spec.gradient} | window={spec.window_size} ====")
        set_seed(spec.seed)

        run_name = _make_run_name(args.run_name_prefix, spec)
        run_dir = log_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # per-spec csv
        spec_csv = SpecCSV(run_dir, spec)

        device = pick_device(spec.device)

        # ---------- Load sensor data dicts ----------
        train_data, test_data, real_data = load_sensor_data(
            training_path=args.train_dir,
            testing_path=args.test_dir,
            real_time_testing_path=args.real_test_dir,
            removed_filtered_columns=["Benzene", "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude"],
        )

        # Optional differencing
        if spec.gradient and spec.gradient > 0:
            train_data = diff_data_like(train_data, periods=spec.gradient)
            test_data  = diff_data_like(test_data,  periods=spec.gradient)

        # ---------- LabelEncoder alignment ----------
        # If we have GCMS CSV (contrastive), use its LabelEncoder to keep indices aligned.
        gcms_scaled, y_encoded, le, scaler = load_gcms_data(args.gcms_csv)

        # ---------- Build sliding-window tensors ----------
        stride = args.stride if args.stride is not None else max(1, spec.window_size // 2)
        Xtr_np, ytr_np = build_sliding_data(train_data, le, spec.window_size, stride)
        Xte_np, yte_np = build_sliding_data(test_data,  le, spec.window_size, stride)

        if spec.fft:
            Xtr_np = highpass_fft_batch(Xtr_np, sampling_rate=spec.sampling_rate, cutoff=spec.fft_cutoff)
            Xte_np = highpass_fft_batch(Xte_np, sampling_rate=spec.sampling_rate, cutoff=spec.fft_cutoff)

        scaler = None if args.no_standardize else fit_standardizer_from_windows(Xtr_np)
        Xtr_np = apply_standardizer(Xtr_np, scaler)
        Xte_np = apply_standardizer(Xte_np, scaler)

        # shapes
        Ntr, T, C = Xtr_np.shape
        K = len(le.classes_)
        print(f"train windows: {Xtr_np.shape}, test windows: {Xte_np.shape}, features per step: {C}, classes: {K}")

        # ---------- DataLoaders ----------
        train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_np), torch.from_numpy(ytr_np)), batch_size=spec.batch_size, shuffle=True, drop_last=False)
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte_np), torch.from_numpy(yte_np)), batch_size=spec.batch_size, shuffle=False, drop_last=False)

        if not spec.contrastive:
            # ====== Classification path ======
            model = get_model(spec.model, num_features=C, num_classes=K, window_size=spec.window_size, channel_last=True)
            train(model, train_loader, epochs=spec.epochs, lr=spec.lr, device=device, dtype=dtype)
            results = evaluate(model, test_loader, device=device, dtype=dtype, ingredient_to_category=ingredient_to_category, class_names=le.classes_)

            spec_csv.write(stage="eval", acc1=results.get("acc@1"), acc5=results.get("acc@5"), extra = json.dumps({
                "per_category": results.get("per_category", {}),
                "fft": getattr(spec, "fft", False),
                "cutoff": getattr(spec, "fft_cutoff", None),
                "gradient": getattr(spec, "gradient", None),
            }, separators=(',', ':'), sort_keys=True))

            append_results_jsonl(
                run_dir, spec, results=results,
                dataset={"train_windows": int(Xtr_np.shape[0]),
                        "test_windows": int(Xte_np.shape[0]),
                        "T": int(Xtr_np.shape[1]), "C": int(Xtr_np.shape[2]),
                        "classes": int(K)},
                checkpoint=str((Path(args.save_dir) / f"{run_name}.pt").resolve())
            )

        else:
            # ====== Contrastive path ======
            # Build sensor encoder using SAME model spec (but we only use features)
            sensor_encoder = get_model(spec.model, num_features=C, num_classes=K, window_size=spec.window_size, channel_last=True)
            # Build GCMS encoder for flat GCMS vectors
            train_pair_data = create_pair_data(Xtr_np, ytr_np, gcms_scaled, le)
            train_dataset = PairedDataset(train_pair_data)
            train_sampler = UniqueGCMSampler(train_dataset.data, batch_size=spec.batch_size)

            train_loader = DataLoader(train_dataset, batch_size=spec.batch_size, sampler=train_sampler)
            test_loader  = DataLoader(TensorDataset(torch.from_numpy(Xte_np), torch.from_numpy(yte_np)), batch_size=spec.batch_size, shuffle=False, drop_last=False)
            
            Dg = gcms_scaled.shape[1]
            gcms_encoder = get_gcms_encoder(in_features=Dg, embedding_dim=256)
            # Train encoders
            contrastive_train(gcms_encoder, sensor_encoder, train_loader, epochs=spec.epochs, lr=spec.lr, device=device, dtype=dtype)

            # Evaluate (window-level)
            results = evaluate_contrastive(
                gcms_encoder, sensor_encoder,
                gcms_data=gcms_scaled,                               # (N_g, Dg)
                sensor_data=torch.from_numpy(Xte_np),           # (N_s, T, C)
                sensor_labels=torch.from_numpy(yte_np),         # (N_s,)
                device=device, dtype=dtype,
                ingredient_to_category=ingredient_to_category,   # <— add
                class_names=le.classes_,   
            )
            # (Optional) save or log 'results'
            spec_csv.write(stage="eval_contrastive", acc1=results.get("acc@1"), acc5=results.get("acc@5"), extra = json.dumps({
                "per_category": results.get("per_category", {}),
                "fft": getattr(spec, "fft", False),
                "cutoff": getattr(spec, "fft_cutoff", None),
                "gradient": getattr(spec, "gradient", None),
            }, separators=(',', ':'), sort_keys=True))

            append_results_jsonl(
                run_dir, spec, results=results,
                dataset={"train_windows": int(Xtr_np.shape[0]),
                        "test_windows": int(Xte_np.shape[0]),
                        "T": int(Xtr_np.shape[1]), "C": int(Xtr_np.shape[2]),
                        "classes": int(K)},
                checkpoint=str((Path(args.save_dir) / f"{run_name}.pt").resolve())
            )

if __name__ == "__main__":
    main()
