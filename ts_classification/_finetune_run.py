import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

import torch
import yaml

from dataloader import create_dataloaders
from models.TimesNet import Model as TimesNet
from models.Transformer import Model as Transformer
from models.TSLANet import Model as TSLANet
# TSCMamba is imported lazily below (mamba_ssm requires a compiled CUDA extension)
from finetune import ClassificationTrainer, TrainerConfig


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to configs/finetune_config.yaml relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "configs" / "finetune_config.yaml"
    else:
        config_path = Path(config_path).expanduser().resolve()
    
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def parse_args(config: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified entry point for training or evaluating TimesNet classifier."
    )
    # Get defaults from config if provided
    config_defaults = config or {}
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default=config_defaults.get("mode", "train"),
        help="Choose 'train' to run fit(), 'eval' to run validate().",
    )
    parser.add_argument(
        "--model",
        choices=["timesnet", "transformer", "tscmamba", "tslanet"],
        default=config_defaults.get("model", "timesnet"),
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(config_defaults.get("data_root", "smell_ts_dataset/SmellNet")),
        help="Root dir containing class folders (e.g., banana/, apple/).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config_defaults.get("classes"),
        help="Class folder names to include (e.g., banana apple).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=config_defaults.get("features"),
        help="Optional list of feature columns to use (e.g., NO2 VOC CO).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=config_defaults.get("seq_len", 512),
        help="Sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config_defaults.get("batch_size", 16),
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config_defaults.get("epochs", 20),
        help="Num epochs (train).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config_defaults.get("learning_rate", 1e-3),
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config_defaults.get("weight_decay", 1e-5),
        help="Weight decay.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=config_defaults.get("train_split", 0.7),
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=config_defaults.get("val_split", 0.15),
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=config_defaults.get("test_split", 0.15),
        help="Test split ratio.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=config_defaults.get("num_workers", 0),
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--normalization",
        choices=["zscore", "minmax", "none"],
        default=config_defaults.get("normalization", "zscore"),
        help="Feature normalization strategy.",
    )
    parser.add_argument(
        "--window-stride",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=config_defaults.get("window_stride", None),
        help="Sliding window stride for training split (None or omit = single window per CSV).",
    )
    parser.add_argument(
        "--temporal-diff",
        type=lambda x: False if x.lower() in ("false", "none", "0") else int(x),
        default=config_defaults.get("temporal_diff", False),
        help=(
            "Temporal differencing lag p (∆xt = xt - xt-p). "
            "Pass a positive integer to enable (e.g. --temporal-diff 1), "
            "or 0/false/none to disable."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config_defaults.get("dropout", 0.1),
        help="TimesNet dropout.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=config_defaults.get("d_model", 64),
        help="TimesNet d_model.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        default=config_defaults.get("d_ff", 128),
        help="TimesNet d_ff.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=config_defaults.get("layers", 2),
        help="Number of TimesBlocks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config_defaults.get("top_k", 5),
        help="Top-k periods.",
    )
    parser.add_argument(
        "--num-kernels",
        type=int,
        default=config_defaults.get("num_kernels", 6),
        help="Inception kernels.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=config_defaults.get("grad_clip", 1.0),
        help="Gradient clip norm.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=config_defaults.get("log_interval", 10),
        help="Logging cadence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config_defaults.get("seed", 42),
        help="Random seed for splits.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(config_defaults.get("save_dir", "artifacts")),
        help="Directory for checkpoints/metadata.",
    )
    checkpoint_default = config_defaults.get("load_pretrained_checkpoint")
    parser.add_argument(
        "--load-pretrained-checkpoint",
        type=Path,
        default=Path(checkpoint_default) if checkpoint_default else None,
        help="Path to a pretrained encoder checkpoint to load (required for eval).",
    )
    metadata_default = config_defaults.get("load_pretrained_metadata")
    parser.add_argument(
        "--load-pretrained-metadata",
        type=Path,
        default=Path(metadata_default) if metadata_default else None,
        help="Metadata JSON saved alongside a pretrained encoder.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable torch.cuda.amp mixed precision.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=config_defaults.get("val_frequency", 1),
        help="Validate every N epochs.",
    )
    parser.add_argument(
        "--eval-at-end",
        action="store_true",
        help="Run a final evaluation after the last training epoch.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default=config_defaults.get("eval_split", "test"),
        help="Which split to evaluate on: 'val' or 'test' (used in eval mode and eval-at-end).",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=config_defaults.get("save_frequency", 1),
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Only save best checkpoint.",
    )
    parser.add_argument(
        "--keep-best-k",
        type=int,
        default=config_defaults.get("keep_best_k", 3),
        help="Number of best checkpoints to keep.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable WandB logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=config_defaults.get("wandb_project", "smell-net"),
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=config_defaults.get("wandb_entity"),
        help="WandB entity (team) name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=config_defaults.get("wandb_run_name"),
        help="Custom WandB run name.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=config_defaults.get("wandb_tags", []),
        help="Tags for WandB run.",
    )

    # ---- Device ----
    parser.add_argument(
        "--device",
        type=str,
        default=config_defaults.get("device", "auto"),
        help=(
            "Device to run on. "
            "'auto' uses CUDA if available, else CPU. "
            "Examples: cpu, cuda, cuda:0, cuda:1."
        ),
    )

    # ---- TSCMamba / TSLANet shared ----
    parser.add_argument(
        "--patch-size",
        type=int,
        default=config_defaults.get("patch_size", 8),
        help="Patch size for TSCMamba (stride=patch_size) and TSLANet (stride=patch_size//2).",
    )

    # ---- TSCMamba-specific ----
    parser.add_argument(
        "--projected-space",
        type=int,
        default=config_defaults.get("projected_space", 64),
        help="TSCMamba: projection/patch output dimension.",
    )
    parser.add_argument(
        "--mamba-d-state",
        type=int,
        default=config_defaults.get("mamba_d_state", 16),
        help="TSCMamba: Mamba SSM state dimension.",
    )
    parser.add_argument(
        "--mamba-dconv",
        type=int,
        default=config_defaults.get("mamba_dconv", 4),
        help="TSCMamba: Mamba local convolution width.",
    )
    parser.add_argument(
        "--mamba-expand",
        type=int,
        default=config_defaults.get("mamba_expand", 2),
        help="TSCMamba: Mamba expansion factor.",
    )
    parser.add_argument(
        "--num-mambas",
        type=int,
        default=config_defaults.get("num_mambas", 2),
        help="TSCMamba: number of stacked Mamba blocks.",
    )

    # ---- TSLANet-specific ----
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=config_defaults.get("emb_dim", 128),
        help="TSLANet: patch embedding dimension.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=config_defaults.get("depth", 2),
        help="TSLANet: number of TSLANet layers.",
    )

    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Convert a device string to a torch.device.

    'auto' → cuda if available, else cpu.
    Any other value is passed directly to torch.device().
    """
    if device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_timesnet_config(
    num_features: int, num_classes: int, args: argparse.Namespace
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=args.seq_len,
        label_len=args.seq_len,
        pred_len=0,
        enc_in=num_features,
        c_out=num_features,
        d_model=args.d_model,
        d_ff=args.d_ff,
        top_k=args.top_k,
        num_kernels=args.num_kernels,
        e_layers=args.layers,
        dropout=args.dropout,
        embed="fixed",
        freq="h",
        num_class=num_classes,
    )


def build_transformer_config(
    num_features: int, num_classes: int, args: argparse.Namespace
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=args.seq_len,
        label_len=args.seq_len,
        pred_len=0,
        enc_in=num_features,
        dec_in=num_features,
        c_out=num_features,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=8,
        e_layers=args.layers,
        d_layers=1,
        dropout=args.dropout,
        embed="fixed",
        freq="h",
        activation="gelu",
        num_class=num_classes,
        factor=1,
        output_attention=False,
    )


def build_tscmamba_config(
    num_features: int, num_classes: int, args: argparse.Namespace
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=args.seq_len,
        enc_in=num_features,
        num_class=num_classes,
        projected_space=args.projected_space,
        patch_size=args.patch_size,
        d_state=args.mamba_d_state,
        dconv=args.mamba_dconv,
        e_fact=args.mamba_expand,
        num_mambas=args.num_mambas,
        initial_focus=1.0,
        dropout=args.dropout,
        max_pooling=0,
        additive_fusion=1,
        only_forward_scan=1,
        flip_dir=1,
        reverse_flip=0,
    )


def build_tslanet_config(
    num_features: int, num_classes: int, args: argparse.Namespace
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="classification",
        seq_len=args.seq_len,
        enc_in=num_features,
        num_class=num_classes,
        emb_dim=args.emb_dim,
        depth=args.depth,
        patch_size=args.patch_size,
        dropout=args.dropout,
        use_icb=True,
        use_asb=True,
        adaptive_filter=True,
    )


def save_metadata(
    save_dir: Path,
    label_map: dict,
    features: List[str],
    args: argparse.Namespace,
    resolved_classes: List[str],
) -> Path:
    metadata = {
        "label_to_id": label_map,
        "features": features,
        "seq_len": args.seq_len,
        "classes": resolved_classes,  # Save resolved classes, not original spec
        "normalization": args.normalization,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "model": args.model,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def load_metadata(metadata_path: Optional[Path]) -> Optional[dict]:
    if not metadata_path:
        return None
    metadata_path = metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # First, do a preliminary parse to get --config if specified
    import sys
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config", type=Path, default=None)
    preliminary_args, _ = preliminary_parser.parse_known_args()
    
    # Load config file
    config = load_config(preliminary_args.config)
    
    # Now parse full args with config as defaults (CLI args will override config)
    args = parse_args(config)
    
    # Handle boolean flags: if config has them as True and CLI didn't set them, use config values
    if not args.mixed_precision and config.get("mixed_precision", False):
        if "--mixed-precision" not in sys.argv:
            args.mixed_precision = True

    if not args.save_best_only and config.get("save_best_only", False):
        if "--save-best-only" not in sys.argv:
            args.save_best_only = True

    if not args.use_wandb and config.get("use_wandb", False):
        if "--use-wandb" not in sys.argv:
            args.use_wandb = True

    if not args.eval_at_end and config.get("eval_at_end", True):
        if "--eval-at-end" not in sys.argv:
            args.eval_at_end = True

    # Validate required arguments
    if not args.classes:
        raise ValueError("--classes must be provided either in config or via CLI.")
    if args.mode == "eval" and not args.load_pretrained_checkpoint:
        raise ValueError("--load-pretrained-checkpoint is required when mode is 'eval'.")

    metadata = load_metadata(args.load_pretrained_metadata)
    if metadata:
        print(
            f"[metadata] Restoring from {args.load_pretrained_metadata}: "
            f"seq_len={metadata.get('seq_len')}, "
            f"normalization={metadata.get('normalization')}, "
            f"features={metadata.get('features')}, "
            f"temporal_diff={metadata.get('temporal_diff', 'not set')}"
        )
        # Keep user overrides if explicitly provided.
        args.features = args.features or metadata.get("features")
        args.seq_len = metadata.get("seq_len", args.seq_len)
        if "classes" in metadata:
            args.classes = metadata["classes"]
        args.normalization = metadata.get("normalization", args.normalization)
        args.train_split = metadata.get("train_split", args.train_split)
        args.val_split = metadata.get("val_split", args.val_split)
        args.test_split = metadata.get("test_split", args.test_split)
        args.seed = metadata.get("seed", args.seed)
        if "temporal_diff" in metadata and "--temporal-diff" not in sys.argv:
            args.temporal_diff = metadata["temporal_diff"]

    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()

    # Resolve features: "all" → None (auto-detect all numeric columns in SmellDataset)
    if isinstance(args.features, list) and len(args.features) == 1:
        if isinstance(args.features[0], str) and args.features[0].lower() == "all":
            args.features = None

    # Prepare classes for resolution (handle YAML parsing quirks)
    classes_input = args.classes
    if isinstance(args.classes, list) and len(args.classes) == 1:
        arg = args.classes[0]
        # Check if it's "all" or a number
        if isinstance(arg, str) and arg.lower() == "all":
            classes_input = "all"
        elif isinstance(arg, str) and arg.isdigit():
            classes_input = int(arg)

    train_loader = None
    val_loader = None
    test_loader = None
    label_map = {}
    features: List[str] = []
    resolved_classes: List[str] = []

    if args.mode == "train":
        train_loader, val_loader, test_loader, label_map, features, resolved_classes = create_dataloaders(
            data_root=data_root,
            classes=classes_input,
            feature_columns=args.features,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=args.num_workers,
            seed=args.seed,
            normalization=args.normalization,
            train_window_stride=args.window_stride,
            temporal_diff=bool(args.temporal_diff),
            diff_lag=int(args.temporal_diff) if args.temporal_diff else 1,
        )
    else:  # eval
        _, val_loader, test_loader, label_map, features, resolved_classes = create_dataloaders(
            data_root=data_root,
            classes=classes_input,
            feature_columns=args.features,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_split=0.0,  # No training set for eval mode
            val_split=args.val_split,
            test_split=args.test_split,
            num_workers=args.num_workers,
            seed=args.seed,
            normalization=args.normalization,
            train_window_stride=None,  # No sliding windows in eval mode
            temporal_diff=bool(args.temporal_diff),
            diff_lag=int(args.temporal_diff) if args.temporal_diff else 1,
        )

    print(f"Using {len(resolved_classes)} classes: {resolved_classes[:5]}{'...' if len(resolved_classes) > 5 else ''}")

    device = resolve_device(args.device)
    print(f"Device: {device}")

    model_name = args.model.lower()
    if model_name == "timesnet":
        model_config = build_timesnet_config(len(features), len(label_map), args)
        model = TimesNet(model_config)
    elif model_name == "transformer":
        model_config = build_transformer_config(len(features), len(label_map), args)
        model = Transformer(model_config)
    elif model_name == "tscmamba":
        from models.TSCMamba import Model as TSCMamba  # lazy: requires mamba_ssm CUDA ext
        model_config = build_tscmamba_config(len(features), len(label_map), args)
        model = TSCMamba(model_config)
    elif model_name == "tslanet":
        model_config = build_tslanet_config(len(features), len(label_map), args)
        model = TSLANet(model_config)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Using model: {model_name}")

    if args.load_pretrained_checkpoint:
        checkpoint = torch.load(args.load_pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded pretrained checkpoint from {args.load_pretrained_checkpoint}")

    wandb_run_config = {
        # Data
        "model": args.model,
        "num_classes": len(label_map),
        "classes": resolved_classes,
        "num_features": len(features),
        "features": features,
        "seq_len": args.seq_len,
        "normalization": args.normalization,
        "window_stride": args.window_stride,
        "temporal_diff": args.temporal_diff,  # False or int lag
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        # Training
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "mixed_precision": args.mixed_precision,
        "device": str(device),
        # Model architecture (shared)
        "dropout": args.dropout,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "layers": args.layers,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
        "patch_size": args.patch_size,
        # TSCMamba
        "projected_space": args.projected_space,
        "mamba_d_state": args.mamba_d_state,
        "mamba_dconv": args.mamba_dconv,
        "mamba_expand": args.mamba_expand,
        "num_mambas": args.num_mambas,
        # TSLANet
        "emb_dim": args.emb_dim,
        "depth": args.depth,
    }

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_dir=args.save_dir.expanduser().resolve(),
        mixed_precision=args.mixed_precision,
        val_frequency=args.val_frequency,
        eval_at_end=args.eval_at_end,
        save_frequency=args.save_frequency,
        save_best_only=args.save_best_only,
        keep_best_k=args.keep_best_k,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_config=wandb_run_config,
    )

    # Create inverse label map for class names
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=str(device),
        test_loader=test_loader,
        class_names=class_names,
    )

    if args.mode == "train":
        history = trainer.fit()
        print("Training complete:", history)
        metadata_path = save_metadata(trainer_config.save_dir, label_map, features, args, resolved_classes)
        print(f"Saved metadata to {metadata_path}")
    else:
        # Choose which split to evaluate on
        if args.eval_split == "test":
            eval_loader = test_loader
            split_name = "test"
        else:
            eval_loader = val_loader
            split_name = "val"

        if eval_loader is None:
            raise ValueError(
                f"--eval-split {args.eval_split!r} was requested but the {split_name} loader is "
                f"empty. Check your --{split_name}-split ratio."
            )

        metrics = trainer.validate(loader=eval_loader, include_per_class=True)
        print(
            f"Evaluation complete ({split_name} split). "
            f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%"
        )


if __name__ == "__main__":
    main()

