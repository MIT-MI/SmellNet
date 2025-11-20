import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

import torch
import yaml

from dataloader import create_dataloaders
from models.TimesNet import Model as TimesNet
from trainer import ClassificationTrainer, TrainerConfig


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to configs/config.yaml relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "configs" / "config.yaml"
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
        "--val-split",
        type=float,
        default=config_defaults.get("val_split", 0.2),
        help="Validation split ratio.",
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
    checkpoint_default = config_defaults.get("checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(checkpoint_default) if checkpoint_default else None,
        help="Checkpoint path to load (required for eval).",
    )
    metadata_default = config_defaults.get("metadata")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(metadata_default) if metadata_default else None,
        help="Metadata JSON from a previous training run.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable torch.cuda.amp mixed precision.",
    )
    return parser.parse_args()


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


def save_metadata(
    save_dir: Path,
    label_map: dict,
    features: List[str],
    args: argparse.Namespace,
) -> Path:
    metadata = {
        "label_to_id": label_map,
        "features": features,
        "seq_len": args.seq_len,
        "classes": args.classes,
        "normalization": args.normalization,
        "val_split": args.val_split,
        "seed": args.seed,
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
    
    # Handle mixed_precision: if config has it as True and CLI didn't set it, use config value
    if not args.mixed_precision and config.get("mixed_precision", False):
        # Check if --mixed-precision was explicitly provided in CLI
        if "--mixed-precision" not in sys.argv:
            args.mixed_precision = True
    
    # Validate required arguments
    if not args.classes:
        raise ValueError("--classes must be provided either in config or via CLI.")
    if args.mode == "eval" and not args.checkpoint:
        raise ValueError("--checkpoint is required when mode is 'eval'.")

    metadata = load_metadata(args.metadata)
    if metadata:
        # Keep user overrides if explicitly provided.
        args.features = args.features or metadata.get("features")
        args.seq_len = metadata.get("seq_len", args.seq_len)
        if "classes" in metadata:
            args.classes = metadata["classes"]
        args.normalization = metadata.get("normalization", args.normalization)
        args.val_split = metadata.get("val_split", args.val_split)
        args.seed = metadata.get("seed", args.seed)

    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()

    train_loader = None
    val_loader = None
    label_map = {}
    features: List[str] = []

    if args.mode == "train":
        train_loader, val_loader, label_map, features = create_dataloaders(
            data_root=data_root,
            classes=args.classes,
            feature_columns=args.features,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed,
            normalization=args.normalization,
        )
    else:  # eval
        _, val_loader, label_map, features = create_dataloaders(
            data_root=data_root,
            classes=args.classes,
            feature_columns=args.features,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            seed=args.seed,
            normalization=args.normalization,
        )

    model_config = build_timesnet_config(len(features), len(label_map), args)
    model = TimesNet(model_config)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_dir=args.save_dir.expanduser().resolve(),
        mixed_precision=args.mixed_precision,
    )

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
    )

    if args.mode == "train":
        history = trainer.fit()
        print("Training complete:", history)
        metadata_path = save_metadata(trainer_config.save_dir, label_map, features, args)
        print(f"Saved metadata to {metadata_path}")
    else:
        metrics = trainer.validate(loader=val_loader)
        print(
            f"Evaluation complete. "
            f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']*100:.2f}%"
        )


if __name__ == "__main__":
    main()

