"""
Contrastive pre-training entry point for SmellNet time-series classifiers.

Mirrors _finetune_run.py for data / model configuration, but runs SupCon pre-training
instead of supervised classification.  After training the backbone weights are
saved and can be loaded by _finetune_run.py via --checkpoint for fine-tuning.

Typical workflow
----------------
# 1. Pre-train
python pretrain_run.py \\
    --model tslanet \\
    --classes all \\
    --data-root smell_ts_dataset/SmellNet \\
    --pretrain-epochs 50

# 2. Fine-tune with the pre-trained encoder
python _finetune_run.py \\
    --model tslanet \\
    --classes all \\
    --data-root smell_ts_dataset/SmellNet \\
    --checkpoint pretrain_ckpts/pretrained_encoder.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dataloader import create_dataloaders
from models.TimesNet import Model as TimesNet
from models.Transformer import Model as Transformer
from models.TSLANet import Model as TSLANet
# TSCMamba is imported lazily below (mamba_ssm requires a compiled CUDA extension)
from pretrain import ContrastiveConfig, ContrastivePretrainer

# Re-use the model config builders from _finetune_run.py
from _finetune_run import (
    build_timesnet_config,
    build_transformer_config,
    build_tscmamba_config,
    build_tslanet_config,
    load_config,
    set_seed,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(config: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    cfg = config or {}
    parser = argparse.ArgumentParser(
        description="Supervised contrastive pre-training for SmellNet backbones."
    )

    # ---- Config file ----
    parser.add_argument("--config", type=Path, default=None)

    # ---- Device ----
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.get("device", "auto"),
        help=(
            "Device to run on. "
            "'auto' uses CUDA if available, else CPU. "
            "Examples: cpu, cuda, cuda:0, cuda:1."
        ),
    )

    # ---- Data ----
    parser.add_argument("--data-root", type=Path,
                        default=Path(cfg.get("data_root", "smell_ts_dataset/SmellNet")))
    parser.add_argument("--classes", nargs="+", default=cfg.get("classes"))
    parser.add_argument("--features", nargs="+", default=cfg.get("features"))
    parser.add_argument("--seq-len", type=int, default=cfg.get("seq_len", 512))
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 16))
    parser.add_argument("--train-split", type=float, default=cfg.get("train_split", 0.8))
    parser.add_argument("--val-split", type=float, default=cfg.get("val_split", 0.1))
    parser.add_argument("--test-split", type=float, default=cfg.get("test_split", 0.1))
    parser.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 0))
    parser.add_argument("--normalization", choices=["zscore", "minmax", "none"],
                        default=cfg.get("normalization", "zscore"))
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))

    # ---- Model ----
    parser.add_argument("--model",
                        choices=["timesnet", "transformer", "tscmamba", "tslanet"],
                        default=cfg.get("model", "timesnet"))
    parser.add_argument("--dropout", type=float, default=cfg.get("dropout", 0.1))
    # TimesNet / Transformer
    parser.add_argument("--d-model", type=int, default=cfg.get("d_model", 64))
    parser.add_argument("--d-ff", type=int, default=cfg.get("d_ff", 128))
    parser.add_argument("--layers", type=int, default=cfg.get("layers", 2))
    parser.add_argument("--top-k", type=int, default=cfg.get("top_k", 5))
    parser.add_argument("--num-kernels", type=int, default=cfg.get("num_kernels", 6))
    # Shared TSCMamba / TSLANet
    parser.add_argument("--patch-size", type=int, default=cfg.get("patch_size", 8))
    # TSCMamba
    parser.add_argument("--projected-space", type=int, default=cfg.get("projected_space", 64))
    parser.add_argument("--mamba-d-state", type=int, default=cfg.get("mamba_d_state", 16))
    parser.add_argument("--mamba-dconv", type=int, default=cfg.get("mamba_dconv", 4))
    parser.add_argument("--mamba-expand", type=int, default=cfg.get("mamba_expand", 2))
    parser.add_argument("--num-mambas", type=int, default=cfg.get("num_mambas", 2))
    # TSLANet
    parser.add_argument("--emb-dim", type=int, default=cfg.get("emb_dim", 128))
    parser.add_argument("--depth", type=int, default=cfg.get("depth", 2))

    # ---- Contrastive pre-training ----
    parser.add_argument("--pretrain-epochs", type=int,
                        default=cfg.get("pretrain_epochs", 50))
    parser.add_argument("--pretrain-lr", type=float,
                        default=cfg.get("pretrain_lr", 1e-3))
    parser.add_argument("--pretrain-weight-decay", type=float,
                        default=cfg.get("pretrain_weight_decay", 1e-4))
    parser.add_argument("--temperature", type=float,
                        default=cfg.get("temperature", 0.07))
    parser.add_argument("--proj-dim", type=int,
                        default=cfg.get("proj_dim", 128))
    parser.add_argument("--proj-hidden", type=int,
                        default=cfg.get("proj_hidden", 256))
    parser.add_argument("--aug-list", nargs="+",
                        default=cfg.get("aug_list",
                                        ["jitter", "scale", "time_shift", "magnitude_warp"]),
                        help="Augmentations to sample from for creating two views.")
    parser.add_argument("--grad-clip", type=float, default=cfg.get("grad_clip", 1.0))
    parser.add_argument("--pretrain-save-dir", type=Path,
                        default=Path(cfg.get("pretrain_save_dir", "pretrain_ckpts")))
    parser.add_argument("--log-interval", type=int, default=cfg.get("log_interval", 10))

    # ---- WandB ----
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str,
                        default=cfg.get("wandb_project", "smell-net-pretrain"))
    parser.add_argument("--wandb-entity", type=str, default=cfg.get("wandb_entity"))
    parser.add_argument("--wandb-run-name", type=str, default=cfg.get("wandb_run_name"))
    parser.add_argument("--wandb-tags", nargs="+", default=cfg.get("wandb_tags", []))

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Two-pass: first grab --config, then full parse with config defaults.
    # Default config for pre-training is pretrain_config.yaml (not finetune_config.yaml).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    config_path = pre_args.config or (
        Path(__file__).parent / "configs" / "pretrain_config.yaml"
    )
    config = load_config(config_path)
    args = parse_args(config)

    # Handle boolean flag from config file
    if not args.use_wandb and config.get("use_wandb", False):
        if "--use-wandb" not in sys.argv:
            args.use_wandb = True

    if not args.classes:
        raise ValueError("--classes must be provided (e.g. --classes all).")

    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()

    # Resolve "all" / integer shorthand
    classes_input = args.classes
    if isinstance(args.classes, list) and len(args.classes) == 1:
        arg = args.classes[0]
        if isinstance(arg, str) and arg.lower() == "all":
            classes_input = "all"
        elif isinstance(arg, str) and arg.isdigit():
            classes_input = int(arg)

    train_loader, val_loader, _, label_map, features, resolved_classes = create_dataloaders(
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
    )

    print(f"Classes ({len(resolved_classes)}): "
          f"{resolved_classes[:5]}{'...' if len(resolved_classes) > 5 else ''}")
    print(f"Features: {features}")

    # Build backbone (num_class is unused during pretraining but required by model __init__)
    num_classes = len(label_map)
    num_features = len(features)
    model_name = args.model.lower()

    if model_name == "timesnet":
        model = TimesNet(build_timesnet_config(num_features, num_classes, args))
    elif model_name == "transformer":
        model = Transformer(build_transformer_config(num_features, num_classes, args))
    elif model_name == "tscmamba":
        from models.TSCMamba import Model as TSCMamba  # lazy: requires mamba_ssm CUDA ext
        model = TSCMamba(build_tscmamba_config(num_features, num_classes, args))
    elif model_name == "tslanet":
        model = TSLANet(build_tslanet_config(num_features, num_classes, args))
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Backbone: {model_name}  |  Device: {args.device}")

    pretrain_config = ContrastiveConfig(
        num_epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
        temperature=args.temperature,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        aug_list=args.aug_list,
        grad_clip=args.grad_clip,
        save_dir=args.pretrain_save_dir.expanduser().resolve(),
        log_interval=args.log_interval,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    trainer = ContrastivePretrainer(
        model=model,
        train_loader=train_loader,
        config=pretrain_config,
        val_loader=val_loader if val_loader is not None else None,
    )

    print(f"\nStarting contrastive pre-training for {args.pretrain_epochs} epochs …")
    history = trainer.fit()
    print(f"\nPre-training complete. Final train loss: {history.get('train_loss', 'N/A'):.4f}")

    encoder_path = trainer.save_encoder()

    # Save metadata alongside the encoder so _finetune_run.py can reconstruct data settings
    meta = {
        "model": model_name,
        "features": features,
        "seq_len": args.seq_len,
        "classes": resolved_classes,
        "normalization": args.normalization,
        "label_to_id": label_map,
    }
    meta_path = pretrain_config.save_dir / "pretrain_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTo fine-tune, run:")
    print(f"  python _finetune_run.py \\")
    print(f"    --model {model_name} \\")
    print(f"    --data-root {data_root} \\")
    print(f"    --classes {' '.join(resolved_classes[:3])}{'  ...' if len(resolved_classes) > 3 else ''} \\")
    print(f"    --checkpoint {encoder_path}")


if __name__ == "__main__":
    main()
