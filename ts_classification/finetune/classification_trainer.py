from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate import compute_metrics


@dataclass
class TrainerConfig:
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0
    log_interval: int = 10
    save_dir: Path = Path("checkpoints")
    mixed_precision: bool = False

    # Validation frequency control
    val_frequency: int = 1  # Validate every N epochs
    eval_at_end: bool = True  # Run a final evaluation after the last training epoch

    # Checkpoint save frequency control
    save_frequency: int = 1  # Save checkpoint every N epochs

    # Save best checkpoint options
    save_best_only: bool = False  # Only keep best checkpoint
    keep_best_k: int = 3  # Keep top-K best checkpoints

    # WandB logging
    use_wandb: bool = False
    wandb_project: str = "smell-net"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_config: Optional[Dict[str, Any]] = None  # Full run config logged to WandB


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        device: Optional[str] = None,
        test_loader: Optional[DataLoader] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = class_names or []

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        self.best_val_acc = 0.0
        self.global_step = 0
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

        # Track best checkpoints for management
        self.best_checkpoints: List[Path] = []

        # Initialize WandB if enabled
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name,
                    tags=self.config.wandb_tags,
                    config=self.config.wandb_config or {
                        "epochs": self.config.num_epochs,
                        "learning_rate": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay,
                        "grad_clip": self.config.grad_clip,
                        "batch_size": train_loader.batch_size if train_loader else None,
                        "val_frequency": self.config.val_frequency,
                        "save_frequency": self.config.save_frequency,
                    },
                )
            except ImportError:
                print("Warning: wandb is not installed. Disabling WandB logging.")
                self.config.use_wandb = False

    def fit(self) -> Dict[str, float]:
        if self.train_loader is None:
            raise ValueError("Training loader is not provided.")

        history: Dict[str, float] = {}
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(epoch)
            history.update({f"train_{k}": v for k, v in train_metrics.items()})

            # Validate only at specified frequency or on last epoch
            should_validate = (
                self.val_loader is not None
                and (epoch % self.config.val_frequency == 0 or epoch == self.config.num_epochs)
            )

            if should_validate:
                val_metrics = self.validate()
                history.update({f"val_{k}": v for k, v in val_metrics.items()})

                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self._save_checkpoint(epoch, best=True)

            # Run test metrics every epoch
            test_metrics = None
            if self.test_loader is not None:
                test_metrics = self.validate(loader=self.test_loader)
                history.update({f"test_{k}": v for k, v in test_metrics.items()})
                # No val set — use test acc as proxy for best checkpoint tracking
                if not should_validate and test_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = test_metrics["accuracy"]
                    self._save_checkpoint(epoch, best=True)

            # Save checkpoint at specified frequency or on last epoch
            if epoch % self.config.save_frequency == 0 or epoch == self.config.num_epochs:
                self._save_checkpoint(epoch, best=False)

            # Log to WandB if enabled
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb_log = {}
                    wandb_log.update({f"train/{k}": v for k, v in train_metrics.items()})
                    if should_validate:
                        wandb_log.update({f"val/{k}": v for k, v in val_metrics.items()})
                    if test_metrics is not None:
                        wandb_log.update({f"test/{k}": v for k, v in test_metrics.items()})
                    if self.best_val_acc > 0.0:
                        wandb_log["best_acc"] = self.best_val_acc
                    wandb.log(wandb_log, step=epoch)
                except ImportError:
                    pass

        # Run final test evaluation with per-class breakdown if eval_at_end is enabled
        if self.config.eval_at_end and self.test_loader is not None:
            test_metrics = self.test(include_per_class=True)
            history.update({f"test_{k}": v for k, v in test_metrics.items()})

            # Log per-class metrics to WandB (scalar metrics already logged per-epoch)
            if self.config.use_wandb:
                try:
                    import wandb
                    per_class = {k: v for k, v in test_metrics.items() if k.startswith("class_")}
                    if per_class:
                        wandb.log({f"test/{k}": v for k, v in per_class.items()}, step=self.config.num_epochs)
                except ImportError:
                    pass

        # Print best checkpoint summary
        if self.best_checkpoints:
            best_path = self.best_checkpoints[-1]
            print(f"\nBest checkpoint: {best_path.name}  (val_acc: {self.best_val_acc*100:.2f}%)")
        else:
            print(f"\nNo best checkpoint saved (val_acc never improved above 0).")

        return history

    def validate(
        self,
        loader: Optional[DataLoader] = None,
        include_per_class: bool = False
    ) -> Dict[str, float]:
        """
        Validate the model with comprehensive metrics.

        Args:
            loader: DataLoader to use (defaults to self.val_loader)
            include_per_class: If True, include per-class metrics

        Returns:
            Dictionary with loss, accuracy, F1, precision, recall (weighted/macro/micro)
            and optionally per-class metrics
        """
        eval_loader = loader or self.val_loader
        if eval_loader is None:
            raise ValueError("Validation loader is not provided.")

        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                inputs, masks, targets = self._move_batch_to_device(batch)
                logits = self.model(inputs, masks, None, None)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_samples += targets.size(0)

        avg_loss = total_loss / max(1, total_samples)

        # Compute comprehensive metrics
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            include_per_class=include_per_class,
            class_names=self.class_names if include_per_class else None
        )
        metrics["loss"] = avg_loss

        return metrics

    def test(self, include_per_class: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on the test set with comprehensive metrics.

        Args:
            include_per_class: If True, include per-class metrics

        Returns:
            Dictionary with comprehensive metrics
        """
        if self.test_loader is None:
            raise ValueError("Test loader is not provided.")

        print("\n" + "=" * 50)
        print("Testing on test set...")
        print("=" * 50)

        test_metrics = self.validate(loader=self.test_loader, include_per_class=include_per_class)

        # Print summary
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted): {test_metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted): {test_metrics['recall_weighted']:.4f}")

        return test_metrics

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        total_samples = 0

        for step, batch in enumerate(self.train_loader, start=1):
            inputs, masks, targets = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                logits = self.model(inputs, masks, None, None)
                loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * targets.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            total_samples += targets.size(0)
            self.global_step += 1

            if step % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(1, total_samples)

        # Compute comprehensive metrics for training
        metrics = compute_metrics(
            np.array(all_predictions),
            np.array(all_targets),
            include_per_class=False  # Don't include per-class for training (too verbose)
        )
        metrics["loss"] = avg_loss

        return metrics

    def _move_batch_to_device(self, batch):
        inputs, masks, targets = batch
        inputs = inputs.to(self.device)
        masks = masks.to(self.device)
        targets = targets.to(self.device)
        return inputs, masks, targets

    def _save_checkpoint(self, epoch: int, best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        suffix = f"epoch_{epoch}_best" if best else f"epoch_{epoch}"
        path = self.config.save_dir / f"classifier_{suffix}.pt"
        torch.save(checkpoint, path)

        # Manage best checkpoint storage
        if best:
            if self.config.save_best_only:
                # Remove all previous best checkpoints
                for old_path in self.best_checkpoints:
                    if old_path.exists():
                        old_path.unlink()
                self.best_checkpoints = [path]
            else:
                # Keep top-K best checkpoints
                self.best_checkpoints.append(path)
                if len(self.best_checkpoints) > self.config.keep_best_k:
                    old_path = self.best_checkpoints.pop(0)
                    if old_path.exists():
                        old_path.unlink()


__all__ = ["ClassificationTrainer", "TrainerConfig"]

