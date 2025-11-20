from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: Optional[float] = 1.0
    log_interval: int = 10
    save_dir: Path = Path("checkpoints")
    mixed_precision: bool = False


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        self.best_val_acc = 0.0
        self.config.save_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> Dict[str, float]:
        if self.train_loader is None:
            raise ValueError("Training loader is not provided.")

        history: Dict[str, float] = {}
        for epoch in range(1, self.config.num_epochs + 1):
            train_metrics = self._run_epoch(epoch)
            history.update({f"train_{k}": v for k, v in train_metrics.items()})

            if self.val_loader is not None:
                val_metrics = self.validate()
                history.update({f"val_{k}": v for k, v in val_metrics.items()})

                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self._save_checkpoint(epoch, best=True)

            self._save_checkpoint(epoch, best=False)

        return history

    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        eval_loader = loader or self.val_loader
        if eval_loader is None:
            raise ValueError("Validation loader is not provided.")

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                inputs, masks, targets = self._move_batch_to_device(batch)
                logits = self.model(inputs, masks, None, None)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {"loss": avg_loss, "accuracy": accuracy}

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
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
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            if step % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {step}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        return {"loss": avg_loss, "accuracy": accuracy}

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
        suffix = "best" if best else f"epoch_{epoch}"
        path = self.config.save_dir / f"classifier_{suffix}.pt"
        torch.save(checkpoint, path)


__all__ = ["ClassificationTrainer", "TrainerConfig"]

