"""
Contrastive pre-training trainer for any SmellNet backbone.

Works with every model in models/ (TimesNet, Transformer, TSCMamba, TSLANet)
as long as it exposes:

    model.encode(x_enc, x_mark_enc) -> torch.Tensor  [B, emb_dim]

The embedding dimension is discovered automatically via a dummy forward pass so
no model-specific config is needed here.

Training loop (one batch):
    x_aug1, x_aug2  = augment(x), augment(x)          # two independent views
    z1 = L2_norm(proj_head(model.encode(x_aug1, mask)))
    z2 = L2_norm(proj_head(model.encode(x_aug2, mask)))
    features = stack([z1, z2], dim=1)                  # [B, 2, proj_dim]
    loss = SupConLoss(features, labels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentations import get_augmentation
from .losses import SupConLoss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    proj_dim: int = 128          # projection head output dimension
    proj_hidden: int = 256       # projection head hidden dimension
    aug_list: List[str] = field(
        default_factory=lambda: ["jitter", "scale", "time_shift", "magnitude_warp"]
    )
    grad_clip: float = 1.0
    save_dir: Path = Path("pretrain_ckpts")
    log_interval: int = 10
    device: str = "auto"   # auto | cpu | cuda | cuda:0 | cuda:1 | ...
    use_wandb: bool = False
    wandb_project: str = "smell-net-pretrain"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP: emb_dim → proj_hidden (BN + ReLU) → proj_dim.

    The projection head is used only during pre-training and discarded
    before fine-tuning.  Only the backbone weights are saved.
    """

    def __init__(self, emb_dim: int, proj_hidden: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContrastivePretrainer:
    """
    Model-agnostic supervised contrastive pre-trainer.

    Args:
        model:        Any backbone with `.encode(x_enc, x_mark_enc) → [B, D]`.
        train_loader: DataLoader that yields (inputs, masks, labels) batches.
        config:       ContrastiveConfig instance.
        val_loader:   Optional validation loader for tracking a proxy loss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: ContrastiveConfig,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        d = config.device.lower()
        if d == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(d)

        self.model.to(self.device)

        # Discover embedding dimension via a dummy forward pass
        emb_dim = self._infer_emb_dim()

        self.proj_head = ProjectionHead(
            emb_dim=emb_dim,
            proj_hidden=config.proj_hidden,
            proj_dim=config.proj_dim,
        ).to(self.device)

        self.criterion = SupConLoss(temperature=config.temperature)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.proj_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.augment: Callable = get_augmentation(config.aug_list)

        config.save_dir.mkdir(parents=True, exist_ok=True)

        # WandB
        if config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=config.wandb_run_name,
                    tags=config.wandb_tags,
                    config={
                        "pretrain_epochs": config.num_epochs,
                        "learning_rate": config.learning_rate,
                        "temperature": config.temperature,
                        "proj_dim": config.proj_dim,
                        "aug_list": config.aug_list,
                    },
                )
            except ImportError:
                print("Warning: wandb not installed. Disabling WandB logging.")
                config.use_wandb = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, float]:
        """Run the full pre-training loop. Returns final-epoch metrics."""
        history: Dict[str, float] = {}
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._run_epoch(epoch, training=True)
            history["train_loss"] = train_loss

            if self.val_loader is not None:
                val_loss = self._run_epoch(epoch, training=False)
                history["val_loss"] = val_loss
                if self.config.use_wandb:
                    self._wandb_log({"val/loss": val_loss, "epoch": epoch})

            if self.config.use_wandb:
                self._wandb_log({"train/loss": train_loss, "epoch": epoch})

        return history

    def save_encoder(self, path: Optional[Path] = None) -> Path:
        """
        Save only the backbone weights (no projection head).

        The saved checkpoint is compatible with `_run.py --checkpoint` for
        fine-tuning: it has the same keys as a standard classification checkpoint
        minus `optimizer_state` / `best_val_acc`.
        """
        out = path or (self.config.save_dir / "pretrained_encoder.pt")
        torch.save({"model_state": self.model.state_dict()}, out)
        print(f"Saved pre-trained encoder to {out}")
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_emb_dim(self) -> int:
        """Run one dummy batch through encode() to find the embedding dimension."""
        # Pull one batch from the loader to get seq_len and num_features
        sample_batch = next(iter(self.train_loader))
        x, mask, _ = sample_batch
        x = x[:1].to(self.device)
        mask = mask[:1].to(self.device)
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode(x, mask)
        self.model.train()
        return emb.shape[-1]

    def _run_epoch(self, epoch: int, training: bool) -> float:
        loader = self.train_loader if training else self.val_loader
        self.model.train(training)
        self.proj_head.train(training)

        total_loss = 0.0
        total_samples = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for step, batch in enumerate(loader, start=1):
                inputs, masks, labels = self._to_device(batch)

                loss = self._step(inputs, masks, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.config.grad_clip:
                        nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.proj_head.parameters()),
                            self.config.grad_clip,
                        )
                    self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

                if training and step % self.config.log_interval == 0:
                    print(
                        f"Epoch {epoch} | Step {step}/{len(loader)} | "
                        f"ContrastiveLoss: {loss.item():.4f}"
                    )

        return total_loss / max(total_samples, 1)

    def _step(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        One contrastive step: two augmented views → encode → project → SupConLoss.
        """
        # Two independently augmented views of the same batch
        x1 = self.augment(inputs)
        x2 = self.augment(inputs)

        # Encode with shared backbone
        e1 = self.model.encode(x1, masks)   # [B, emb_dim]
        e2 = self.model.encode(x2, masks)   # [B, emb_dim]

        # Project and L2-normalise
        z1 = F.normalize(self.proj_head(e1), dim=-1)   # [B, proj_dim]
        z2 = F.normalize(self.proj_head(e2), dim=-1)   # [B, proj_dim]

        # Stack into [B, 2, proj_dim] for SupConLoss
        features = torch.stack([z1, z2], dim=1)

        return self.criterion(features, labels)

    def _to_device(self, batch):
        inputs, masks, labels = batch
        return (
            inputs.to(self.device),
            masks.to(self.device),
            labels.to(self.device),
        )

    def _wandb_log(self, d: dict) -> None:
        try:
            import wandb
            wandb.log(d)
        except ImportError:
            pass


__all__ = ["ContrastiveConfig", "ContrastivePretrainer", "ProjectionHead"]
