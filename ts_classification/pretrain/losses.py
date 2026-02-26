"""
Supervised Contrastive Loss (Khosla et al., 2020).

Reference: https://arxiv.org/abs/2004.11362

Given a batch of N samples with class labels, two augmented views are created
per sample (n_views = 2).  The loss maximises agreement between embeddings of
the same class across both views while repelling embeddings of different classes.

  L = -Σᵢ (1/|P(i)|) Σ_{p ∈ P(i)} log [
          exp(zᵢ · zₚ / τ)
         ─────────────────────────────────
          Σ_{a ∈ A(i)} exp(zᵢ · zₐ / τ)
      ]

where:
  A(i) = all indices in the 2N-sample set excluding i itself
  P(i) = {a ∈ A(i) : y_a = yᵢ}   (same-class positives)
  τ    = temperature (default 0.07)
  zᵢ   = L2-normalised projection of sample i
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature: Logit scale τ.  Smaller values produce sharper distributions.
        base_temperature: Reference temperature used for loss normalisation
                          (kept at 0.07 following the original paper).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalised projections of shape [B, n_views, proj_dim].
                      Typically n_views = 2 (two augmented views per sample).
            labels:   Class indices of shape [B].

        Returns:
            Scalar loss.
        """
        device = features.device
        B, n_views, _ = features.shape

        # Flatten to [B * n_views, proj_dim] and repeat labels accordingly
        features_flat = features.view(B * n_views, -1)         # [2B, D]
        labels_rep = labels.repeat(n_views)                    # [2B]

        # Similarity matrix: [2B, 2B]
        sim = torch.matmul(features_flat, features_flat.T) / self.temperature

        # Mask out self-similarities (diagonal)
        N = B * n_views
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim.masked_fill_(self_mask, float("-inf"))

        # Log-sum-exp denominator (over all non-self entries per row)
        log_denom = torch.logsumexp(sim, dim=1)                # [2B]

        # Positive mask: same class, different index
        labels_eq = labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1)  # [2B, 2B]
        pos_mask = labels_eq & ~self_mask                      # exclude self

        # Number of positives per anchor (avoid division by zero)
        num_pos = pos_mask.sum(dim=1).clamp(min=1).float()    # [2B]

        # Per-anchor loss: mean over positives of (sim - log_denom)
        pos_sim = (sim * pos_mask.float()).sum(dim=1)          # [2B]
        loss_per_anchor = -(self.temperature / self.base_temperature) * (
            pos_sim / num_pos - log_denom
        )

        # Average only over anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)
        if has_pos.sum() == 0:
            return features_flat.sum() * 0.0   # differentiable zero

        return loss_per_anchor[has_pos].mean()


__all__ = ["SupConLoss"]
