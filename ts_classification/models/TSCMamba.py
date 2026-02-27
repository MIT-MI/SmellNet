"""
TSC-Mamba: Mamba SSM-based time series classifier for SmellNet.

Adapted from misc/tsc_mambda/tsc_mamba.py for the ts_classification pipeline.

Key changes from the original:
- Replaces the CWT spectrogram + 2D ConversionLayer with a 1D patch embedding
  that operates directly on raw sensor time series.
- Removes ROCKET, RevIN, pywt, skimage dependencies.
- Conforms to the standard forward signature:
    forward(x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None)
  where x_enc is [B, T, C] (batch, seq_len, num_features).
- All architecture config is passed via a configs object (SimpleNamespace).

Required configs fields:
    task_name        : str   = "classification"
    seq_len          : int               (time steps)
    enc_in           : int               (number of sensor channels)
    num_class        : int               (number of output classes)
    projected_space  : int   = 64        (projection / patch output dimension)
    patch_size       : int   = 8         (1D conv patch size & stride)
    d_state          : int   = 16        (Mamba SSM state dimension)
    dconv            : int   = 4         (Mamba local convolution width)
    e_fact           : int   = 2         (Mamba expansion factor)
    num_mambas       : int   = 2         (number of stacked Mamba blocks)
    initial_focus    : float = 1.0       (learnable fusion weight initialisation)
    dropout          : float = 0.1
    max_pooling      : int   = 0         (0=mean pooling, 1=max pooling)
    additive_fusion  : int   = 1         (1=additive, 0=multiplicative fusion)
    only_forward_scan: int   = 1         (1=unidirectional, 0=bidirectional)
    flip_dir         : int   = 1         (dimension to flip for backward scan)
    reverse_flip     : int   = 0         (0=sum, 1=flip-back before sum)
"""

import math

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class PatchEmbedding1D(nn.Module):
    """
    1D patch embedding that replaces the original 2D CWT-based ConversionLayer.

    Uses a depthwise Conv1d to extract local patches from each sensor channel
    independently, then projects to the desired output dimension.

    Input : [B, C, T]
    Output: [B, C, projected_space]
    """

    def __init__(self, enc_in: int, seq_len: int, patch_size: int, projected_space: int):
        super().__init__()
        # Depthwise convolution: each channel processed independently
        num_patches = math.floor((seq_len - patch_size) / patch_size) + 1
        self.conv = nn.Conv1d(
            enc_in, enc_in,
            kernel_size=patch_size, stride=patch_size,
            groups=enc_in, padding=0,
        )
        self.proj = nn.Linear(num_patches, projected_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, C, num_patches] -> [B, C, projected_space]
        x = self.conv(x)
        x = self.proj(x)
        return x


class Model(nn.Module):
    """
    TSC-Mamba classifier for smell time series data.

    Architecture:
        Input [B, T, C]
            ↓  transpose → [B, C, T]
        1D Patch Embedding  → [B, C, projected_space]   (local features)
        Linear Projection   → [B, C, projected_space]   (global features)
            ↓  learnable fusion + concat → [B, C, projected_space*3]
        LayerNorm + GELU
            ↓
        Mamba1 (temporal scan over C, d_model=projected_space*3)
        Mamba2 (feature scan  over projected_space*3, d_model=C)
            ↓  residual sum → [B, C, projected_space*3]
        Mean/Max pool over C → [B, projected_space*3]
        MLP classifier       → [B, num_class]
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if configs.task_name == "classification":
            self.patcher = PatchEmbedding1D(
                enc_in=configs.enc_in,
                seq_len=configs.seq_len,
                patch_size=configs.patch_size,
                projected_space=configs.projected_space,
            )
            self.projector = nn.Linear(configs.seq_len, configs.projected_space)

            self.ln = nn.LayerNorm(configs.projected_space * 3)
            self.dropout = nn.Dropout(configs.dropout)
            self.learnable_focus = nn.Parameter(torch.tensor([configs.initial_focus]))
            self.gelu = nn.GELU()

            self.mamba1 = nn.ModuleList([
                Mamba(
                    d_model=configs.projected_space * 3,
                    d_state=configs.d_state,
                    d_conv=configs.dconv,
                    expand=configs.e_fact,
                ) for _ in range(configs.num_mambas)
            ])
            self.mamba2 = nn.ModuleList([
                Mamba(
                    d_model=configs.enc_in,
                    d_state=configs.d_state,
                    d_conv=configs.dconv,
                    expand=configs.e_fact,
                ) for _ in range(configs.num_mambas)
            ])

            self.flatten = nn.Flatten(start_dim=1)
            self.classifier = nn.Sequential(
                nn.Linear(configs.projected_space * 3, (configs.projected_space * 3) // 2),
                nn.Dropout(configs.dropout),
                nn.Linear((configs.projected_space * 3) // 2, configs.num_class),
            )

    def _backbone(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Shared backbone → [B, projected_space*3] pooled representation."""
        # x_enc: [B, T, C] → [B, C, T]
        x = x_enc.transpose(1, 2)

        x_patched = self.patcher(x)       # [B, C, projected_space]
        x_projected = self.projector(x)   # [B, C, projected_space]

        if self.configs.additive_fusion == 1:
            x_fused = (self.learnable_focus * x_projected) + (2.0 - self.learnable_focus) * x_patched
        else:
            x_fused = (self.learnable_focus * x_projected) * (2.0 - self.learnable_focus) * x_patched

        x_fused = self.gelu(x_fused)
        concatenated_x = torch.cat([x_patched, x_fused, x_projected], dim=2)  # [B, C, projected_space*3]
        concatenated_x = self.ln(concatenated_x)

        if self.configs.num_mambas != 0:
            # --- Temporal scan (over sensor dimension C) ---
            x1 = concatenated_x.clone()  # [B, C, projected_space*3]
            for i in range(self.configs.num_mambas):
                x1 = self.mamba1[i](x1) + x1.clone()

            if self.configs.only_forward_scan == 0:
                concatenated_x = torch.flip(concatenated_x, dims=[self.configs.flip_dir])
                x1_flipped = concatenated_x.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
                if self.configs.reverse_flip == 0:
                    x1 = x1 + x1_flipped
                else:
                    x1 = x1 + torch.flip(x1_flipped, dims=[self.configs.flip_dir])
                concatenated_x = torch.flip(concatenated_x, dims=[self.configs.flip_dir])

            # --- Feature scan (over projected dimension) ---
            # [B, C, projected_space*3] → [B, projected_space*3, C]
            concatenated_x = concatenated_x.permute(0, 2, 1)
            x2 = concatenated_x.clone()
            for i in range(self.configs.num_mambas):
                x2 = self.mamba2[i](x2) + x2.clone()
            x2 = x2.permute(0, 2, 1)  # [B, C, projected_space*3]

            if self.configs.only_forward_scan == 0:
                x2_flipped = torch.flip(concatenated_x.clone(), dims=[self.configs.flip_dir])
                for i in range(self.configs.num_mambas):
                    x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
                x2_flipped = x2_flipped.permute(0, 2, 1)  # [B, C, projected_space*3]
                if self.configs.reverse_flip == 0:
                    x2 = x2 + x2_flipped
                else:
                    x2 = x2 + torch.flip(x2_flipped, dims=[self.configs.flip_dir])

            x3 = x2 + x1  # [B, C, projected_space*3]
        else:
            x3 = concatenated_x

        if self.configs.max_pooling == 0:
            x3 = x3.mean(1)   # [B, projected_space*3]
        else:
            x3, _ = x3.max(1)

        return self.flatten(x3)  # [B, projected_space*3]

    def encode(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Backbone embedding for contrastive pre-training → [B, projected_space*3]."""
        return self._backbone(x_enc, x_mark_enc)

    def classification(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._backbone(x_enc, x_mark_enc))  # [B, num_class]

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ) -> torch.Tensor:
        if self.configs.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None
