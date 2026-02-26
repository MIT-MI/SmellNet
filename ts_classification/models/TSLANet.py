"""
TSLANet: Time Series Local-Adaptive Network for SmellNet.

Adapted from misc/tsla/tsla_model.py for the ts_classification pipeline.

Key changes from the original:
- Removes PyTorch Lightning dependency; all classes inherit from nn.Module.
- Replaces global `args` namespace with a `configs` object passed to __init__.
- Conforms to the standard forward signature:
    forward(x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None)
  where x_enc is [B, T, C] (batch, seq_len, num_features).
- Removes the Lightning training/pretraining wrapper classes; only the core
  TSLANet backbone and classification head are exposed.

Required configs fields:
    task_name       : str   = "classification"
    seq_len         : int               (time steps)
    enc_in          : int               (number of sensor channels)
    num_class       : int               (number of output classes)
    emb_dim         : int   = 128       (patch embedding dimension)
    depth           : int   = 2         (number of TSLANet layers)
    patch_size      : int   = 8         (patch size; stride = patch_size // 2)
    dropout         : float = 0.15
    use_icb         : bool  = True      (enable Inverse Channel Block)
    use_asb         : bool  = True      (enable Adaptive Spectral Block)
    adaptive_filter : bool  = True      (enable adaptive high-freq masking in ASB)
"""

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_


class ICB(nn.Module):
    """
    Inverse Channel Block: two parallel Conv1d streams with cross-multiplication.

    Input / Output: [B, N, C] where N = num_patches, C = emb_dim.
    """

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)   # [B, C, N]

        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out = self.conv3(x1 * x2_2 + x2 * x1_2)
        return out.transpose(1, 2)   # [B, N, C]


class PatchEmbed(nn.Module):
    """
    Overlapping 1D patch embedding via Conv1d.

    Input : [B, C_in, T]   (channels-first)
    Output: [B, num_patches, emb_dim]
    """

    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, emb_dim, num_patches] -> [B, num_patches, emb_dim]
        return self.proj(x).transpose(1, 2)


class Adaptive_Spectral_Block(nn.Module):
    """
    FFT-based spectral mixing with learnable complex weights and optional
    adaptive high-frequency masking.

    Input / Output: [B, N, C].
    """

    def __init__(self, dim: int, adaptive_filter: bool = True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        trunc_normal_(self.complex_weight_high, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def _adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        median_energy = energy.view(B, -1).median(dim=1, keepdim=True)[0].view(B, 1)
        normalized = energy / (median_energy + 1e-6)
        mask = ((normalized > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return mask.unsqueeze(-1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            freq_mask = self._adaptive_high_freq_mask(x_fft)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted = x_weighted + (x_fft * freq_mask.to(x.device)) * weight_high

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm="ortho")
        return x.to(dtype).view(B, N, C)


class TSLANet_layer(nn.Module):
    """
    Single TSLANet block: optional ASB (spectral) + optional ICB (local channel mixing).
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_asb: bool = True,
        use_icb: bool = True,
        adaptive_filter: bool = True,
    ):
        super().__init__()
        self.use_asb = use_asb
        self.use_icb = use_icb
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=adaptive_filter)
        self.icb = ICB(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_asb and self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class Model(nn.Module):
    """
    TSLANet classifier for smell time series data.

    Architecture:
        Input [B, T, C]
            ↓  transpose → [B, C, T]
        PatchEmbed (overlapping Conv1d)  → [B, num_patches, emb_dim]
        + Positional Embedding
            ↓
        Stack of TSLANet_layer (ASB + ICB)
            ↓
        Mean pool over patches           → [B, emb_dim]
        Linear head                      → [B, num_class]
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        use_asb = getattr(configs, "use_asb", True)
        use_icb = getattr(configs, "use_icb", True)
        adaptive_filter = getattr(configs, "adaptive_filter", True)
        dropout = getattr(configs, "dropout", 0.15)
        emb_dim = getattr(configs, "emb_dim", 128)
        depth = getattr(configs, "depth", 2)
        patch_size = getattr(configs, "patch_size", 8)

        self.patch_embed = PatchEmbed(
            seq_len=configs.seq_len,
            patch_size=patch_size,
            in_chans=configs.enc_in,
            embed_dim=emb_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(
                dim=emb_dim,
                drop=dropout,
                drop_path=dpr[i],
                use_asb=use_asb,
                use_icb=use_icb,
                adaptive_filter=adaptive_filter,
            )
            for i in range(depth)
        ])

        self.head = nn.Linear(emb_dim, configs.num_class)

        # Weight initialisation
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _backbone(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Shared backbone → [B, emb_dim] mean-pooled patch representation."""
        # x_enc: [B, T, C] → [B, C, T] for Conv1d patch embedding
        x = x_enc.transpose(1, 2)
        x = self.patch_embed(x)    # [B, num_patches, emb_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.tsla_blocks:
            x = blk(x)
        return x.mean(1)           # [B, emb_dim]

    def encode(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        """Backbone embedding for contrastive pre-training → [B, emb_dim]."""
        return self._backbone(x_enc, x_mark_enc)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ) -> torch.Tensor:
        return self.head(self._backbone(x_enc, x_mark_enc))  # [B, num_class]
