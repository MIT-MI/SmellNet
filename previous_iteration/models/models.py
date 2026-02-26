import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable, List, Sequence


# ---------- Positional encoding ----------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ---------- Transformer ----------
class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_cls_token: bool = False,
        pool: str = "mean",  # 'mean' or 'cls'
    ):
        super().__init__()
        assert pool in ("mean", "cls"), "pool must be 'mean' or 'cls'"

        # Project to model_dim then norm (pre-norm encoder will handle the rest)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        self.pos = SinusoidalPositionalEncoding(model_dim) if use_positional_encoding else nn.Identity()
        self.use_cls_token = use_cls_token
        self.pool = pool

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def _key_padding_mask(self, lengths: torch.Tensor, T: int) -> torch.Tensor:
        # True = PAD position
        device = lengths.device
        rng = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        return rng >= lengths.unsqueeze(1)                  # (B, T)

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, F), lengths: (B,) actual lengths (optional)
        returns: (B, D) pooled token embedding
        """
        x = self.input_proj(x)     # (B, T, D)
        x = self.pos(x)            # (B, T, D)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._key_padding_mask(lengths, x.size(1))  # (B, T)

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)                 # (B, 1, D)
            x = torch.cat([cls, x], dim=1)                                 # (B, 1+T, D)
            if key_padding_mask is not None:
                pad0 = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([pad0, key_padding_mask], dim=1)  # (B, 1+T)

        h = self.transformer(x, src_key_padding_mask=key_padding_mask)     # (B, L, D)

        if self.use_cls_token and self.pool == "cls":
            feat = h[:, 0]  # (B, D)
        else:
            tokens = h[:, 1:] if self.use_cls_token else h                 # (B, T, D)
            if key_padding_mask is None:
                feat = tokens.mean(dim=1)
            else:
                mask = (~key_padding_mask).float()
                if self.use_cls_token:
                    mask = mask[:, 1:]                                     # drop CLS column
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)      # (B, 1)
                feat = (tokens * mask.unsqueeze(-1)).sum(dim=1) / denom

        return feat  # (B, D)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.forward_features(x, lengths)     # (B, D)
        feat = self.dropout(feat)
        return self.classifier(feat)                 # (B, num_classes)


# ---------- LSTM ----------
class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pool: str = "mean",  # 'last' | 'mean' | 'max'  (default keeps your current behavior)
    ):
        super().__init__()
        assert pool in ("last", "mean", "max")
        self.pool = pool
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.embedding_layer = nn.Linear(feat_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _masked_mean(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mask = (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B,T)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, _) = self.lstm(packed)
            if self.pool == "last":
                if self.bidirectional:
                    feat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)
                else:
                    feat = h[-1]                               # (B, H)
            else:
                out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,D)
                if self.pool == "mean":
                    feat = self._masked_mean(out, lengths)
                else:  # max
                    B, T, D = out.shape
                    mask = (torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1))
                    out = out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                    feat = out.max(dim=1).values
        else:
            out, (h, _) = self.lstm(x)  # (B,T,D_lstm)
            if self.pool == "last":
                feat = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
            elif self.pool == "mean":
                feat = out.mean(dim=1)
            else:
                feat = out.max(dim=1).values

        emb = self.embedding_layer(feat)  # (B, embedding_dim)
        return emb

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        embedding = self.forward_features(x, lengths)
        out = self.classifier(embedding)
        return out  # <- return only logits


def _ensure_3d(x: torch.Tensor, channel_last: bool = True) -> tuple[torch.Tensor, bool]:
    """
    Ensure x is (N, C, T) for convs.
    Accepts:
      - (N, C)
      - (N, T, C)
      - (N, C, T)
    Returns:
      x_cf: (N, C, T)
      was_2d: whether input was (N, C)
    """
    if x.dim() == 2:                 # (N, C)
        x = x.unsqueeze(-1)          # (N, C, 1)
        return x, True
    if x.dim() == 3:
        N, A, B = x.shape
        if channel_last:             # assume (N, T, C) -> (N, C, T)
            if A < B:                # heuristic: usually T < C? Not reliable; prefer explicit flag.
                # We rely on flag; but if channel_last=True, reshape accordingly:
                x = x.transpose(1, 2)  # (N, C, T)
            else:
                # If user passed (N, C, T) while channel_last=True, do nothing.
                pass
        else:                         # channel_first=True, expect (N, C, T)
            # Ensure (N, C, T); if (N, T, C), swap:
            # A crude check: time usually bigger than channels; but stick to flag.
            pass
        if x.shape[1] > x.shape[2] and channel_last:
            # If we mis-detected, swap to make second dim channels and third dim time.
            x = x.transpose(1, 2)
        return x, False
    raise ValueError(f"Expected input with 2 or 3 dims, got {x.shape}")


def _temporal_pool_3d(x: torch.Tensor, mode: str = "mean", channel_last: bool = True) -> torch.Tensor:
    """
    Reduce (N, T, C) or (N, C, T) to (N, C) by pooling over time.
    """
    if x.dim() == 2:
        return x  # already (N, C)
    # reformat to (N, T, C) for easy pooling if channel_last; else use (N, C, T)
    if channel_last:
        if x.shape[1] != x.shape[2]:  # try to detect (N, T, C)
            pass
        # assume (N, T, C)
        if mode == "mean":
            return x.mean(dim=1)      # (N, C)
        elif mode == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")
    else:
        # (N, C, T)
        if mode == "mean":
            return x.mean(dim=2)
        elif mode == "max":
            return x.max(dim=2).values
        else:
            raise ValueError(f"Unsupported pool mode: {mode}")


# -----------------------------
# MLP
# -----------------------------
class MLPClassifier(nn.Module):
    """
    Simple MLP classifier.

    Input:
      - (N, C)   : directly used
      - (N, T, C): pooled over time first (default: mean) to (N, C)
      - (N, C, T): set channel_last=False to pool along T to (N, C)

    Tip: Using pool='mean' makes this robust to variable window sizes.
    If you want to flatten (requires fixed T), set pool='flatten' and pass in_features=C*T.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_sizes: Sequence[int] = (256, 256),
        dropout: float = 0.2,
        pool: str = "mean",        # 'mean' | 'max' | 'flatten'
        channel_last: bool = True, # True if your sequences are (N, T, C)
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.pool = pool
        self.channel_last = channel_last
        self.use_batchnorm = use_batchnorm

        layers: List[nn.Module] = []
        last = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(last, num_classes)

    def forward_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # Normalize input shape:
        if x.dim() == 3 and self.pool in ("mean", "max"):
            # Pool temporal dim to get (N, C)
            if self.channel_last:
                # (N, T, C) -> (N, C)
                x = _temporal_pool_3d(x, mode=self.pool, channel_last=True)
            else:
                # (N, C, T) -> (N, C)
                x = _temporal_pool_3d(x, mode=self.pool, channel_last=False)
        elif x.dim() == 3 and self.pool == "flatten":
            # Flatten time and channels
            if self.channel_last:  # (N, T, C) -> (N, T*C)
                N, T, C = x.shape
                x = x.reshape(N, T * C)
            else:                  # (N, C, T) -> (N, C*T)
                N, C, T = x.shape
                x = x.reshape(N, C * T)
        elif x.dim() == 2:
            # (N, C)
            pass
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)} for MLP")

        return self.backbone(x)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.forward_features(x)  # lengths not used for MLP
        return self.head(feats)


# -----------------------------
# 1D CNN
# -----------------------------
class CNN1DClassifier(nn.Module):
    """
    Lightweight 1D CNN for time series classification.

    Accepts:
      - (N, T, C) with channel_last=True (default)
      - (N, C, T) with channel_last=False
      - (N, C) will be treated as T=1

    Uses global average pooling -> fully-connected head, so it works with
    variable window sizes (e.g., 50/100/500).
    """
    def __init__(
        self,
        in_channels: int,           # number of features (C)
        num_classes: int,
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int | Sequence[int] = 5,
        dropout: float = 0.2,
        channel_last: bool = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.channel_last = channel_last

        if isinstance(kernel_size, int):
            k_sizes = [kernel_size] * len(channels)
        else:
            assert len(kernel_size) == len(channels), "kernel_size list must match channels length"
            k_sizes = list(kernel_size)

        convs: List[nn.Module] = []
        c_in = in_channels
        for c_out, k in zip(channels, k_sizes):
            pad = k // 2  # "same" padding
            convs.append(nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad))
            if use_batchnorm:
                convs.append(nn.BatchNorm1d(c_out))
            convs.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                convs.append(nn.Dropout(dropout))
            c_in = c_out

        self.conv = nn.Sequential(*convs)
        self.gap = nn.AdaptiveAvgPool1d(1)  # (N, C, T) -> (N, C, 1)
        self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # Ensure (N, C, T)
        x_cf, _ = _ensure_3d(x, channel_last=self.channel_last)  # (N, C, T)
        h = self.conv(x_cf)          # (N, C', T)
        h = self.gap(h).squeeze(-1)  # (N, C')
        return h

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)


class GCMSMLPEncoder(nn.Module):
    """
    GCMS encoder for flat vectors.
    Input:  x (B, N)  # N = number of GCMS points per sample
    Output: z (B, embedding_dim)
    """
    def __init__(
        self,
        in_features: int,                 # N
        embedding_dim: int = 256,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,       # per-sample feature normalization at input
        use_batchnorm: bool = False,      # BN between hidden layers (off by default for small batches)
        l2_normalize: bool = False,       # set True if your contrastive loss expects unit vectors
    ):
        super().__init__()
        layers: list[nn.Module] = []

        if use_layernorm:
            layers.append(nn.LayerNorm(in_features))

        last = in_features
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize

    def forward_features(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 2:
            # Allow accidental (B, T, C) by flattening, but prefer giving (B, N)
            x = x.view(x.size(0), -1)
        z = self.net(x)  # (B, D)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward_features(x, lengths)