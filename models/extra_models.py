
#!/usr/bin/env python3
# extra_models.py
"""
Two drop-in model architectures for your odor-mixture project:
- SmellBiLSTM: a compact bidirectional LSTM
- SmellTransformer: a lightweight Transformer encoder

Both follow the same interface as your existing TCN:
    __init__(in_ch, num_classes, ...)
    forward(x) -> (logits, presence_logits)

They accept inputs shaped either as (B, 1, T, F) or (B, T, F).
"""

from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_BTF(x: torch.Tensor) -> torch.Tensor:
    """Convert input to shape (B, T, F). Accepts (B,1,T,F) or (B,T,F)."""
    if x.dim() == 4:   # (B,1,T,F) -> (B,T,F)
        x = x.squeeze(1)
    assert x.dim() == 3, f"Expected (B,1,T,F) or (B,T,F), got {tuple(x.shape)}"
    return x


class SmellBiLSTM(nn.Module):
    """
    Bidirectional LSTM baseline.
    - Encodes sequence with BiLSTM
    - Mean-pools over time
    - Two heads: mixture proportion (logits) and presence (logits)
    """
    def __init__(self, in_ch: int, num_classes: int, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden * 2, 128)
        self.head = nn.Linear(128, num_classes)
        self.presence_head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        x = _to_BTF(x)                # (B,T,F)
        out, _ = self.lstm(x)         # (B,T,2H)
        h = out.mean(dim=1)           # (B,2H) mean pooling over time
        h = F.relu(self.proj(h), inplace=True)  # (B,128)
        logits = self.head(h)               # mixture proportion logits
        presence_logits = self.presence_head(h)  # presence logits
        return logits, presence_logits


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard Transformer-style sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)     # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,D) -> add PE of length T"""
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)    # (1,T,D) broadcast


class SmellTransformer(nn.Module):
    """
    Lightweight Transformer encoder baseline.
    - Linear projection from features -> d_model
    - Sinusoidal positional encoding
    - N-layer TransformerEncoder with batch_first=True
    - Mean-pooling over time
    - Two heads: mixture proportion (logits) and presence (logits)
    """
    def __init__(self, in_ch: int, num_classes: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_ch, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = _SinusoidalPositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.presence_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor):
        x = _to_BTF(x)                        # (B,T,F)
        h = self.input_proj(x)                # (B,T,D)
        h = self.posenc(h)                    # add positional encodings
        h = self.encoder(h)                   # (B,T,D)
        h = self.norm(h.mean(dim=1))          # mean-pool over time -> (B,D)
        logits = self.head(h)                 # mixture proportion logits
        presence_logits = self.presence_head(h)
        return logits, presence_logits


# Optional: simple factory if you want to import and call get_model_class(name)
def get_model_class(name: str):
    name = (name or '').lower()
    if name in ('lstm', 'bilstm'):
        return SmellBiLSTM
    if name in ('transformer', 'tfm'):
        return SmellTransformer
    raise ValueError(f"Unknown architecture '{name}'. Use one of: 'lstm', 'transformer'.")
