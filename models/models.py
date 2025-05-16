import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_classes, num_heads=8, num_layers=4, dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),  # normalize input per time step
            nn.Linear(input_dim, model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)  # (B, T, D)
        x = self.transformer(x)  # (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.pool(x).squeeze(-1)  # (B, D)
        x = self.dropout(x)
        return self.classifier(x)  # (B, num_classes)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=50):
        super().__init__()

        self.output_dim = output_dim
        # Base encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )


    def forward(self, x):
        x = self.encoder(x)
        return x


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq, hidden)
        pooled = torch.mean(lstm_out, dim=1)  # average over time
        embedding = self.embedding_layer(pooled)
        out = self.classifier(embedding)
        return out, embedding


class FusionModelWithGCMSDropout(nn.Module):
    def __init__(self, smell_encoder, gcms_encoder, combined_dim, output_dim, gcms_dropout_p=0.3):
        super().__init__()
        self.smell_encoder = smell_encoder
        self.gcms_encoder = gcms_encoder

        self.gcms_dropout_p = gcms_dropout_p

        self.combined_fc = nn.Linear(smell_encoder.output_dim + gcms_encoder.output_dim, combined_dim)
        self.classifier = nn.Linear(combined_dim, output_dim)

    def forward(self, smell_input, gcms_input):
        # Apply GCMS dropout only during training
        if self.training and torch.rand(1).item() < self.gcms_dropout_p:
            gcms_input = torch.zeros_like(gcms_input)

        smell_feat = self.smell_encoder(smell_input)
        gcms_feat = self.gcms_encoder(gcms_input)

        combined = torch.cat([smell_feat, gcms_feat], dim=-1)
        combined = F.relu(self.combined_fc(combined))
        return self.classifier(combined)
    
    
class TranslationModel(nn.Module):
    def __init__(self, smell_encoder, gcms_dim, num_classes, hidden_dim=None, dropout=0.1):
        """
        smell_encoder: a neural network that maps smell_input → latent embedding
        gcms_dim: dimensionality of GC-MS vector to predict
        num_classes: number of output classes
        hidden_dim: optional hidden dimension for intermediate projection
        dropout: dropout rate (applied before heads)
        """
        super().__init__()
        self.smell_encoder = smell_encoder

        # Optional projection layer
        self.projection = nn.Identity()
        proj_dim = smell_encoder.output_dim if hasattr(smell_encoder, "output_dim") else gcms_dim
        if hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(proj_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            proj_dim = hidden_dim

        # GC-MS regression head
        self.gcms_head = nn.Linear(proj_dim, gcms_dim)

        # Classification head
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, smell_input):
        """
        Returns:
            gcms_pred: predicted GC-MS vector
            class_logits: classification logits
        """
        smell_feat = self.smell_encoder(smell_input)
        feat = self.projection(smell_feat)
        gcms_pred = self.gcms_head(feat)
        class_logits = self.classifier(feat)
        return gcms_pred, class_logits