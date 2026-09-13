import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout, output_window):
        super().__init__()
        self.model_dim = model_dim
        self.output_window = output_window

        # 🔢 Projection for input features
        self.input_proj = nn.Linear(input_dim, model_dim)

        # 🧠 Positional Encoding
        self.positional_encoding = PositionalEncoding(model_dim, dropout)

        # 🔁 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔚 Final regressor
        self.fc_out = nn.Linear(model_dim, output_window)

    def forward(self, src):
        # src: [B, T_in, input_dim]
        x = self.input_proj(src)                         # [B, T_in, model_dim]
        x = self.positional_encoding(x)                 # [B, T_in, model_dim]
        x = self.transformer_encoder(x)                 # [B, T_in, model_dim]
        x = x[:, -1, :]                                 # فقط آخرین زمان مهمه
        out = self.fc_out(x)                            # [B, output_window]
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
