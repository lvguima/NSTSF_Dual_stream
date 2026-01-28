import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalEncoderTCN(nn.Module):
    def __init__(self, d_model, num_layers, kernel_size, dilation_base, dropout):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(num_layers):
            dilation = dilation_base ** i
            layers.append(TemporalBlock(in_channels, d_model, kernel_size, dilation, dropout))
            in_channels = d_model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, L, C]
        bsz, seq_len, n_vars = x.shape
        x = x.permute(0, 2, 1).contiguous().reshape(bsz * n_vars, 1, seq_len)
        out = self.network(x)  # [B*C, D, L]
        out = out.permute(0, 2, 1).contiguous().reshape(bsz, n_vars, seq_len, -1)
        return out


class TemporalEncoderTransformer(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff, dropout, activation="gelu",
                 attn_factor=1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        from layers.SelfAttention_Family import FullAttention, AttentionLayer

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attn_factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: [B, L, C]
        bsz, seq_len, n_vars = x.shape
        x = x.permute(0, 2, 1).contiguous().reshape(bsz * n_vars, seq_len, 1)
        x = self.in_proj(x)
        x = x + self.position_embedding(x)
        x = self.dropout(x)

        x, _ = self.encoder(x, attn_mask=None)
        x = x.reshape(bsz, n_vars, x.shape[1], -1)
        return x
