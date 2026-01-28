import torch.nn as nn

from layers.TemporalEncoder import TemporalEncoderTCN, TemporalEncoderTransformer


class TemporalEncoderWrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        temporal_encoder = getattr(configs, "temporal_encoder", "tcn").lower()
        if temporal_encoder == "transformer":
            self.encoder = TemporalEncoderTransformer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                num_layers=configs.e_layers,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=getattr(configs, "activation", "gelu"),
                attn_factor=getattr(configs, "factor", 1),
            )
        elif temporal_encoder == "tcn":
            tcn_kernel = int(getattr(configs, "tcn_kernel", 3))
            tcn_dilation = int(getattr(configs, "tcn_dilation", 2))
            self.encoder = TemporalEncoderTCN(
                d_model=configs.d_model,
                num_layers=configs.e_layers,
                kernel_size=tcn_kernel,
                dilation_base=tcn_dilation,
                dropout=configs.dropout,
            )
        else:
            raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder}")

    def forward(self, x):
        return self.encoder(x)
