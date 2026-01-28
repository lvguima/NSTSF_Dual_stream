import torch.nn as nn


class ForecastHead(nn.Module):
    def __init__(self, d_model, num_tokens, pred_len):
        super().__init__()
        self.pred_len = int(pred_len)
        self.in_dim = int(d_model) * int(num_tokens)
        self.proj = nn.Linear(self.in_dim, self.pred_len)

    def forward(self, x):
        # x: [B, C, N, D]
        bsz, n_vars, num_tokens, dim = x.shape
        x = x.reshape(bsz, n_vars, num_tokens * dim)
        out = self.proj(x)
        return out.permute(0, 2, 1)
