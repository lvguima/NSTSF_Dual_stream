import torch
import torch.nn as nn
import torch.nn.functional as F


class DecompGate(nn.Module):
    def __init__(self, hidden_dim=16, bias_init=2.0):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.fc2.bias, float(bias_init))

    def forward(self, x_enc, x_res):
        # x_enc, x_res: [B, L, C]
        eps = 1e-8
        e_res = x_res.abs().mean(dim=1)
        diff = x_enc[:, 1:, :] - x_enc[:, :-1, :]
        e_diff = diff.abs().mean(dim=1)
        rho = e_res / (e_res + e_diff + eps)
        feats = torch.stack(
            [rho, torch.log(e_res + eps), torch.log(e_diff + eps)], dim=-1
        )
        out = self.fc2(F.gelu(self.fc1(feats)))
        return torch.sigmoid(out).squeeze(-1)
