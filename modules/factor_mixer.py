import math
import torch
import torch.nn as nn


class FactorMixer(nn.Module):
    def __init__(self, num_vars, rank, alpha_init=-8.0):
        super().__init__()
        self.num_vars = int(num_vars)
        self.rank = max(1, int(rank))
        self.p = nn.Parameter(torch.randn(self.num_vars, self.rank) * 0.02)
        self.q = nn.Parameter(torch.randn(self.num_vars, self.rank) * 0.02)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

        self.last_alpha = None
        self.last_entropy = None
        self.last_adj = None
        self.last_reg_loss = None

    def _adjacency(self):
        scores = torch.matmul(self.p, self.q.transpose(-1, -2)) * (1.0 / math.sqrt(self.rank))
        return torch.softmax(scores, dim=-1)

    def forward(self, x):
        # x: [B, C, N, D]
        adj = self._adjacency()
        mixed = torch.einsum("ij,bjnd->bind", adj, x)
        alpha = torch.sigmoid(self.alpha_logit)
        out = x + alpha * mixed

        eps = 1e-12
        entropy = -(adj * (adj + eps).log()).sum(-1)
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        reg = (adj - eye).pow(2).sum()

        self.last_alpha = alpha.detach()
        self.last_entropy = entropy.detach()
        self.last_adj = adj.detach()
        self.last_reg_loss = reg
        return out
