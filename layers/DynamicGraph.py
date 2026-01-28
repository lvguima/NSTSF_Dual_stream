import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankGraphGenerator(nn.Module):
    def __init__(self, in_dim, rank, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.proj_u = nn.Linear(in_dim, rank, bias=False)
        self.proj_v = nn.Linear(in_dim, rank, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # z: [B, C, D]
        u = self.proj_u(z)
        v = self.proj_v(z)
        scores = torch.matmul(u, v.transpose(-1, -2)) * (1.0 / math.sqrt(self.rank))
        scores = self.dropout(scores)
        adj = F.softmax(scores, dim=-1)
        return adj, u, v


class LinearGraphMixing(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, x):
        # adj: [B, C, C], x: [B, C, T, D]
        mixed = torch.einsum("bij,bjtd->bitd", adj, x)
        mixed = self.dropout(mixed)
        return x + mixed
