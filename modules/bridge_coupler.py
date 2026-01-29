import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableTokenDetrend(nn.Module):
    def __init__(self, window=16):
        super().__init__()
        self.window = max(1, int(window))

    def _moving_average(self, x):
        # x: [B, C, N, D]
        if self.window <= 1:
            return x
        k = self.window
        left = k // 2
        right = k // 2
        if k % 2 == 0:
            left = max(0, left - 1)
        bsz, n_vars, num_tokens, dim = x.shape
        flat = x.permute(0, 1, 3, 2).contiguous()
        flat = flat.reshape(bsz * n_vars * dim, 1, num_tokens)
        flat = F.pad(flat, (left, right), mode="replicate")
        flat = F.avg_pool1d(flat, kernel_size=k, stride=1, padding=0)
        flat = flat.reshape(bsz, n_vars, dim, num_tokens).permute(0, 1, 3, 2)
        return flat

    def forward(self, x):
        trend = self._moving_average(x)
        return x - trend


class StatsFiLM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        hidden = max(1, int(hidden))
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, mu, sigma, h):
        # mu, sigma: [B, C], h: [B, C, N, D]
        eps = 1e-8
        feats = torch.stack([mu, torch.log(sigma + eps)], dim=-1)
        out = self.fc2(F.gelu(self.fc1(feats)))
        gamma = out[..., 0]
        beta = out[..., 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + gamma) + beta


class BridgeCouplingAttention(nn.Module):
    def __init__(self, d_model, rank=8, scale=8, topk=0, alpha_init=-20.0):
        super().__init__()
        self.rank = max(1, int(rank))
        self.scale = max(1, int(scale))
        self.topk = int(topk)
        self.q_proj = nn.Linear(d_model, self.rank, bias=False)
        self.k_proj = nn.Linear(d_model, self.rank, bias=False)
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)))

        self.last_alpha = None
        self.last_entropy = None
        self.last_topk_mass = None
        self.last_adj_diff = None

    def _sparsify(self, adj):
        if self.topk <= 0:
            return adj, None
        k = min(self.topk, adj.shape[-1])
        vals, idx = torch.topk(adj, k, dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, idx, 1.0)
        masked = adj * mask
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_mass = masked.sum(dim=-1)
        return masked / denom, topk_mass

    def forward(self, h_stable, h_content):
        # h_*: [B, C, N, D]
        bsz, n_vars, num_tokens, dim = h_stable.shape
        scale = min(self.scale, num_tokens)
        alpha = torch.sigmoid(self.alpha_logit)

        segments = []
        entropies = []
        topk_masses = [] if self.topk > 0 else None
        adj_diffs = []
        prev_adj = None
        eps = 1e-12

        for start in range(0, num_tokens, scale):
            end = min(start + scale, num_tokens)
            h_s = h_stable[:, :, start:end, :]
            h_c = h_content[:, :, start:end, :]
            z = h_s.mean(dim=2)
            q = self.q_proj(z)
            k = self.k_proj(z)
            scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.rank))
            adj = torch.softmax(scores, dim=-1)
            adj, topk_mass = self._sparsify(adj)
            mixed = torch.einsum("bij,bjtd->bitd", adj, h_c)
            segments.append(h_c + alpha * mixed)

            entropy = -(adj * (adj + eps).log()).sum(-1)
            entropies.append(entropy)
            if topk_masses is not None and topk_mass is not None:
                topk_masses.append(topk_mass)
            if prev_adj is not None:
                diff = (adj - prev_adj).abs().mean(dim=(-1, -2))
                adj_diffs.append(diff)
            prev_adj = adj

        out = torch.cat(segments, dim=2)
        self.last_alpha = alpha.detach()
        self.last_entropy = torch.stack(entropies, dim=0).detach()
        if topk_masses is not None:
            self.last_topk_mass = torch.stack(topk_masses, dim=0).detach()
        else:
            self.last_topk_mass = None
        if adj_diffs:
            self.last_adj_diff = torch.stack(adj_diffs, dim=0).detach()
        else:
            self.last_adj_diff = None
        return out
