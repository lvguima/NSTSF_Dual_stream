import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMapNormalizer(nn.Module):
    def __init__(self, mode="none", window=16, alpha=0.3):
        super().__init__()
        self.mode = str(mode).lower()
        self.window = max(1, int(window))
        self.alpha = float(alpha)
        if self.mode not in ("none", "ma_detrend", "diff1", "ema_detrend"):
            raise ValueError(f"Unsupported graph_map_norm: {self.mode}")

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

    def _diff1(self, x):
        diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        pad = torch.zeros_like(x[:, :, :1, :])
        return torch.cat([pad, diff], dim=2)

    def _ema_trend(self, x):
        alpha = max(0.0, min(1.0, float(self.alpha)))
        num_tokens = x.shape[2]
        if num_tokens == 0:
            return x
        trend = []
        prev = x[:, :, 0, :]
        trend.append(prev)
        for idx in range(1, num_tokens):
            prev = alpha * x[:, :, idx, :] + (1.0 - alpha) * prev
            trend.append(prev)
        return torch.stack(trend, dim=2)

    def forward(self, x):
        if x is None:
            return None
        if self.mode == "none":
            return x
        if self.mode == "ma_detrend":
            trend = self._moving_average(x)
            return x - trend
        if self.mode == "diff1":
            return self._diff1(x)
        if self.mode == "ema_detrend":
            trend = self._ema_trend(x)
            return x - trend
        raise ValueError(f"Unsupported graph_map_norm: {self.mode}")
