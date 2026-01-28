import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    def __init__(self, seq_len, use_patch=False, patch_len=16, patch_stride=None):
        super().__init__()
        self.seq_len = int(seq_len)
        self.use_patch = bool(use_patch)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride) if patch_stride is not None else self.patch_len

        if self.patch_len <= 0:
            self.patch_len = self.seq_len
        self.patch_len = min(self.patch_len, self.seq_len)

        if self.patch_stride <= 0:
            self.patch_stride = self.patch_len
        self.patch_stride = min(self.patch_stride, self.patch_len)

        if self.use_patch:
            self.num_tokens = (self.seq_len - self.patch_len) // self.patch_stride + 1
        else:
            self.num_tokens = self.seq_len

    def forward(self, h_time):
        # h_time: [B, C, L, D]
        if not self.use_patch:
            return h_time
        bsz, n_vars, seq_len, dim = h_time.shape
        patch_len = min(self.patch_len, seq_len)
        stride = min(self.patch_stride, patch_len)

        h = h_time.reshape(bsz * n_vars, seq_len, dim)
        patches = h.unfold(dimension=1, size=patch_len, step=stride)
        h_pooled = patches.mean(dim=-1)
        return h_pooled.reshape(bsz, n_vars, -1, dim)
