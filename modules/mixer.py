import torch
import torch.nn as nn


class GraphMixer(nn.Module):
    def __init__(self, dropout=0.0, gate_mode="none", num_vars=None, num_tokens=None, gate_init=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate_mode = str(gate_mode).lower()
        self.gate = None
        if self.gate_mode == "none":
            self.gate = None
        elif self.gate_mode == "scalar":
            self.gate = nn.Parameter(torch.tensor(gate_init))
        elif self.gate_mode == "per_var":
            if num_vars is None:
                raise ValueError("num_vars is required for per_var gate_mode")
            self.gate = nn.Parameter(torch.full((1, int(num_vars), 1, 1), gate_init))
        elif self.gate_mode == "per_token":
            if num_vars is None or num_tokens is None:
                raise ValueError("num_vars and num_tokens are required for per_token gate_mode")
            self.gate = nn.Parameter(torch.full((1, int(num_vars), int(num_tokens), 1), gate_init))
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

    def _get_gate(self, x, token_offset=0):
        if self.gate is None:
            return None
        gate = torch.sigmoid(self.gate)
        if self.gate_mode == "per_token":
            start = int(token_offset) if token_offset is not None else 0
            end = start + x.shape[2]
            gate = gate[:, :, start:end, :]
        return gate

    def forward(self, adj, x, token_offset=0):
        # adj: [B, C, C], x: [B, C, T, D]
        mixed = torch.einsum("bij,bjtd->bitd", adj, x)
        gate = self._get_gate(x, token_offset=token_offset)
        if gate is not None:
            mixed = mixed * gate
        mixed = self.dropout(mixed)
        return x + mixed
