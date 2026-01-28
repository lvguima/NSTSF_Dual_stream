import torch.nn as nn

from layers.DynamicGraph import LowRankGraphGenerator


class LowRankGraphLearner(nn.Module):
    def __init__(self, in_dim, rank, dropout=0.0):
        super().__init__()
        self.generator = LowRankGraphGenerator(in_dim, rank, dropout)

    def forward(self, z):
        return self.generator(z)
