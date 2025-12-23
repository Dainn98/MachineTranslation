import torch
import torch.nn as nn
from config import EPS

class Norm(nn.Module):
    # EPS = 1e-6
    def __init__(self, d_model, eps=EPS):
        super().__init__()
        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias  = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.alpha * (x - norm) / (std + self.eps) + self.bias
