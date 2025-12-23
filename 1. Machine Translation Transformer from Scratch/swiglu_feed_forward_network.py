import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT,D_SwiGLU_FF

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=D_SwiGLU_FF, dropout=DROPOUT):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.w1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.silu(x1) * x2   # SwiGLU
        x = self.dropout(x)
        return self.w2(x)
