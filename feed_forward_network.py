import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT,D_FF

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        return self.linear_2(x)
