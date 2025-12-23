import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import EPS, DROPOUT

def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=DROPOUT):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        assert d_model % heads == 0
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # linear projection + split into heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1,2)

        # apply attention
        scores = attention(q, k, v, mask, self.dropout) #(batch,head,length,d_k)

        # concat heads
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model) #(B, L, h, d_k)-> (B, L, D_model) 

        # output projection
        return self.out(concat)
