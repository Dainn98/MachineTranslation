import torch.nn as nn

from embedder import Embedder
from positional_encoder import PositionalEncoder
from attention import MultiHeadAttention
from swiglu_feed_forward_network import SwiGLUFeedForward
from norm import Norm
from config import DROPOUT

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=DROPOUT):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.attention = MultiHeadAttention(heads, d_model, dropout=dropout)
        # self.ff = FeedForward(d_model, dropout=dropout)
        self.ff = SwiGLUFeedForward(d_model, dropout=dropout)
        

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attention(x2, x2, x2, mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout=DROPOUT):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, dropout) for _ in range(N)
        ])

        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)
