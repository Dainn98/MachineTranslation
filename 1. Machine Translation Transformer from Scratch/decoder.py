
import torch.nn as nn
from embedder import Embedder
from positional_encoder import PositionalEncoder
from attention import MultiHeadAttention
from swiglu_feed_forward_network import SwiGLUFeedForward
from norm import Norm
from config import DROPOUT

# import norm, attention, feed_forward_network
# 

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=DROPOUT):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)

        # self.ff = FeedForward(d_model, dropout=dropout)
        self.ff = SwiGLUFeedForward(d_model, dropout=dropout)
        

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x2 = self.norm_1(x)
        x  = x + self.dropout_1(self.attn_1(x2, x2, x2, tgt_mask))

        x2 = self.norm_2(x)
        x  = x + self.dropout_2(self.attn_2(x2, enc_out, enc_out, src_mask))

        x2 = self.norm_3(x)
        x  = x + self.dropout_3(self.ff(x2))

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout=DROPOUT):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, heads, dropout) for _ in range(N)
        ])

        self.norm = Norm(d_model)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embed(tgt)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x, enc_out, src_mask, tgt_mask)

        return self.norm(x)
