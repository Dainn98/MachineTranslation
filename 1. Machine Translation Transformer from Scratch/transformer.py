
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from config import D_MODEL, NUM_LAYERS, NUM_HEADS, DROPOUT

class Transformer(nn.Module):
    
    def __init__(self, src_vocab, tgt_vocab, d_model=D_MODEL, N=NUM_LAYERS, heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        e = self.encoder(src, src_mask)
        d = self.decoder(tgt, e, src_mask, tgt_mask)
        return self.out(d)
