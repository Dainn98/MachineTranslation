import torch
import math
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i]   = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0) # (max_seq_len, d_model) => (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

        print(f'PositionalEncoder d_model:', d_model)
        print(f'PositionalEncoder max_seq_len:',max_seq_len)

    def forward(self, x): #shape input: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
