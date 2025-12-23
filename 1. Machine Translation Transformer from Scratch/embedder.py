import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        print(f'Embedder vocab_size, d_model',vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
