import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.position = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        tok = self.token(x)
        pos = self.position(positions)
        return self.dropout(tok + pos)