import torch
import torch.nn as nn

from model.embeddings import Embeddings
from model.block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = Embeddings(config)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = nn.functional.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx