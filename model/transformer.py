import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.head.weight = self.embedding.token.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        if idx.size(1) > self.config.block_size:
            idx = idx[:, -self.config.block_size:]

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
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=None
    ):
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty != 1.0:
                for batch_idx in range(idx.size(0)):
                    used_tokens = torch.unique(idx[batch_idx])
                    logits[batch_idx, used_tokens] /= repetition_penalty

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    logits[logits < values[:, [-1]]] = float("-inf")

                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_remove = cumulative_probs > top_p
                    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                    sorted_remove[..., 0] = False
                    remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                    logits = logits.masked_fill(remove, float("-inf"))

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return idx
