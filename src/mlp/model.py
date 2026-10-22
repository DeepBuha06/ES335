"""Bengio-style fixed-context MLP language model: embedding -> concat -> hidden -> vocab logits."""
import torch
from torch import nn


class MLPLM(nn.Module):
    def __init__(self, vocab_size, context, emb_dim, hidden, dropout=0.0):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc1 = nn.Linear(context * emb_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, ctx):
        """ctx: (B, context) long -> (B, V) logits."""
        h = self.emb(ctx).flatten(1)
        return self.out(self.drop(torch.nn.functional.gelu(self.fc1(h))))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
