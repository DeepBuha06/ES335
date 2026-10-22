"""Fixed-context windows over a list of token-id sequences, shared by the n-gram and MLP models.

Every function is padded on the left with `context` <BOS> and ends with <EOS>. The predicted
positions are every real token plus the final <EOS>, so all models are scored on the same targets.
"""
import numpy as np


class Windows:
    def __init__(self, seqs, context, bos_id, eos_id):
        self.context = context
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        padded = lens + context + 1
        starts = np.cumsum(padded) - padded
        self.flat = np.full(int(padded.sum()), bos_id, dtype=np.uint16)
        for s, st, n in zip(seqs, starts, lens):
            self.flat[st + context:st + context + n] = s
            self.flat[st + context + n] = eos_id
        n_pred = lens + 1
        total = int(n_pred.sum())
        self.seq_id = np.repeat(np.arange(len(seqs), dtype=np.int32), n_pred)
        within = np.arange(total) - np.repeat(np.cumsum(n_pred) - n_pred, n_pred)
        self.pos = np.repeat(starts + context, n_pred) + within
        self._offsets = np.arange(-context, 1)

    def __len__(self):
        return len(self.pos)

    def window(self, idx):
        """(m, context + 1) uint16: the context tokens (oldest first) followed by the target."""
        return self.flat[self.pos[idx][:, None] + self._offsets]

    def contexts(self, idx):
        return self.window(idx)[:, :-1]

    def targets(self, idx):
        return self.flat[self.pos[idx]]
