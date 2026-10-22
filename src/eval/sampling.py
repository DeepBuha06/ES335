"""Prompt extraction and batched autoregressive sampling shared by every model.

A model is a `step_fn(ctx) -> probs`: ctx is (m, context) token ids (left-padded with <BOS> exactly as in
training), probs is (m, V) float64 with rows summing to 1 (temperature already applied).
"""
import numpy as np


def make_prompt(ids, ids_of, k):
    """Function header (decorators + signature line) + <INDENT> + first k body tokens, or None.

    ids_of: dict with vocabulary ids of 'def', '<NEWLINE>', '<INDENT>', '<DEDENT>'.
    Rejects one-line functions and functions that end inside the prompt.
    """
    ids = np.asarray(ids)
    d = np.nonzero(ids == ids_of["def"])[0]
    if len(d) == 0:
        return None
    nl = np.nonzero(ids[d[0]:] == ids_of["<NEWLINE>"])[0]
    if len(nl) == 0:
        return None
    nl = d[0] + nl[0]
    end = nl + 2 + k
    if end >= len(ids) or ids[nl + 1] != ids_of["<INDENT>"]:
        return None
    depth = np.cumsum((ids[:end] == ids_of["<INDENT>"]).astype(int) - (ids[:end] == ids_of["<DEDENT>"]))
    if depth[nl + 1:].min() < 1:
        return None
    return ids[:end].astype(np.uint16)


def prompt_depth(prompt, ids_of):
    return int((prompt == ids_of["<INDENT>"]).sum() - (prompt == ids_of["<DEDENT>"]).sum())


def generate(step_fn, context, prompts, max_new, rng, ids_of, bos_id, eos_id, record_top=0):
    """Sample one completion per prompt.

    Stops at <EOS> or when <DEDENT> returns to function level (status 1, natural) or after `max_new`
    tokens (status 2, truncated). Returns (tokens, status, rec) with tokens a list of uint16 arrays
    (the generated tokens including the stopping one) and, if record_top > 0, rec = dict of per-step
    sampled-token probabilities and top-k (ids, probs).
    """
    B = len(prompts)
    ctx = np.full((B, context), bos_id, dtype=np.uint16)
    for i, p in enumerate(prompts):
        tail = np.asarray(p[-context:], dtype=np.uint16)
        ctx[i, context - len(tail):] = tail
    depth = np.array([prompt_depth(p, ids_of) for p in prompts])
    out = np.zeros((B, max_new), dtype=np.uint16)
    length = np.zeros(B, dtype=np.int64)
    status = np.zeros(B, dtype=np.int8)
    rec = None
    if record_top:
        rec = {"p": np.zeros((B, max_new), dtype=np.float32), "top_ids": np.zeros((B, max_new, record_top), dtype=np.uint16),
               "top_p": np.zeros((B, max_new, record_top), dtype=np.float32)}
    for t in range(max_new):
        act = np.nonzero(status == 0)[0]
        if len(act) == 0:
            break
        probs = step_fn(ctx[act])
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(len(act)) * cdf[:, -1]
        tok = np.minimum((cdf < u[:, None]).sum(axis=1), probs.shape[1] - 1)
        if record_top:
            rec["p"][act, t] = probs[np.arange(len(act)), tok]
            top = np.argpartition(-probs, record_top, axis=1)[:, :record_top]
            tp = np.take_along_axis(probs, top, axis=1)
            o = np.argsort(-tp, axis=1)
            rec["top_ids"][act, t] = np.take_along_axis(top, o, axis=1)
            rec["top_p"][act, t] = np.take_along_axis(tp, o, axis=1)
        out[act, t] = tok
        length[act] = t + 1
        ctx[act, :-1] = ctx[act, 1:]
        ctx[act, -1] = tok
        depth[act] += (tok == ids_of["<INDENT>"]).astype(int) - (tok == ids_of["<DEDENT>"])
        status[act[(tok == eos_id) | (depth[act] <= 0)]] = 1
    status[status == 0] = 2
    return [out[i, :length[i]] for i in range(B)], status, rec


def ngram_step(model, temperature):
    def step(ctx):
        p = model.dense(ctx)
        if temperature != 1.0:
            p = p ** (1.0 / temperature)
            p /= p.sum(axis=1, keepdims=True)
        return p
    return step


def mlp_step(model, temperature, device="cuda"):
    import torch

    @torch.no_grad()
    def step(ctx):
        logits = model(torch.from_numpy(ctx.astype(np.int64)).to(device)).float()
        return torch.softmax(logits / temperature, dim=1).double().cpu().numpy()
    return step
