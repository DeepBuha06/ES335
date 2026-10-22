"""Token categories, per-function aggregation and bootstrap confidence intervals."""
import keyword

import numpy as np

from src.data.tokenizer import BOS, DEDENT, EOS, INDENT, NEWLINE, NUM, PAD, STR, UNK

CATEGORIES = ["keyword", "identifier", "operator/punct", "layout", "literal", "unk", "eos"]


def token_category(t):
    if t == UNK:
        return "unk"
    if t == EOS:
        return "eos"
    if t in (NEWLINE, INDENT, DEDENT):
        return "layout"
    if t in (STR, NUM):
        return "literal"
    if t in (BOS, PAD):
        return "eos"  # never a target; kept only so every id has a category
    if keyword.iskeyword(t):
        return "keyword"
    return "identifier" if t.isidentifier() else "operator/punct"


def category_ids(vocab):
    return np.array([CATEGORIES.index(token_category(t)) for t in vocab.itos])


def per_function(nll, seq_id, n_functions):
    """Sum of NLL and number of predicted tokens per function."""
    return np.bincount(seq_id, weights=nll, minlength=n_functions), np.bincount(seq_id, minlength=n_functions)


def category_loss(nll, targets, cat_of_id):
    cats = cat_of_id[targets]
    n = len(cats)
    return {c: {"share": float((cats == i).sum() / n),
                "mean_nll": float(nll[cats == i].mean()) if (cats == i).any() else None}
            for i, c in enumerate(CATEGORIES)}


def bootstrap_ppl(fn_nll_sum, fn_n, rng, n_boot=1000, alpha=0.05):
    """95% CI of perplexity, resampling functions with replacement."""
    idx = rng.integers(0, len(fn_n), size=(n_boot, len(fn_n)))
    ppl = np.exp(fn_nll_sum[idx].sum(axis=1) / fn_n[idx].sum(axis=1))
    return [float(np.quantile(ppl, alpha / 2)), float(np.quantile(ppl, 1 - alpha / 2))]


def bootstrap_rate(ok, natural, rng, n_boot=2000, alpha=0.05):
    """Validity rate = sum(ok) / sum(natural) with a CI from resampling prompts.
    ok, natural: (reps, prompts) boolean arrays (replicates share prompts)."""
    P = ok.shape[1]
    idx = rng.integers(0, P, size=(n_boot, P))
    num, den = ok[:, idx].sum(axis=(0, 2)), natural[:, idx].sum(axis=(0, 2))
    ok_boot = num[den > 0] / den[den > 0]
    rate = ok.sum() / max(natural.sum(), 1)
    return float(rate), [float(np.quantile(ok_boot, alpha / 2)), float(np.quantile(ok_boot, 1 - alpha / 2))]


def bootstrap_ratio(a_sum, a_n, b_sum, b_n, rng, n_boot=1000, alpha=0.05):
    """95% CI of ppl_b / ppl_a - 1 (relative gap) with the two function sets resampled independently."""
    ia = rng.integers(0, len(a_n), size=(n_boot, len(a_n)))
    ib = rng.integers(0, len(b_n), size=(n_boot, len(b_n)))
    ppl_a = np.exp(a_sum[ia].sum(axis=1) / a_n[ia].sum(axis=1))
    ppl_b = np.exp(b_sum[ib].sum(axis=1) / b_n[ib].sum(axis=1))
    gap = ppl_b / ppl_a - 1
    return [float(np.quantile(gap, alpha / 2)), float(np.quantile(gap, 1 - alpha / 2))]
