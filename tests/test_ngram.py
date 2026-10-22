from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.data.windows import Windows
from src.ngram.kn import KNModel

BOS, EOS, V = 2, 3, 12


def toy_seqs(seed=0, n=60):
    rng = np.random.default_rng(seed)
    base = [rng.integers(4, V, size=rng.integers(3, 9)) for _ in range(6)]
    seqs = []
    for _ in range(n):  # repeated fragments so higher-order contexts recur
        parts = [base[i] for i in rng.integers(0, len(base), size=rng.integers(1, 4))]
        seqs.append(np.concatenate(parts).astype(np.uint16))
    return seqs


def ref_discounts(counts, kind):
    n = [sum(c == k for c in counts) for k in (1, 2, 3, 4)]
    y = n[0] / (n[0] + 2 * n[1]) if n[0] + n[1] else 0.5
    if kind == "single" or min(n[:3]) == 0:
        return (y, y, y)
    d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1], 3 - 4 * y * n[3] / n[2])
    return tuple(min(max(v, 1e-3), k - 1e-3) for v, k in zip(d, (1, 2, 3)))


def ref_prob(seqs, N, kind="modified"):
    """Dict-based interpolated (modified) KN straight from the definition (slow, for tiny corpora)."""
    raw = Counter()
    for s in seqs:
        t = [BOS] * (N - 1) + list(s) + [EOS]
        for i in range(N - 1, len(t)):
            raw[tuple(t[i - N + 1:i + 1])] += 1
    tables = {N: raw}
    for k in range(N - 1, 0, -1):
        tables[k] = Counter(g[1:] for g in tables[k + 1])
    D = {k: ref_discounts(list(tab.values()), kind) for k, tab in tables.items()}

    def disc(k, c):
        return 0.0 if c == 0 else D[k][min(c, 3) - 1]

    def p(k, h, w):
        if k == 0:
            return 1 / V
        lower = p(k - 1, h[1:], w)
        match = {g: c for g, c in tables[k].items() if g[:-1] == h}
        if not match:
            return lower
        tot = sum(match.values())
        gamma = sum(disc(k, c) for c in match.values())
        c = match.get(h + (w,), 0)
        return (max(c - disc(k, c), 0) + gamma * lower) / tot

    return lambda h, w: p(N, tuple(h), w)


@pytest.mark.parametrize("kind", ["single", "modified"])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_matches_reference_and_normalizes(order, kind):
    seqs = toy_seqs()
    model = KNModel.fit(Windows(seqs, order - 1, BOS, EOS), order, V, seed=1, kind=kind)
    if kind == "modified":
        assert len({round(float(d), 6) for d in model.levels[-1]["D"]}) == 3  # three different discounts in use
    ref = ref_prob(seqs, order, kind)
    rng = np.random.default_rng(5)
    real = Windows(seqs[:10], order - 1, BOS, EOS)
    ctxs = np.concatenate([real.contexts(slice(0, len(real))), rng.integers(0, V, size=(30, order - 1))])
    dist = model.dist(ctxs)
    assert dist.shape == (len(ctxs), V)
    assert np.abs(dist.sum(axis=1) - 1).max() < 1e-9
    for row, ctx in zip(dist, ctxs):
        for w in range(V):
            assert row[w] == pytest.approx(ref(ctx, w), rel=1e-9)


@pytest.mark.parametrize("kind", ["single", "modified"])
@pytest.mark.parametrize("order", [2, 3, 5])
def test_dense_equals_dist(order, kind):
    seqs = toy_seqs(seed=3)
    model = KNModel.fit(Windows(seqs, order - 1, BOS, EOS), order, V, kind=kind)
    rng = np.random.default_rng(1)
    w = Windows(seqs[:8], order - 1, BOS, EOS)
    ctxs = np.concatenate([w.contexts(slice(0, len(w))), rng.integers(0, V, size=(20, order - 1))])
    assert np.allclose(model.dense(ctxs), model.dist(ctxs), rtol=1e-10, atol=0)
    assert np.abs(model.dense(ctxs).sum(axis=1) - 1).max() < 1e-9


def test_prob_matches_dist_and_roundtrip(tmp_path):
    seqs = toy_seqs()
    model = KNModel.fit(Windows(seqs, 2, BOS, EOS), 3, V)
    w = Windows(seqs[:5], 2, BOS, EOS)
    sl = slice(0, len(w))
    ctx, tgt = w.contexts(sl), w.targets(sl)
    assert np.allclose(model.prob(ctx, tgt), model.dist(ctx)[np.arange(len(tgt)), tgt])
    assert np.allclose(model.nll(w), -np.log(model.prob(ctx, tgt)))
    model.save(tmp_path / "m.npz")
    assert np.allclose(KNModel.load(tmp_path / "m.npz").dense(ctx), model.dist(ctx))


@pytest.mark.skipif(not Path("data/processed/ngram_5.npz").exists(), reason="n-gram models not built")
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_real_model_normalizes_over_full_vocab(order):
    from src.data.dataset import load_ids
    model = KNModel.load(f"data/processed/ngram_{order}.npz")
    w = Windows(load_ids("val")[:200], order - 1, BOS, EOS)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(w), size=10)
    ctxs = np.concatenate([w.contexts(idx), rng.integers(0, model.V, size=(5, order - 1))])  # 5 mostly unseen
    sums = model.dist(ctxs).sum(axis=1)
    assert model.V == 10004 and np.abs(sums - 1).max() < 1e-6
