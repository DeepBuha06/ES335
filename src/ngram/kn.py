"""Interpolated Kneser-Ney n-gram language model in NumPy, with one discount per order ("single") or
three discounts per order ("modified", Chen & Goodman 1998).

An order-N model predicts a token from the N-1 previous tokens. For k < N the counts are
continuation counts (number of distinct tokens that precede the k-gram); order N uses raw counts:

    P_k(w|h) = max(c(h,w) - D(c), 0) / c(h) + gamma(h) * P_{k-1}(w|h')     if c(h) > 0
    P_k(w|h) = P_{k-1}(w|h')                                                otherwise
    P_0(w)   = 1 / V
    gamma(h) = (D1 N1(h) + D2 N2(h) + D3 N3+(h)) / c(h)

where N1(h), N2(h), N3+(h) count the followers of h seen once, twice, three or more times, and D(c) is D1, D2
or D3 for c = 1, 2, >= 3. From the count-of-counts n1..n4 of level k, with Y = n1 / (n1 + 2 n2):
single: D1 = D2 = D3 = Y;  modified: D1 = 1 - 2Y n2/n1, D2 = 2 - 3Y n3/n2, D3 = 3 - 4Y n4/n3.
gamma(h) is exactly the mass removed by discounting, and the recursion ends in a uniform distribution over
the full vocabulary, so probabilities over the vocabulary sum to 1.

N-grams are keyed by a 64-bit linear hash sum((token + 1) * M[distance]) so tables are sorted uint64
arrays searched with np.searchsorted (collision probability ~1e-6 for ~1e7 types).
"""
import numpy as np

CHUNK = 500_000


def _multipliers(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 2**62, size=n, dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def _hash(tok, mult, first_dist):
    """tok: (n, c) columns oldest -> newest; the newest column sits at distance `first_dist`."""
    n, c = tok.shape
    h = np.zeros(n, dtype=np.uint64)
    for col in range(c):
        h += (tok[:, col].astype(np.uint64) + np.uint64(1)) * mult[first_dist + c - 1 - col]
    return h


def _find(keys, q):
    idx = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return idx, keys[idx] == q


def discounts(fc, kind):
    """(D1, D2, D3) from the count-of-counts of one level."""
    n1, n2, n3, n4 = (int((fc == k).sum()) for k in (1, 2, 3, 4))
    y = n1 / (n1 + 2 * n2) if n1 + n2 else 0.5
    if kind == "single" or min(n1, n2, n3) == 0:
        return (y, y, y)
    d = (1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3)
    return tuple(float(min(max(v, 1e-3), k - 1e-3)) for v, k in zip(d, (1, 2, 3)))


def _make_level(fk, fc, types, mult, kind="modified"):
    """fk/fc: sorted n-gram keys and their counts; types: (n, k) token matrix aligned with fk."""
    k = types.shape[1]
    ctx = _hash(types[:, :-1], mult, 1) if k > 1 else np.zeros(len(types), dtype=np.uint64)
    ck, inv = np.unique(ctx, return_inverse=True)
    inv = inv.reshape(-1)
    D = discounts(fc, kind)
    cn1 = np.bincount(inv).astype(np.int32)  # number of distinct followers of each context (CSR lengths)
    n_1 = np.bincount(inv, weights=(fc == 1))
    n_2 = np.bincount(inv, weights=(fc == 2))
    n_3 = np.bincount(inv, weights=(fc >= 3))
    by_ctx = np.argsort(inv, kind="stable")  # CSR view: the followers of each context are contiguous
    return {
        "fk": fk, "fc": fc.astype(np.int32), "ck": ck,
        "ctot": np.bincount(inv, weights=fc).astype(np.float64),
        "cn1": cn1, "gnum": D[0] * n_1 + D[1] * n_2 + D[2] * n_3, "D": np.array(D),
        "cw": types[by_ctx, -1].astype(np.uint16), "cc": fc[by_ctx].astype(np.int32),
        "cstart": np.concatenate([[0], np.cumsum(cn1)]).astype(np.int64),
    }


def _disc(D, c):
    """Discount for counts c (0 for unseen)."""
    return np.where(c >= 3, D[2], np.where(c == 2, D[1], np.where(c == 1, D[0], 0.0)))


class KNModel:
    def __init__(self, order, vocab_size, levels, mult, kind="modified"):
        self.order, self.V, self.levels, self.mult, self.kind = order, vocab_size, levels, mult, kind

    @classmethod
    def fit(cls, windows, order, vocab_size, seed=0, kind="modified"):
        assert windows.context == order - 1
        mult = _multipliers(order, seed)
        n = len(windows)
        keys = np.empty(n, dtype=np.uint64)
        for s in range(0, n, CHUNK):
            sl = slice(s, min(s + CHUNK, n))
            keys[sl] = _hash(windows.window(sl), mult, 0)
        fk, idx, fc = np.unique(keys, return_index=True, return_counts=True)
        del keys
        types = np.concatenate([windows.window(idx[s:s + CHUNK]) for s in range(0, len(idx), CHUNK)])
        levels = [None] * order
        levels[order - 1] = _make_level(fk, fc, types, mult, kind)
        for k in range(order - 1, 0, -1):
            sub = types[:, 1:]  # drop the oldest token: continuation counts of the k-gram
            fk, idx, fc = np.unique(_hash(sub, mult, 0), return_index=True, return_counts=True)
            types = sub[idx]
            levels[k - 1] = _make_level(fk, fc, types, mult, kind)
        return cls(order, vocab_size, levels, mult, kind)

    def _prob(self, ctx, w):
        """ctx: (B, order-1) token ids; w: (B,) or (B, V) targets. Returns probabilities shaped like w."""
        N = self.order
        ctx = np.asarray(ctx)
        w = np.asarray(w)
        extra = (1,) * (w.ndim - 1)
        wterm = (w.astype(np.uint64) + np.uint64(1)) * self.mult[0]
        p = np.full(w.shape, 1.0 / self.V)
        ck = np.zeros(len(ctx), dtype=np.uint64)
        for k in range(1, N + 1):
            if k > 1:
                ck += (ctx[:, N - k].astype(np.uint64) + np.uint64(1)) * self.mult[k - 1]
            L = self.levels[k - 1]
            ci, cfound = _find(L["ck"], ck)
            fi, ffound = _find(L["fk"], ck.reshape(-1, *extra) + wterm)
            tot = np.where(cfound, L["ctot"][ci], 1.0).reshape(-1, *extra)
            gnum = L["gnum"][ci].reshape(-1, *extra)
            c = np.where(ffound, L["fc"][fi], 0)
            pk = (np.maximum(c - _disc(L["D"], c), 0.0) + gnum * p) / tot
            p = np.where(cfound.reshape(-1, *extra), pk, p)
        return p

    def prob(self, ctx, w):
        return self._prob(ctx, w)

    def dist(self, ctx):
        """(B, V) probability of every vocabulary token given each context row."""
        ctx = np.asarray(ctx)
        return self._prob(ctx, np.broadcast_to(np.arange(self.V), (len(ctx), self.V)))

    def _scatter_level(self, L, rows, ci, new):
        """Add the discounted seen-follower mass A(w|h) = (c - D) / c(h) of contexts `ci` to rows of `new`."""
        lens = L["cn1"][ci]
        starts = L["cstart"][ci]
        total = int(lens.sum())
        within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        idx = np.repeat(starts, lens) + within
        cc = L["cc"][idx]
        new[np.repeat(rows, lens), L["cw"][idx]] += (cc - _disc(L["D"], cc)) / np.repeat(L["ctot"][ci], lens)

    def dense(self, ctx):
        """Same distribution as `dist`, computed by scattering each context's seen followers on top of the
        lower-order distribution (fast enough to sample batches of hundreds of sequences)."""
        ctx = np.asarray(ctx)
        N, B, V = self.order, len(ctx), self.V
        if not hasattr(self, "_p1"):
            L = self.levels[0]
            p1 = np.full((1, V), L["gnum"][0] / L["ctot"][0] / V)
            self._scatter_level(L, np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), p1)
            self._p1 = p1[0]
        p = np.broadcast_to(self._p1, (B, V))
        ck = np.zeros(B, dtype=np.uint64)
        for k in range(2, N + 1):
            ck += (ctx[:, N - k].astype(np.uint64) + np.uint64(1)) * self.mult[k - 1]
            L = self.levels[k - 1]
            ci, found = _find(L["ck"], ck)
            lam = np.where(found, L["gnum"][ci] / np.where(found, L["ctot"][ci], 1.0), 1.0)
            new = lam[:, None] * p
            rows = np.nonzero(found)[0]
            self._scatter_level(L, rows, ci[rows], new)
            p = new
        return np.array(p)

    def nll(self, windows, chunk=CHUNK):
        """Per-position negative log-likelihood (nats) of every predicted token in `windows`."""
        out = np.empty(len(windows), dtype=np.float64)
        for s in range(0, len(windows), chunk):
            sl = slice(s, min(s + chunk, len(windows)))
            out[sl] = -np.log(self._prob(windows.contexts(sl), windows.targets(sl)))
        return out

    def sizes(self):
        return [{"k": k + 1, "ngram_types": len(L["fk"]), "contexts": len(L["ck"]), "discounts": [round(float(d), 4) for d in L["D"]]}
                for k, L in enumerate(self.levels)]

    def save(self, path):
        arrays = {"order": np.array(self.order), "V": np.array(self.V), "mult": self.mult, "kind": np.array(self.kind)}
        for k, L in enumerate(self.levels, 1):
            arrays.update({f"{name}_{k}": np.asarray(v) for name, v in L.items()})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        order = int(z["order"])
        names = ("fk", "fc", "ck", "ctot", "cn1", "gnum", "D", "cw", "cc", "cstart")
        levels = [{n: z[f"{n}_{k}"] for n in names} for k in range(1, order + 1)]
        return cls(order, int(z["V"]), levels, z["mult"], str(z["kind"]))
