import numpy as np

from src.data.tokenizer import DEDENT, INDENT, NEWLINE, NUM, STR, UNK, Vocab, code_to_tokens
from src.eval import metrics
from src.eval.sampling import generate, make_prompt
from src.eval.syntax import check_tokens

SRC = "def f(a, b):\n    x = g(a, [1, 2])\n    if x:\n        return b\n    return x\n"
TOKS = code_to_tokens(SRC)
VOCAB = Vocab.build([TOKS], 1000)[0]
IDS_OF = {t: VOCAB.stoi[t] for t in ("def", NEWLINE, INDENT, DEDENT)}


def test_check_tokens_valid_and_categories():
    assert check_tokens(TOKS)["ok"]
    cut = TOKS[:TOKS.index("[") + 2]  # ... g ( a , [ 1   (never closed, then ends)
    r = check_tokens(cut)
    assert not r["ok"] and r["category"] in ("unbalanced brackets", "incomplete statement")
    bad_indent = ["def", "f", "(", ")", ":", NEWLINE, "return", NUM, NEWLINE]
    assert check_tokens(bad_indent)["category"] == "indentation"
    assert check_tokens(["def", "f", "(", ")", ":", NEWLINE, INDENT, "x", "=", NEWLINE, DEDENT])["category"] in ("incomplete statement", "other")
    r = check_tokens(["x", "=", "(", "1", ")", ")", NEWLINE])
    assert r["category"] == "unbalanced brackets" 
    assert check_tokens(["def", "f", "(", ")", ":", NEWLINE, INDENT, "return", UNK, NEWLINE, DEDENT])["ok"]


def test_failing_token_points_at_error():
    toks = ["x", "=", NUM, NEWLINE, "y", "=", "=", "=", NEWLINE]
    r = check_tokens(toks)
    assert not r["ok"] and toks[r["token_index"]] in ("=", NEWLINE)


def test_make_prompt():
    ids = np.array(VOCAB.encode(TOKS))
    p = make_prompt(ids, IDS_OF, 5)
    assert VOCAB.decode(p)[-6] == INDENT and len(p) == TOKS.index(INDENT) + 1 + 5
    assert make_prompt(ids, IDS_OF, 500) is None
    assert make_prompt(np.array(VOCAB.encode(code_to_tokens("def f(): return 1\n"))), IDS_OF, 1) is None
    assert make_prompt(np.array(VOCAB.encode(code_to_tokens("def f():\n    return 1\n"))), IDS_OF, 4) is None  # function ends inside the prompt
    assert make_prompt(np.array(VOCAB.encode(code_to_tokens("def f():\n    return 1\n"))), IDS_OF, 2) is not None


def test_generate_stop_rules():
    V = len(VOCAB)
    bos, eos = VOCAB.stoi["<BOS>"], VOCAB.stoi["<EOS>"]
    prompt = np.array(VOCAB.encode(TOKS[:TOKS.index(INDENT) + 3]), dtype=np.uint16)  # depth 1

    def script(seq):
        it = iter(seq)

        def step(ctx):
            p = np.zeros((len(ctx), V))
            p[:, VOCAB.stoi[next(it)]] = 1.0
            return p
        return step

    rng = np.random.default_rng(0)
    toks, st, _ = generate(script(["a", INDENT, "b", DEDENT, DEDENT, "c"]), 3, [prompt], 20, rng, IDS_OF, bos, eos)
    assert st[0] == 1 and VOCAB.decode(toks[0])[-1] == DEDENT and len(toks[0]) == 5  # back to function level
    toks, st, _ = generate(script(["a", "b", "<EOS>", "c"]), 3, [prompt], 20, rng, IDS_OF, bos, eos)
    assert st[0] == 1 and len(toks[0]) == 3
    toks, st, _ = generate(script(["a"] * 10), 3, [prompt], 4, rng, IDS_OF, bos, eos)
    assert st[0] == 2 and len(toks[0]) == 4


def test_categories_and_bootstrap():
    v = Vocab.build([["def", "x", "(", NEWLINE, STR, "self"]], 100)[0]
    cat = metrics.category_ids(v)
    name = lambda t: metrics.CATEGORIES[cat[v.stoi[t]]]
    assert [name(t) for t in ("def", "x", "(", NEWLINE, STR, UNK, "<EOS>")] == \
        ["keyword", "identifier", "operator/punct", "layout", "literal", "unk", "eos"]
    rng = np.random.default_rng(0)
    ok = np.array([[1, 1, 0, 0] * 50] * 2, dtype=bool)
    rate, ci = metrics.bootstrap_rate(ok, np.ones_like(ok), rng)
    assert rate == 0.5 and ci[0] < 0.5 < ci[1]
    s, n = np.array([10.0, 20, 30]), np.array([5, 5, 5])
    lo, hi = metrics.bootstrap_ppl(s, n, rng)
    assert lo < np.exp(60 / 15) < hi
