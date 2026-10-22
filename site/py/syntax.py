"""Syntax check of generated token sequences and a small failure taxonomy."""
import ast
import warnings

from tokenizer import BOS, DEDENT, EOS, INDENT, NEWLINE, PAD, tokens_to_code_map

CATEGORIES = ["unbalanced brackets", "indentation", "incomplete statement", "other"]
_NOT_REAL = {NEWLINE, INDENT, DEDENT, BOS, EOS, PAD}


def classify(err, n_lines, at_last_token):
    msg = err.msg or ""
    if isinstance(err, IndentationError) or "indent" in msg:
        return "indentation"
    if any(s in msg for s in ("never closed", "unmatched", "does not match opening", "closing parenthesis")):
        return "unbalanced brackets"
    if msg.startswith("expected") or "unexpected EOF" in msg or at_last_token or (err.lineno or 0) > n_lines:
        return "incomplete statement"
    return "other"


def failing_token(tokens, spans, lineno, offset):
    """Index of the last real token starting at or before the (1-based) error position."""
    pos = (lineno - 1, max((offset or 1) - 1, 0))
    best = None
    for i, (t, s) in enumerate(zip(tokens, spans)):
        if t in _NOT_REAL:
            continue
        if s <= pos:
            best = i
        else:
            break
    return best


def check_tokens(tokens):
    """tokens: list of token strings. Returns dict(ok, ...); on failure also category, msg, token_index."""
    code, spans = tokens_to_code_map(tokens)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return {"ok": True}
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        if not isinstance(e, SyntaxError):
            return {"ok": False, "category": "other", "msg": type(e).__name__, "token_index": None}
        n_lines = code.count("\n")
        idx = failing_token(tokens, spans, e.lineno or n_lines, e.offset)
        real = [i for i, t in enumerate(tokens) if t not in _NOT_REAL]
        at_last = bool(real) and idx == real[-1]
        return {"ok": False, "category": classify(e, n_lines, at_last), "msg": e.msg, "token_index": idx,
                "lineno": e.lineno, "offset": e.offset}
