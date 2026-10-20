"""Python-code tokenizer built on the stdlib `tokenize` module.

code_to_tokens: source -> list of token strings (docstrings stripped, COMMENT/NL dropped,
                string/number literals mapped to <STR>/<NUM>, layout kept as <NEWLINE>/<INDENT>/<DEDENT>).
tokens_to_code: token strings -> source that `ast.parse` can check.
"""
import ast
import io
import json
import tokenize
from collections import Counter

STR, NUM = "<STR>", "<NUM>"
NEWLINE, INDENT, DEDENT = "<NEWLINE>", "<INDENT>", "<DEDENT>"
UNK, BOS, EOS, PAD = "<UNK>", "<BOS>", "<EOS>", "<PAD>"
SPECIALS = [PAD, UNK, BOS, EOS]  # ids 0..3

SUBSTITUTE = {STR: '"s"', NUM: "0"}
UNK_NAME = "unk_tok"

_DROP = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
_LAYOUT = {tokenize.NEWLINE: NEWLINE, tokenize.INDENT: INDENT, tokenize.DEDENT: DEDENT}
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def _docstring_nodes(tree):
    """Yield (Expr node, is_only_statement) for every function/class docstring."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                yield first, len(node.body) == 1


def _char_col(line, byte_col):
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def _docstring_edits(code, toks, tree, stats):
    """Return (indices to drop, {index: replacement token}) that strip docstrings."""
    lines = code.split("\n")
    pos = {t.start: i for i, t in enumerate(toks)}
    drop, replace = set(), {}
    for node, only in _docstring_nodes(tree):
        start = (node.lineno, _char_col(lines[node.lineno - 1], node.col_offset))
        i = pos.get(start)
        # previous meaningful token must open the statement (else e.g. a parenthesised docstring)
        j = i - 1 if i is not None else None
        while j is not None and j >= 0 and toks[j].type in (tokenize.COMMENT, tokenize.NL):
            j -= 1
        ok_prev = j is not None and j >= 0 and (
            toks[j].type in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT)
            or (toks[j].type == tokenize.OP and toks[j].string == ":"))
        if i is None or not ok_prev:
            stats["docstring_skipped"] += 1
            continue
        k = i
        while toks[k].type != tokenize.NEWLINE:
            k += 1
        span = range(i, k)
        if any(toks[m].type == tokenize.OP and toks[m].string == ";" for m in span):
            stats["docstring_skipped"] += 1
            continue
        drop.update(span)
        if only:
            replace[i] = "pass"
            drop.discard(i)
            stats["pass_inserted"] += 1
        else:
            drop.add(k)  # the docstring's own NEWLINE
        stats["docstring_stripped"] += 1
    return drop, replace


def code_to_tokens(code, stats=None):
    """Tokenize one function. Raises SyntaxError / tokenize.TokenError on invalid code."""
    stats = stats if stats is not None else Counter()
    tree = ast.parse(code)
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    drop, replace = _docstring_edits(code, toks, tree, stats)
    out, fdepth = [], 0
    for i, t in enumerate(toks):
        if i in drop:
            continue
        if i in replace:
            out.append(replace[i])
            continue
        if _FSTRING_START is not None and t.type == _FSTRING_START:
            if fdepth == 0:
                out.append(STR)
            fdepth += 1
            continue
        if fdepth:
            fdepth -= t.type == _FSTRING_END
            continue
        if t.type in _DROP:
            continue
        if t.type in _LAYOUT:
            out.append(_LAYOUT[t.type])
        elif t.type == tokenize.STRING:
            out.append(STR)
        elif t.type == tokenize.NUMBER:
            out.append(NUM)
        else:
            out.append(t.string)
    return out


def tokens_to_code(tokens, indent="    "):
    """Rebuild parseable source; placeholders become "s" / 0 / unk_tok."""
    lines, cur, level = [], [], 0
    for t in tokens:
        if t == NEWLINE:
            lines.append(indent * level + " ".join(cur))
            cur = []
        elif t == INDENT:
            level += 1
        elif t == DEDENT:
            level = max(level - 1, 0)
        elif t in (BOS, EOS, PAD):
            continue
        else:
            cur.append(SUBSTITUTE.get(t, UNK_NAME if t == UNK else t))
    if cur:
        lines.append(indent * level + " ".join(cur))
    return "\n".join(lines) + "\n"


class Vocab:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]

    @classmethod
    def build(cls, token_lists, size):
        counts = Counter(t for toks in token_lists for t in toks)
        top = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if t not in SPECIALS][:size]
        return cls(SPECIALS + top), counts

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
