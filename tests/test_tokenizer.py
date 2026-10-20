import ast

import pytest

from src.data.tokenizer import (DEDENT, INDENT, NEWLINE, NUM, STR, UNK, Vocab,
                                code_to_tokens, tokens_to_code)

CASES = [
    'def f(x):\n    """doc"""\n    return x + 1\n',
    'def f():\n    "only a docstring"\n',
    'def f(): "doc"\n',
    'def f(a, *b, **c) -> int:\n    # comment\n    y = f"{a!r:>{10}}" + \'x\' \'y\'\n    return [i ** 2 for i in (\n        1, 2)]\n',
    'class A:\n    def m(self):\n        """über"""\n        class B:\n            "doc"\n        return B\n',
    '@deco(1)\nasync def g():\n    async with a as b:\n        await b\n',
    '    def method(self):\n        return self.x\n'.strip() + "\n",
]


@pytest.mark.parametrize("src", CASES)
def test_roundtrip_parses(src):
    ast.parse(tokens_to_code(code_to_tokens(src)))


def test_docstrings_comments_literals():
    toks = code_to_tokens(CASES[0])
    assert STR not in toks and "#" not in " ".join(toks)
    assert toks[-4:] == ["+", NUM, NEWLINE, DEDENT]
    assert code_to_tokens(CASES[1]) == ["def", "f", "(", ")", ":", NEWLINE, INDENT, "pass", NEWLINE, DEDENT]
    assert code_to_tokens(CASES[2]) == ["def", "f", "(", ")", ":", "pass", NEWLINE]


def test_fstring_is_one_str():
    toks = code_to_tokens('def f(x):\n    return f"a{x}b{f\'{x}\'}"\n')
    assert toks.count(STR) == 1 and "x" not in toks[toks.index("return"):]


def test_unk_roundtrip():
    toks = code_to_tokens(CASES[0])
    vocab = Vocab.build([["def", "(", ")", ":", NEWLINE, INDENT, DEDENT, "return", "+", NUM]], 100)[0]
    ids = vocab.encode(toks)
    assert vocab.unk_id in ids
    ast.parse(tokens_to_code(vocab.decode(ids)))
