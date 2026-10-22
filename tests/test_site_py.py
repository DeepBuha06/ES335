"""The Pyodide copies in site/py must behave exactly like src/."""
import importlib.util
import sys
from pathlib import Path

import pytest

from src.data.tokenizer import code_to_tokens, tokens_to_code_map
from src.eval.syntax import check_tokens

SITE = Path("site/py")


@pytest.fixture(scope="module")
def web():
    sys.path.insert(0, str(SITE))
    try:
        for name in ("tokenizer", "syntax"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("web_api", SITE / "web_api.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path.remove(str(SITE))
        for name in ("tokenizer", "syntax", "web_api"):
            sys.modules.pop(name, None)


def test_copies_identical(web):
    assert (SITE / "tokenizer.py").read_text() == Path("src/data/tokenizer.py").read_text()
    assert (SITE / "syntax.py").read_text() == Path("src/eval/syntax.py").read_text().replace("src.data.tokenizer", "tokenizer")


def test_check_and_prompt_agree_with_src(web):
    code = 'def f(a, b):\n    """doc"""\n    x = g(a, [1, 2])\n    return x\n'
    toks = code_to_tokens(code)
    assert web.check_tokens(toks) == check_tokens(toks)
    bad = toks[:-6] + ["(", "(", "<NEWLINE>"]
    assert web.check_tokens(bad) == check_tokens(bad)
    assert web.prompt_to_tokens(code)[:5] == toks[:5]


def test_incomplete_prompt(web):
    t = web.prompt_to_tokens("def f(x):\n    y = foo(x, ")
    assert t[:6] == ["def", "f", "(", "x", ")", ":"] and t[-1] == "," and "<DEDENT>" not in t[-2:]
    t = web.prompt_to_tokens("def f(x):\n    if x:\n        return 1\n")
    assert t[-1] == "<NEWLINE>"
