"""Entry points called from the page (Pyodide). Uses the same tokenizer and syntax checker as the experiments."""
import io
import json
import tokenize

from syntax import check_tokens
from tokenizer import (DEDENT, NEWLINE, NUM, STR, _DROP, _FSTRING_END, _FSTRING_START, _LAYOUT, code_to_tokens)


def _loose_tokens(code):
    """Tokens of incomplete code: same mapping as code_to_tokens but no docstring stripping (needs a full parse)."""
    out, fdepth = [], 0
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
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
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def prompt_to_tokens(code):
    """Token strings for a prompt. The model continues from the end, so trailing DEDENTs (and a NEWLINE the
    tokenizer invents when the text does not end with a line break) are removed."""
    try:
        toks = code_to_tokens(code)
    except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
        toks = _loose_tokens(code)
    while toks and toks[-1] == DEDENT:
        toks.pop()
    if toks and toks[-1] == NEWLINE and not code.endswith("\n"):
        toks.pop()
    return toks


def prompt_json(code):
    return json.dumps(prompt_to_tokens(code))


def check_json(tokens_json):
    return json.dumps(check_tokens(json.loads(tokens_json)))
