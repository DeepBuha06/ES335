"""Copy the exact tokenizer and syntax checker into site/py for Pyodide (one source of truth).
tests/test_site_py.py checks the copies behave identically to src/."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "site" / "py"
out.mkdir(parents=True, exist_ok=True)
(out / "tokenizer.py").write_text((root / "src/data/tokenizer.py").read_text())
syn = (root / "src/eval/syntax.py").read_text().replace("from src.data.tokenizer import", "from tokenizer import")
assert "src." not in syn
(out / "syntax.py").write_text(syn)
print("wrote", sorted(p.name for p in out.glob("*.py")))
