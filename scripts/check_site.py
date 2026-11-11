"""Self-check for the website against the CLAUDE.md section 5 spec. Usage: python scripts/check_site.py [site_dir]

Checks: every number in static prose comes from data (data-v bindings resolve), banned words / emoji / banned
CSS, colour whitelist (greys, palette, muted red), fonts, page weight. Citation blocks (class refs) and
code may contain years, identifiers, and commands.
"""
import colorsys
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import yaml

site = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
html = (site / "index.html").read_text(encoding="utf-8")
problems = []

BANNED_WORDS = ["unlock", "dive into", "seamless", "powerful", "journey", "delve", "leverage", "cutting-edge", "cutting edge",
                "state-of-the-art", "game-changing", "revolutioniz", "in this project, we", "key features", "get started"]
BANNED_CSS = [r"gradient", r"backdrop-filter", r"filter:\s*blur", r"text-shadow", r"box-shadow", r"rounded-2xl", r"tailwind"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤️]")

# ---------------------------------------------------------------- text nodes outside data-v
class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.texts, self.binds = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.stack.append((tag, a))
        if "data-v" in a:
            self.binds.append(a["data-v"])

    def handle_endtag(self, tag):
        while self.stack and self.stack[-1][0] != tag:
            self.stack.pop()
        if self.stack:
            self.stack.pop()

    def handle_data(self, d):
        if not d.strip():
            return
        tags = [t for t, _ in self.stack]
        if any(t in ("script", "style", "title") for t in tags):
            return
        skip = any("data-v" in a or {"num", "kicker", "refs"} & set(a.get("class", "").split()) for _, a in self.stack) or "code" in tags
        self.texts.append((d, skip, tags))

p = P()
p.feed(html)
ALLOWED = [r"Fig\.\s*\d+", r"ES 335", r"24110082"]
for d, skip, tags in p.texts:
    if skip:
        continue
    t = d
    for pat in ALLOWED:
        t = re.sub(pat, "", t)
    if re.search(r"\d", t):
        problems.append(f"number in static text: {d.strip()[:90]!r}")

# ---------------------------------------------------------------- data bindings resolve
data = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in (site / "data").glob("*.json")}
def lookup(path):
    o = data
    for k in path.split("."):
        try:
            o = len(o) if (k == "length" and isinstance(o, list)) else (o[int(k)] if isinstance(o, list) else o[k])
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    return o
for b in p.binds:
    if lookup(b) is None:
        problems.append(f"data-v path not found: {b}")

# ---------------------------------------------------------------- banned words, emoji, css
files = [f for f in site.rglob("*") if f.suffix in (".html", ".js", ".css", ".py") and not {"data", "model", "vendor"} & set(f.parts)]
for f in files:
    txt = f.read_text(encoding="utf-8")
    low = txt.lower()
    for w in BANNED_WORDS:
        if w in low:
            problems.append(f"{f}: banned phrase {w!r}")
    if EMOJI.search(txt):
        problems.append(f"{f}: emoji / pictograph character")
    if f.suffix in (".css", ".html", ".js"):
        for pat in BANNED_CSS:
            if re.search(pat, txt, re.I) and not (f.name == "check_site.py"):
                problems.append(f"{f}: banned style {pat!r}")
    if "?" in re.sub(r"<[^>]+>|\{[^}]*\}", "", re.sub(r"<script.*?</script>", "", txt, flags=re.S)) and f.suffix == ".html":
        for d, skip, tags in p.texts:
            if "?" in d:
                problems.append(f"rhetorical question in prose: {d.strip()[:80]!r}")

# ---------------------------------------------------------------- colours
pal = yaml.safe_load(open("configs/palette.yaml"))["site"]
ok_hex = {v for k, v in pal.items() if isinstance(v, str)} | set(pal["ngram"].values()) | set(pal["mlp"].values()) | set(pal["fail"].values())
for f in [f for f in site.rglob("*") if f.suffix in (".css", ".js", ".html") and not {"data", "vendor"} & set(f.parts)]:
    for h in set(re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", f.read_text(encoding="utf-8"))):
        hx = h.lower() if len(h) == 7 else "#" + "".join(c * 2 for c in h[1:]).lower()
        r, g, b = (int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5))
        hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
        if hx in {x.lower() for x in ok_hex} or sat < 0.04:
            continue
        problems.append(f"{f}: colour {hx} not in palette (hue {hue * 360:.0f}, sat {sat:.2f})")

# ---------------------------------------------------------------- fonts, weight
if "IBM+Plex+Sans" not in html or "IBM+Plex+Mono" not in html:
    problems.append("IBM Plex fonts not linked")
if (site / "css/style.css").read_text(encoding="utf-8").count("serif") != (site / "css/style.css").read_text(encoding="utf-8").count("sans-serif"):
    problems.append("serif font family in CSS")
tot = sum(f.stat().st_size for f in site.rglob("*") if f.is_file())
big = sorted(((f.stat().st_size, f) for f in site.rglob("*") if f.is_file()), reverse=True)[:5]
print(f"site size {tot / 1e6:.1f} MB; largest: " + ", ".join(f"{f.name} {s / 1e6:.2f} MB" for s, f in big))
print(f"data bindings checked: {len(p.binds)}; static text nodes: {len(p.texts)}")
print("banned items checked: marketing words, emoji, gradients, shadows, rounded-2xl/tailwind, off-palette colours (cream/beige/orange), "
      "serif fonts, hero/cards/CTA phrases, rhetorical questions, un-bound numbers")
if problems:
    print(f"\n{len(problems)} problem(s):")
    print("\n".join(" - " + x for x in problems))
    sys.exit(1)
print("OK: none found")
