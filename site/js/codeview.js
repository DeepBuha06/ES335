// Render model tokens as readable Python: real line breaks and indentation, normal spacing, and small tags for
// placeholders. Every token keeps its own <span data-i> so visuals can highlight, shade or mark it.
const LAYOUT = new Set(["<NEWLINE>", "<INDENT>", "<DEDENT>"]);
const OPEN = new Set(["(", "[", "{"]), CLOSE = new Set([")", "]", "}"]);
const NO_SPACE_BEFORE = new Set([")", "]", "}", ",", ":", ".", ";"]);
const NO_SPACE_AFTER = new Set(["(", "[", "{", ".", "@", "~"]);
const isName = (t) => /^[A-Za-z_]\w*$/.test(t) || t === "<UNK>" || t === "<STR>" || t === "<NUM>";

export const PLACEHOLDER = {
  "<UNK>": { text: "name", cls: "ph unk", title: "an identifier that is not in the 10,004-token vocabulary" },
  "<STR>": { text: '"…"', cls: "ph str", title: "a string literal (all strings are replaced by one token)" },
  "<NUM>": { text: "0", cls: "ph num", title: "a number literal (all numbers are replaced by one token)" },
  "<EOS>": { text: "end", cls: "ph eos", title: "end of function" },
};

/** tokens -> lines: [{ indent, items: [{ i, text, cls, title, space }] }]; layout tokens are attached to lines. */
export function layoutTokens(tokens) {
  const lines = [];
  let cur = { indent: 0, items: [], layout: [] }, level = 0, depth = 0, prev = null;
  const push = () => { lines.push(cur); cur = { indent: level, items: [], layout: [] }; prev = null; };
  tokens.forEach((t, i) => {
    if (t === "<NEWLINE>") { cur.layout.push({ i, kind: "newline" }); push(); return; }
    if (t === "<INDENT>") { level += 1; cur.indent = level; cur.layout.push({ i, kind: "indent" }); return; }
    if (t === "<DEDENT>") { level = Math.max(0, level - 1); cur.indent = level; cur.layout.push({ i, kind: "dedent" }); return; }
    let space = prev !== null;
    if (NO_SPACE_BEFORE.has(t) || (prev !== null && NO_SPACE_AFTER.has(prev))) space = false;
    if ((t === "(" || t === "[") && prev !== null && (isName(prev) || CLOSE.has(prev)) && !["if", "in", "return", "and", "or", "not", "for", "else", "while", "yield"].includes(prev)) space = false;
    if ((t === "=" || prev === "=") && depth > 0) space = false; // keyword arguments: f(a=1)
    if (t === ":" ) space = false;
    const ph = PLACEHOLDER[t];
    cur.items.push({ i, text: ph ? ph.text : t, cls: ph ? ph.cls : "", title: ph?.title, space });
    if (OPEN.has(t)) depth++;
    if (CLOSE.has(t)) depth = Math.max(0, depth - 1);
    prev = t;
  });
  if (cur.items.length || cur.layout.length) lines.push(cur);
  return lines;
}

/** Build the DOM for a token list. Returns { root, spans } where spans[i] is the element of token i (layout tokens get
 *  small invisible markers so a highlight can still include them). */
export function renderCode(tokens, { lineNumbers = true, markLayout = true } = {}) {
  const root = document.createElement("div");
  root.className = "codeview";
  const spans = new Array(tokens.length).fill(null);
  layoutTokens(tokens).forEach((ln, k) => {
    const row = document.createElement("div");
    row.className = "cv-line";
    if (lineNumbers) {
      const no = document.createElement("span");
      no.className = "cv-no";
      no.textContent = String(k + 1);
      row.appendChild(no);
    }
    const body = document.createElement("span");
    body.className = "cv-body";
    body.style.paddingLeft = `${ln.indent * 2}ch`;
    ln.layout.filter((l) => l.kind !== "newline").forEach((l) => {
      const m = document.createElement("span");
      m.className = "cv-layout";
      m.dataset.i = l.i;
      m.title = l.kind === "indent" ? "indent token" : "dedent token";
      m.textContent = markLayout ? (l.kind === "indent" ? "⇥" : "⇤") : "";
      body.appendChild(m);
      spans[l.i] = m;
    });
    ln.items.forEach((it) => {
      if (it.space) body.appendChild(document.createTextNode(" "));
      const s = document.createElement("span");
      s.className = `tok ${it.cls}`.trim();
      s.dataset.i = it.i;
      s.textContent = it.text;
      if (it.title) s.title = it.title;
      body.appendChild(s);
      spans[it.i] = s;
    });
    ln.layout.filter((l) => l.kind === "newline").forEach((l) => {
      const m = document.createElement("span");
      m.className = "cv-layout nl";
      m.dataset.i = l.i;
      m.title = "line-break token";
      m.textContent = markLayout ? "↵" : "";
      body.appendChild(m);
      spans[l.i] = m;
    });
    row.appendChild(body);
    root.appendChild(row);
  });
  return { root, spans };
}

/** For a failing completion: the bracket the parser complained about and the opening bracket it pairs with. */
export function bracketPair(tokens, failAt) {
  const stack = [];
  const match = { ")": "(", "]": "[", "}": "{" };
  for (let i = 0; i <= Math.min(failAt, tokens.length - 1); i++) {
    const t = tokens[i];
    if (OPEN.has(t)) stack.push(i);
    else if (CLOSE.has(t)) {
      if (i === failAt) return { open: stack.length ? stack[stack.length - 1] : null, close: i, kind: stack.length && tokens[stack[stack.length - 1]] !== match[t] ? "mismatch" : "unmatched" };
      if (stack.length && tokens[stack[stack.length - 1]] === match[t]) stack.pop();
    }
  }
  return stack.length ? { open: stack[stack.length - 1], close: null, kind: "unclosed" } : null;
}

/** Count real tokens (not layout) between two positions. */
export function tokenDistance(tokens, a, b) {
  let n = 0;
  for (let i = a + 1; i <= b; i++) if (!LAYOUT.has(tokens[i])) n++;
  return n;
}

/** Readable Python source for a token list (placeholders become valid code), used to fill the editor. */
export function toSource(tokens) {
  const SUB = { "<UNK>": "unk_tok", "<STR>": '"s"', "<NUM>": "0", "<EOS>": "" };
  return layoutTokens(tokens).map((ln) => "    ".repeat(ln.indent) +
    ln.items.map((it, k) => (k && it.space ? " " : "") + (SUB[tokens[it.i]] ?? tokens[it.i])).join("")).join("\n");
}
