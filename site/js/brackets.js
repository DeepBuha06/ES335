// "Why the code breaks": real failed samples with the opening bracket, the bracket the parser rejected, an arc between
// them, and the model's context window at the moment it wrote the bad bracket. Data: failure_examples.json.
import { renderCode, bracketPair } from "./codeview.js";
import { modelColor, modelLabel, SET_LABEL } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const LAYOUT = new Set(["<NEWLINE>", "<INDENT>", "<DEDENT>"]);

// Candidate examples: every saved failed sample from the test run (failure_examples.json and the explorer samples)
// whose error is a bracket that we can pair with the bracket that opened it.
function usable(D, model) {
  const pool = D.failure_examples.examples.filter((e) => e.model === model);
  D.explorer.prompts.forEach((pr, i) => {
    const run = D.explorer.runs[model][i];
    if (run.check?.ok !== false) return;
    pool.push({ model, T: D.explorer.temperature, category: run.check.category, msg: run.check.msg, token_index: run.check.token_index,
      n_prompt_tokens: pr.prompt_tokens.length, set: pr.set, tokens: [...pr.prompt_tokens, ...run.tokens.filter((t) => t !== "<EOS>")] });
  });
  const seen = new Set();
  return pool.filter((e) => e.category === "unbalanced brackets" && e.token_index != null).map((e) => {
    const toks = e.tokens, at = e.token_index;
    let p = bracketPair(toks, at);
    if (["(", "[", "{"].includes(toks[at])) p = { open: at, close: null, kind: "unclosed" };
    if (!p || p.open == null || p.open < e.n_prompt_tokens - 2) return null;
    const end = p.close ?? toks.length - 1;
    if (end - p.open < 3) return null;
    const key = toks.slice(0, end + 1).join(" ");
    if (seen.has(key)) return null;
    seen.add(key);
    return { e, p, end };
  }).filter(Boolean).sort((a, b) => (a.p.kind === "mismatch" ? 0 : 1) - (b.p.kind === "mismatch" ? 0 : 1) || (b.end - b.p.open) - (a.end - a.p.open));
}

/** cut the token list to whole lines from a little before `from` to a little after `to`; keep indentation */
function excerpt(tokens, from, to) {
  let s = from;
  let lines = 0;
  while (s > 0 && lines < 2) { s--; if (tokens[s] === "<NEWLINE>") lines++; }
  if (s > 0) s++;
  let e = to;
  lines = 0;
  while (e < tokens.length - 1 && lines < 1) { e++; if (tokens[e] === "<NEWLINE>") lines++; }
  let level = 0;
  for (let i = 0; i < s; i++) level += tokens[i] === "<INDENT>" ? 1 : tokens[i] === "<DEDENT>" ? -1 : 0;
  const pre = Array(Math.max(0, level)).fill("<INDENT>");
  return { toks: [...pre, ...tokens.slice(s, e + 1)], offset: s - pre.length };
}

export function initBrackets(fig, cap, D) {
  const F = D.failure_examples, pal = D.palette;
  const models = ["ngram5", "mlp20"];
  const sets = Object.fromEntries(models.map((m) => [m, usable(D, m)]));
  const st = { model: sets.ngram5.length ? "ngram5" : "mlp20", k: 0 };
  const head = el("div", "br-head");
  const seg = el("div", "seg");
  models.forEach((m) => {
    const b = el("button");
    b.type = "button";
    const dot = el("i");
    dot.style.background = modelColor(m, pal);
    dot.style.borderRadius = m.startsWith("ngram") ? "1px" : "50%";
    b.append(dot, document.createTextNode(modelLabel(m)));
    b.dataset.m = m;
    b.disabled = !sets[m].length;
    b.addEventListener("click", () => { st.model = m; st.k = 0; draw(); });
    seg.appendChild(b);
  });
  const nav = el("div", "ctl");
  const prev = Object.assign(el("button", "btn icon", "←"), { type: "button", title: "Previous example" });
  const next = Object.assign(el("button", "btn icon", "→"), { type: "button", title: "Next example" });
  const count = el("span", "lab");
  nav.append(prev, count, next);
  head.append(seg, nav);
  const code = el("div", "br-code");
  const facts = el("div", "br-facts");
  fig.prepend(head, code, facts);
  prev.addEventListener("click", () => { st.k = (st.k - 1 + sets[st.model].length) % sets[st.model].length; draw(); });
  next.addEventListener("click", () => { st.k = (st.k + 1) % sets[st.model].length; draw(); });
  new ResizeObserver(() => drawArc()).observe(code);

  let arc = null;
  function drawArc() {
    if (!arc) return;
    code.querySelector("svg.arc")?.remove();
    const { a, b } = arc;
    if (!a) return;
    const box = code.getBoundingClientRect();
    const pa = a.getBoundingClientRect(), pb = b ? b.getBoundingClientRect() : null;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "arc");
    svg.setAttribute("width", code.scrollWidth);
    svg.setAttribute("height", code.scrollHeight);
    const yMid = (r) => r.bottom - box.top + code.scrollTop + 1; // run just under the line, not through the text
    const xl = (r) => r.left - box.left + code.scrollLeft - 3;
    const gx = 6; // connector runs down the left margin, like an editor's bracket guide
    const path = (d, color, w = 1.8) => { const p = document.createElementNS(ns, "path"); p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", color); p.setAttribute("stroke-width", w); svg.appendChild(p); };
    if (pb) path(`M${xl(pa)},${yMid(pa)} H${gx} V${yMid(pb)} H${xl(pb)}`, pal.ink_secondary);
    else path(`M${xl(pa)},${yMid(pa)} H${gx} V${code.scrollHeight - 10}`, pal.ink_secondary);
    code.appendChild(svg);
  }

  function draw() {
    const list = sets[st.model];
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.m === st.model));
    count.textContent = `example ${st.k + 1} of ${list.length}`;
    const { e, p, end } = list[st.k];
    const ctx = st.model.startsWith("ngram") ? +st.model.slice(5) - 1 : +st.model.slice(3);
    const { toks, offset } = excerpt(e.tokens, Math.min(p.open, end - ctx), end);
    const r = renderCode(toks, { lineNumbers: false, markLayout: false });
    code.replaceChildren(r.root);
    code.style.setProperty("--win", modelColor(st.model, pal));
    const at = (i) => r.spans[i - offset];
    for (let i = end - ctx; i < end; i++) at(i)?.classList.add("win");
    for (let i = end + 1; i - offset < toks.length; i++) at(i)?.classList.add("after");
    at(p.open)?.classList.add("open");
    if (p.close != null) at(p.close)?.classList.add("close");
    arc = { a: at(p.open), b: p.close != null ? at(p.close) : null };
    requestAnimationFrame(drawArc);
    const dist = end - p.open;
    const realDist = e.tokens.slice(p.open + 1, end + 1).filter((t) => !LAYOUT.has(t)).length;
    const what = p.kind === "unclosed" ? `opens <code>${e.tokens[p.open]}</code> and never closes it`
      : `closes <code>${e.tokens[p.open]}</code> with <code>${e.tokens[p.close]}</code>`;
    facts.innerHTML = `<div><b>${dist} tokens</b>between the opening bracket and the error</div>` +
      `<div><b>${ctx} tokens</b>visible to ${modelLabel(st.model)}</div>` +
      `<div><b>${dist > ctx ? "out of view" : "in view"}</b>the opening bracket, when the model wrote the error</div>`;
    cap.innerHTML = `${modelLabel(st.model)}, prompt from ${SET_LABEL[e.set] === "ID" ? "a seen" : "an unseen"} repository: the sample ${what}. ` +
      `Blue outline: opening bracket. Red outline: where Python gave up (${e.msg}). Shaded: what the model could see. ` +
      `${realDist} code tokens apart, not counting line breaks and indentation.`;
  }
  draw();
}
