// "Why the code breaks": real failed samples with the opening bracket, the bracket the parser rejected, an arc between
// them, and a window slider: the yellow band is a window of that many tokens ending where the model wrote the bad
// bracket, so the reader can see whether the opening bracket was inside it. Data: failure_examples.json, explorer.json.
import { renderCode, bracketPair, markWindow } from "./codeview.js";
import { SET_LABEL, choices, stopLeft } from "./util.js";

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

export function initBrackets(fig, D) {
  const pal = D.palette, info = D.perplexity.model_info;
  const models = ["ngram5", "mlp20"];
  const sets = Object.fromEntries(models.map((m) => [m, usable(D, m)]));
  const maxDist = Math.max(...Object.values(sets).flat().map(({ p, end }) => end - p.open));
  const W_MAX = Math.min(60, Math.max(30, maxDist + 4));
  const st = { model: sets.ngram5.length ? "ngram5" : "mlp20", k: 0, w: 0 };
  const name = (m) => (m.startsWith("ngram") ? `n-gram, order ${m.slice(5)}` : `MLP, context ${m.slice(3)}`);

  const modelCh = choices(models.map((m) => [m, name(m)]), st.model, (m) => { st.model = m; st.k = 0; load(); }, { label: "Model" });
  modelCh.el.querySelectorAll("button").forEach((b) => { b.classList.add(b.dataset.opt.startsWith("ngram") ? "is-ngram" : "is-mlp"); b.disabled = !sets[b.dataset.opt].length; });
  const prev = Object.assign(el("button", "btn", "Previous example"), { type: "button" });
  const next = Object.assign(el("button", "btn", "Next example"), { type: "button" });
  const count = el("span", "note");
  const nav = el("div", "br-nav");
  nav.append(prev, next, count);
  const head = el("div", "fig-ctl");
  const mf = el("div", "st-field"); mf.append(el("span", "", "Sample from"), modelCh.el);
  head.append(mf, nav);

  // window slider; two markers on its track: the model's real window (above) and the opening bracket (below)
  const slideRow = el("div", "br-slide");
  const sHead = el("div", "br-slide-head");
  const lab = el("label", "field-lab", "Window, in tokens");
  lab.htmlFor = "br-window";
  const wLab = el("span", "br-wlab");
  const reset = Object.assign(el("button", "btn small", ""), { type: "button" });
  sHead.append(lab, wLab, reset);
  const track = el("div", "br-track");
  const range = Object.assign(el("input"), { type: "range", min: 1, max: W_MAX, value: 1, id: "br-window" });
  const mkModel = el("span", "mk mk-model"), mkDist = el("span", "mk mk-dist");
  mkModel.appendChild(el("span")); mkDist.appendChild(el("span"));
  track.append(mkModel, mkDist, range);
  slideRow.append(sHead, track);
  const place = (mk, v, html) => {
    const frac = (Math.min(W_MAX, Math.max(1, v)) - 1) / (W_MAX - 1);
    mk.style.left = stopLeft(frac);
    mk.dataset.align = frac > 0.7 ? "end" : "start";
    mk.firstChild.innerHTML = html;
  };

  const code = el("div", "br-code");
  const read = el("p", "br-read");
  const about = el("p", "br-about");
  read.setAttribute("aria-live", "polite");
  fig.prepend(head, about, slideRow, code, read);
  range.setAttribute("aria-describedby", "br-read");
  read.id = "br-read";
  prev.addEventListener("click", () => { st.k = (st.k - 1 + sets[st.model].length) % sets[st.model].length; load(); });
  next.addEventListener("click", () => { st.k = (st.k + 1) % sets[st.model].length; load(); });
  range.addEventListener("input", () => { st.w = +range.value; mark(); });
  reset.addEventListener("click", () => { st.w = info[st.model].context; range.value = st.w; mark(); });
  new ResizeObserver(() => drawArc()).observe(code);

  let arc = null, cur = null;
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
    const yB = (r) => r.bottom - box.top + code.scrollTop + 1; // run just under the line, not through the text
    const xl = (r) => r.left - box.left + code.scrollLeft - 3;
    const gx = 4; // connector down the left margin, like an editor's bracket guide
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", pb ? `M${xl(pa)},${yB(pa)} H${gx} V${yB(pb)} H${xl(pb)}` : `M${xl(pa)},${yB(pa)} H${gx} V${code.scrollHeight - 8}`);
    p.setAttribute("fill", "none"); p.setAttribute("stroke", pal.ink_secondary); p.setAttribute("stroke-width", 1.5);
    svg.appendChild(p);
    code.appendChild(svg);
  }

  // new example: render the excerpt once, wide enough for the largest window
  function load() {
    const list = sets[st.model];
    modelCh.set(st.model);
    count.textContent = `${st.k + 1} of ${list.length}`;
    const { e, p, end } = list[st.k];
    const { toks, offset } = excerpt(e.tokens, Math.min(p.open, end - W_MAX), end);
    const r = renderCode(toks, { lineNumbers: false, markLayout: false });
    code.replaceChildren(r.root);
    const at = (i) => r.spans[i - offset];
    for (let i = end + 1; i - offset < toks.length; i++) at(i)?.classList.add("later");
    at(p.open)?.classList.add("open");
    if (p.close != null) at(p.close)?.classList.add("reject");
    arc = { a: at(p.open), b: p.close != null ? at(p.close) : null };
    cur = { e, p, end, at, offset, n: toks.length };
    st.w = info[st.model].context;
    range.value = st.w;
    reset.textContent = `Back to this model’s window (${info[st.model].context})`;
    const isNg = st.model.startsWith("ngram");
    mkModel.classList.toggle("is-ngram", isNg); mkModel.classList.toggle("is-mlp", !isNg);
    place(mkModel, info[st.model].context, `this model sees <b>${info[st.model].context}</b>`);
    place(mkDist, end - p.open, `opening bracket: <b>${end - p.open}</b> back`);
    const what = p.kind === "unclosed" ? `opens <code>${e.tokens[p.open]}</code> and never closes it`
      : `closes <code>${e.tokens[p.open]}</code> with <code>${e.tokens[p.close]}</code>`;
    about.innerHTML = `A real failed sample from the test run, prompt from ${SET_LABEL[e.set] === "ID" ? "a seen" : "an unseen"} repository. It ${what}. Python says: <i></i>.`;
    about.querySelector("i").textContent = e.msg;
    mark();
    requestAnimationFrame(drawArc);
  }

  // window change: move the highlighter only
  function mark() {
    const { p, end, at, offset, n } = cur;
    for (let i = offset; i < offset + n; i++) at(i)?.classList.remove("hl", "hl-join");
    const spans = Array.from({ length: n }, (_, j) => at(offset + j));
    markWindow(spans, end - st.w - offset, end - offset);
    const dist = end - p.open, inside = dist <= st.w;
    wLab.textContent = `${st.w} token${st.w === 1 ? "" : "s"}`;
    wLab.classList.toggle("in", inside);
    read.innerHTML = `The opening bracket is <b>${dist} tokens</b> back from where the model wrote ${p.close != null ? "the wrong bracket" : "its last token"}. ` +
      (inside ? `A window of ${st.w} tokens <b>includes</b> it.` : `A window of ${st.w} tokens <b>does not reach</b> it.`) +
      ` This model actually saw ${info[st.model].context} tokens` +
      (dist <= info[st.model].context ? ", so here it could see the opening bracket and still got it wrong." : ".");
  }
  load();
}
