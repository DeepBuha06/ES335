// "Writing code one token at a time": step through a real validation function and show, at every step, what each
// model can see (its context window) and its five most likely next tokens. Data: site/data/traces.json.
import { renderCode, PLACEHOLDER } from "./codeview.js";
import { modelColor, modelLabel, fmt } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const show = (t) => PLACEHOLDER[t]?.text ?? ({ "<NEWLINE>": "line break", "<INDENT>": "indent", "<DEDENT>": "dedent" }[t] ?? t);
const NGRAM = "ngram5";

export function initPredict(host, D) {
  const T = D.traces, pal = D.palette;
  const st = { f: 0, ctx: 20, pos: 0, playing: null };
  const mlps = [3, 5, 10, 20];

  // header: example picker + context switch
  const head = el("div", "pv-head");
  const pick = el("select");
  pick.setAttribute("aria-label", "Example function");
  T.functions.forEach((f, i) => pick.appendChild(Object.assign(el("option", "", `${f.func_name}  (${f.repository_name})`), { value: i })));
  const ctlCtx = el("div", "ctl");
  const lab = el("span", "lab", "MLP context");
  const seg = el("div", "seg");
  seg.setAttribute("role", "group");
  seg.setAttribute("aria-label", "MLP context length");
  mlps.forEach((c) => {
    const b = el("button");
    b.type = "button";
    const dot = el("i");
    dot.style.background = pal.mlp[c];
    dot.style.borderRadius = "50%";
    b.append(dot, document.createTextNode(`${c}`));
    b.addEventListener("click", () => { st.ctx = c; draw(); });
    b.dataset.c = c;
    seg.appendChild(b);
  });
  ctlCtx.append(lab, seg);
  head.append(pick, ctlCtx);

  const codeBox = el("div", "pv-code");
  const play = el("div", "pv-play");
  const bPrev = Object.assign(el("button", "btn icon", "←"), { type: "button", title: "Previous token" });
  const bPlay = Object.assign(el("button", "btn icon", "▶"), { type: "button", title: "Play / pause" });
  const bNext = Object.assign(el("button", "btn icon", "→"), { type: "button", title: "Next token" });
  const slider = Object.assign(el("input"), { type: "range", min: 0, value: 0 });
  slider.setAttribute("aria-label", "Position in the function");
  const posLab = el("span", "pv-pos");
  play.append(bPrev, bPlay, bNext, slider, posLab);

  const verdict = el("p", "pv-verdict");
  const bars = el("div", "pv-bars");
  const colNg = el("div"), colMl = el("div");
  bars.append(colNg, colMl);
  const score = el("div", "pv-score");
  host.append(head, codeBox, play, verdict, bars, score);

  let spans = [], bos = null, fnRendered = -1;
  function renderFn() {
    const f = T.functions[st.f];
    codeBox.replaceChildren();
    bos = el("span", "bos", "start of function");
    bos.title = "padding token: the window reaches back before the first token";
    codeBox.appendChild(bos);
    const r = renderCode(f.tokens, { lineNumbers: true, markLayout: true });
    codeBox.appendChild(r.root);
    spans = r.spans;
    slider.max = f.tokens.length - 1;
    fnRendered = st.f;
  }

  function barCol(col, id, k) {
    const f = T.functions[st.f], m = f.models[id], truth = f.tokens[k];
    col.replaceChildren();
    const h = el("h4");
    const sw = el("span", `swatch ${id.startsWith("ngram") ? "sq" : "ci"}`);
    sw.style.background = modelColor(id, pal);
    sw.style.margin = "0";
    const sees = id.startsWith("ngram") ? +id.slice(5) - 1 : +id.slice(3);
    h.append(sw, document.createTextNode(`${modelLabel(id)} · sees ${sees} tokens`));
    col.appendChild(h);
    let hit = false;
    m.top[k].forEach(([tok, p]) => {
      const row = el("div", "bar-row" + (tok === truth ? " true" : ""));
      if (tok === truth) hit = true;
      const fill = el("div", "fill");
      fill.style.width = `${(100 * p).toFixed(1)}%`;
      fill.style.background = modelColor(id, pal);
      const track = el("div", "track");
      track.appendChild(fill);
      row.append(el("span", "lbl", show(tok)), track, el("span", "val", `${(100 * p).toFixed(0)}%`));
      row.title = tok === truth ? "the token that actually comes next" : "";
      col.appendChild(row);
    });
    if (!hit) col.appendChild(el("div", "bar-miss", `real next token ${show(truth)}: ${(100 * m.p_true[k]).toFixed(1)}%, not in the top five`));
  }

  function draw() {
    const f = T.functions[st.f];
    if (fnRendered !== st.f) renderFn();
    const k = st.pos, mlp = `mlp${st.ctx}`;
    codeBox.style.setProperty("--win", pal.mlp[st.ctx]);
    spans.forEach((s, i) => {
      if (!s) return;
      s.classList.toggle("future", i > k);
      s.classList.toggle("target", i === k);
      s.classList.toggle("in-mlp", i < k && i >= k - st.ctx);
      s.classList.toggle("in-ng", i < k && i >= k - 4);
    });
    bos.classList.toggle("in-mlp", k - st.ctx < 0);
    const t = spans[k];
    if (t) { // keep the current token visible inside the code box without scrolling the page
      const r = t.getBoundingClientRect(), b = codeBox.getBoundingClientRect();
      if (r.top < b.top + 8 || r.bottom > b.bottom - 8) codeBox.scrollTop += r.top - b.top - b.height / 2;
    }
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.c === st.ctx));
    slider.value = k;
    posLab.textContent = `token ${k + 1} of ${f.tokens.length}`;
    const ord = (n) => (n === 1 ? "1st" : n === 2 ? "2nd" : n === 3 ? "3rd" : `${n}th`);
    const part = (id) => {
      const m = f.models[id], r = m.rank[k];
      return `${modelLabel(id)} ranked it <b>${ord(r)}</b> (${fmt.pct0(m.p_true[k])})${r === 1 ? ", its top guess" : ""}`;
    };
    verdict.innerHTML = `The real code continues with <code>${show(f.tokens[k])}</code>. The models do not write it here, they only guess: ` +
      `${part(NGRAM)}; ${part(mlp)}.`;
    barCol(colNg, NGRAM, k);
    barCol(colMl, mlp, k);
    // perplexity over the tokens so far: exp(mean surprise)
    const ppl = (id) => Math.exp(f.models[id].p_true.slice(0, k + 1).reduce((s, p) => s - Math.log(Math.max(p, 1e-6)), 0) / (k + 1));
    score.innerHTML = `<div>${modelLabel(NGRAM)}, perplexity on this function so far<b>${fmt.f2(ppl(NGRAM))}</b></div>` +
      `<div>${modelLabel(mlp)}, perplexity on this function so far<b>${fmt.f2(ppl(mlp))}</b></div>`;
  }

  const stop = () => { clearInterval(st.playing); st.playing = null; bPlay.textContent = "▶"; };
  const step = (d) => { st.pos = Math.max(0, Math.min(T.functions[st.f].tokens.length - 1, st.pos + d)); draw(); };
  bPrev.addEventListener("click", () => { stop(); step(-1); });
  bNext.addEventListener("click", () => { stop(); step(1); });
  bPlay.addEventListener("click", () => {
    if (st.playing) return stop();
    if (st.pos >= T.functions[st.f].tokens.length - 1) st.pos = 0;
    bPlay.textContent = "\u2016";
    st.playing = setInterval(() => { if (st.pos >= T.functions[st.f].tokens.length - 1) stop(); else step(1); }, 520);
  });
  slider.addEventListener("input", () => { stop(); st.pos = +slider.value; draw(); });
  pick.addEventListener("change", () => { stop(); st.f = +pick.value; st.pos = 0; draw(); });
  host.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { stop(); step(1); e.preventDefault(); }
    if (e.key === "ArrowLeft") { stop(); step(-1); e.preventDefault(); }
  });
  host.tabIndex = -1;
  st.pos = Math.min(18, T.functions[0].tokens.length - 1);
  draw();
}
