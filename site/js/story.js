// Results, "Guess the next token": step through a real validation function (traces.json). At every step both models
// are given the real code so far; the yellow band is the MLP's window, the orange underline the n-gram's, and the
// bars are each model's five most likely next tokens. "Guess first" hides the next token until the reader picks one.
import { renderCode, PLACEHOLDER, markWindow } from "./codeview.js";
import { modelColor, modelLabel, fmt, ctxState, choices } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const show = (t) => PLACEHOLDER[t]?.text ?? ({ "<NEWLINE>": "line break", "<INDENT>": "indent", "<DEDENT>": "dedent" }[t] ?? t);
const NGRAM = "ngram5";
const MLPS = ["mlp3", "mlp5", "mlp10", "mlp20"];
const ord = (n) => (n % 10 === 1 && n !== 11 ? `${n}st` : n % 10 === 2 && n !== 12 ? `${n}nd` : n % 10 === 3 && n !== 13 ? `${n}rd` : `${n}th`);

export function initPredict(host, D) {
  const T = D.traces, pal = D.palette, info = D.perplexity.model_info;
  const ngCtx = info[NGRAM].context;
  const st = { f: 0, pos: 0, playing: null, guess: false, picked: null };
  const mlpId = () => `mlp${ctxState.value}`;

  // controls
  const head = el("div", "st-head");
  const fnWrap = el("label", "st-field");
  const pick = el("select");
  T.functions.forEach((f, i) => pick.appendChild(Object.assign(el("option", "", `${f.func_name} (${f.repository_name})`), { value: i })));
  fnWrap.append(el("span", "", "Function"), pick);
  const ctxCh = choices(MLPS.map((m) => [info[m].context, `${info[m].context}`]), ctxState.value, (v) => ctxState.set(v), { kind: "mlp", label: "MLP window in tokens" });
  const ctxWrap = el("div", "st-field");
  ctxWrap.append(el("span", "", "MLP window"), ctxCh.el);
  const modeCh = choices([["watch", "Watch"], ["guess", "Guess first"]], "watch", (v) => { stop(); st.guess = v === "guess"; st.picked = null; draw(); }, { label: "Mode" });
  const modeWrap = el("div", "st-field");
  modeWrap.append(el("span", "", "Mode"), modeCh.el);
  head.append(fnWrap, ctxWrap, modeWrap);

  const codeBox = el("div", "st-code");
  const play = el("div", "st-play");
  const bPrev = Object.assign(el("button", "btn", "Previous"), { type: "button" });
  const bPlay = Object.assign(el("button", "btn primary", "Play"), { type: "button" });
  const bNext = Object.assign(el("button", "btn", "Next"), { type: "button" });
  const slider = Object.assign(el("input"), { type: "range", min: 0, value: 0 });
  slider.setAttribute("aria-label", "Position in the function");
  const posLab = el("span", "st-pos");
  play.append(bPrev, bPlay, bNext, slider, posLab);

  const guessBox = el("div", "st-guess");
  const verdict = el("p", "st-verdict");
  verdict.setAttribute("aria-live", "polite");
  const bars = el("div", "st-bars");
  const colNg = el("div"), colMl = el("div");
  bars.append(colNg, colMl);
  const score = el("div", "st-score");
  host.append(head, codeBox, play, guessBox, verdict, bars, score);

  let spans = [], bos = null, fnRendered = -1;
  function renderFn() {
    const f = T.functions[st.f];
    codeBox.replaceChildren();
    bos = el("span", "bos", "start of function");
    codeBox.appendChild(bos);
    const r = renderCode(f.tokens, { lineNumbers: true, markLayout: false });
    codeBox.appendChild(r.root);
    spans = r.spans;
    slider.max = f.tokens.length - 1;
    fnRendered = st.f;
  }

  function barCol(col, id, k) {
    const f = T.functions[st.f], m = f.models[id], truth = f.tokens[k];
    col.replaceChildren();
    const sees = info[id].context;
    const h = el("p", "st-col-head");
    h.innerHTML = `<span class="${id.startsWith("ngram") ? "k-ngram" : "k-mlp"}">${id.startsWith("ngram") ? "n-gram" : "MLP"}</span> sees ${sees} tokens. Its five best guesses:`;
    col.appendChild(h);
    let hit = false;
    m.top[k].forEach(([tok, p]) => {
      const row = el("div", "bar-row" + (tok === truth ? " true" : "") + (tok === st.picked ? " picked" : ""));
      if (tok === truth) hit = true;
      const fill = el("div", "fill");
      fill.style.width = `${(100 * p).toFixed(1)}%`;
      fill.style.background = modelColor(id, pal);
      const track = el("div", "track");
      track.appendChild(fill);
      row.append(el("span", "lbl", show(tok)), track, el("span", "val", `${(100 * p).toFixed(0)}%`));
      if (tok === truth) row.title = "the token that actually comes next";
      col.appendChild(row);
    });
    if (!hit) col.appendChild(el("p", "bar-miss", `The real next token, ${show(truth)}, got ${(100 * m.p_true[k]).toFixed(1)}%: not in its top five.`));
  }

  function draw() {
    const f = T.functions[st.f];
    if (fnRendered !== st.f) renderFn();
    const k = st.pos, ctx = ctxState.value, mlp = mlpId();
    const asking = st.guess && st.picked == null;
    ctxCh.set(ctx);
    spans.forEach((s, i) => {
      if (!s) return;
      s.classList.remove("hl", "hl-join", "ng", "target", "later", "masked", "veil");
      if (i > k) s.classList.add(asking ? "veil" : "later");
    });
    spans[k]?.classList.add(asking ? "masked" : "target");
    markWindow(spans, k - ctx, k);
    bos.classList.toggle("hl", k - ctx < 0);
    for (let i = Math.max(0, k - ngCtx); i < k; i++) spans[i]?.classList.add("ng");
    const t = spans[k];
    if (t && !t.classList.contains("cv-layout")) { // keep the current token in view inside the code box
      const r = t.getBoundingClientRect(), b = codeBox.getBoundingClientRect();
      if (r.top < b.top + 8 || r.bottom > b.bottom - 8) codeBox.scrollTop += r.top - b.top - b.height / 2;
    }
    slider.value = k;
    posLab.textContent = `token ${k + 1} of ${f.tokens.length}`;

    // guess-first: candidate tokens are both models' top five, in a neutral (alphabetical) order
    guessBox.replaceChildren();
    if (asking) {
      const cands = [...new Set([...f.models[NGRAM].top[k], ...f.models[mlp].top[k]].map(([tok]) => tok))].sort((a, b) => show(a).localeCompare(show(b)));
      guessBox.appendChild(el("p", "", "What comes next? Pick one:"));
      const row = el("div", "st-cands");
      cands.forEach((tok) => {
        const b = Object.assign(el("button", "cand", show(tok)), { type: "button" });
        b.addEventListener("click", () => { st.picked = tok; draw(); });
        row.appendChild(b);
      });
      guessBox.appendChild(row);
      verdict.textContent = "";
      bars.hidden = true; score.hidden = true;
      return;
    }
    bars.hidden = false; score.hidden = false;
    const truth = f.tokens[k];
    const part = (id) => {
      const m = f.models[id], r = m.rank[k];
      return `the <span class="${id.startsWith("ngram") ? "k-ngram" : "k-mlp"}">${id.startsWith("ngram") ? "n-gram" : "MLP"}</span> ranked it <b>${ord(r)}</b> (${fmt.pct0(m.p_true[k])})`;
    };
    let lead = "";
    if (st.guess) {
      lead = st.picked === truth ? "Right. " : `You picked <code>${show(st.picked)}</code>. `;
      if (![...f.models[NGRAM].top[k], ...f.models[mlp].top[k]].some(([tok]) => tok === truth)) lead += "The real token was in neither model’s top five. ";
    }
    verdict.innerHTML = `${lead}The real next token is <code>${show(truth)}</code>: ${part(NGRAM)}; ${part(mlp)}.`;
    barCol(colNg, NGRAM, k);
    barCol(colMl, mlp, k);
    // perplexity over the tokens so far: exp(mean surprise)
    const ppl = (id) => Math.exp(f.models[id].p_true.slice(0, k + 1).reduce((s, p) => s - Math.log(Math.max(p, 1e-6)), 0) / (k + 1));
    score.innerHTML = `<p>Perplexity on this function so far: <span class="k-ngram">n-gram</span> <b>${fmt.f2(ppl(NGRAM))}</b>, <span class="k-mlp">MLP</span> <b>${fmt.f2(ppl(mlp))}</b>. Lower is better.</p>`;
  }

  const stop = () => { clearInterval(st.playing); st.playing = null; bPlay.textContent = "Play"; };
  const step = (d) => { st.picked = null; st.pos = Math.max(0, Math.min(T.functions[st.f].tokens.length - 1, st.pos + d)); draw(); };
  bPrev.addEventListener("click", () => { stop(); step(-1); });
  bNext.addEventListener("click", () => { stop(); step(1); });
  bPlay.addEventListener("click", () => {
    if (st.playing) return stop();
    if (st.guess) { st.guess = false; modeCh.set("watch"); }
    if (st.pos >= T.functions[st.f].tokens.length - 1) st.pos = 0;
    bPlay.textContent = "Pause";
    st.playing = setInterval(() => { if (st.pos >= T.functions[st.f].tokens.length - 1) stop(); else step(1); }, 520);
  });
  slider.addEventListener("input", () => { stop(); st.picked = null; st.pos = +slider.value; draw(); });
  pick.addEventListener("change", () => { stop(); st.f = +pick.value; st.pos = 0; st.picked = null; draw(); });
  codeBox.tabIndex = 0;
  codeBox.setAttribute("aria-label", "Code; use the left and right arrow keys to step");
  codeBox.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { stop(); step(1); e.preventDefault(); }
    if (e.key === "ArrowLeft") { stop(); step(-1); e.preventDefault(); }
  });
  ctxState.on(() => draw());
  st.pos = Math.min(18, T.functions[0].tokens.length - 1);
  draw();
}
