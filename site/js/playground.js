// Playground: compare an n-gram and an MLP continuation of a test function, rendered as readable code.
// Preset samples come from explorer.json; an edited prompt is continued live by the MLP (live.js).
import { renderCode, PLACEHOLDER, toSource } from "./codeview.js";
import { MODEL_IDS, isNgram, modelColor, tooltip, fmt, choices, btn as button, field } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const show = (t) => PLACEHOLDER[t]?.text ?? ({ "<NEWLINE>": "line break", "<INDENT>": "indent", "<DEDENT>": "dedent" }[t] ?? t);
const PANEL = "#FFFFFF"; // confidence shading mixes the model colour into the white page
const rgb = (h) => { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
const lum = (c) => c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }).reduce((t, v, i) => t + v * [0.2126, 0.7152, 0.0722][i], 0);
export function shade(color, p) {
  const a = 0.08 + 0.72 * p, b = rgb(PANEL), c = rgb(color);
  const m = b.map((v, i) => Math.round(v + (c[i] - v) * a));
  return { bg: `rgb(${m.join(",")})`, fg: lum(m) > 0.3 ? "#16161A" : "#FFFFFF" };
}

/** Plain-English reason for a parser failure. */
export function explain(run) {
  if (run.status === "truncated") return { bad: false, text: "Did not finish within 256 tokens, so it is not counted." };
  const c = run.check;
  if (c.ok) return { bad: false, text: "Python accepts this code." };
  const m = c.msg || "";
  let r;
  if ((r = m.match(/closing parenthesis '(.)' does not match opening parenthesis '(.)'/))) return { bad: true, text: `Closes ‘${r[2]}’ with ‘${r[1]}’.` };
  if ((r = m.match(/unmatched '(.)'/))) return { bad: true, text: `Closes ‘${r[1]}’ that was never opened.` };
  if ((r = m.match(/'(.)' was never closed/))) return { bad: true, text: `Opens ‘${r[1]}’ and never closes it.` };
  if (c.category === "indentation") return { bad: true, text: "The indentation does not line up." };
  if (m.startsWith("expected ':'")) return { bad: true, text: "A ‘:’ is missing." };
  if (c.category === "incomplete statement") return { bad: true, text: "The code stops in the middle of a statement." };
  return { bad: true, text: "Python cannot read this line." };
}

function output(host, title, cls) {
  host.replaceChildren();
  const head = el("div", "head");
  const name = el("span", `name ${cls}`.trim());
  name.innerHTML = title;
  const verdict = el("span", "verdict");
  head.append(name, verdict);
  const body = el("div", "body"), why = el("div", "explain");
  host.append(head, body, why);
  return { body, why, verdict };
}

function renderRun(ui, promptTokens, run, color, tip) {
  const gen = run.tokens.filter((t) => t !== "<EOS>");
  const all = [...promptTokens, ...gen];
  const r = renderCode(all, { lineNumbers: true, markLayout: false });
  ui.body.replaceChildren(r.root);
  const n0 = promptTokens.length;
  gen.forEach((t, j) => {
    const s = r.spans[n0 + j];
    if (!s || s.classList.contains("cv-layout")) return;
    const p = run.p[j], top = run.top[j];
    const sh = shade(color, p);
    s.classList.add("gen");
    s.style.background = sh.bg;
    s.style.color = sh.fg;
    const html = `<b class="mono">${show(t)}</b> · the model gave it <span class="mono">${fmt.pct1(p)}</span><br>its top five guesses here:<br>` +
      top.map(([u, q]) => `<span class="mono">${show(u).padEnd(12, " ")} ${fmt.pct1(q)}</span>`).join("<br>");
    s.addEventListener("mousemove", (ev) => tip.show(html, ev));
    s.addEventListener("mouseleave", () => tip.hide());
  });
  const ex = explain(run);
  if (run.check?.ok === false && run.check.token_index != null) {
    const s = r.spans[run.check.token_index];
    if (s) { s.classList.add("err"); s.style.color = ""; }
  }
  ui.verdict.textContent = run.status === "truncated" ? "unfinished" : run.check.ok ? "valid Python" : "not valid Python";
  ui.verdict.classList.toggle("bad", ex.bad);
  ui.verdict.classList.toggle("open", run.status === "truncated");
  ui.why.textContent = ex.text;
  ui.why.classList.toggle("bad", ex.bad);
}

async function makeEditor(host, onChange) {
  const { EditorView, keymap, lineNumbers, EditorState, python, history, historyKeymap, defaultKeymap, HighlightStyle, syntaxHighlighting, tags } =
    await import("../vendor/cm.js");
  const style = HighlightStyle.define([{ tag: tags.keyword, fontWeight: "700" }, { tag: tags.comment, color: "#5C5C66" }]);
  let quiet = false;
  const view = new EditorView({
    parent: host,
    state: EditorState.create({
      doc: "",
      extensions: [lineNumbers(), history(), keymap.of([...defaultKeymap, ...historyKeymap]), python(), syntaxHighlighting(style), EditorView.lineWrapping,
        EditorView.updateListener.of((u) => u.docChanged && !quiet && onChange(u.state.doc.toString())),
        EditorView.theme({ "&": { color: "#16161A", backgroundColor: "#FFFFFF" }, ".cm-gutters": { backgroundColor: "#FFFFFF", color: "#5C5C66", border: "none" },
          ".cm-activeLine": { backgroundColor: "transparent" }, ".cm-activeLineGutter": { backgroundColor: "transparent" }, ".cm-cursor": { borderLeftColor: "#16161A" },
          ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "#FFE066" } }, { dark: false })],
    }),
  });
  return {
    set(text) { quiet = true; view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } }); quiet = false; },
    get: () => view.state.doc.toString(),
  };
}

export async function initPlayground(root, D) {
  const ex = D.explorer, pal = D.palette;
  const tip = tooltip(root);
  const st = { prompt: 0, ngram: "ngram5", mlp: "mlp20" };
  const name = (id) => (isNgram(id) ? `<span class="k-ngram">n-gram</span> order ${id.slice(5)}` : `<span class="k-mlp">MLP</span> context ${id.slice(3)}`);
  // function picker: the saved test prompts, grouped by seen / unseen repository
  const pick = el("select");
  pick.setAttribute("aria-label", "Test function");
  [["test_id", "From repositories seen in training"], ["test_ood", "From repositories never seen"]].forEach(([set, title]) => {
    const g = el("optgroup"); g.label = title;
    ex.prompts.forEach((p, i) => { if (p.set === set) g.appendChild(Object.assign(el("option", "", `${p.func_name} (${p.repository_name})`), { value: i })); });
    pick.appendChild(g);
  });
  pick.addEventListener("change", () => { st.prompt = +pick.value; show(); });
  const ngCh = choices(MODEL_IDS.filter(isNgram).map((id) => [id, id.slice(5)]), st.ngram, (v) => { st.ngram = v; show(); }, { kind: "ngram", label: "n-gram order" });
  const mlCh = choices(MODEL_IDS.filter((m) => !isNgram(m)).map((id) => [id, id.slice(3)]), st.mlp, (v) => { st.mlp = v; show(); }, { kind: "mlp", label: "MLP context" });
  // 1. what to compare
  const ctl = el("div", "fig-ctl pg-bar");
  const fnField = field("Test function", pick);
  fnField.classList.add("pg-fn");
  ctl.append(fnField, field("n-gram order", ngCh.el), field("MLP window, in tokens", mlCh.el));
  // 2. the start both models were given, as an editor
  const ed = el("div", "pg-ed");
  const edHead = el("div", "pg-ed-head");
  const edLab = el("span", "field-lab", "Start of the function, given to both models. Edit it to write your own.");
  const edTag = el("span", "pg-tag", "edited");
  edTag.hidden = true;
  const reset = button("Undo my edits", "small");
  reset.hidden = true;
  edHead.append(edLab, edTag, reset);
  const edHost = el("div", "cm-host");
  ed.append(edHead, edHost);
  // 3. run the MLP live on an edited start
  const live = el("div", "pg-live");
  const temp = el("select");
  temp.setAttribute("aria-label", "Temperature for live sampling");
  D.validity.config.temperatures.forEach((t) => temp.appendChild(Object.assign(el("option", "", fmt.t(t)), { value: t })));
  const btn = button("Continue with the MLP", "primary");
  btn.disabled = true;
  const note = el("p", "note pg-status", "");
  note.setAttribute("aria-live", "polite");
  live.append(field("Temperature", temp), btn, note);
  // 4. what each model wrote
  const pair = el("div", "pair");
  const outN = el("div", "out is-ngram"), outM = el("div", "out is-mlp");
  pair.append(outN, outM);
  const scale = el("div", "scale");
  scale.append(document.createTextNode("model was unsure"));
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => { const s = el("span", "step"); s.style.background = shade(pal.mlp[20], p).bg; scale.append(s); });
  scale.append(document.createTextNode("model was sure"));
  root.append(ctl, ed, live, pair, scale);

  let preset = "";
  const editor = await makeEditor(edHost, (text) => {
    const custom = text !== preset;
    btn.disabled = !custom;
    edTag.hidden = reset.hidden = !custom;
    if (custom) {
      note.textContent = "The n-gram samples exist only for the saved starts. Press the button to let the MLP continue your version.";
      const ui = output(outN, name(st.ngram), "");
      ui.body.appendChild(el("p", "note", "No n-gram sample for an edited start."));
    }
  });
  const maxNew = D.validity.config.max_new_tokens ?? 256;
  function show() {
    const p = ex.prompts[st.prompt];
    preset = toSource(p.prompt_tokens);
    editor.set(preset);
    btn.disabled = true;
    edTag.hidden = reset.hidden = true;
    pick.value = st.prompt;
    note.textContent = "Showing the samples saved by the test run. Edit the start to run the MLP yourself.";
    ngCh.set(st.ngram); mlCh.set(st.mlp);
    for (const [box, id] of [[outN, st.ngram], [outM, st.mlp]]) {
      const ui = output(box, name(id), "");
      renderRun(ui, p.prompt_tokens, ex.runs[id][st.prompt], modelColor(id, pal), tip);
    }
  }
  show();

  reset.addEventListener("click", () => show());
  let liveMod = null, seed = 0;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      liveMod ??= await import("./live.js");
      const id = st.mlp;
      const res = await liveMod.generate({ ctx: +id.slice(3), code: editor.get(), temperature: +temp.value, seed: seed++, maxNew, onStatus: (s) => (note.textContent = s) });
      const ui = output(outM, `${name(id)}, written live in your browser`, "");
      renderRun(ui, res.promptTokens, { tokens: res.tokens, p: res.p, top: res.top, status: res.status, check: res.check }, modelColor(id, pal), tip);
      note.textContent = `Written in your browser by the compressed model in ${fmt.int(res.ms)} ms.`;
    } catch (e) {
      note.textContent = `Live sampling failed: ${e.message}`;
    }
    btn.disabled = false;
  });
}
