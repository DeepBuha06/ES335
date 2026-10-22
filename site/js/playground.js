// Playground: compare an n-gram and an MLP continuation of a test function, rendered as readable code.
// Preset samples come from explorer.json; an edited prompt is continued live by the MLP (live.js).
import { renderCode, PLACEHOLDER, toSource } from "./codeview.js";
import { MODEL_IDS, isNgram, modelLabel, modelColor, tooltip, fmt } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const show = (t) => PLACEHOLDER[t]?.text ?? ({ "<NEWLINE>": "line break", "<INDENT>": "indent", "<DEDENT>": "dedent" }[t] ?? t);
const PANEL = "#131519";
const rgb = (h) => { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
const lum = (c) => c.map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }).reduce((t, v, i) => t + v * [0.2126, 0.7152, 0.0722][i], 0);
export function shade(color, p) {
  const a = 0.08 + 0.72 * p, b = rgb(PANEL), c = rgb(color);
  const m = b.map((v, i) => Math.round(v + (c[i] - v) * a));
  return { bg: `rgb(${m.join(",")})`, fg: lum(m) > 0.3 ? "#0c0d0f" : "#e6e8eb" };
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

function output(host, title, color, square) {
  host.replaceChildren();
  const head = el("div", "head");
  const sw = el("span", `swatch ${square ? "sq" : "ci"}`);
  sw.style.background = color;
  const verdict = el("span", "verdict");
  head.append(sw, document.createTextNode(title), verdict);
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
  ui.why.textContent = ex.text;
  ui.why.classList.toggle("bad", ex.bad);
}

async function makeEditor(host, onChange) {
  const { EditorView, keymap, lineNumbers, EditorState, python, history, historyKeymap, defaultKeymap, HighlightStyle, syntaxHighlighting, tags } =
    await import("../vendor/cm.js");
  const style = HighlightStyle.define([{ tag: tags.keyword, fontWeight: "600" }, { tag: tags.comment, color: "#9aa0a8" }]);
  let quiet = false;
  const view = new EditorView({
    parent: host,
    state: EditorState.create({
      doc: "",
      extensions: [lineNumbers(), history(), keymap.of([...defaultKeymap, ...historyKeymap]), python(), syntaxHighlighting(style), EditorView.lineWrapping,
        EditorView.updateListener.of((u) => u.docChanged && !quiet && onChange(u.state.doc.toString())),
        EditorView.theme({ "&": { color: "#e6e8eb", backgroundColor: "#0c0d0f" }, ".cm-gutters": { backgroundColor: "#0c0d0f", color: "#6c727b", border: "none" },
          ".cm-activeLine": { backgroundColor: "#1a1d22" }, ".cm-cursor": { borderLeftColor: "#e6e8eb" } }, { dark: true })],
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
  const side = el("div", "pg-side"), main = el("div", "pg-main");
  root.append(side, main);
  const items = [];
  [["test_id", "From seen repositories"], ["test_ood", "From unseen repositories"]].forEach(([set, title]) => {
    side.appendChild(el("div", "pg-group", title));
    ex.prompts.forEach((p, i) => {
      if (p.set !== set) return;
      const b = el("button", "pg-item");
      b.type = "button";
      b.append(el("span", "fn", p.func_name), el("span", "rp", p.repository_name));
      b.addEventListener("click", () => { st.prompt = i; show(); });
      side.appendChild(b);
      items[i] = b;
    });
  });
  const ctl = el("div", "ctl");
  const seg = (ids, key, label) => {
    const s = el("div", "seg");
    s.setAttribute("role", "group");
    s.setAttribute("aria-label", label);
    ids.forEach((id) => {
      const b = el("button");
      b.type = "button";
      const dot = el("i");
      dot.style.background = modelColor(id, pal);
      dot.style.borderRadius = isNgram(id) ? "1px" : "50%";
      b.append(dot, document.createTextNode(isNgram(id) ? id.slice(5) : id.slice(3)));
      b.dataset.id = id;
      b.addEventListener("click", () => { st[key] = id; show(); });
      s.appendChild(b);
    });
    return s;
  };
  ctl.append(el("span", "lab", "n-gram order"), seg(MODEL_IDS.filter(isNgram), "ngram", "n-gram order"),
    el("span", "lab", "MLP context"), seg(MODEL_IDS.filter((m) => !isNgram(m)), "mlp", "MLP context"));
  const edHost = el("div", "cm-host");
  const live = el("div", "pg-live");
  const note = el("p", "note", "Edit the start of the function above to let the MLP continue your version.");
  const temp = el("select");
  temp.setAttribute("aria-label", "Temperature for live sampling");
  D.validity.config.temperatures.forEach((t) => temp.appendChild(Object.assign(el("option", "", `temperature ${fmt.t(t)}`), { value: t })));
  const btn = Object.assign(el("button", "btn primary", "Continue with the MLP"), { type: "button", disabled: true });
  live.append(note, temp, btn);
  const pair = el("div", "pair");
  const outN = el("div", "out"), outM = el("div", "out");
  pair.append(outN, outM);
  const scale = el("div", "scale");
  scale.append(document.createTextNode("model was unsure"));
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => { const s = el("span", "step"); s.style.background = shade(pal.mlp[20], p).bg; scale.append(s); });
  scale.append(document.createTextNode("model was sure"));
  main.append(ctl, edHost, live, pair, scale);

  let preset = "";
  const editor = await makeEditor(edHost, (text) => {
    const custom = text !== preset;
    btn.disabled = !custom;
    if (custom) {
      note.textContent = "Edited: the n-gram samples exist only for the preset functions. Press the button to let the MLP continue your version.";
      outN.replaceChildren(el("p", "note", "No n-gram output for an edited start."));
    }
  });
  const maxNew = D.validity.config.max_new_tokens ?? 256;
  function show() {
    const p = ex.prompts[st.prompt];
    preset = toSource(p.prompt_tokens);
    editor.set(preset);
    btn.disabled = true;
    note.textContent = "Edit the start of the function above to let the MLP continue your version.";
    items.forEach((b, i) => b.setAttribute("aria-pressed", i === st.prompt));
    root.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.id === st.ngram || b.dataset.id === st.mlp));
    for (const [box, id] of [[outN, st.ngram], [outM, st.mlp]]) {
      const ui = output(box, modelLabel(id), modelColor(id, pal), isNgram(id));
      renderRun(ui, p.prompt_tokens, ex.runs[id][st.prompt], modelColor(id, pal), tip);
    }
  }
  show();

  let liveMod = null, seed = 0;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      liveMod ??= await import("./live.js");
      const id = st.mlp;
      const res = await liveMod.generate({ ctx: +id.slice(3), code: editor.get(), temperature: +temp.value, seed: seed++, maxNew, onStatus: (s) => (note.textContent = s) });
      const ui = output(outM, `${modelLabel(id)} · live`, modelColor(id, pal), false);
      renderRun(ui, res.promptTokens, { tokens: res.tokens, p: res.p, top: res.top, status: res.status, check: res.check }, modelColor(id, pal), tip);
      note.textContent = `Written in your browser by the compressed model in ${fmt.int(res.ms)} ms.`;
    } catch (e) {
      note.textContent = `Live sampling failed: ${e.message}`;
    }
    btn.disabled = false;
  });
}
