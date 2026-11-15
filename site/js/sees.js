// Baseline & methods, "What each model sees": one real validation function (traces.json). The tick slider sets the
// MLP's window (yellow band); the n-gram's window is always the orange underline. Clicking a token moves the guess there.
import { renderCode, markWindow } from "./codeview.js";
import { MODEL_IDS, isNgram, tickSlider } from "./util.js";

export function initSees(root, D) {
  const f = D.traces.functions[0], info = D.perplexity.model_info;
  const ctxs = MODEL_IDS.filter((m) => !isNgram(m)).map((m) => info[m].context), ng = info.ngram5.context;
  const read = root.querySelector("#sees-read");
  const st = { c: ctxs.length - 1, k: 0 };
  // start the guess about two lines in, so even the longest window has code to cover
  st.k = f.tokens.findIndex((t, i) => i >= 2 * ctxs.at(-1) && !t.startsWith("<"));
  if (st.k < 0) st.k = Math.min(f.tokens.length - 1, 30);

  const slider = tickSlider(ctxs, st.c, (i) => { st.c = i; draw(); }, { kind: "mlp", label: "Size of the MLP's window, in tokens", text: (c) => `${c}` });
  slider.input.id = "sees-range";
  root.querySelector("#sees-slider").appendChild(slider.el);

  const box = root.querySelector("#sees-code");
  const bos = document.createElement("span");
  bos.className = "bos"; bos.textContent = "start of function";
  const r = renderCode(f.tokens, { lineNumbers: true, markLayout: false });
  box.append(bos, r.root);
  r.spans.forEach((s, i) => {
    if (!s || s.classList.contains("cv-layout")) return;
    s.addEventListener("click", () => { if (i > 0) { st.k = i; draw(); } });
  });
  const lineOf = (s) => [...r.root.children].indexOf(s.closest(".cv-line")) + 1;

  function draw() {
    const c = ctxs[st.c], k = st.k;
    r.spans.forEach((s, i) => { if (s) { s.classList.remove("hl", "hl-join", "ng", "target", "later"); if (i > k) s.classList.add("later"); } });
    r.spans[k]?.classList.add("target");
    const first = markWindow(r.spans, k - c, k);
    bos.classList.toggle("hl", k - c < 0);
    for (let i = Math.max(0, k - ng); i < k; i++) r.spans[i]?.classList.add("ng");
    const reach = k - c < 0 ? "reaching the start of the function" : `reaching back to line ${lineOf(first)}`;
    read.innerHTML =
      `<li><span class="key key-target" aria-hidden="true"></span><span>The token being guessed, on line ${lineOf(r.spans[k])}. Click any other token to guess there.</span></li>` +
      `<li><span class="key key-hl" aria-hidden="true"></span><span><span class="k-mlp">MLP</span>: the last <b>${c}</b> tokens, ${reach}.</span></li>` +
      `<li><span class="key key-ng" aria-hidden="true"></span><span><span class="k-ngram">n-gram</span>: the last <b>${ng}</b> tokens, whatever the slider says.</span></li>`;
  }
  draw();
}
