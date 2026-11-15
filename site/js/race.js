// Hero, "The race": the n-gram and the MLP continue the same real test-function start, revealed token by token from
// the completions saved by the test run (explorer.json, the same samples as the Try it panel). Nothing is generated
// here. The yellow band is each model's context window while it writes; a magenta wavy underline marks the token
// where Python's parser gave up.
import { renderCode, markWindow } from "./codeview.js";
import { explain } from "./playground.js";
import { SET_LABEL } from "./util.js";

const RATE_MS = 83; // about 12 tokens per second
const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function initRace(root, D) {
  const ex = D.explorer, info = D.perplexity.model_info;
  const panes = [...root.querySelectorAll(".pane")].map((el) => ({
    el, id: el.dataset.model, code: el.querySelector(".pane-code"), status: el.querySelector(".pane-status"),
  }));
  const which = root.querySelector("#race-which");
  let fn = 0, timer = null;

  function draw(p, n, done) {
    const prompt = ex.prompts[fn].prompt_tokens, run = ex.runs[p.id][fn];
    const gen = run.tokens.filter((t) => t !== "<EOS>");
    const all = [...prompt, ...gen.slice(0, n)];
    const r = renderCode(all, { lineNumbers: false, markLayout: false });
    r.spans.forEach((s, i) => s && i < prompt.length && s.classList.add("given"));
    if (!done) markWindow(r.spans, all.length - info[p.id].context, all.length);
    const bad = run.check?.ok === false ? run.check.token_index : null;
    if (bad != null && bad < all.length) r.spans[bad]?.classList.add("reject");
    p.code.replaceChildren(r.root);
    if (done) {
      const target = bad != null ? r.spans[bad] : null;
      p.code.scrollTop = target ? target.offsetTop - p.code.offsetTop - p.code.clientHeight / 2 : p.code.scrollHeight;
      const ex2 = explain(run);
      p.status.textContent = run.status === "truncated" ? ex2.text : run.check.ok ? "Valid Python." : `Python rejects this. ${ex2.text}`;
      p.status.className = `pane-status ${run.status === "truncated" ? "" : run.check.ok ? "good" : "bad"}`;
    } else {
      p.code.scrollTop = p.code.scrollHeight;
      p.status.textContent = "Writing…";
      p.status.className = "pane-status";
    }
    return gen.length;
  }

  function play() {
    clearInterval(timer);
    const pr = ex.prompts[fn];
    which.innerHTML = `Function ${fn + 1} of ${ex.prompts.length}: <code></code>, from ${SET_LABEL[pr.set] === "ID" ? "a repository seen in training" : "a repository neither model saw"}.`;
    which.querySelector("code").textContent = pr.func_name;
    const len = panes.map((p) => ex.runs[p.id][fn].tokens.filter((t) => t !== "<EOS>").length);
    if (reduced()) { panes.forEach((p) => draw(p, Infinity, true)); return; }
    let n = 0;
    panes.forEach((p) => draw(p, 0, false));
    timer = setInterval(() => {
      n += 1;
      panes.forEach((p, k) => { if (n <= len[k]) draw(p, n, n === len[k]); });
      if (n >= Math.max(...len)) clearInterval(timer);
    }, RATE_MS);
  }

  root.querySelector("#race-replay").addEventListener("click", play);
  root.querySelector("#race-next").addEventListener("click", () => { fn = (fn + 1) % ex.prompts.length; play(); });
  play();
}
