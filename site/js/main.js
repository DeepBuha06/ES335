import { loadData, bindNumbers, fmt, MODEL_IDS, isNgram } from "./util.js";
import * as V from "./viz.js";
import { initRace } from "./race.js";
import { initSees } from "./sees.js";
import { initPredict } from "./story.js";
import { initBrackets } from "./brackets.js";

const $ = (id) => document.getElementById(id);

function modelsTable(D) {
  const I = D.perplexity.model_info;
  const name = (id) => (isNgram(id) ? `<span class="k-ngram">n-gram</span> order ${id.slice(5)}` : `<span class="k-mlp">MLP</span> context ${id.slice(3)}`);
  return `<table><thead><tr><th>model</th><th class="n">sees</th><th class="n">size</th></tr></thead><tbody>` +
    MODEL_IDS.map((id) => `<tr><td>${name(id)}</td><td class="n">${I[id].context} tokens</td>` +
      `<td class="n">${isNgram(id) ? `${fmt.int(I[id].ngram_types)} n-grams` : `${fmt.int(I[id].n_params)} weights`}</td></tr>`).join("") +
    "</tbody></table>";
}

// 2px reading-progress line in the top bar
function wireProgress() {
  const bar = document.querySelector(".bar-progress i");
  let ticking = false;
  const update = () => {
    ticking = false;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = `${(100 * Math.min(1, window.scrollY / Math.max(1, max))).toFixed(2)}%`;
  };
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  window.addEventListener("resize", update);
  update();
}

// Sticky table of contents: underline the section being read and keep its link in view on narrow screens
function wireToc() {
  const toc = document.querySelector(".toc");
  if (!toc) return;
  const links = [...toc.querySelectorAll("a")];
  const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
  let ticking = false, cur = null;
  const update = () => {
    ticking = false;
    const line = toc.getBoundingClientRect().bottom + 24;
    let i = -1;
    secs.forEach((s, k) => { if (s && s.getBoundingClientRect().top <= line) i = k; });
    const a = links[i] || null;
    if (a === cur) return;
    if (cur) cur.classList.remove("on");
    cur = a;
    if (a) {
      a.classList.add("on");
      const l = a.offsetLeft, r = l + a.offsetWidth;
      if (l < toc.scrollLeft || r > toc.scrollLeft + toc.clientWidth) toc.scrollTo({ left: l - 16, behavior: "smooth" });
    }
  };
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
  window.addEventListener("resize", update);
  update();
}

// Reproduce: copy one command (the <code> next to the button) or all of them
function wireCopy() {
  const copy = async (b, text) => {
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
    catch { b.textContent = "Select and copy"; }
    setTimeout(() => (b.textContent = b.dataset.label), 1600);
  };
  document.querySelectorAll("[data-copy]").forEach((b) => {
    b.dataset.label = b.textContent;
    b.addEventListener("click", () => copy(b, b.parentElement.querySelector("code").textContent));
  });
  const all = document.getElementById("copy-all");
  all.dataset.label = all.textContent;
  all.addEventListener("click", () => copy(all, [...document.querySelectorAll(".repro .r-cmd code")].map((c) => c.textContent).join("\n")));
}

// Interactive figures: once built, everything but the caption goes in a framed box whose header says
// "Interactive" and what to do (the figure's data-how), so a demo never looks like the page around it.
function frameDemos() {
  document.querySelectorAll("figure[data-how]").forEach((fig) => {
    const box = document.createElement("div");
    box.className = "demo-box";
    const head = document.createElement("div");
    head.className = "demo-head";
    head.innerHTML = `<span class="demo-tag">Interactive</span><span class="demo-how"></span>`;
    head.lastChild.textContent = fig.dataset.how;
    box.appendChild(head);
    [...fig.childNodes].filter((n) => n.nodeName !== "FIGCAPTION").forEach((n) => box.appendChild(n));
    fig.prepend(box);
    fig.classList.add("demo");
  });
}

// run each figure on its own, so one failure cannot blank the rest of the page
function mount(name, fn) {
  try { fn(); } catch (e) { console.error(`${name} failed:`, e); }
}

async function main() {
  const D = await loadData();
  const missing = Object.entries(D).filter(([, v]) => v == null).map(([k]) => k);
  if (missing.length) console.warn("missing data files:", missing);
  bindNumbers(document, D);
  mount("race", () => initRace($("race"), D));
  mount("splits", () => { V.splitBar($("fig-splits"), D); $("tbl-splits").innerHTML = V.tableSplits(D); });
  mount("funnel", () => V.funnel($("fig-funnel"), D));
  mount("tokens", () => V.tokensFigure($("fig-tokens"), D));
  mount("sees", () => initSees($("sees"), D));
  mount("models", () => { $("tbl-models").innerHTML = modelsTable(D); V.curvesChart($("fig-curves"), D); });
  mount("stepper", () => initPredict($("pv"), D));
  mount("perplexity", () => { V.pplChart($("fig-ppl"), D); $("tbl-ppl").innerHTML = V.tablePerplexity(D); });
  mount("waffles", () => { V.waffles($("fig-waffle"), D); $("tbl-validity").innerHTML = V.tableValidity(D); });
  mount("slope", () => V.slopeChart($("fig-slope"), D));
  mount("failures", () => V.failureBars($("fig-fail"), D));
  mount("brackets", () => initBrackets($("fig-bracket"), D));
  mount("copy", wireCopy);
  mount("demos", frameDemos);
  wireProgress();
  wireToc();
  const pg = await import("./playground.js");
  await pg.initPlayground($("x-root"), D);
  document.dispatchEvent(new Event("site-ready"));
}

main().catch((e) => { console.error(e); document.body.insertAdjacentHTML("beforeend", `<p class="col note">Failed to load: ${e.message}</p>`); });
