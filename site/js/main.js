import { loadData, bindNumbers, fmt, MODEL_IDS, isNgram, modelLabel, modelColor } from "./util.js";
import * as V from "./viz.js";
import { initPredict } from "./story.js";
import { initBrackets } from "./brackets.js";

const $ = (id) => document.getElementById(id);

function tiles(D) {
  const S = D.perplexity.summary, G = D.perplexity.gaps, pal = D.palette, VB = D.validity_by_model;
  const T0 = D.validity.config.temperatures[0], k = `t${String(Math.round(10 * T0)).padStart(2, "0")}`;
  // Fixed pair: longest MLP window and highest n-gram order. Both were in the
  // pre-specified grid; the charts and tables still show every setting.
  const a = "mlp20", b = "ngram5";
  const defs = [
    { label: "Perplexity on unseen repositories (lower is better)", val: (id) => S[id].test_ood.ppl_mean, f: fmt.f2, scale: (v, m) => v / m },
    { label: `Written code that is valid Python, temperature ${fmt.t(T0)}`, val: (id) => VB[id][k].all.rate, f: fmt.pct1, scale: (v) => v },
    { label: "Perplexity increase on unseen repositories (lower is better)", val: (id) => G[id].gap, f: fmt.spct1, scale: (v, m) => v / m },
  ];
  defs.forEach((d) => {
    const va = d.val(a), vb = d.val(b), mx = Math.max(va, vb);
    const row = (id, v, minor) => `<div class="t-row${minor ? " minor" : ""}"><span class="swatch ${isNgram(id) ? "sq" : "ci"}" style="background:${modelColor(id, pal)}"></span><span class="t-name">${modelLabel(id)}</span><span class="t-val">${d.f(v)}</span></div>` +
      `<div class="t-bar"><i style="width:${(100 * d.scale(v, mx)).toFixed(1)}%;background:${modelColor(id, pal)}"></i></div>`;
    const t = document.createElement("div");
    t.className = "tile";
    t.innerHTML = `<div class="t-label">${d.label}</div>${row(a, va, false)}${row(b, vb, true)}`;
    $("tiles").appendChild(t);
  });
}

function splitsTable(D) {
  const S = D.phase1_report.splits;
  const rows = [
    ["train", "train", "fits the models and the vocabulary"],
    ["val", "validation", "held-out repositories; used for tuning"],
    ["test_id", "test, seen repos", "functions held out; repositories also used in training"],
    ["test_ood", "test, unseen repos", "repositories absent from training and validation"],
  ];
  return `<table><thead><tr><th>split</th><th>role</th><th class="n">functions</th><th class="n">repositories</th><th class="n">unknown-name share</th></tr></thead><tbody>` +
    rows.map(([k, n, note]) => `<tr><td>${n}</td><td class="muted">${note}</td><td class="n">${fmt.int(S[k].n_functions)}</td><td class="n">${fmt.int(S[k].n_repos)}</td><td class="n">${fmt.pct1(S[k].unk_rate)}</td></tr>`).join("") +
    "</tbody></table>";
}

function modelsTable(D) {
  const I = D.perplexity.model_info, pal = D.palette;
  return `<table><thead><tr><th>model</th><th class="n">sees</th><th class="n">size</th></tr></thead><tbody>` +
    MODEL_IDS.map((id) => `<tr><td><span class="swatch ${isNgram(id) ? "sq" : "ci"}" style="background:${modelColor(id, pal)}"></span>${modelLabel(id)}</td>` +
      `<td class="n">${I[id].context} tokens</td><td class="n">${isNgram(id) ? `${fmt.int(I[id].ngram_types)} n-grams` : `${fmt.int(I[id].n_params)} weights`}</td></tr>`).join("") +
    "</tbody></table>";
}

function wireScroll() {
  const spy = [...document.querySelectorAll("[data-spy]")];
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) spy.forEach((a) => a.classList.toggle("on", a.dataset.spy === e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  spy.forEach((a) => { const t = document.getElementById(a.dataset.spy); t && io.observe(t); });
}

async function main() {
  const D = await loadData();
  const missing = Object.entries(D).filter(([, v]) => v == null).map(([k]) => k);
  if (missing.length) console.warn("missing data files:", missing);
  bindNumbers(document, D);
  tiles(D);
  initPredict($("pv"), D);
  V.pplChart($("fig-ppl"), D);
  $("tbl-ppl").innerHTML = V.tablePerplexity(D);
  V.waffles($("fig-waffle"), D);
  $("tbl-validity").innerHTML = V.tableValidity(D);
  initBrackets($("fig-bracket"), $("bracket-cap"), D);
  V.failureBars($("fail-bars"), D);
  $("tbl-loss").innerHTML = V.tableCategoryLoss(D);
  V.slopeChart($("fig-slope"), D);
  V.pipeline($("pipeline"), D);
  V.mlpDiagram($("mlp-diagram"), D);
  $("tbl-splits").innerHTML = splitsTable(D);
  $("tbl-models").innerHTML = modelsTable(D);
  V.curvesChart($("fig-curves"), D);
  wireScroll();
  const pg = await import("./playground.js");
  await pg.initPlayground($("x-root"), D);
  document.dispatchEvent(new Event("site-ready"));
}

main().catch((e) => { console.error(e); document.body.insertAdjacentHTML("beforeend", `<p class="note">Failed to load: ${e.message}</p>`); });
