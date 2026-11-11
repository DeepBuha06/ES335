// Result visuals. Same colour mapping everywhere (palette.json): n-gram = greys, MLP = one blue ramp, invalid = red.
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";
import { MODEL_IDS, FAIL_CATS, isNgram, modelLabel, modelColor, tooltip, fmt } from "./util.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const NG = MODEL_IDS.filter(isNgram), ML = MODEL_IDS.filter((m) => !isNgram(m));
const ctxOf = (id) => (isNgram(id) ? +id.slice(5) - 1 : +id.slice(3));
const short = (id) => (isNgram(id) ? `order ${id.slice(5)}` : `context ${id.slice(3)}`);

function segmented(options, value, onChange, label) {
  const seg = el("div", "seg");
  seg.setAttribute("role", "group");
  if (label) seg.setAttribute("aria-label", label);
  const set = (v) => seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === String(v)));
  options.forEach(([v, text]) => {
    const b = el("button", "", text);
    b.type = "button";
    b.dataset.v = v;
    b.addEventListener("click", () => { set(v); onChange(v); });
    seg.appendChild(b);
  });
  set(value);
  return seg;
}

function responsive(host, draw) {
  let w = 0;
  const run = () => { const nw = host.clientWidth; if (nw > 0 && nw !== w) { w = nw; draw(w); } };
  new ResizeObserver(run).observe(host);
  run();
  return () => draw(host.clientWidth);
}

function mark(g, id, pal, x, y, r = 5.5) {
  const c = modelColor(id, pal);
  return isNgram(id)
    ? g.append("rect").attr("x", x - r).attr("y", y - r).attr("width", 2 * r).attr("height", 2 * r).attr("fill", c).attr("stroke", pal.panel).attr("stroke-width", 2)
    : g.append("circle").attr("cx", x).attr("cy", y).attr("r", r).attr("fill", c).attr("stroke", pal.panel).attr("stroke-width", 2);
}

// ---------------------------------------------------------------- perplexity vs context
export function pplChart(fig, D) {
  const pal = D.palette, S = D.perplexity.summary;
  const box = el("div", "chart");
  fig.prepend(box);
  const ctl = el("div", "ctl");
  let split = "test_ood";
  ctl.append(el("span", "lab", "Functions from"), segmented([["test_ood", "unseen repositories"], ["test_id", "seen repositories"]], split, (v) => { split = v; redraw(); }, "Split"));
  fig.prepend(ctl);
  const tip = tooltip(box);
  const all = MODEL_IDS.flatMap((m) => ["test_id", "test_ood"].map((s) => S[m][s].ppl_mean));
  const redraw = responsive(box, (W) => {
    const H = Math.min(420, Math.max(300, W * 0.6)), m = { l: 46, r: 92, t: 18, b: 44 };
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).insert("svg", ".tip").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const x = d3.scaleLog().domain([0.85, 24]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([Math.floor(d3.min(all) - 0.6), Math.ceil(d3.max(all) + 0.6)]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(6)).join("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([1, 2, 3, 4, 5, 10, 20]).tickFormat(d3.format("d")).tickSize(0).tickPadding(10)).call((s) => s.select(".domain").remove());
    svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text("tokens of code the model can see");
    svg.append("text").attr("transform", `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("perplexity (lower is better)");
    // same-context guide
    svg.append("line").attr("x1", x(3)).attr("x2", x(3)).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", pal.rule);
    svg.append("text").attr("x", x(3) + 6).attr("y", m.t + 12).text("same 3 tokens");
    for (const [ids, lineColor, name] of [[NG, pal.ngram[5], "n-gram"], [ML, pal.mlp[10], "MLP"]]) {
      const pts = ids.map((id) => ({ id, x: ctxOf(id), v: S[id][split] }));
      svg.append("path").attr("d", d3.line().x((d) => x(d.x)).y((d) => y(d.v.ppl_mean))(pts)).attr("fill", "none").attr("stroke", lineColor).attr("stroke-width", 2.5).attr("stroke-linejoin", "round");
      pts.forEach((p) => {
        if (p.v.ppl_std > 0) svg.append("line").attr("x1", x(p.x)).attr("x2", x(p.x)).attr("y1", y(p.v.ppl_mean - p.v.ppl_std)).attr("y2", y(p.v.ppl_mean + p.v.ppl_std)).attr("stroke", modelColor(p.id, pal)).attr("stroke-width", 2);
        mark(svg, p.id, pal, x(p.x), y(p.v.ppl_mean));
        svg.append("circle").attr("cx", x(p.x)).attr("cy", y(p.v.ppl_mean)).attr("r", 16).attr("fill", "transparent")
          .on("mousemove", (ev) => tip.show(`<b>${modelLabel(p.id)}</b> · sees ${p.x} tokens<br>perplexity <span class="mono">${fmt.f2(p.v.ppl_mean)}</span>` +
            (p.v.n_seeds > 1 ? ` ± <span class="mono">${fmt.f2(p.v.ppl_std)}</span> over ${p.v.n_seeds} seeds` : ""), ev))
          .on("mouseleave", () => tip.hide());
      });
      const last = pts[pts.length - 1];
      svg.append("text").attr("class", "lab-strong").attr("x", x(last.x) + 12).attr("y", y(last.v.ppl_mean) + 4).text(name);
      svg.append("text").attr("x", x(last.x) + 12).attr("y", y(last.v.ppl_mean) + 20).text(fmt.f2(last.v.ppl_mean));
    }
    const m3 = S.mlp3[split].ppl_mean;
    svg.append("text").attr("x", x(3) + 10).attr("y", y(m3) + 18).text(fmt.f2(m3));
  });
}

// ---------------------------------------------------------------- valid code: 10x10 unit charts
export function waffles(fig, D) {
  const pal = D.palette, V = D.validity_by_model, temps = D.validity.config.temperatures;
  const key = (t) => `t${String(Math.round(10 * t)).padStart(2, "0")}`;
  let T = temps[0];
  const ctl = el("div", "ctl");
  ctl.append(el("span", "lab", "Temperature"), segmented(temps.map((t) => [t, fmt.t(t)]), T, (v) => { T = +v; update(); }, "Sampling temperature"));
  const rows = el("div", "waffle-rows");
  fig.prepend(ctl, rows);
  const cells = {};
  [["n-gram", NG], ["MLP", ML]].forEach(([name, ids]) => {
    const row = el("div", "waffle-row");
    row.appendChild(el("h4", "", name === "n-gram" ? "n-gram models (order 2 to 5)" : "MLP models (context 3 to 20)"));
    const wrap = el("div", "waffles");
    ids.forEach((id) => {
      const w = el("div", "waffle");
      const g = el("div", "grid10");
      g.setAttribute("role", "img");
      const sq = Array.from({ length: 100 }, () => g.appendChild(el("i")));
      const val = el("div", "w-val"), nm = el("div", "w-name", `${name} ${short(id)}`);
      w.append(g, val, nm);
      wrap.appendChild(w);
      cells[id] = { sq, val, g };
    });
    row.appendChild(wrap);
    rows.appendChild(row);
  });
  function update() {
    MODEL_IDS.forEach((id) => {
      const r = V[id][key(T)].all, n = Math.round(100 * r.rate);
      cells[id].sq.forEach((s, i) => { s.style.background = i < n ? modelColor(id, pal) : ""; });
      cells[id].val.textContent = fmt.pct0(r.rate);
      cells[id].g.setAttribute("aria-label", `${modelLabel(id)}: ${n} of 100 finished functions are valid Python`);
    });
  }
  update();
}

// ---------------------------------------------------------------- seen vs unseen repositories
export function slopeChart(fig, D) {
  const pal = D.palette, S = D.perplexity.summary, G = D.perplexity.gaps;
  const box = el("div", "chart");
  fig.prepend(box);
  const tip = tooltip(box);
  responsive(box, (W) => {
    const H = 440, m = { l: 70, r: 190, t: 34, b: 20 };
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).insert("svg", ".tip").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const xs = [m.l, Math.max(m.l + 140, W - m.r)];
    const vals = MODEL_IDS.flatMap((id) => [S[id].test_id.ppl_mean, S[id].test_ood.ppl_mean]);
    const y = d3.scaleLog().domain([d3.min(vals) * 0.95, d3.max(vals) * 1.03]).range([H - m.b, m.t]);
    [["seen repositories", xs[0]], ["unseen repositories", xs[1]]].forEach(([t, x]) => {
      svg.append("line").attr("x1", x).attr("x2", x).attr("y1", m.t - 6).attr("y2", H - m.b).attr("stroke", pal.hairline);
      svg.append("text").attr("class", "lab-strong").attr("x", x).attr("y", 16).attr("text-anchor", "middle").text(t);
    });
    // right-hand labels, pushed apart so they never overlap
    const labs = MODEL_IDS.map((id) => ({ id, y: y(S[id].test_ood.ppl_mean) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labs.length; i++) labs[i].y = Math.max(labs[i].y, labs[i - 1].y + 17);
    const labsL = MODEL_IDS.map((id) => ({ id, y: y(S[id].test_id.ppl_mean) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labsL.length; i++) labsL[i].y = Math.max(labsL[i].y, labsL[i - 1].y + 15);
    const strong = new Set(["ngram5", "mlp20"]);
    MODEL_IDS.forEach((id) => {
      const a = S[id].test_id.ppl_mean, b = S[id].test_ood.ppl_mean, c = modelColor(id, pal);
      const g = svg.append("g").attr("opacity", strong.has(id) ? 1 : 0.75);
      g.append("line").attr("x1", xs[0]).attr("x2", xs[1]).attr("y1", y(a)).attr("y2", y(b)).attr("stroke", c).attr("stroke-width", strong.has(id) ? 3 : 1.6);
      mark(g, id, pal, xs[0], y(a), 4.5);
      mark(g, id, pal, xs[1], y(b), 4.5);
      g.append("text").attr("x", xs[0] - 10).attr("y", labsL.find((l) => l.id === id).y + 4).attr("text-anchor", "end").text(fmt.f2(a));
      const L = labs.find((l) => l.id === id);
      const t = g.append("text").attr("x", xs[1] + 12).attr("y", L.y + 4).attr("class", strong.has(id) ? "lab-strong" : "");
      t.append("tspan").text(`${fmt.f2(b)}  `);
      t.append("tspan").text(`${fmt.spct1(G[id].gap)}  ${modelLabel(id)}`);
      g.on("mousemove", (ev) => tip.show(`<b>${modelLabel(id)}</b><br>seen repositories <span class="mono">${fmt.f2(a)}</span><br>unseen repositories <span class="mono">${fmt.f2(b)}</span><br>change <span class="mono">${fmt.spct1(G[id].gap)}</span> (95% CI ${fmt.f1(100 * G[id].ci[0])} to ${fmt.f1(100 * G[id].ci[1])}%)`, ev))
        .on("mouseleave", () => tip.hide());
    });
  });
}

// ---------------------------------------------------------------- what goes wrong: share of finished samples
export function failureBars(host, D) {
  const pal = D.palette, V = D.validity_by_model, T = D.validity.config.temperatures[0];
  const k = `t${String(Math.round(10 * T)).padStart(2, "0")}`;
  const wrap = el("div", "fb");
  wrap.appendChild(el("p", "note", `Finished samples at temperature ${fmt.t(T)}, by outcome:`));
  const cats = [["valid", null], ...FAIL_CATS.map((c) => [c, pal.fail[c]])];
  ["ngram5", "mlp20"].forEach((id) => {
    const r = V[id][k].all;
    const row = el("div", "fb-row", modelLabel(id));
    const bar = el("div", "fb-bar");
    cats.forEach(([c, col]) => {
      const n = c === "valid" ? r.valid : r.failures[c];
      if (!n) return;
      const i = el("i");
      i.style.width = `${(100 * n) / r.natural}%`;
      i.style.background = col ?? modelColor(id, pal);
      i.title = `${c}: ${fmt.int(n)} of ${fmt.int(r.natural)} (${fmt.pct1(n / r.natural)})`;
      bar.appendChild(i);
    });
    row.appendChild(bar);
    wrap.appendChild(row);
  });
  const lg = el("div", "fb-legend");
  cats.forEach(([c, col]) => {
    const s = el("span");
    s.innerHTML = `<span class="swatch sq" style="background:${col ?? pal.ink_secondary}"></span>${c === "valid" ? "valid (model colour)" : c}`;
    lg.appendChild(s);
  });
  wrap.appendChild(lg);
  host.appendChild(wrap);
}

// ---------------------------------------------------------------- training curves
export function curvesChart(fig, D) {
  const pal = D.palette, T = D.training_curves;
  const box = el("div", "chart");
  fig.prepend(box);
  responsive(box, (W) => {
    const H = 300, m = { l: 50, r: 90, t: 10, b: 40 };
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const all = Object.values(T).flat();
    const x = d3.scaleLinear().domain([1, d3.max(all, (c) => c.epochs.at(-1))]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([d3.min(all, (c) => d3.min(c.val_loss)) - 0.03, d3.max(all, (c) => d3.max(c.val_loss)) + 0.02]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(12).tickFormat(d3.format("d")).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 4).attr("text-anchor", "middle").text("epoch");
    Object.entries(T).forEach(([c, runs]) => {
      runs.forEach((r, k) => svg.append("path").attr("d", d3.line().x((_, i) => x(r.epochs[i])).y((v) => y(v))(r.val_loss)).attr("fill", "none").attr("stroke", pal.mlp[c]).attr("stroke-width", 2).attr("opacity", k ? 0.5 : 1));
      const r0 = runs[0];
      svg.append("text").attr("x", x(r0.epochs.at(-1)) + 8).attr("y", y(r0.val_loss.at(-1)) + 4).text(`context ${c}`);
    });
  });
}

// ---------------------------------------------------------------- MLP diagram
export function mlpDiagram(host, D) {
  const info = D.perplexity.model_info.mlp20, V = D.phase1_report.vocab.size_incl_specials, pal = D.palette;
  const W = 600, H = 220;
  const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("role", "img")
    .attr("aria-label", "MLP: context tokens to embeddings to one hidden layer to probabilities over the vocabulary");
  const toks = ["self", ".", "name", "(", "x"];
  toks.forEach((t, i) => {
    const yy = 26 + i * 34;
    svg.append("rect").attr("x", 2).attr("y", yy - 13).attr("width", 64).attr("height", 24).attr("rx", 3).attr("fill", pal.chip);
    svg.append("text").attr("x", 34).attr("y", yy + 4).attr("text-anchor", "middle").attr("font-family", "IBM Plex Mono").attr("fill", pal.ink).text(t);
    svg.append("line").attr("x1", 68).attr("x2", 112).attr("y1", yy).attr("y2", yy).attr("stroke", pal.rule);
    for (let k = 0; k < 8; k++) svg.append("rect").attr("x", 116 + k * 6).attr("y", yy - 9).attr("width", 4).attr("height", 18).attr("fill", pal.mlp[10]).attr("opacity", 0.35 + 0.08 * ((i * 3 + k * 5) % 8));
    svg.append("line").attr("x1", 166).attr("x2", 262).attr("y1", yy).attr("y2", 94).attr("stroke", pal.rule);
  });
  svg.append("rect").attr("x", 264).attr("y", 14).attr("width", 24).attr("height", 160).attr("rx", 3).attr("fill", pal.mlp[20]).attr("opacity", 0.85);
  svg.append("line").attr("x1", 290).attr("x2", 350).attr("y1", 94).attr("y2", 94).attr("stroke", pal.rule);
  const shape = [0.42, 0.18, 0.09, 0.06, 0.04, 0.03, 0.02, 0.02, 0.015, 0.01, 0.01, 0.008, 0.006, 0.005]; // schematic, not data
  shape.forEach((p, i) => svg.append("rect").attr("x", 356 + i * 15).attr("y", 174 - p * 330).attr("width", 11).attr("height", p * 330).attr("fill", pal.mlp[20]));
  const lab = (x, y, t, a = "middle") => svg.append("text").attr("x", x).attr("y", y).attr("text-anchor", a).text(t);
  lab(2, 200, `last ${info.context} tokens`, "start");
  lab(141, 200, `${info.emb_dim} numbers each`);
  lab(276, 200, `${info.hidden} units`);
  lab(462, 200, `chance of each of ${fmt.int(V)} tokens`);
  lab(462, 216, "schematic", "middle").attr("font-size", 10);
}

// ---------------------------------------------------------------- data pipeline
export function pipeline(host, D) {
  const R0 = D.phase0_recon, R1 = D.phase1_report, pal = D.palette;
  const S = R1.splits, order = ["train", "val", "test_id", "test_ood"];
  const cols = { train: pal.ink_secondary, val: pal.rule, test_id: pal.mlp[5], test_ood: pal.mlp[20] };
  const names = { train: "train", val: "validation", test_id: "test, seen repos", test_ood: "test, unseen repos" };
  const tot = order.reduce((s, k) => s + S[k].n_functions, 0);
  const box = (num, lab, extra = "") => { const d = el("div", "pipe"); d.innerHTML = `<div class="p-num">${num}</div><div class="p-lab">${lab}</div>${extra}`; return d; };
  host.append(
    box(fmt.int(R0.n_functions_total), `Python functions in CodeSearchNet, from ${fmt.int(R0.n_repos_total)} repositories`),
    box(fmt.int(R1.sampling.n_sampled), `kept after removing unparseable, very long and duplicate functions, at most ${R1.sampling.max_per_repo} per repository`),
    box("4 splits", "validation and the unseen-repository test use new repositories. The seen-repository test holds out functions from training repositories",
      `<div class="p-split">${order.map((k) => `<i style="width:${(100 * S[k].n_functions) / tot}%;background:${cols[k]}" title="${names[k]}: ${fmt.int(S[k].n_functions)} functions"></i>`).join("")}</div>` +
      `<div class="p-lab">${order.map((k) => `<span class="swatch sq" style="background:${cols[k]}"></span>${names[k]}`).join("<br>")}</div>`),
    box(fmt.int(S.train.n_tokens), "training tokens; each function becomes a list like:",
      `<div class="p-code">def name ( self , key ) : ↵ ⇥ return self . data [ key ] ↵ ⇤</div>`),
    box(`${NG.length} + ${ML.length}`, `models: n-grams of order ${NG.map((id) => id.slice(5)).join(", ")} and MLPs with context ${ML.map((id) => id.slice(3)).join(", ")} (three seeds each)`),
  );
}

// ---------------------------------------------------------------- tables
const th = (s, cls = "") => `<th class="${cls}">${s}</th>`;
export function tablePerplexity(D) {
  const S = D.perplexity.summary, G = D.perplexity.gaps;
  let h = `<table><thead><tr>${th("model")}${th("validation", "n")}${th("seen repos", "n")}${th("unseen repos", "n")}${th("change", "n")}</tr></thead><tbody>`;
  MODEL_IDS.forEach((id) => {
    const f = (s) => `${fmt.f2(S[id][s].ppl_mean)}${S[id][s].n_seeds > 1 ? ` ± ${fmt.f2(S[id][s].ppl_std)}` : ""}`;
    h += `<tr><td>${modelLabel(id)}</td><td class="n">${f("val")}</td><td class="n">${f("test_id")}</td><td class="n">${f("test_ood")}</td><td class="n">${fmt.spct1(G[id].gap)}</td></tr>`;
  });
  return h + "</tbody></table>";
}
export function tableValidity(D) {
  const V = D.validity_by_model;
  let h = `<table><thead><tr>${th("model")}${th("T", "n")}${th("of finished", "n")}${th("of every sample", "n")}${th("95% CI", "n")}${th("finished", "n")}${th("cut off", "n")}</tr></thead><tbody>`;
  MODEL_IDS.forEach((id) => Object.values(V[id]).forEach((byT) => {
    const r = byT.all;
    const ofAll = r.completions ? r.valid / r.completions : 0;
    h += `<tr><td>${modelLabel(id)}</td><td class="n">${fmt.t(r.T)}</td><td class="n">${fmt.pct1(r.rate)}</td><td class="n">${fmt.pct1(ofAll)}</td><td class="n">${fmt.pct1(r.ci[0])}–${fmt.pct1(r.ci[1])}</td><td class="n">${fmt.int(r.natural)}</td><td class="n">${fmt.int(r.truncated)}</td></tr>`;
  }));
  return h + "</tbody></table>";
}
export function tableCategoryLoss(D) {
  const C = D.category_loss;
  const cats = ["identifier", "keyword", "operator/punct", "layout", "literal", "unk", "eos"];
  const ids = ["ngram5", "mlp3", "mlp20"];
  let h = `<table><thead><tr>${th("token kind, unseen repositories")}${ids.map((id) => th(modelLabel(id), "n")).join("")}</tr></thead><tbody>`;
  cats.forEach((c) => {
    h += `<tr><td>${c}</td>${ids.map((id) => `<td class="n">${fmt.f2(C[id].test_ood[c].mean_nll)}</td>`).join("")}</tr>`;
  });
  return h + `</tbody></table><p class="vis-cap">Mean next-token loss in nats on unseen repositories. A higher number means less probability on the true token.</p>`;
}
