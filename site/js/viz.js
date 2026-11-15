// Figures. One colour per model family everywhere (css/style.css, via util.themePalette): n-gram orange, MLP blue.
// Charts label lines directly and draw short annotations instead of legends (DESIGN.md).
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm";
import { MODEL_IDS, FAIL_CATS, isNgram, modelLabel, modelColor, tooltip, fmt, ctxState, choices } from "./util.js";
import { renderCode } from "./codeview.js";

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const NG = MODEL_IDS.filter(isNgram), ML = MODEL_IDS.filter((m) => !isNgram(m));
const ctxOf = (id, D) => D.perplexity.model_info[id].context;
const family = (id) => (isNgram(id) ? `<span class="k-ngram">n-gram</span> order ${id.slice(5)}` : `<span class="k-mlp">MLP</span> context ${id.slice(3)}`);
const tkey = (t) => `t${String(Math.round(10 * t)).padStart(2, "0")}`;

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
    ? g.append("rect").attr("x", x - r).attr("y", y - r).attr("width", 2 * r).attr("height", 2 * r).attr("fill", c).attr("stroke", pal.background).attr("stroke-width", 2)
    : g.append("circle").attr("cx", x).attr("cy", y).attr("r", r).attr("fill", c).attr("stroke", pal.background).attr("stroke-width", 2);
}

function ctlRow(...items) {
  const row = el("div", "fig-ctl");
  items.forEach(([label, node]) => { const f = el("div", "st-field"); f.append(el("span", "", label), node); row.appendChild(f); });
  return row;
}

// ---------------------------------------------------------------- Data: the four splits, to scale
export function splitBar(host, D) {
  const S = D.phase1_report.splits, pal = D.palette;
  const order = [
    ["train", "training", pal.ink, pal.background],
    ["test_id", "test, seen repositories", pal.ink_secondary, pal.background],
    ["val", "validation", pal.hairline, pal.ink],
    ["test_ood", "test, unseen repositories", pal.background, pal.ink],
  ];
  const tot = order.reduce((s, [k]) => s + S[k].n_functions, 0);
  const box = el("div", "chart");
  host.prepend(box);
  responsive(box, (W) => {
    const narrow = W < 600;
    const barY = 44, barH = 44, lineH = 20, step = narrow ? 62 : 52;
    const H = barY + barH + 24 + 3 * step;
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", order.map(([k, n]) => `${n}: ${fmt.int(S[k].n_functions)} functions`).join("; "));
    let x = 0;
    const seg = {};
    order.forEach(([k, , fill, stroke]) => {
      const w = (W * S[k].n_functions) / tot;
      seg[k] = { x, w };
      svg.append("rect").attr("x", x + 0.5).attr("y", barY).attr("width", Math.max(1, w - 2)).attr("height", barH).attr("fill", fill).attr("stroke", fill === pal.background ? stroke : "none");
      x += w;
    });
    // which repositories each pair of splits comes from
    const brace = (x0, x1, text, anchor) => {
      svg.append("path").attr("d", `M${x0 + 1},${barY - 6} V${barY - 14} H${x1 - 2} V${barY - 6}`).attr("fill", "none").attr("stroke", pal.ink_secondary);
      svg.append("text").attr("x", anchor === "end" ? x1 - 2 : x0 + 1).attr("y", barY - 22).attr("text-anchor", anchor).text(text);
    };
    brace(seg.train.x, seg.test_id.x + seg.test_id.w, "repositories used in training", "start");
    brace(seg.val.x, seg.test_ood.x + seg.test_ood.w, "never in training", "end");
    // training: label inside the bar
    svg.append("text").attr("x", 12).attr("y", barY + 28).attr("fill", pal.background).attr("class", "on-dark")
      .text(`training: ${fmt.int(S.train.n_functions)} functions`);
    // the three small splits: leader lines down to staggered labels, text to the left of each leader
    ["test_id", "val", "test_ood"].forEach((k, i) => {
      const [, name] = order.find((o) => o[0] === k);
      const cx = seg[k].x + seg[k].w / 2, ly = barY + barH + 22 + i * step;
      svg.append("line").attr("x1", cx).attr("x2", cx).attr("y1", barY + barH + 2).attr("y2", ly + 4).attr("stroke", pal.ink_secondary);
      const t = svg.append("text").attr("x", cx - 8).attr("y", ly).attr("text-anchor", "end");
      t.append("tspan").attr("class", "lab-strong").text(name);
      t.append("tspan").attr("x", cx - 8).attr("dy", lineH).text(`${fmt.int(S[k].n_functions)} functions, ${fmt.int(S[k].n_repos)} repos`);
    });
  });
}

// ---------------------------------------------------------------- Preprocessing: a waterfall, every bar to one scale
// Each row is a bar on the same scale (all pooled functions = full width). A removal is drawn where it comes off the
// end of the bar before it; the parse failures are magenta because they are the code Python rejects.
export function funnel(host, D) {
  const R = D.phase1_report;
  const full = R.n_pooled;
  const rows = [
    { kind: "total", text: "Python functions, after pooling the official files", n: R.n_pooled, to: R.n_pooled },
    { kind: "cut reject", text: "Current Python cannot parse them", n: R.filter.ast_fail },
    { kind: "cut", text: `Longer than ${fmt.int(R.config.max_tokens)} tokens`, n: R.filter.too_long },
    { kind: "cut", text: "Near-duplicates of another function", n: R.dedup.near_removed },
    { kind: "total", text: `Left, from ${fmt.int(R.dedup.repos_remaining)} repositories`, n: R.dedup.remaining, to: R.dedup.remaining },
    { kind: "total kept", text: `Kept: at most ${fmt.int(R.sampling.max_per_repo)} per repository, from ${fmt.int(R.sampling.n_repos)} repositories`, n: R.sampling.n_sampled, to: R.sampling.n_sampled, ghost: R.dedup.remaining },
  ];
  const pct = (v) => `${(100 * v) / full}%`;
  const wrap = el("div", "wf");
  wrap.setAttribute("role", "img");
  wrap.setAttribute("aria-label", rows.map((r) => `${r.text}: ${r.kind.startsWith("cut") ? "minus " : ""}${fmt.int(r.n)}`).join("; "));
  let level = full;
  rows.forEach((r) => {
    const row = el("div", `wf-row ${r.kind}`);
    const lab = el("div", "wf-lab");
    lab.append(el("span", "wf-text", r.text), el("span", "wf-num", `${r.kind.startsWith("cut") ? "−" : ""}${fmt.int(r.n)}`));
    const track = el("div", "wf-track");
    if (r.kind.startsWith("cut")) {
      const ghost = el("i", "wf-ghost"); ghost.style.width = pct(level - r.n);
      const cut = el("i", "wf-cut"); cut.style.left = pct(level - r.n); cut.style.width = `max(3px, ${pct(r.n)})`;
      track.append(ghost, cut);
      level -= r.n;
    } else {
      if (r.ghost) {
        const g = el("i", "wf-capped"); g.style.left = pct(r.to); g.style.width = pct(r.ghost - r.to);
        const t = el("span", "wf-note", "removed by the cap"); t.style.left = pct(r.to);
        track.append(g, t);
      }
      const bar = el("i", "wf-bar"); bar.style.width = pct(r.to);
      track.appendChild(bar);
      level = r.to;
    }
    row.append(lab, track);
    wrap.appendChild(row);
  });
  host.prepend(wrap);
}

// ---------------------------------------------------------------- Preprocessing: code <-> tokens, linked
// The first lines of a real function, and the tokens the models read. Hovering or clicking a token on either side
// lights up the same token on the other side.
const TOKEN_NOTE = {
  "<UNK>": "a name outside the vocabulary, so it becomes <code>&lt;UNK&gt;</code>",
  "<STR>": "a string; every string becomes <code>&lt;STR&gt;</code>",
  "<NUM>": "a number; every number becomes <code>&lt;NUM&gt;</code>",
  "<NEWLINE>": "the end of a line, which is a token too",
  "<INDENT>": "one level of indentation more, also a token",
  "<DEDENT>": "one level of indentation less, also a token",
};
export function tokensFigure(host, D) {
  const F = D.traces.functions;
  const st = { f: 0, pin: 1 };
  const pick = el("select");
  pick.setAttribute("aria-label", "Function");
  F.forEach((f, i) => pick.appendChild(Object.assign(el("option", "", `${f.func_name} (${f.repository_name})`), { value: i })));
  const ctl = el("div", "fig-ctl");
  const fld = el("div", "st-field"); fld.append(el("span", "", "Function"), pick); ctl.append(fld);
  const grid = el("div", "tk-grid");
  const left = el("div", "tk-side"), right = el("div", "tk-side");
  const codeHead = el("p", "tk-head"), tokHead = el("p", "tk-head");
  const codeBox = el("div", "tk-code"), list = el("div", "tk-list");
  left.append(codeHead, codeBox);
  right.append(tokHead, list);
  grid.append(left, right);
  const read = el("p", "tk-read");
  read.setAttribute("aria-live", "polite");
  host.prepend(ctl, grid, read);

  let spans = [], chips = [], toks = [];
  const light = (i) => {
    spans.forEach((s) => s?.classList.remove("hl"));
    chips.forEach((c) => c.classList.remove("hl"));
    if (i == null) return;
    spans[i]?.classList.add("hl"); chips[i]?.classList.add("hl");
  };
  const say = (i) => {
    const t = toks[i];
    const what = TOKEN_NOTE[t] ?? "in the vocabulary, so it stays as it is";
    read.innerHTML = `Token <b>${i + 1}</b> of ${toks.length}: <code></code>, ${what}.`;
    read.querySelector("code").textContent = t;
  };
  const focus = (i) => { light(i); say(i); };
  const unfocus = () => { light(st.pin); say(st.pin); };

  function load() {
    const f = F[st.f];
    const end = f.tokens.indexOf("<NEWLINE>", f.tokens.indexOf("<NEWLINE>") + 1); // the first two lines
    toks = f.tokens.slice(0, end + 1);
    const r = renderCode(toks, { lineNumbers: true, markLayout: true });
    codeBox.replaceChildren(r.root);
    spans = r.spans;
    list.replaceChildren();
    chips = toks.map((t, i) => {
      const b = el("button", t.startsWith("<") ? "tk special" : "tk");
      b.type = "button";
      b.append(el("span", "tk-i", String(i + 1)), el("span", "tk-t", t));
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("mouseenter", () => focus(i));
      b.addEventListener("mouseleave", unfocus);
      b.addEventListener("focus", () => focus(i));
      b.addEventListener("blur", unfocus);
      b.addEventListener("click", () => { st.pin = i; chips.forEach((c, j) => c.setAttribute("aria-pressed", j === i)); focus(i); });
      list.appendChild(b);
      return b;
    });
    spans.forEach((s, i) => {
      if (!s) return;
      s.addEventListener("mouseenter", () => focus(i));
      s.addEventListener("mouseleave", unfocus);
      s.addEventListener("click", () => chips[i].click());
    });
    const lines = r.root.children.length;
    codeHead.innerHTML = `The first ${lines} lines of the function`;
    tokHead.innerHTML = `What the models read: <b>${toks.length} tokens</b>`;
    st.pin = Math.max(0, toks.indexOf("<UNK>"));
    chips[st.pin].setAttribute("aria-pressed", "true");
    unfocus();
  }
  pick.addEventListener("change", () => { st.f = +pick.value; load(); });
  load();
}

// ---------------------------------------------------------------- Results: perplexity against context
export function pplChart(fig, D) {
  const pal = D.palette, S = D.perplexity.summary;
  let split = "test_ood";
  const splitCh = choices([["test_ood", "unseen repositories"], ["test_id", "seen repositories"]], split, (v) => { split = v; redraw(); }, { label: "Which test functions" });
  const ctxCh = choices(ML.map((m) => [ctxOf(m, D), `${ctxOf(m, D)}`]), ctxState.value, (v) => ctxState.set(v), { kind: "mlp", label: "MLP window in tokens" });
  const box = el("div", "chart");
  fig.prepend(ctlRow(["MLP window", ctxCh.el], ["Test functions from", splitCh.el]), box);
  const tip = tooltip(box);
  const all = MODEL_IDS.flatMap((m) => ["test_id", "test_ood"].map((s) => S[m][s].ppl_mean));
  const redraw = responsive(box, (W) => {
    const narrow = W < 600;
    const H = narrow ? 340 : 420, m = { l: 48, r: narrow ? 64 : 120, t: 24, b: 52 };
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).insert("svg", ".tip").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Perplexity against the number of tokens each model can see");
    const x = d3.scaleLog().domain([0.85, 24]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([Math.floor(d3.min(all) - 0.6), Math.ceil(d3.max(all) + 0.6)]).range([H - m.b, m.t]);
    // highlighter: the selected MLP window
    const c = ctxState.value;
    svg.append("rect").attr("x", x(c) - 9).attr("y", m.t).attr("width", 18).attr("height", H - m.b - m.t).attr("fill", pal.highlight);
    svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues([1, 2, 3, 4, 5, 10, 20]).tickFormat(d3.format("d")).tickSize(0).tickPadding(10)).call((s) => s.select(".domain").remove());
    svg.append("text").attr("x", m.l).attr("y", H - 8).text("tokens of code the model can see");
    svg.append("text").attr("x", m.l - 40).attr("y", 14).text("perplexity, lower is better");
    // annotation: the same 3 tokens of context
    const m3 = S.mlp3[split].ppl_mean, n4 = S.ngram4[split].ppl_mean;
    if (!narrow) {
      svg.append("line").attr("x1", x(3)).attr("x2", x(3)).attr("y1", y(n4) - 8).attr("y2", y(m3) + 8).attr("stroke", pal.ink_secondary).attr("stroke-dasharray", "3 3");
      svg.append("text").attr("x", x(3)).attr("y", Math.max(y(n4), y(m3)) + 30).attr("text-anchor", "middle").text("same 3 tokens of context");
    }
    for (const [ids, color, name] of [[NG, pal.ngram[5], "n-gram"], [ML, pal.mlp[20], "MLP"]]) {
      const pts = ids.map((id) => ({ id, x: ctxOf(id, D), v: S[id][split] }));
      svg.append("path").attr("d", d3.line().x((d) => x(d.x)).y((d) => y(d.v.ppl_mean))(pts)).attr("fill", "none").attr("stroke", color).attr("stroke-width", 2.5);
      pts.forEach((p) => {
        if (p.v.ppl_std > 0) svg.append("line").attr("x1", x(p.x)).attr("x2", x(p.x)).attr("y1", y(p.v.ppl_mean - p.v.ppl_std)).attr("y2", y(p.v.ppl_mean + p.v.ppl_std)).attr("stroke", color).attr("stroke-width", 2);
        const sel = !isNgram(p.id) && p.x === c;
        mark(svg, p.id, pal, x(p.x), y(p.v.ppl_mean), sel ? 8 : 5.5);
        svg.append("circle").attr("cx", x(p.x)).attr("cy", y(p.v.ppl_mean)).attr("r", 18).attr("fill", "transparent")
          .on("mousemove", (ev) => tip.show(`${family(p.id)}, sees ${p.x} tokens<br>perplexity <b>${fmt.f2(p.v.ppl_mean)}</b>` +
            (p.v.n_seeds > 1 ? ` &plusmn; ${fmt.f2(p.v.ppl_std)} over ${p.v.n_seeds} seeds` : ""), ev))
          .on("mouseleave", () => tip.hide());
      });
      const last = pts.at(-1);
      svg.append("text").attr("class", "lab-strong").attr("fill", color).attr("x", x(last.x) + 12).attr("y", y(last.v.ppl_mean) + 5).text(narrow && name === "n-gram" ? `n-gram ${fmt.f2(last.v.ppl_mean)}` : name);
    }
    // selected MLP point: its value, written next to it
    const sv = S[`mlp${c}`][split].ppl_mean;
    svg.append("text").attr("class", "lab-strong").attr("x", x(c) + 12).attr("y", y(sv) - 12).text(fmt.f2(sv));
    // annotation: where the n-gram ends up
    const n5 = S.ngram5[split].ppl_mean;
    if (!narrow) svg.append("line").attr("x1", x(4)).attr("x2", x(4)).attr("y1", y(n5) - 9).attr("y2", y(n5) - 34).attr("stroke", pal.ink_secondary);
    if (!narrow) svg.append("text").attr("x", x(4)).attr("y", y(n5) - 40).attr("text-anchor", "middle").text(`longest n-gram: ${fmt.f2(n5)}`);
  });
  ctxState.on(() => { ctxCh.set(ctxState.value); redraw(); });
}

// ---------------------------------------------------------------- Results: valid code, 10x10 unit charts
export function waffles(fig, D) {
  const pal = D.palette, V = D.validity_by_model, temps = D.validity.config.temperatures;
  let T = temps[0];
  const tCh = choices(temps.map((t) => [t, fmt.t(t)]), T, (v) => { T = +v; update(); }, { label: "Sampling temperature" });
  const rows = el("div", "waffle-rows");
  fig.prepend(ctlRow(["Temperature", tCh.el]), rows);
  const cells = {};
  [["n-gram", NG, "k-ngram", "order"], ["MLP", ML, "k-mlp", "context"]].forEach(([name, ids, cls, what]) => {
    const row = el("div", "waffle-row");
    const h = el("p", "waffle-head");
    h.innerHTML = `<span class="${cls}">${name}</span>, by ${what}`;
    row.appendChild(h);
    const wrap = el("div", "waffles");
    ids.forEach((id) => {
      const w = el("div", "waffle");
      const g = el("div", "grid10");
      g.setAttribute("role", "img");
      const sq = Array.from({ length: 100 }, () => g.appendChild(el("i")));
      const val = el("div", "w-val"), nm = el("div", "w-name", `${what} ${id.replace(/\D+/, "")}`);
      w.append(g, val, nm);
      wrap.appendChild(w);
      cells[id] = { sq, val, g };
    });
    row.appendChild(wrap);
    rows.appendChild(row);
  });
  function update() {
    MODEL_IDS.forEach((id) => {
      const r = V[id][tkey(T)].all, n = Math.round(100 * r.rate);
      cells[id].sq.forEach((s, i) => { s.style.background = i < n ? modelColor(id, pal) : ""; });
      cells[id].val.textContent = fmt.pct0(r.rate);
      cells[id].val.style.color = modelColor(id, pal);
      cells[id].g.setAttribute("aria-label", `${modelLabel(id)}: ${n} of 100 finished functions are valid Python`);
    });
  }
  update();
}

// ---------------------------------------------------------------- Results: seen vs unseen repositories
export function slopeChart(fig, D) {
  const pal = D.palette, S = D.perplexity.summary, G = D.perplexity.gaps;
  const box = el("div", "chart");
  fig.prepend(box);
  const tip = tooltip(box);
  // One row per model. On each row: perplexity on seen repositories (hollow mark) and on unseen ones (filled mark),
  // joined by a line, then the two values and the change written out in labelled columns.
  responsive(box, (W) => {
    const narrow = W < 600;
    const rowH = 36, groupH = 34, top = 58, axisH = 44;
    const labW = narrow ? 84 : 150, colW = narrow ? 0 : 96, chW = narrow ? 64 : 100;
    const x0 = labW, x1 = W - 2 * colW - chW - 20;
    const groups = [["n-gram", NG, "order"], ["MLP", ML, "context"]];
    const H = top + groups.length * groupH + MODEL_IDS.length * rowH + axisH;
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).insert("svg", ".tip").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`)
      .attr("role", "img").attr("aria-label", "Perplexity of every model on test functions from seen and from unseen repositories");
    const vals = MODEL_IDS.flatMap((id) => [S[id].test_id.ppl_mean, S[id].test_ood.ppl_mean]);
    const x = d3.scaleLinear().domain([Math.floor(d3.min(vals) - 0.5), Math.ceil(d3.max(vals) + 0.5)]).range([x0 + 10, x1 - 10]);
    const bottom = H - axisH;
    // vertical grid + axis
    svg.append("g").attr("class", "grid").selectAll("line").data(x.ticks(narrow ? 4 : 7)).join("line")
      .attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", top - 6).attr("y2", bottom);
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${bottom})`)
      .call(d3.axisBottom(x).ticks(narrow ? 4 : 7).tickSize(0).tickPadding(10)).call((s) => s.select(".domain").remove());
    svg.append("text").attr("x", narrow ? 0 : x0 + 10).attr("y", H - 4).text(narrow ? "perplexity, lower is better" : "perplexity on test functions, lower is better");
    // column heads, each with the mark it stands for
    const hy = 22;
    const headMark = (cx, filled) => svg.append("circle").attr("cx", cx).attr("cy", hy - 5).attr("r", 5)
      .attr("fill", filled ? pal.ink : pal.background).attr("stroke", pal.ink).attr("stroke-width", 2);
    if (!narrow) {
      const cs = x1 + 20 + colW, cu = x1 + 20 + 2 * colW;
      svg.append("text").attr("class", "lab-strong").attr("x", cs).attr("y", hy).attr("text-anchor", "end").text("seen");
      svg.append("text").attr("class", "lab-strong").attr("x", cu).attr("y", hy).attr("text-anchor", "end").text("unseen");
      headMark(cs - 46, false); headMark(cu - 62, true);
      svg.append("text").attr("x", cu).attr("y", hy + 20).attr("text-anchor", "end").text("repositories");
    }
    svg.append("text").attr("class", "lab-strong").attr("x", W).attr("y", hy).attr("text-anchor", "end").text("change");
    svg.append("text").attr("x", W).attr("y", hy + 20).attr("text-anchor", "end").text("on unseen");
    if (narrow) {
      headMark(x0 + 16, false); svg.append("text").attr("x", x0 + 26).attr("y", hy).text("seen");
      headMark(x0 + 86, true); svg.append("text").attr("x", x0 + 96).attr("y", hy).text("unseen");
    }
    let yy = top;
    for (const [name, ids, what] of groups) {
      const col = name === "n-gram" ? pal.ngram[5] : pal.mlp[20];
      svg.append("text").attr("x", 0).attr("y", yy + 22).attr("font-weight", 700).attr("fill", col).text(narrow ? name : `${name}, by ${what}`);
      yy += groupH;
      ids.forEach((id) => {
        const a = S[id].test_id.ppl_mean, b = S[id].test_ood.ppl_mean, c = modelColor(id, pal), cy = yy + rowH / 2;
        const g = svg.append("g").attr("class", "slope-row");
        g.append("rect").attr("x", 0).attr("y", yy).attr("width", W).attr("height", rowH).attr("fill", "transparent");
        g.append("line").attr("x1", 0).attr("x2", W).attr("y1", yy + rowH).attr("y2", yy + rowH).attr("stroke", pal.hairline);
        g.append("text").attr("x", 12).attr("y", cy + 5).text(`${what} ${id.replace(/\D+/, "")}`);
        g.append("line").attr("x1", x(a)).attr("x2", x(b)).attr("y1", cy).attr("y2", cy).attr("stroke", c).attr("stroke-width", 3);
        g.append("circle").attr("cx", x(a)).attr("cy", cy).attr("r", 6).attr("fill", pal.background).attr("stroke", c).attr("stroke-width", 2.5);
        g.append("circle").attr("cx", x(b)).attr("cy", cy).attr("r", 6).attr("fill", c).attr("stroke", c).attr("stroke-width", 2.5);
        if (!narrow) {
          g.append("text").attr("x", x1 + 20 + colW).attr("y", cy + 5).attr("text-anchor", "end").style("font-variant-numeric", "tabular-nums").text(fmt.f2(a));
          g.append("text").attr("x", x1 + 20 + 2 * colW).attr("y", cy + 5).attr("text-anchor", "end").style("font-variant-numeric", "tabular-nums").text(fmt.f2(b));
        }
        g.append("text").attr("class", "lab-strong").attr("x", W).attr("y", cy + 5).attr("text-anchor", "end").style("font-variant-numeric", "tabular-nums").text(fmt.spct1(G[id].gap));
        g.on("mousemove", (ev) => tip.show(`${family(id)}<br>seen repositories <b>${fmt.f2(a)}</b><br>unseen repositories <b>${fmt.f2(b)}</b><br>change <b>${fmt.spct1(G[id].gap)}</b> (95% interval ${fmt.f1(100 * G[id].ci[0])} to ${fmt.f1(100 * G[id].ci[1])}%)`, ev))
          .on("mouseleave", () => tip.hide());
        yy += rowH;
      });
    }
  });
}

// ---------------------------------------------------------------- Errors: what happens to finished samples
export function failureBars(host, D) {
  const pal = D.palette, V = D.validity_by_model, T = D.validity.config.temperatures[0];
  const cats = [["valid", null], ["unbalanced brackets", pal.invalid], ["indentation", pal.ink_secondary], ["incomplete statement", pal.hairline], ["other", pal.ink]];
  const wrap = el("div", "fb");
  ["ngram5", "mlp20"].forEach((id) => {
    const r = V[id][tkey(T)].all;
    const row = el("div", "fb-row");
    const h = el("p", "fb-head");
    h.innerHTML = family(id);
    const bar = el("div", "fb-bar");
    cats.forEach(([c, col]) => {
      const n = c === "valid" ? r.valid : r.failures[c];
      if (!n) return;
      const i = el("i");
      i.style.width = `${(100 * n) / r.natural}%`;
      i.style.background = col ?? modelColor(id, pal);
      i.title = `${c}: ${fmt.int(n)} of ${fmt.int(r.natural)} finished (${fmt.pct1(n / r.natural)})`;
      if (n / r.natural > 0.09) i.textContent = `${c === "unbalanced brackets" ? "brackets" : c} ${fmt.pct0(n / r.natural)}`;
      if (col === pal.hairline) i.classList.add("light");
      bar.appendChild(i);
    });
    // phones: the bar is too narrow for in-bar labels, so the same shares are written out underneath
    const txt = el("p", "fb-text", cats.map(([c]) => [c, c === "valid" ? r.valid : r.failures[c]]).filter(([, n]) => n / r.natural >= 0.01)
      .map(([c, n]) => `${c} ${fmt.pct0(n / r.natural)}`).join(", "));
    row.append(h, bar, txt);
    wrap.appendChild(row);
  });
  const key = el("p", "fb-key");
  key.innerHTML = cats.slice(1).map(([c, col]) => `<span><i style="background:${col}"></i>${c}</span>`).join("") + `<span>valid: the model&rsquo;s own colour</span>`;
  wrap.appendChild(key);
  host.prepend(wrap);
}

// ---------------------------------------------------------------- training curves
export function curvesChart(fig, D) {
  const pal = D.palette, T = D.training_curves;
  const box = el("div", "chart");
  fig.prepend(box);
  responsive(box, (W) => {
    const H = 300, m = { l: 50, r: 100, t: 10, b: 44 };
    d3.select(box).select("svg").remove();
    const svg = d3.select(box).append("svg").attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    const all = Object.values(T).flat();
    const x = d3.scaleLinear().domain([1, d3.max(all, (c) => c.epochs.at(-1))]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([d3.min(all, (c) => d3.min(c.val_loss)) - 0.03, d3.max(all, (c) => d3.max(c.val_loss)) + 0.02]).range([H - m.b, m.t]);
    svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")).tickSize(0).tickPadding(8)).call((s) => s.select(".domain").remove());
    svg.append("text").attr("x", m.l).attr("y", H - 6).text("epoch; validation loss on the left");
    Object.entries(T).forEach(([c, runs]) => {
      runs.forEach((r, k) => svg.append("path").attr("d", d3.line().x((_, i) => x(r.epochs[i])).y((v) => y(v))(r.val_loss)).attr("fill", "none").attr("stroke", pal.mlp[c]).attr("stroke-width", 2).attr("opacity", k ? 0.4 : 1));
      const r0 = runs[0];
      svg.append("text").attr("x", x(r0.epochs.at(-1)) + 8).attr("y", y(r0.val_loss.at(-1)) + 5).text(`context ${c}`);
    });
  });
}

// ---------------------------------------------------------------- tables for "Full numbers"
const th = (s, cls = "") => `<th class="${cls}">${s}</th>`;
export function tablePerplexity(D) {
  const S = D.perplexity.summary, G = D.perplexity.gaps;
  let h = `<table><thead><tr>${th("model")}${th("validation", "n")}${th("seen repos", "n")}${th("unseen repos", "n")}${th("change", "n")}</tr></thead><tbody>`;
  MODEL_IDS.forEach((id) => {
    const f = (s) => `${fmt.f2(S[id][s].ppl_mean)}${S[id][s].n_seeds > 1 ? ` &plusmn; ${fmt.f2(S[id][s].ppl_std)}` : ""}`;
    h += `<tr><td>${family(id)}</td><td class="n">${f("val")}</td><td class="n">${f("test_id")}</td><td class="n">${f("test_ood")}</td><td class="n">${fmt.spct1(G[id].gap)} <span class="muted">(${fmt.f1(100 * G[id].ci[0])} to ${fmt.f1(100 * G[id].ci[1])})</span></td></tr>`;
  });
  return h + "</tbody></table>";
}
export function tableValidity(D) {
  const V = D.validity_by_model;
  let h = `<table><thead><tr>${th("model")}${th("temperature", "n")}${th("valid", "n")}${th("95% CI", "n")}${th("finished", "n")}${th("cut off", "n")}</tr></thead><tbody>`;
  MODEL_IDS.forEach((id) => Object.values(V[id]).forEach((byT) => {
    const r = byT.all;
    h += `<tr><td>${family(id)}</td><td class="n">${fmt.t(r.T)}</td><td class="n">${fmt.pct1(r.rate)}</td><td class="n">${fmt.pct1(r.ci[0])}–${fmt.pct1(r.ci[1])}</td><td class="n">${fmt.int(r.natural)}</td><td class="n">${fmt.int(r.truncated)}</td></tr>`;
  }));
  return h + "</tbody></table>";
}
export function tableSplits(D) {
  const S = D.phase1_report.splits;
  const rows = [["train", "training"], ["val", "validation"], ["test_id", "test, seen repositories"], ["test_ood", "test, unseen repositories"]];
  return `<table><thead><tr>${th("split")}${th("functions", "n")}${th("repositories", "n")}${th("tokens", "n")}${th("unknown-name share", "n")}</tr></thead><tbody>` +
    rows.map(([k, n]) => `<tr><td>${n}</td><td class="n">${fmt.int(S[k].n_functions)}</td><td class="n">${fmt.int(S[k].n_repos)}</td><td class="n">${fmt.int(S[k].n_tokens)}</td><td class="n">${fmt.pct1(S[k].unk_rate)}</td></tr>`).join("") +
    "</tbody></table>";
}
