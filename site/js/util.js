// Shared helpers: data loading, number formatting, model metadata.
export const MODEL_IDS = ["ngram2", "ngram3", "ngram4", "ngram5", "mlp3", "mlp5", "mlp10", "mlp20"];
export const SPLITS = ["val", "test_id", "test_ood"];
export const SPLIT_LABEL = { val: "val (held-out repos)", test_id: "test ID (training repos)", test_ood: "test OOD (unseen repos)" };
export const SET_LABEL = { test_id: "ID", test_ood: "OOD" };
export const FAIL_CATS = ["unbalanced brackets", "indentation", "incomplete statement", "other"];

export const isNgram = (id) => id.startsWith("ngram");
export const modelLabel = (id) => (isNgram(id) ? `n-gram order ${id.slice(5)}` : `MLP context ${id.slice(3)}`);
export const modelColor = (id, pal) => (isNgram(id) ? pal.ngram[id.slice(5)] : pal.mlp[id.slice(3)]);

const NAMES = ["perplexity", "category_loss", "validity", "failure_examples", "explorer", "palette",
  "training_curves", "phase0_recon", "phase1_report", "phase2_ngram", "mlp_summary", "phase3_dropout_search", "validity_by_model", "traces"];

export async function loadData() {
  const entries = await Promise.all(NAMES.map(async (n) => {
    const r = await fetch(`data/${n}.json`);
    return [n, r.ok ? await r.json() : null];
  }));
  return Object.fromEntries(entries);
}

export const fmt = {
  int: (x) => Math.round(x).toLocaleString("en-US"),
  f1: (x) => x.toFixed(1), f2: (x) => x.toFixed(2), f3: (x) => x.toFixed(3),
  spct1: (x) => `${x >= 0 ? "+" : "\u2212"}${(100 * Math.abs(x)).toFixed(1)}%`,
  pct0: (x) => `${(100 * x).toFixed(0)}%`, pct1: (x) => `${(100 * x).toFixed(1)}%`,
  t: (x) => Number(x).toFixed(1),
  mb: (x) => `${(x / 1e6).toFixed(1)}`, m: (x) => `${(x / 1e6).toFixed(2)}M`,
};

// Resolve "a.b.c" in the data object; numeric-looking keys work on objects and arrays.
export function lookup(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : k === "length" && Array.isArray(o) ? o.length : o[k]), obj);
}

export function bindNumbers(root, D) {
  root.querySelectorAll("[data-v]").forEach((el) => {
    const v = lookup(D, el.dataset.v);
    const f = fmt[el.dataset.f || "int"];
    el.textContent = v == null || Number.isNaN(v) ? "n/a" : f(v);
    if (v == null) el.title = `missing: ${el.dataset.v}`;
  });
}

// Token strings -> display text for chips.
export const GLYPH = { "<NEWLINE>": "↵", "<INDENT>": "⇥", "<DEDENT>": "⇤" };
export const isLayout = (t) => t in GLYPH;
export function chipText(t) {
  return GLYPH[t] ?? t;
}

// Port of src/data/tokenizer.py:tokens_to_code_map so the page can show code and mark the failing line.
const SUBST = { "<STR>": '"s"', "<NUM>": "0", "<UNK>": "unk_tok" };
export function tokensToCode(tokens, indent = "    ") {
  const lines = []; const spans = new Array(tokens.length).fill(null);
  let cur = [], idx = [], level = 0;
  const flush = () => {
    let col = indent.length * level;
    idx.forEach((i, k) => { spans[i] = [lines.length, col]; col += cur[k].length + 1; });
    lines.push(indent.repeat(level) + cur.join(" "));
    return cur.length ? col - 1 : indent.length * level;
  };
  tokens.forEach((t, i) => {
    if (t === "<NEWLINE>") { const end = flush(); spans[i] = [lines.length - 1, end]; cur = []; idx = []; }
    else if (t === "<INDENT>") { level += 1; spans[i] = [lines.length, 0]; }
    else if (t === "<DEDENT>") { level = Math.max(level - 1, 0); spans[i] = [lines.length, 0]; }
    else if (["<BOS>", "<EOS>", "<PAD>"].includes(t)) { spans[i] = [lines.length, 0]; }
    else { cur.push(SUBST[t] ?? t); idx.push(i); }
  });
  if (cur.length) flush();
  return { code: lines.join("\n") + "\n", spans, lines };
}

export function tooltip(host) {
  const el = document.createElement("div");
  el.className = "tip"; el.style.display = "none"; host.appendChild(el);
  return {
    show(html, ev) {
      el.innerHTML = html; el.style.display = "block";
      const r = host.getBoundingClientRect();
      let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
      if (x + el.offsetWidth > r.width) x = ev.clientX - r.left - el.offsetWidth - 14;
      if (x < 0) x = 4;
      el.style.left = `${x}px`; el.style.top = `${y}px`;
    },
    hide() { el.style.display = "none"; },
  };
}
