// Live sampling in the browser: Pyodide (exact Python tokenize/ast) + ONNX Runtime Web (int8 MLP).
// Loaded lazily the first time someone edits a prompt and asks for a live sample.
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

let pending = null;
const sessions = {};

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src; s.onload = resolve; s.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(s);
  });
}

async function boot(onStatus) {
  onStatus("Loading ONNX Runtime Web and Pyodide (first use only)…");
  await Promise.all([loadScript(`${ORT_BASE}ort.min.js`), loadScript(`${PYODIDE_BASE}pyodide.js`)]);
  window.ort.env.wasm.wasmPaths = ORT_BASE;
  window.ort.env.wasm.numThreads = 1;
  const py = await window.loadPyodide({ indexURL: PYODIDE_BASE });
  for (const f of ["tokenizer.py", "syntax.py", "web_api.py"]) {
    py.FS.writeFile(f, await (await fetch(`py/${f}`)).text());
  }
  py.runPython("import web_api");
  const vocab = await (await fetch("model/vocab.json")).json();
  return { py, vocab, stoi: Object.fromEntries(vocab.map((t, i) => [t, i])) };
}

async function session(ctx, onStatus) {
  if (!sessions[ctx]) {
    onStatus(`Loading the context-${ctx} model…`);
    sessions[ctx] = window.ort.InferenceSession.create(`model/mlp_ctx${ctx}.int8.onnx`, { executionProviders: ["wasm"] });
  }
  return sessions[ctx];
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function softmax(logits, T) {
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
  const p = new Float64Array(logits.length);
  let s = 0;
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / T); s += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
}

function topK(p, k) {
  const best = [];
  for (let i = 0; i < p.length; i++) {
    if (best.length < k || p[i] > best[best.length - 1][1]) {
      best.push([i, p[i]]);
      best.sort((a, b) => b[1] - a[1]);
      if (best.length > k) best.pop();
    }
  }
  return best;
}

export async function generate({ ctx, code, temperature = 1, seed = 0, maxNew = 256, onStatus = () => {} }) {
  pending ??= boot(onStatus);
  const { py, vocab, stoi } = await pending;
  const sess = await session(ctx, onStatus);
  onStatus("Sampling…");
  py.globals.set("code_in", code);
  const promptTokens = JSON.parse(py.runPython("web_api.prompt_json(code_in)"));
  const unk = stoi["<UNK>"], eos = stoi["<EOS>"], bos = stoi["<BOS>"], INDENT = stoi["<INDENT>"], DEDENT = stoi["<DEDENT>"];
  const hist = Array(ctx).fill(bos).concat(promptTokens.map((t) => stoi[t] ?? unk));
  let depth = promptTokens.filter((t) => t === "<INDENT>").length - promptTokens.filter((t) => t === "<DEDENT>").length;
  let entered = depth > 0;
  const rand = mulberry32(seed * 7919 + 12345);
  const t0 = performance.now();
  const out = { tokens: [], p: [], top: [] };
  let status = "truncated";
  for (let step = 0; step < maxNew; step++) {
    const input = new window.ort.Tensor("int64", BigInt64Array.from(hist.slice(-ctx), (v) => BigInt(v)), [1, ctx]);
    const logits = (await sess.run({ ctx: input })).logits.data;
    const p = softmax(logits, temperature);
    let u = rand(), tok = p.length - 1;
    for (let i = 0; i < p.length; i++) { u -= p[i]; if (u <= 0) { tok = i; break; } }
    out.tokens.push(vocab[tok]); out.p.push(p[tok]);
    out.top.push(topK(p, 5).map(([i, q]) => [vocab[i], q]));
    hist.push(tok);
    if (tok === INDENT) depth++; else if (tok === DEDENT) depth--;
    if (depth > 0) entered = true;
    if (tok === eos || (entered && depth <= 0)) { status = "natural"; break; }
  }
  const ms = performance.now() - t0;
  let check = { ok: null };
  if (status === "natural") {
    py.globals.set("toks_in", JSON.stringify(promptTokens.concat(out.tokens.filter((t) => t !== "<EOS>"))));
    check = JSON.parse(py.runPython("web_api.check_json(toks_in)"));
  }
  return { promptTokens, ...out, status, check, ms };
}
