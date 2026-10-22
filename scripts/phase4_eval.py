"""Phase 4: final evaluation. The ONLY script allowed to load test_id / test_ood (see src/data/dataset.py).

Stages: ppl (perplexity, per-category loss, ID-vs-OOD gap), sample (syntax validity, failure taxonomy,
explorer completions), plots (figures + site data from the saved JSON). --dev substitutes the validation
split for the test splits and writes to results/phase4_dev, to test the pipeline without touching test data.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_ids, load_split
from src.data.tokenizer import BOS, DEDENT, EOS, INDENT, NEWLINE, Vocab
from src.data.windows import Windows
from src.eval import metrics
from src.eval.sampling import generate, make_prompt, mlp_step, ngram_step
from src.eval.syntax import CATEGORIES as FAIL_CATS
from src.eval.syntax import check_tokens
from src.mlp.model import MLPLM
from src.mlp.train import GPUWindows, nll_array
from src.ngram.kn import KNModel

DEV = "cuda"


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_mlp(path):
    ck = torch.load(path, map_location=DEV)
    c = ck["cfg"]
    m = MLPLM(ck["vocab_size"], ck["context"], c["emb_dim"], c["hidden"], c["dropout"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def model_ids(cfg, dev):
    ids = [f"ngram{o}" for o in cfg["ngram_orders"]]
    for c in cfg["mlp_contexts"]:
        have = [s for s in cfg["mlp_seeds"] if (Path(cfg["ckpt_dir"]) / f"mlp_ctx{c}_seed{s}.pt").exists()]
        if len(have) < len(cfg["mlp_seeds"]) and not dev:
            raise FileNotFoundError(f"missing MLP checkpoints for context {c}: have seeds {have}")
        if have:
            ids.append(f"mlp{c}")
    return ids


def seeds_of(cfg, mid):
    return [None] if mid.startswith("ngram") else [s for s in cfg["mlp_seeds"]
                                                   if (Path(cfg["ckpt_dir"]) / f"mlp_ctx{mid[3:]}_seed{s}.pt").exists()]


# ------------------------------------------------------------------ stage: perplexity
def stage_ppl(cfg, vocab, data, out):
    rng = np.random.default_rng(cfg["seed"])
    cat_of_id = metrics.category_ids(vocab)
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    mids = model_ids(cfg, args.dev)
    rows, fn_acc, cat_acc = [], {}, {}
    for split in cfg["splits"]:
        ids = data[split]
        cache = {}

        def win(ctx):
            if ctx not in cache:
                cache[ctx] = Windows(ids, ctx, bos, eos)
            return cache[ctx]

        base = win(3)
        targets, seq_id = base.targets(slice(None)), base.seq_id
        for mid in mids:
            for seed in seeds_of(cfg, mid):
                if mid.startswith("ngram"):
                    order = int(mid[5:])
                    nll = KNModel.load(Path(cfg["ngram_dir"]) / f"ngram_{order}.npz").nll(win(order - 1))
                else:
                    c = int(mid[3:])
                    model, _ = load_mlp(Path(cfg["ckpt_dir"]) / f"mlp_ctx{c}_seed{seed}.pt")
                    nll = nll_array(model, GPUWindows(win(c), DEV), amp=False)
                    del model
                s, n = metrics.per_function(nll, seq_id, len(ids))
                fn_acc.setdefault((mid, split), []).append((s, n))
                cat_acc.setdefault((mid, split), []).append(metrics.category_loss(nll, targets, cat_of_id))
                rows.append({"model": mid, "seed": seed, "split": split, "ppl": float(np.exp(nll.mean())),
                             "nll": float(nll.mean()), "n_tokens": int(len(nll))})
                say(f"{split:9s} {mid:8s} seed={seed}: ppl={rows[-1]['ppl']:.3f}")
        torch.cuda.empty_cache()

    summary, cat_summary = {}, {}
    for (mid, split), lst in fn_acc.items():
        ppls = np.array([r["ppl"] for r in rows if r["model"] == mid and r["split"] == split])
        s = np.mean([a for a, _ in lst], axis=0)  # seed-averaged per-function NLL sums
        n = lst[0][1]
        summary.setdefault(mid, {})[split] = {
            "ppl_mean": float(ppls.mean()), "ppl_std": float(ppls.std(ddof=1)) if len(ppls) > 1 else 0.0,
            "n_seeds": len(ppls), "ci": metrics.bootstrap_ppl(s, n, rng, cfg["bootstrap"]["n_boot_ppl"]),
            "n_functions": int(len(n)), "n_tokens": int(n.sum())}
        cats = cat_acc[(mid, split)]
        cat_summary.setdefault(mid, {})[split] = {
            c: {"share": cats[0][c]["share"], "mean_nll": float(np.mean([x[c]["mean_nll"] for x in cats]))
                if cats[0][c]["mean_nll"] is not None else None} for c in metrics.CATEGORIES}
    gaps = {}
    for mid in summary:
        if "test_id" in summary[mid] and "test_ood" in summary[mid]:
            (si, ni), (so, no) = ([np.mean([a for a, _ in fn_acc[(mid, sp)]], axis=0), fn_acc[(mid, sp)][0][1]]
                                  for sp in ("test_id", "test_ood"))
            gaps[mid] = {"gap": summary[mid]["test_ood"]["ppl_mean"] / summary[mid]["test_id"]["ppl_mean"] - 1,
                         "ci": metrics.bootstrap_ratio(si, ni, so, no, rng, cfg["bootstrap"]["n_boot_ppl"])}
    info = {}
    for mid in mids:
        if mid.startswith("mlp"):
            _, ck = load_mlp(Path(cfg["ckpt_dir"]) / f"mlp_ctx{mid[3:]}_seed{seeds_of(cfg, mid)[0]}.pt")
            info[mid] = {"context": ck["context"], "n_params": sum(v.numel() for v in ck["state_dict"].values()),
                         "emb_dim": ck["cfg"]["emb_dim"], "hidden": ck["cfg"]["hidden"], "dropout": ck["cfg"]["dropout"]}
        else:
            z = np.load(Path(cfg["ngram_dir"]) / f"ngram_{mid[5:]}.npz")
            info[mid] = {"order": int(mid[5:]), "context": int(mid[5:]) - 1,
                         "ngram_types": sum(len(z[f"fk_{k}"]) for k in range(1, int(mid[5:]) + 1))}
    result = {"rows": rows, "summary": summary, "gaps": gaps, "model_info": info, "seed": cfg["seed"]}
    (out / "perplexity.json").write_text(json.dumps(result, indent=1))
    (out / "category_loss.json").write_text(json.dumps(cat_summary, indent=1))
    with open(out / "perplexity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "ppl_mean", "ppl_std", "ci_lo", "ci_hi", "n_seeds"])
        for mid, d in summary.items():
            for sp, v in d.items():
                w.writerow([mid, sp, f"{v['ppl_mean']:.4f}", f"{v['ppl_std']:.4f}", f"{v['ci'][0]:.4f}", f"{v['ci'][1]:.4f}", v["n_seeds"]])


# ------------------------------------------------------------------ stage: sampling
def stage_sample(cfg, vocab, data, out, meta_df):
    sc = cfg["sampling"]
    ids_of = {t: vocab.stoi[t] for t in ("def", NEWLINE, INDENT, DEDENT)}
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    prompts, pmeta = [], []
    for split in ("test_id", "test_ood"):
        order = np.random.default_rng(cfg["seed"]).permutation(len(data[split]))
        got = 0
        for i in order:
            p = make_prompt(data[split][i], ids_of, sc["prompt_body_tokens"])
            if p is None:
                continue
            prompts.append(p)
            pmeta.append({"set": split, "index": int(i), **{k: str(meta_df[split][k].iloc[i]) for k in
                                                            ("repository_name", "func_path_in_repository", "func_name")}})
            got += 1
            if got == sc["prompts_per_set"]:
                break
    sets = np.array([m["set"] for m in pmeta])
    say(f"prompts: {len(prompts)} ({Counter(sets)}); example: {' '.join(vocab.decode(prompts[0]))}")

    mids = model_ids(cfg, args.dev)
    runs, examples = [], []
    for mid in mids:
        for T in sc["temperatures"]:
            ok = np.zeros((len(sc["reps"]), len(prompts)), bool)
            nat = np.zeros_like(ok)
            cats = np.full((len(sc["reps"]), len(prompts)), "", dtype=object)
            for ri, rep in enumerate(sc["reps"]):
                rng = np.random.default_rng(sc["sample_seed_base"] + rep)
                if mid.startswith("ngram"):
                    model = KNModel.load(Path(cfg["ngram_dir"]) / f"ngram_{mid[5:]}.npz")
                    step, ctx_len = ngram_step(model, T), int(mid[5:]) - 1
                else:
                    if rep not in seeds_of(cfg, mid):
                        continue
                    model, _ = load_mlp(Path(cfg["ckpt_dir"]) / f"mlp_ctx{mid[3:]}_seed{rep}.pt")
                    step, ctx_len = mlp_step(model, T), int(mid[3:])
                toks, status, _ = generate(step, ctx_len, prompts, sc["max_new_tokens"], rng, ids_of, bos, eos)
                for j, (p, g) in enumerate(zip(prompts, toks)):
                    full = vocab.decode(p) + vocab.decode(g[g != eos])
                    r = check_tokens(full)
                    nat[ri, j] = status[j] == 1
                    ok[ri, j] = r["ok"] and nat[ri, j]
                    if nat[ri, j] and not r["ok"]:
                        cats[ri, j] = r["category"]
                        if rep == sc["reps"][0] and sum(e["model"] == mid and e["T"] == T and e["category"] == r["category"]
                                                        for e in examples) < cfg["examples_per_category"]:
                            examples.append({"model": mid, "T": T, "category": r["category"], "msg": r["msg"],
                                             "token_index": r["token_index"], "n_prompt_tokens": len(p), "set": pmeta[j]["set"],
                                             "prompt_id": j, "tokens": full})
                del model
            done = nat.any(axis=1)
            for st in ("test_id", "test_ood", "all"):
                m = np.ones(len(prompts), bool) if st == "all" else sets == st
                o, n = ok[done][:, m], nat[done][:, m]
                rate, ci = metrics.bootstrap_rate(o, n, np.random.default_rng(cfg["seed"]), cfg["bootstrap"]["n_boot_rate"])
                cc = Counter(cats[done][:, m].ravel()) if True else None
                runs.append({"model": mid, "T": T, "set": st, "reps": int(done.sum()), "n_prompts": int(m.sum()),
                             "completions": int(o.size), "natural": int(n.sum()), "truncated": int(o.size - n.sum()),
                             "valid": int(o.sum()), "rate": rate if n.sum() else None, "ci": ci,
                             "failures": {c: int(cc.get(c, 0)) for c in FAIL_CATS}})
            r = runs[-1]
            say(f"{mid:8s} T={T}: natural={r['natural']}/{r['completions']} valid|natural={r['rate']:.3f} {r['failures']}")
    (out / "validity.json").write_text(json.dumps({"config": sc, "rows": runs}, indent=1))
    (out / "failure_examples.json").write_text(json.dumps({"prompts": pmeta, "examples": examples}, indent=1))
    return prompts, pmeta, ids_of


def stage_explorer(cfg, vocab, data, out, prompts, pmeta, ids_of):
    ex = cfg["explorer"]
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    pick = []
    for st in ("test_id", "test_ood"):
        pick += [i for i, m in enumerate(pmeta) if m["set"] == st][:ex["prompts_per_set"]]
    P = [prompts[i] for i in pick]
    result = {"prompts": [{**pmeta[i], "prompt_tokens": vocab.decode(prompts[i]),
                           "original_tokens": vocab.decode(data[pmeta[i]["set"]][pmeta[i]["index"]][:ex["max_original_tokens"]])}
                          for i in pick], "temperature": 1.0, "seed": ex["seed"], "top_k": ex["top_k"], "runs": {}}
    for mid in model_ids(cfg, args.dev):
        rng = np.random.default_rng(ex["seed"])
        if mid.startswith("ngram"):
            step, ctx_len = ngram_step(KNModel.load(Path(cfg["ngram_dir"]) / f"ngram_{mid[5:]}.npz"), 1.0), int(mid[5:]) - 1
        else:
            model, _ = load_mlp(Path(cfg["ckpt_dir"]) / f"mlp_ctx{mid[3:]}_seed{seeds_of(cfg, mid)[0]}.pt")
            step, ctx_len = mlp_step(model, 1.0), int(mid[3:])
        toks, status, rec = generate(step, ctx_len, P, cfg["sampling"]["max_new_tokens"], rng, ids_of, bos, eos, record_top=ex["top_k"])
        runs = []
        for j, (p, g) in enumerate(zip(P, toks)):
            n = len(g)
            gen = vocab.decode(g)
            check = check_tokens(vocab.decode(p) + vocab.decode(g[g != eos])) if status[j] == 1 else {"ok": None}
            runs.append({"tokens": gen, "p": [round(float(x), 4) for x in rec["p"][j, :n]],
                         "top": [[[vocab.itos[int(t)], round(float(q), 4)] for t, q in zip(rec["top_ids"][j, s], rec["top_p"][j, s])]
                                 for s in range(n)],
                         "status": "natural" if status[j] == 1 else "truncated", "check": check})
        result["runs"][mid] = runs
        say(f"explorer {mid}: {sum(r['check'].get('ok') is True for r in runs)}/{len(runs)} valid")
    (out / "explorer.json").write_text(json.dumps(result, separators=(",", ":")))


# ------------------------------------------------------------------ stage: plots + site data
def stage_plots(cfg, out):
    from src.eval import plots
    perp = json.loads((out / "perplexity.json").read_text())
    cat = json.loads((out / "category_loss.json").read_text())
    val = json.loads((out / "validity.json").read_text())
    fig = Path("results/figures" if not args.dev else out / "figures")
    plots.fig_ppl_vs_context(perp["summary"], fig / "phase4_perplexity.png")
    plots.fig_id_ood_gap(perp["summary"], perp["gaps"], fig / "phase4_id_ood_gap.png")
    plots.fig_validity([r for r in val["rows"] if r["set"] != "all"], fig / "phase4_validity.png")
    fail_rows = [{"model": r["model"], "T": r["T"], "valid_share": r["valid"] / r["natural"] if r["natural"] else 0,
                  "categories": {c: n / r["natural"] for c, n in r["failures"].items()} if r["natural"] else {}}
                 for r in val["rows"] if r["set"] == "all"]
    plots.fig_failures(fail_rows, fig / "phase4_failures.png")
    plots.fig_category_loss(cat, "test_ood", fig / "phase4_category_loss.png")
    curves = {}
    chosen = json.loads(Path("results/phase3/final.json").read_text())  # dropout chosen per context
    for c in cfg["mlp_contexts"]:
        for f in sorted(Path("results/phase3/curves").glob(f"ctx{c}_seed*_do{chosen[str(c)]}.csv")):
            rows = list(csv.DictReader(open(f)))
            curves.setdefault(c, []).append(([int(r["epoch"]) for r in rows], [float(r["val_loss"]) for r in rows]))
    plots.fig_training_curves(curves, fig / "phase3_training_curves.png")
    by_model = {}  # validity keyed by model / temperature / prompt set, so the page can bind numbers by name
    for r in val["rows"]:
        by_model.setdefault(r["model"], {}).setdefault(f"t{int(round(10 * r['T'])):02d}", {})[r["set"]] = r
    (out / "validity_by_model.json").write_text(json.dumps(by_model, indent=1))
    if not args.dev:
        site = Path(cfg["site_dir"])
        site.mkdir(parents=True, exist_ok=True)
        (site / "validity_by_model.json").write_text((out / "validity_by_model.json").read_text())
        for name in ("perplexity", "category_loss", "validity", "failure_examples", "explorer"):
            (site / f"{name}.json").write_text((out / f"{name}.json").read_text())
        (site / "palette.json").write_text(json.dumps(yaml.safe_load(open("configs/palette.yaml"))["site"], indent=1))
        (site / "training_curves.json").write_text(json.dumps(
            {str(c): [{"epochs": e, "val_loss": v} for e, v in lst] for c, lst in curves.items()}))
        for name in ("phase0_recon", "phase1_report", "phase2_ngram"):
            src = Path("results") / f"{name}.json"
            if src.exists():
                (site / f"{name}.json").write_text(src.read_text())
        if Path("results/phase3/summary.csv").exists():
            (site / "mlp_summary.json").write_text(json.dumps(list(csv.DictReader(open("results/phase3/summary.csv")))))
        for name in ("dropout_search", "search_summary"):
            src = Path("results/phase3") / f"{name}.json"
            if src.exists():
                (site / f"phase3_{name}.json").write_text(src.read_text())
    say(f"figures -> {fig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="ppl,sample,plots")
    ap.add_argument("--dev", action="store_true", help="use the val split instead of test splits (pipeline test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/phase4.yaml"))
    out = Path(cfg["out_dir"] + ("_dev" if args.dev else ""))
    out.mkdir(parents=True, exist_ok=True)
    vocab = Vocab.load("data/processed/vocab.json")
    stages = args.stages.split(",")
    if "ppl" in stages or "sample" in stages:
        src = {s: ("val" if args.dev else s) for s in cfg["splits"]}
        data = {s: load_ids(src[s]) for s in cfg["splits"]}
    if "ppl" in stages:
        stage_ppl(cfg, vocab, data, out)
    if "sample" in stages:
        meta = {s: load_split(src[s], columns=["repository_name", "func_path_in_repository", "func_name"]) for s in ("test_id", "test_ood")}
        prompts, pmeta, ids_of = stage_sample(cfg, vocab, data, out, meta)
        stage_explorer(cfg, vocab, data, out, prompts, pmeta, ids_of)
    if "plots" in stages:
        stage_plots(cfg, out)
