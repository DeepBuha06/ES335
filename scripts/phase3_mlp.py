"""Phase 3: MLP language models, val only (test splits are not touched here). Resumable: finished runs are skipped.

(1) learning rate / hidden size / dropout search at context 10 (short budget) -> lr and hidden size, fixed for all contexts;
(2) dropout chosen per context length from a small grid, full training at seed 0 (longer contexts memorise more);
(3) remaining seeds for each context with its chosen dropout; (4) publish the chosen checkpoints under the names
    Phase 4 reads (data/checkpoints/mlp_ctx{c}_seed{s}.pt) and aggregate.
"""
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_ids
from src.data.tokenizer import BOS, EOS, Vocab
from src.data.windows import Windows
from src.mlp.train import GPUWindows, train_model

COLS = ["context", "seed", "dropout", "n_params", "best_epoch", "epochs", "best_val_loss", "val_ppl", "train_s"]


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def gpu_windows(ids, context, bos, eos):
    return GPUWindows(Windows(ids, context, bos, eos), "cuda")


def pick(rows, tol):
    """Best validation loss wins, but among configs within `tol` nats of it the smallest hidden size is
    preferred (differences that small are seed noise; a smaller model also limits the parameter confound)."""
    best = min(float(r["best_val_loss"]) for r in rows)
    near = [r for r in rows if float(r["best_val_loss"]) <= best + tol]
    r = min(near, key=lambda r: (int(r["hidden"]), float(r["best_val_loss"])))
    return {"lr": float(r["lr"]), "hidden": int(r["hidden"]), "dropout": float(r["dropout"])}


def search(cfg, vocab, train, val, out):
    path = out / "search.csv"
    if path.exists():
        return pick(list(csv.DictReader(open(path))), cfg["search"]["tolerance"])
    s = cfg["search"]
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    gtr, gva = gpu_windows(train, s["context"], bos, eos), gpu_windows(val, s["context"], bos, eos)
    rows = []
    for lr, hidden, dropout in itertools.product(s["grid"]["lr"], s["grid"]["hidden"], s["grid"]["dropout"]):
        m = {**cfg["model"], "lr": lr, "hidden": hidden, "dropout": dropout}
        r = train_model(m, gtr, gva, len(vocab), s["context"], cfg["seed"], out / "search_curves" / f"lr{lr}_h{hidden}_d{dropout}.csv",
                        max_epochs=s["max_epochs"])
        rows.append({"lr": lr, "hidden": hidden, "dropout": dropout, **r})
        say(f"search lr={lr} hidden={hidden} dropout={dropout}: val_loss={r['best_val_loss']:.4f} ppl={np.exp(r['best_val_loss']):.3f} ({r['train_s']}s)")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return pick([{k: str(v) for k, v in r.items()} for r in rows], s["tolerance"])


def tag(c, seed, d):
    return f"ctx{c}_seed{seed}_do{d}"


def aggregate(cfg, out, chosen):
    rows = list(csv.DictReader(open(out / "runs.csv")))
    summary = []
    for c in cfg["contexts"]:
        rs = [r for r in rows if int(r["context"]) == c and float(r["dropout"]) == chosen.get(c, -1)]
        if not rs:
            continue
        v = np.array([float(r["val_ppl"]) for r in rs])
        summary.append({"context": c, "dropout": chosen[c], "n_seeds": len(rs), "n_params": rs[0]["n_params"],
                        "val_ppl_mean": round(float(v.mean()), 4), "val_ppl_std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
                        "mean_best_epoch": float(np.mean([int(r["best_epoch"]) for r in rs]))})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    grid = {c: {float(r["dropout"]): float(r["val_ppl"]) for r in rows if int(r["context"]) == c and int(r["seed"]) == cfg["seeds"][0]}
            for c in cfg["contexts"]}
    (out / "dropout_search.json").write_text(json.dumps({"seed": cfg["seeds"][0], "val_ppl": {str(c): {str(d): v for d, v in sorted(g.items())}
                                                                                          for c, g in grid.items()}, "chosen": {str(c): d for c, d in chosen.items()}}, indent=1))


def main():
    cfg = yaml.safe_load(open("configs/phase3.yaml"))
    out, ckdir = Path(cfg["results_dir"]), Path(cfg["ckpt_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)
    vocab = Vocab.load("data/processed/vocab.json")
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    train, val = load_ids("train"), load_ids("val")

    base = search(cfg, vocab, train, val, out)
    say(f"lr / hidden from the search at context {cfg['search']['context']}: {base}")
    runs = out / "runs.csv"
    if not runs.exists():
        with open(runs, "w", newline="") as f:
            csv.writer(f).writerow(COLS)

    def done():
        return {(int(r["context"]), int(r["seed"]), float(r["dropout"])): float(r["best_val_loss"]) for r in csv.DictReader(open(runs))}

    def run(c, seed, d):
        if (c, seed, d) in done():
            return
        m = {**cfg["model"], "lr": base["lr"], "hidden": base["hidden"], "dropout": d}
        gtr, gva = gpu_windows(train, c, bos, eos), gpu_windows(val, c, bos, eos)
        r = train_model(m, gtr, gva, len(vocab), c, seed, out / "curves" / f"{tag(c, seed, d)}.csv", ckdir / f"mlp_{tag(c, seed, d)}.pt")
        with open(runs, "a", newline="") as f:
            csv.writer(f).writerow([c, seed, d, r["n_params"], r["best_epoch"], r["epochs"], f"{r['best_val_loss']:.5f}",
                                    f"{np.exp(r['best_val_loss']):.4f}", r["train_s"]])
        say(f"ctx={c} seed={seed} dropout={d}: params={r['n_params']:,} best_epoch={r['best_epoch']}/{r['epochs']} "
            f"val_ppl={np.exp(r['best_val_loss']):.3f} ({r['train_s']}s)")
        del gtr, gva
        torch.cuda.empty_cache()

    s0 = cfg["seeds"][0]
    for c in cfg["contexts"]:                       # (2) per-context dropout at seed 0
        for d in cfg["dropout_grid"]:
            run(c, s0, d)
    for c in cfg["contexts"]:                       # if the largest value wins, extend the grid one step (<= max)
        tried = list(cfg["dropout_grid"])
        while True:
            res = done()
            best = min(tried, key=lambda d: res[(c, s0, d)])
            nxt = round(best + cfg["dropout_step"], 2)
            if best != max(tried) or nxt > cfg["dropout_max"]:
                break
            run(c, s0, nxt)
            tried.append(nxt)
    res = done()
    chosen = {c: min([d for (cc, s, d) in res if cc == c and s == s0], key=lambda d: res[(c, s0, d)]) for c in cfg["contexts"]}
    say(f"chosen dropout per context (val, seed {s0}): {chosen}")
    (out / "chosen.json").write_text(json.dumps({"model": {**cfg["model"], "lr": base["lr"], "hidden": base["hidden"]},
                                                 "dropout_per_context": {str(c): d for c, d in chosen.items()},
                                                 "seed": cfg["seed"], "seeds": cfg["seeds"]}, indent=2))
    for seed in cfg["seeds"][1:]:                   # (3) remaining seeds
        for c in cfg["contexts"]:
            run(c, seed, chosen[c])
    for c in cfg["contexts"]:                       # (4) publish under the names Phase 4 reads
        for seed in cfg["seeds"]:
            src, dst = ckdir / f"mlp_{tag(c, seed, chosen[c])}.pt", ckdir / f"mlp_ctx{c}_seed{seed}.pt"
            dst.unlink(missing_ok=True)
            os.link(src, dst)
    (out / "final.json").write_text(json.dumps({str(c): d for c, d in chosen.items()}))
    aggregate(cfg, out, chosen)
    say("done")


if __name__ == "__main__":
    main()
