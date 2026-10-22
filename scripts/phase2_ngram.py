"""Phase 2: interpolated Kneser-Ney n-grams, orders 2-5; val perplexity, build time, peak memory.

Parent mode runs one subprocess per order (clean peak-RSS per model) and aggregates the results.
Test splits are not touched here.
"""
import argparse
import csv
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_ids
from src.data.tokenizer import BOS, EOS, Vocab
from src.data.windows import Windows
from src.ngram.kn import KNModel


def run_order(cfg, order, kind):
    vocab = Vocab.load("data/processed/vocab.json")
    bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
    train, val = load_ids("train"), load_ids("val")
    t0 = time.time()
    model = KNModel.fit(Windows(train, order - 1, bos, eos), order, len(vocab), cfg["hash_seed"], kind=kind)
    build_s = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    wv = Windows(val, order - 1, bos, eos)
    val_nll = model.nll(wv)
    rng = np.random.default_rng(cfg["seed"])
    sub = [train[i] for i in rng.choice(len(train), cfg["train_subset_functions"], replace=False)]
    train_nll = model.nll(Windows(sub, order - 1, bos, eos))

    idx = rng.integers(0, len(wv), size=cfg["norm_check_real_contexts"])
    ctxs = np.concatenate([wv.contexts(idx), rng.integers(0, len(vocab), size=(cfg["norm_check_random_contexts"], order - 1))])
    max_dev = float(np.abs(model.dist(ctxs).sum(axis=1) - 1).max())

    main = kind == cfg["kind"]
    path = Path(cfg["model_dir"]) / (f"ngram_{order}.npz" if main else f"ngram_{order}_{kind}.npz")
    model.save(path)
    res = {"order": order, "kind": kind, "context_tokens": order - 1, "seed": cfg["seed"], "hash_seed": cfg["hash_seed"],
           "val_ppl": float(np.exp(val_nll.mean())), "val_nll": float(val_nll.mean()), "val_tokens": len(val_nll),
           "train_subset_ppl": float(np.exp(train_nll.mean())), "build_s": round(build_s, 1),
           "peak_rss_gb": round(peak_gb, 2), "model_file_mb": round(path.stat().st_size / 1e6, 1),
           "max_abs_sum_deviation": max_dev, "levels": model.sizes()}
    Path(f"results/phase2_ngram_order{order}{'' if main else '_' + kind}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "levels"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int)
    ap.add_argument("--kind")
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/phase2.yaml"))
    if args.order:
        return run_order(cfg, args.order, args.kind or cfg["kind"])
    mem_kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
    for order in cfg["orders"]:
        for kind in (cfg["kind"], cfg["ablation_kind"]):
            print(f"order {order} ({kind}) ...", flush=True)
            subprocess.run([sys.executable, __file__, "--order", str(order), "--kind", kind], check=True)
    rows = [json.loads(Path(f"results/phase2_ngram_order{o}.json").read_text()) for o in cfg["orders"]]
    abl = [json.loads(Path(f"results/phase2_ngram_order{o}_{cfg['ablation_kind']}.json").read_text()) for o in cfg["orders"]]
    Path("results/phase2_ngram.json").write_text(json.dumps({"mem_total_gb": round(mem_kb / 1e6, 2), "kind": cfg["kind"], "models": rows,
                                                             "ablation": {"kind": cfg["ablation_kind"], "val_ppl": {str(r["order"]): r["val_ppl"] for r in abl}}}, indent=2))
    with open("results/phase2_ngram.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "kind", "val_ppl", "train_subset_ppl", "build_s", "peak_rss_gb", "model_file_mb", "max_abs_sum_deviation"])
        for r in rows + abl:
            w.writerow([r[k] for k in ("order", "kind", "val_ppl", "train_subset_ppl", "build_s", "peak_rss_gb", "model_file_mb", "max_abs_sum_deviation")])


if __name__ == "__main__":
    main()
