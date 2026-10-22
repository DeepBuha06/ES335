"""Phase 5: next-token prediction traces for the website's "watch it predict" visual.

For a few short validation functions (validation split only; test splits are not touched), record at every
position the probability each model gives to the true next token and its top-5 candidates. Functions are
picked at random (fixed seed) among short, readable ones, not chosen by how well any model does.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_split
from src.data.tokenizer import BOS, EOS, UNK, Vocab
from src.data.windows import Windows
from src.mlp.model import MLPLM
from src.ngram.kn import KNModel

cfg = yaml.safe_load(open("configs/phase5.yaml"))["traces"]
p4 = yaml.safe_load(open("configs/phase4.yaml"))
vocab = Vocab.load("data/processed/vocab.json")
bos, eos, unk = vocab.stoi[BOS], vocab.stoi[EOS], vocab.stoi[UNK]

df = load_split("val", columns=["ids", "repository_name", "func_name"])
lens = df["ids"].map(len)
n_unk = df["ids"].map(lambda x: int(np.sum(np.asarray(x) == unk)))
has = lambda tok: df["ids"].map(lambda x: vocab.stoi[tok] in set(x))
ok = (lens >= cfg["min_tokens"]) & (lens <= cfg["max_tokens"]) & (n_unk <= cfg["max_unk"]) & has("(") & has("<INDENT>") & has("[")
cand = np.flatnonzero(ok.values)
rng = np.random.default_rng(cfg["seed"])
pick = rng.choice(cand, size=cfg["n_functions"], replace=False)
print(f"{len(cand)} candidate functions, picked {sorted(pick.tolist())}")

models = {}
for o in p4["ngram_orders"]:
    models[f"ngram{o}"] = ("ngram", KNModel.load(Path(p4["ngram_dir"]) / f"ngram_{o}.npz"), o - 1)
for c in p4["mlp_contexts"]:
    ck = torch.load(Path(p4["ckpt_dir"]) / f"mlp_ctx{c}_seed0.pt", map_location="cuda")
    m = MLPLM(ck["vocab_size"], c, ck["cfg"]["emb_dim"], ck["cfg"]["hidden"], ck["cfg"]["dropout"]).cuda()
    m.load_state_dict(ck["state_dict"])
    models[f"mlp{c}"] = ("mlp", m.eval(), c)

out = {"split": "val", "seed": cfg["seed"], "top_k": cfg["top_k"], "functions": []}
for i in pick:
    ids = np.asarray(df["ids"].iloc[i], dtype=np.uint16)
    fn = {"repository_name": df["repository_name"].iloc[i], "func_name": df["func_name"].iloc[i],
          "tokens": vocab.decode(ids) + [EOS], "models": {}}
    for mid, (kind, model, ctx) in models.items():
        w = Windows([ids], ctx, bos, eos)
        sl = slice(0, len(w))
        if kind == "ngram":
            probs = model.dense(w.contexts(sl))
        else:
            with torch.no_grad():
                logits = model(torch.from_numpy(w.contexts(sl).astype(np.int64)).cuda()).float()
                probs = torch.softmax(logits, 1).double().cpu().numpy()
        tgt = w.targets(sl).astype(np.int64)
        top = np.argsort(-probs, axis=1)[:, :cfg["top_k"]]
        fn["models"][mid] = {
            "p_true": [round(float(x), 4) for x in probs[np.arange(len(tgt)), tgt]],
            "rank": [int((probs[j] > probs[j, t]).sum()) + 1 for j, t in enumerate(tgt)],
            "top": [[[vocab.itos[int(k)], round(float(probs[j, k]), 4)] for k in top[j]] for j in range(len(tgt))],
        }
    nll = {mid: float(-np.mean(np.log(np.maximum(m["p_true"], 1e-6)))) for mid, m in fn["models"].items()}
    fn["mean_nll"] = {k: round(v, 4) for k, v in nll.items()}
    out["functions"].append(fn)
    print(fn["func_name"], len(ids), {k: round(np.exp(v), 2) for k, v in nll.items()})
Path("site/data/traces.json").write_text(json.dumps(out, separators=(",", ":")))
print("site/data/traces.json", round(Path("site/data/traces.json").stat().st_size / 1e6, 2), "MB")
