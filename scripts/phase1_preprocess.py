"""Phase 1: filter, deduplicate, sample, split, tokenize, build vocab.

Outputs: data/processed/{train,val,test_id,test_ood}.parquet, data/processed/vocab.json,
results/phase1_report.json
"""
import ast
import hashlib
import json
import sys
import time
import tokenize
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load import load_config, load_splits
from src.data.splits import sample_by_repo, split_repos
from src.data.tokenizer import UNK, Vocab, code_to_tokens, tokens_to_code

META = ["repository_name", "func_path_in_repository", "func_name", "split_name"]
SPLITS = ["train", "val", "test_id", "test_ood"]
warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escapes in source strings; harmless


def pass1(code):
    """(status, n_tokens, hash, stats) without keeping the token list."""
    stats = Counter()
    try:
        toks = code_to_tokens(code, stats)
    except (SyntaxError, ValueError):
        return "ast_fail", 0, b"", stats
    except (tokenize.TokenError, IndentationError):
        return "tokenize_fail", 0, b"", stats
    return "ok", len(toks), hashlib.blake2b(" ".join(toks).encode(), digest_size=8).digest(), stats


def tokenize_and_roundtrip(code):
    toks = code_to_tokens(code)
    try:
        ast.parse(tokens_to_code(toks))
        ok = True
    except SyntaxError:
        ok = False
    return toks, ok


def roundtrip_vocab(toks, vocab):
    """Round trip after vocab mapping (<UNK> substituted by a placeholder identifier)."""
    try:
        ast.parse(tokens_to_code(vocab.decode(vocab.encode(toks))))
        return True
    except SyntaxError:
        return False


def main():
    t0 = time.time()
    with open("configs/phase1.yaml") as f:
        cfg = yaml.safe_load(f)
    pool = Pool(cfg["workers"])  # fork before loading data
    raw = load_splits(load_config(), columns=META + ["whole_func_string"])
    df = pd.concat(raw.values(), ignore_index=True)
    rep = {"seed": cfg["seed"], "config": cfg, "n_pooled": len(df)}

    # ---- pass 1: parse/tokenize everything, keep only length + hash
    res = pool.map(pass1, df["whole_func_string"].tolist(), chunksize=500)
    df["status"] = [r[0] for r in res]
    df["n_tokens"] = [r[1] for r in res]
    df["hash"] = [r[2] for r in res]
    tstats = sum((r[3] for r in res), Counter())
    rep["filter"] = {"ast_fail": int((df.status == "ast_fail").sum()),
                     "tokenize_fail": int((df.status == "tokenize_fail").sum()),
                     "docstrings": dict(tstats)}
    df = df[df.status == "ok"]
    ok_len = df["n_tokens"].values
    rep["filter"]["processed_len_percentiles"] = {str(p): float(np.percentile(ok_len, p)) for p in (1, 5, 50, 95, 99, 99.9)}
    rep["filter"]["max_tokens_is_percentile"] = float((ok_len <= cfg["max_tokens"]).mean() * 100)
    rep["filter"]["too_short"] = int((ok_len < cfg["min_tokens"]).sum())
    rep["filter"]["too_long"] = int((ok_len > cfg["max_tokens"]).sum())
    df = df[(df.n_tokens >= cfg["min_tokens"]) & (df.n_tokens <= cfg["max_tokens"])]

    # ---- dedup on raw text, then on processed token sequence
    df = df.sort_values(["repository_name", "func_path_in_repository", "func_name"], kind="stable")
    n0 = len(df)
    df = df[~df["whole_func_string"].duplicated()]
    n1 = len(df)
    dup = df["hash"].duplicated()
    cross_repo = df[df["hash"].isin(df.loc[dup, "hash"])].groupby("hash")["repository_name"].nunique()
    df = df[~dup]
    rep["dedup"] = {"exact_removed": n0 - n1, "near_removed": int(dup.sum()),
                    "near_dup_groups_spanning_repos": int((cross_repo > 1).sum()),
                    "remaining": len(df), "repos_remaining": int(df.repository_name.nunique())}

    # ---- sample ~100k by repo, then split
    sample, k = sample_by_repo(df, cfg["target_total"], cfg["seed"])
    rep["sampling"] = {"repo_cap_K": int(k), "n_sampled": len(sample),
                       "n_repos": int(sample.repository_name.nunique()),
                       "max_repo_share": float(sample.repository_name.value_counts(normalize=True).iloc[0])}
    sample = split_repos(sample, cfg["test_ood_repo_frac"], cfg["val_repo_frac"], cfg["test_id_size"], cfg["seed"])

    # ---- pass 2: tokenize the sample and round-trip check
    res = pool.map(tokenize_and_roundtrip, sample["whole_func_string"].tolist(), chunksize=200)
    sample["tokens"] = [r[0] for r in res]
    rt_ok = np.array([r[1] for r in res])

    train_tokens = sample.loc[sample.split == "train", "tokens"]
    vocab, counts = Vocab.build(train_tokens, cfg["vocab_size"])
    rt_vocab = np.array([roundtrip_vocab(t, vocab) for t in sample["tokens"]])
    rep["roundtrip"] = {"rate_raw_tokens": float(rt_ok.mean()), "fail_raw_tokens": int((~rt_ok).sum()),
                        "rate_after_vocab_unk": float(rt_vocab.mean()), "fail_after_vocab_unk": int((~rt_vocab).sum())}
    rep["roundtrip"]["fail_examples"] = sample.loc[~rt_ok, "whole_func_string"].head(3).str[:300].tolist()
    rep["vocab"] = {"size_incl_specials": len(vocab), "train_types": len(counts),
                    "train_token_coverage": float(sum(counts[t] for t in vocab.itos if t in counts) / sum(counts.values())),
                    "min_count_in_vocab": int(min(counts[t] for t in vocab.itos[4:]))}

    # ---- per-split stats and save
    out = Path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    vocab.save(out / "vocab.json")
    rep["splits"] = {}
    for s in SPLITS:
        part = sample[sample.split == s]
        ids = [vocab.encode(t) for t in part["tokens"]]
        n_tok = sum(len(x) for x in ids)
        n_unk = sum(x.count(vocab.unk_id) for x in ids)
        rep["splits"][s] = {"n_functions": len(part), "n_repos": int(part.repository_name.nunique()),
                            "n_tokens": n_tok, "unk_rate": n_unk / n_tok,
                            "functions_with_unk": float(np.mean([vocab.unk_id in x for x in ids])),
                            "orig_csn_split": part["split_name"].value_counts().to_dict()}
        part.assign(ids=ids).drop(columns=["status", "hash"]).to_parquet(out / f"{s}.parquet", index=False)

    rs = {s: set(sample.loc[sample.split == s, "repository_name"]) for s in SPLITS}
    rep["repo_overlap"] = {f"{a}&{b}": len(rs[a] & rs[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    rep["runtime_s"] = round(time.time() - t0, 1)
    rep["peak_rss_gb_main"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    Path("results").mkdir(exist_ok=True)
    Path("results/phase1_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("filter", "dedup", "sampling", "roundtrip", "vocab", "repo_overlap", "runtime_s")}, indent=2, default=str))
    for s, v in rep["splits"].items():
        print(s, v)
    pool.close()


if __name__ == "__main__":
    main()
