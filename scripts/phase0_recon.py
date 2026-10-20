"""Phase 0: data recon for CodeSearchNet Python. Writes results/phase0_recon.json + length figure."""
import ast
import io
import json
import sys
import time
import tokenize
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load import load_config, load_splits

SKIP = {tokenize.ENCODING, tokenize.ENDMARKER}


def analyse(code):
    """Return (n_tokens or -1 if tokenize fails, n_tokens excl. COMMENT/NL, ast_ok)."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type not in SKIP]
        n_all = len(toks)
        n_core = sum(t.type not in (tokenize.COMMENT, tokenize.NL) for t in toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        n_all = n_core = -1
    try:
        ast.parse(code)
        ok = True
    except (SyntaxError, ValueError):
        ok = False
    return n_all, n_core, ok


def pct(a, ps):
    return {str(p): float(np.percentile(a, p)) for p in ps}


def main():
    cfg = load_config()
    t0 = time.time()
    pool = Pool(cfg["recon"]["workers"])  # fork before loading data to keep worker RSS small
    splits = load_splits(cfg)
    ps = cfg["recon"]["length_percentiles"]
    first = next(iter(splits.values()))
    report = {
        "source": {k: cfg[k] for k in ("hf_repo", "revision", "files")},
        "seed": cfg["seed"],
        "fields": {c: str(first[c].dtype) for c in first.columns},
        "example_row": {c: str(first[c].iloc[0])[:200] for c in first.columns},
        "splits": {},
    }
    repos = {}
    code_sets = {}
    lengths = {}
    with pool:
        for name, df in splits.items():
            code = df["whole_func_string"].fillna("")
            empty = int((code.str.strip() == "").sum())
            res = pool.map(analyse, code.tolist(), chunksize=500)
            n_all = np.array([r[0] for r in res])
            n_core = np.array([r[1] for r in res])
            ast_ok = np.array([r[2] for r in res])
            good = n_all >= 0
            lengths[name] = n_all[good]
            per_repo = df.groupby("repository_name").size()
            repos[name] = set(per_repo.index)
            code_sets[name] = set(code)
            report["splits"][name] = {
                "n_functions": int(len(df)),
                "n_repos": int(per_repo.size),
                "functions_per_repo": pct(per_repo.values, [50, 90, 99]) | {"max": int(per_repo.max())},
                "top5_repos": {k: int(v) for k, v in per_repo.nlargest(5).items()},
                "empty": empty,
                "exact_duplicates_within": int(code.duplicated().sum()),
                "tokenize_fail": int((~good).sum()),
                "ast_parse_fail": int((~ast_ok).sum()),
                "tokens_all": pct(n_all[good], ps) | {"mean": float(n_all[good].mean()), "max": int(n_all[good].max())},
                "tokens_excl_comment_nl": pct(n_core[good], ps) | {"mean": float(n_core[good].mean())},
                "total_tokens_all": int(n_all[good].sum()),
                "split_name_values": sorted(df["split_name"].unique().tolist()) if "split_name" in df else None,
                "language_values": sorted(df["language"].unique().tolist()) if "language" in df else None,
            }
    names = list(splits)
    report["repo_overlap"] = {f"{a}&{b}": len(repos[a] & repos[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    report["code_overlap_exact"] = {f"{a}&{b}": len(code_sets[a] & code_sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    report["n_repos_total"] = len(set().union(*repos.values()))
    report["n_functions_total"] = int(sum(len(d) for d in splits.values()))
    report["runtime_s"] = round(time.time() - t0, 1)
    report["peak_rss_gb_main"] = round(psutil.Process().memory_info().rss / 1e9, 2)

    out = Path("results")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "phase0_recon.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.logspace(0, np.log10(max(l.max() for l in lengths.values()) + 1), 60)
    greys = {"train": "#111111", "validation": "#777777", "test": "#bbbbbb"}
    for name, l in lengths.items():
        ax.hist(l, bins=bins, histtype="step", density=True, color=greys.get(name), label=f"{name} (n={len(l):,})")
    ax.set_xscale("log")
    ax.set_xlabel("tokens per function (Python tokenize, incl. COMMENT/NL)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "phase0_lengths.png", dpi=150)
    print(json.dumps({k: report[k] for k in ("repo_overlap", "code_overlap_exact", "n_repos_total", "n_functions_total", "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
