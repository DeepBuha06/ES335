"""Matplotlib figures for Phase 4. Colours come from configs/palette.yaml (same mapping everywhere):
n-gram orders 2-5 = grey family, MLP contexts 3-20 = one blue ramp, invalid syntax = muted red."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.lines import Line2D

PAL = yaml.safe_load(open("configs/palette.yaml"))["print"]
NGRAM_ORDERS, MLP_CONTEXTS = [2, 3, 4, 5], [3, 5, 10, 20]
MODELS = [f"ngram{o}" for o in NGRAM_ORDERS] + [f"mlp{c}" for c in MLP_CONTEXTS]
FAIL_COLORS = {"unbalanced brackets": "#7a2c2c", "indentation": "#b04a4a", "incomplete statement": "#d08f8f", "other": "#ebc9c9"}
SPLIT_TITLES = {"val": "val (held-out repos)", "test_id": "test ID (training repos)", "test_ood": "test OOD (unseen repos)"}


def color(mid):
    return PAL["ngram"][int(mid[5:])] if mid.startswith("ngram") else PAL["mlp"][int(mid[3:])]


def label(mid):
    return f"n-gram order {mid[5:]}" if mid.startswith("ngram") else f"MLP context {mid[3:]}"


def style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "IBM Plex Sans" if "IBM Plex Sans" in have else "DejaVu Sans", "font.size": 9,
        "text.color": PAL["ink"], "axes.labelcolor": PAL["ink_secondary"], "xtick.color": PAL["ink_secondary"],
        "ytick.color": PAL["ink_secondary"], "axes.edgecolor": PAL["hairline"], "axes.linewidth": 0.8,
        "axes.facecolor": PAL["background"], "figure.facecolor": PAL["background"], "axes.grid": True,
        "grid.color": PAL["hairline"], "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
    })


def _legend_models(fig, y=-0.02, ncol=8):
    handles = [Line2D([], [], marker="s" if m.startswith("ngram") else "o", ls="", color=color(m), markersize=7,
                      markeredgecolor="white", markeredgewidth=1, label=label(m)) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), fontsize=8, handlelength=1,
               columnspacing=1.2, labelcolor=PAL["ink_secondary"])


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_ppl_vs_context(summary, path):
    """summary[model_id][split] = dict(ppl_mean, ppl_std, ci)."""
    style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, split in zip(axes, ["val", "test_id", "test_ood"]):
        for fam, ms, xs in (("ngram", [f"ngram{o}" for o in NGRAM_ORDERS], [o - 1 for o in NGRAM_ORDERS]),
                            ("mlp", [f"mlp{c}" for c in MLP_CONTEXTS], MLP_CONTEXTS)):
            ys = [summary[m][split]["ppl_mean"] for m in ms]
            err = [summary[m][split]["ppl_std"] for m in ms]
            ax.plot(xs, ys, color=color(ms[-1]), lw=2, zorder=2)
            for m, x, y, e in zip(ms, xs, ys, err):
                if e:
                    ax.errorbar(x, y, yerr=e, color=color(m), lw=1, capsize=2, zorder=3)
                ax.plot(x, y, "s" if fam == "ngram" else "o", color=color(m), markersize=7, markeredgecolor="white",
                        markeredgewidth=1, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(SPLIT_TITLES[split], fontsize=9, color=PAL["ink"], loc="left")
        ax.set_xlabel("context tokens (n-gram order n uses n-1)")
    axes[0].set_ylabel("perplexity (lower is better)")
    _legend_models(fig, y=-0.08)
    _save(fig, path)


def fig_id_ood_gap(summary, gaps, path):
    style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, m in enumerate(MODELS[::-1]):
        a, b = summary[m]["test_id"]["ppl_mean"], summary[m]["test_ood"]["ppl_mean"]
        ax.plot([a, b], [i, i], color=PAL["hairline"], lw=2, zorder=1)
        ax.plot(a, i, "o", color=color(m), markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.plot(b, i, "o", markerfacecolor="white", markeredgecolor=color(m), markeredgewidth=2, markersize=8, zorder=3)
        ax.text(max(a, b) + 0.025 * (ax.get_xlim()[1] - ax.get_xlim()[0]) + 0.35, i, f"+{100 * gaps[m]['gap']:.1f}%",
                va="center", fontsize=8, color=PAL["ink_secondary"])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([label(m) for m in MODELS[::-1]])
    ax.set_xlabel("perplexity")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=PAL["ink_secondary"], markersize=7, label="test ID"),
                       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=PAL["ink_secondary"],
                              markeredgewidth=1.5, markersize=7, label="test OOD")], loc="lower right", fontsize=8)
    _save(fig, path)


def fig_validity(rows, path):
    """rows: list of dicts(model, T, set, rate, ci)."""
    style()
    temps = sorted({r["T"] for r in rows}, reverse=True)
    fig, axes = plt.subplots(1, len(temps), figsize=(4.8 * len(temps), 3.8), sharey=True)
    for ax, T in zip(np.atleast_1d(axes), temps):
        for i, m in enumerate(MODELS[::-1]):
            for dy, st, filled in ((0.13, "test_id", True), (-0.13, "test_ood", False)):
                r = next(r for r in rows if r["model"] == m and r["T"] == T and r["set"] == st)
                ax.plot(r["ci"], [i + dy] * 2, color=color(m), lw=1.5, zorder=2)
                ax.plot(r["rate"], i + dy, "o", color=color(m), markerfacecolor=color(m) if filled else "white",
                        markeredgecolor=color(m), markeredgewidth=1.5 if not filled else 1, markersize=7, zorder=3)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([label(m) for m in MODELS[::-1]])
        ax.set_title(f"temperature {T}", fontsize=9, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of completed samples that parse")
        ax.grid(axis="y", visible=False)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=PAL["ink_secondary"], markersize=7, label="test ID prompts"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=PAL["ink_secondary"], markeredgewidth=1.5,
               markersize=7, label="test OOD prompts")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), fontsize=8,
               labelcolor=PAL["ink_secondary"])
    _save(fig, path)


def fig_failures(fail_rows, path):
    """fail_rows: list of dicts(model, T, valid_share, categories={cat: share of natural completions})."""
    style()
    temps = sorted({r["T"] for r in fail_rows}, reverse=True)
    fig, axes = plt.subplots(1, len(temps), figsize=(4.8 * len(temps), 3.8), sharey=True)
    for ax, T in zip(np.atleast_1d(axes), temps):
        for i, m in enumerate(MODELS[::-1]):
            r = next(r for r in fail_rows if r["model"] == m and r["T"] == T)
            left = 0
            for cat, c in list(FAIL_COLORS.items()) + [("valid", "#ececec")]:
                v = r["valid_share"] if cat == "valid" else r["categories"].get(cat, 0.0)
                ax.barh(i, v, left=left, height=0.55, color=c, edgecolor="white", linewidth=1.2, label=cat)
                left += v
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([label(m) for m in MODELS[::-1]])
        ax.set_xlim(0, 1)
        ax.set_title(f"temperature {T}", fontsize=9, loc="left")
        ax.set_xlabel("share of completed samples")
        ax.grid(axis="y", visible=False)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8,
               labelcolor=PAL["ink_secondary"])
    _save(fig, path)


def fig_category_loss(cat_summary, split, path):
    """cat_summary[model][split][category] = dict(share, mean_nll)."""
    style()
    cats = list(next(iter(cat_summary.values()))[split])
    cats = [c for c in cats if c != "eos"] + ["eos"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, c in enumerate(cats[::-1]):
        for k, m in enumerate(MODELS):
            v = cat_summary[m][split][c]["mean_nll"]
            ax.plot(v, j + (3.5 - k) * 0.07, "s" if m.startswith("ngram") else "o", color=color(m), markersize=7, markeredgecolor="white",
                    markeredgewidth=1, zorder=3)
    ax.set_yticks(range(len(cats)))
    first = cat_summary[MODELS[0]][split]
    ax.set_yticklabels([f"{c} ({100 * first[c]['share']:.1f}% of tokens)" for c in cats[::-1]])
    ax.set_xlabel(f"mean cross-entropy on {SPLIT_TITLES[split].split(' (')[0]} targets (nats)")
    ax.grid(axis="y", visible=False)
    _legend_models(fig, y=-0.1, ncol=4)
    _save(fig, path)


def fig_training_curves(curves, path):
    """curves[context] = list over seeds of (epochs, val_loss)."""
    style()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for c in MLP_CONTEXTS:
        for ep, v in curves.get(c, []):
            ax.plot(ep, v, color=PAL["mlp"][c], lw=1.5, alpha=0.9)
        if curves.get(c):
            ep, v = curves[c][0]
            ax.text(ep[-1] + 0.1, v[-1], f"context {c}", color=PAL["ink_secondary"], fontsize=8, va="center")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (nats)")
    _save(fig, path)
