# Syntax learning and generalization in MLP-based Python code generation

ES 335 project (Buha Deep Maheshbhai, 24110082; Panchal Heet Chirag, 24110235).

I train fixed-context MLP language models (context 3, 5, 10, 20 tokens) and modified Kneser–Ney n-gram models
(order 2–5) on Python functions from CodeSearchNet, and compare them on next-token perplexity, on the share of
sampled completions that Python's parser accepts, and on how much worse they do on repositories excluded from
training. The website in `site/` shows the results and a completion explorer.

Research question and plan: `docs/proposal.md`. Every number in the results comes from the files in `results/`
and `site/data/`, written by the scripts below.

## Requirements

- Linux or WSL2, Python 3.12, about 10 GB RAM, 3 GB disk for data and checkpoints.
- An NVIDIA GPU with at least 6 GB is used for MLP training (tested on an RTX 3050 Laptop GPU, CUDA 12.8 wheels).
- [uv](https://docs.astral.sh/uv/) for the virtual environment (plain `pip` works too).

## Setup

```bash
git clone https://github.com/DeepBuha06/ES335.git
cd ES335
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock   # exact versions; requirements.txt has the ranges
.venv/bin/python -m pytest -q                                   # unit tests (some skip until data is built)
```

## Data access

The data is the Python part of CodeSearchNet on Hugging Face, `code-search-net/code_search_net`, pinned to
revision `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` (`configs/data.yaml`). It is public, so no account or token
is needed. Step 0 below downloads the parquet files over HTTPS into `data/raw/` and checks them against the
SHA-256 hashes recorded in `configs/data.yaml`.
`results/splits_repos.json` lists every repository in each split, with its number of functions.

## Reproduction, step by step

Run times are what I measured on my laptop (RTX 3050 6 GB, 10 GB RAM in WSL2). All scripts use fixed seeds
(recorded in the result files) and are safe to re-run.

| Step | Command | Time | Writes |
|---|---|---|---|
| 0. Download + data recon | `.venv/bin/python scripts/phase0_recon.py` | 31 s after a ~580 MB download | `data/raw/`, `results/phase0_recon.json` |
| 1. Filter, split, tokenize | `.venv/bin/python scripts/phase1_preprocess.py` | 47 s | `data/processed/*.parquet`, `vocab.json`, `results/phase1_report.json` |
| 2. n-gram models | `.venv/bin/python scripts/phase2_ngram.py` | 36 s | `data/processed/ngram_*.npz`, `results/phase2_ngram.*` |
| 3. MLP search + training | `.venv/bin/python scripts/phase3_mlp.py` | about 6.5 h (see below) | `data/checkpoints/`, `results/phase3/` |
| 4. Final evaluation (test sets) | `.venv/bin/python scripts/phase4_eval.py` | 15 min | `results/phase4/`, `results/figures/`, `site/data/` |
| 5. Browser models | `.venv/bin/python scripts/phase5_export_onnx.py` | 16 min | `site/model/*.int8.onnx`, `results/phase5_onnx.json` |
| 5b. Prediction traces for the site | `.venv/bin/python scripts/phase5_traces.py` | under 1 min | `site/data/traces.json` (validation split only) |
| Site checks | `.venv/bin/python scripts/build_site_py.py && .venv/bin/python scripts/check_site.py site` | seconds | `site/py/` |

Step 3 is resumable: if it is interrupted, run the same command again and finished runs are skipped. It does,
in order: a learning-rate / hidden-size / dropout search at context 10 (8 short runs, about 37 min); a dropout
search for every context length at seed 0 (dropout 0.1–0.65, about 2.5 h); the remaining two seeds per context
with the chosen dropout (8 runs, about 1.7 h). Step 5 needs the seed-0 checkpoints from step 3 and is
independent of step 4.

Only `scripts/phase4_eval.py` may load the two test splits (`src/data/dataset.py` refuses otherwise), and it
was run once for the reported results. `--dev` runs the whole evaluation on the validation split instead.

To view the website locally:

```bash
cd site && python3 -m http.server 8000      # then open http://localhost:8000
```

It is deployed to GitHub Pages by `.github/workflows/pages.yml` (repository Settings → Pages → Source:
GitHub Actions).

## Layout

```
src/data/     loading, filtering, splits, tokenizer (tokenize-based), windows shared by all models, test-set guard
src/ngram/    interpolated Kneser–Ney (single and modified discounts), NumPy only
src/mlp/      the MLP and its training loop (AMP, AdamW, early stopping)
src/eval/     metrics, sampling, syntax check and failure taxonomy, Matplotlib figures
src/export/   ONNX export for the browser
scripts/      one entry point per phase, plus site checks
configs/      YAML settings for every phase and the colour palette
results/      JSON/CSV results and figures (committed)
site/         the website: index.html, css/, js/, data/ (results), model/ (int8 MLPs), py/ (tokenizer for Pyodide)
tests/        pytest: tokenizer round trip, n-gram normalization and reference check, split disjointness, test guard, sampling and syntax check
```

## Known limitations

- Parsing is a weak test of code quality: code can parse and still be useless. Nothing is executed.
- Identifiers outside the 10,000-token vocabulary become one `<UNK>` token (about 9% of test tokens), and
  docstrings, comments and literal values are removed, so the task is easier than modelling raw source.
- The MLPs with longer context also have more parameters (the first layer grows with the context), so part of
  the gain from context could come from model size.
- Every MLP run has the same 12-epoch budget; the runs with dropout 0.5 were still improving slightly at the
  end, so longer training might lower perplexity a little further.
- Learning rate and hidden size were chosen at one context length with a short budget.
- The ID and OOD test sets differ in size and in which repositories they contain; their gap includes sampling
  noise beyond the function-level bootstrap intervals. `test_id` holds out functions from repositories that also
  appear in training. `test_ood` holds out repositories. Validation and `test_ood` share no repository with training.
- The official CodeSearchNet train, validation, and test files were pooled before this split. The training split
  includes 4,090 functions from the official test file. Reported numbers evaluate this repository split.
- The length cap (870 tokens) and the removal of 35 near-duplicate functions (21 groups crossed repositories)
  were applied to the pooled data before the split. The vocabulary was counted on training functions only.
- At temperature 0.5 the context-20 MLP leaves 428 of 6,000 samples unfinished, against 1 for the order-5 n-gram.
  The 93.5% validity rate is among samples that finished. At temperature 1.0 the unfinished share is similar
  across models, and unfinished samples are excluded from the rate in both cases.

## AI use

AI coding assistants were used for coding, debugging, editing, and visualization, including Cursor for the
website text and Claude Code for the page design. Displayed numbers were checked against `results/` and `site/data/`. I am responsible for the
code, the numbers, and the claims.

## Sources

- Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. CodeSearchNet Challenge:
  Evaluating the State of Semantic Code Search. arXiv:1909.09436, 2019. https://arxiv.org/abs/1909.09436
- Stanley F. Chen and Joshua Goodman. An empirical study of smoothing techniques for language modeling.
  Computer Speech and Language 13(4), 1999. The n-gram uses their modified Kneser–Ney discounts.
- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model.
  Journal of Machine Learning Research 3, 2003. https://www.jmlr.org/papers/v3/bengio03a.html
- Page libraries: CodeMirror 6 (MIT, `site/vendor/LICENSE-codemirror.txt`), D3, ONNX Runtime Web, and Pyodide.
- Fonts: Atkinson Hyperlegible Next and Atkinson Hyperlegible Mono (SIL Open Font License, `site/vendor/fonts/OFL.txt`).
  Training uses PyTorch. The n-gram is NumPy only.
