"""Phase 5: export seed-0 MLPs to int8 ONNX for the website and report the cost of quantization (val split only)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_ids
from src.data.tokenizer import BOS, EOS, Vocab
from src.data.windows import Windows
from src.export.onnx_export import export_fp32, quantize, session_nll
from src.mlp.model import MLPLM
from src.mlp.train import GPUWindows, nll_array

cfg = yaml.safe_load(open("configs/phase4.yaml"))
vocab = Vocab.load("data/processed/vocab.json")
bos, eos = vocab.stoi[BOS], vocab.stoi[EOS]
val = load_ids("val")
tmp, site = Path("data/onnx"), Path("site/model")
tmp.mkdir(parents=True, exist_ok=True)
site.mkdir(parents=True, exist_ok=True)
vocab.save(site / "vocab.json")
report = {}
chosen = json.loads(Path("results/phase3/chosen.json").read_text())["dropout_per_context"]  # dropout chosen per context
for c in cfg["mlp_contexts"]:
    ck = torch.load(Path(cfg["ckpt_dir"]) / f"mlp_ctx{c}_seed0_do{chosen[str(c)]}.pt", map_location="cpu")
    m = ck["cfg"]
    model = MLPLM(ck["vocab_size"], c, m["emb_dim"], m["hidden"], m["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    fp32, int8 = tmp / f"mlp_ctx{c}.fp32.onnx", site / f"mlp_ctx{c}.int8.onnx"
    export_fp32(model, c, fp32)
    quantize(fp32, int8)
    w = Windows(val, c, bos, eos)
    ref = nll_array(model.cuda(), GPUWindows(w, "cuda"), amp=False)
    nll_f, top_f = session_nll(fp32, w)
    nll_q, top_q = session_nll(int8, w)
    report[f"mlp{c}"] = {"context": c, "fp32_mb": round(fp32.stat().st_size / 1e6, 2), "int8_mb": round(int8.stat().st_size / 1e6, 2),
                         "val_ppl_torch": float(np.exp(ref.mean())), "val_ppl_onnx_fp32": float(np.exp(nll_f.mean())),
                         "val_ppl_onnx_int8": float(np.exp(nll_q.mean())), "top1_agreement_int8_vs_fp32": float((top_f == top_q).mean())}
    print(report[f"mlp{c}"], flush=True)
Path("results/phase5_onnx.json").write_text(json.dumps({"seed": 0, "split": "val", "models": report}, indent=1))
