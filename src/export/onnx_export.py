"""Export trained MLPs to int8 ONNX for the browser and measure the quantization cost on val."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ExportMLP(nn.Module):
    """Same function as MLPLM, with pre-transposed weights so the linear layers export as plain MatMul + Add
    (which onnxruntime's dynamic quantizer converts to int8)."""

    def __init__(self, model):
        super().__init__()
        self.emb = nn.Parameter(model.emb.weight.detach().clone())
        self.w1 = nn.Parameter(model.fc1.weight.detach().t().contiguous())
        self.b1 = nn.Parameter(model.fc1.bias.detach().clone())
        self.w2 = nn.Parameter(model.out.weight.detach().t().contiguous())
        self.b2 = nn.Parameter(model.out.bias.detach().clone())

    def forward(self, ctx):
        h = self.emb[ctx].flatten(1)
        h = torch.nn.functional.gelu(h @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


def export_fp32(model, context, path):
    wrapper = ExportMLP(model).cpu().eval()
    dummy = torch.zeros(1, context, dtype=torch.int64)
    torch.onnx.export(wrapper, dummy, str(path), input_names=["ctx"], output_names=["logits"],
                      dynamic_axes={"ctx": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)


def quantize(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])


def session_nll(path, windows, batch=4096):
    """Per-position NLL of every target in `windows` with an onnxruntime session; also returns top-1 ids."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = np.empty(len(windows))
    top1 = np.empty(len(windows), dtype=np.int64)
    for s in range(0, len(windows), batch):
        sl = slice(s, min(s + batch, len(windows)))
        logits = sess.run(None, {"ctx": windows.contexts(sl).astype(np.int64)})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        tgt = windows.targets(sl).astype(np.int64)
        out[sl] = -logp[np.arange(len(tgt)), tgt]
        top1[sl] = logp.argmax(axis=1)
    return out, top1
