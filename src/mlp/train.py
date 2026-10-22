"""Training / evaluation of the fixed-context MLP. All windows live on the GPU; batches are gathered there."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.mlp.model import MLPLM


class GPUWindows:
    """Same predicted positions as src.data.windows.Windows, indexed on the GPU."""

    def __init__(self, windows, device):
        self.flat = torch.from_numpy(windows.flat.astype(np.int32)).to(device)
        self.pos = torch.from_numpy(windows.pos).to(device)
        self.offsets = torch.arange(-windows.context, 0, device=device)
        self.n = len(windows)

    def batch(self, idx):
        p = self.pos[idx]
        return self.flat[p[:, None] + self.offsets].long(), self.flat[p].long()


@torch.no_grad()
def nll_array(model, gw, batch=8192, amp=True):
    """Per-position NLL (nats, float64 on CPU) for every predicted token."""
    model.eval()
    out = torch.empty(gw.n, dtype=torch.float64)
    for s in range(0, gw.n, batch):
        idx = torch.arange(s, min(s + batch, gw.n), device=gw.pos.device)
        ctx, tgt = gw.batch(idx)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(ctx)
        out[s:s + len(idx)] = F.cross_entropy(logits.float(), tgt, reduction="none").double().cpu()
    return out.numpy()


def train_model(cfg, train_gw, val_gw, vocab_size, context, seed, log_path, ckpt_path=None, max_epochs=None):
    """cfg: dict(emb_dim, hidden, dropout, lr, weight_decay, batch_size, max_epochs, patience, warmup_steps).
    Returns dict(best_val_loss, best_epoch, epochs, n_params, train_s). Early stops on val loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = train_gw.pos.device
    model = MLPLM(vocab_size, context, cfg["emb_dim"], cfg["hidden"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    max_epochs = max_epochs or cfg["max_epochs"]
    bs = cfg["batch_size"]
    steps_per_epoch = train_gw.n // bs
    total = steps_per_epoch * max_epochs

    def lr_at(step):  # linear warmup then cosine decay to 5% of peak
        if step < cfg["warmup_steps"]:
            return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
        t = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
        return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t)))

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    best, best_epoch, bad, step, t0 = float("inf"), -1, 0, 0, time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_s"])
        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(train_gw.n, device=dev, generator=gen)
            run = torch.zeros((), device=dev)
            for b in range(steps_per_epoch):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                ctx, tgt = train_gw.batch(perm[b * bs:(b + 1) * bs])
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(ctx)
                loss = F.cross_entropy(logits.float(), tgt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run += loss.detach()
                step += 1
            train_loss = (run / steps_per_epoch).item()
            val_loss = float(nll_array(model, val_gw).mean())
            w.writerow([epoch + 1, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{np.exp(val_loss):.4f}",
                        f"{lr_at(step - 1):.6g}", f"{time.time() - t0:.1f}"])
            f.flush()
            if val_loss < best - 1e-4:
                best, best_epoch, bad = val_loss, epoch + 1, 0
                if ckpt_path:
                    torch.save({"state_dict": model.state_dict(), "context": context, "seed": seed, "cfg": cfg,
                                "vocab_size": vocab_size, "best_epoch": best_epoch, "val_loss": best}, ckpt_path)
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    return {"best_val_loss": best, "best_epoch": best_epoch, "epochs": epoch + 1,
            "n_params": model.n_params(), "train_s": round(time.time() - t0, 1)}
