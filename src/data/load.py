"""Download and load CodeSearchNet Python parquet files from the official HF repo."""
import hashlib
from pathlib import Path
import urllib.request

import pandas as pd
import yaml

URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def load_config(path="configs/data.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(cfg, verify=False):
    raw = Path(cfg["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, rel in cfg["files"].items():
        dest = raw / Path(rel).name
        if not dest.exists():
            url = URL.format(repo=cfg["hf_repo"], rev=cfg["revision"], path=rel)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            verify = True
        if verify and sha256(dest) != cfg["sha256"][split]:
            raise ValueError(f"checksum mismatch for {dest}")
        out[split] = dest
    return out


def load_splits(cfg, columns=None):
    """Return {split: DataFrame} for the official train/validation/test splits."""
    return {s: pd.read_parquet(p, columns=columns) for s, p in download(cfg).items()}
