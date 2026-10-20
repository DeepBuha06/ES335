from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.splits import repo_cap, sample_by_repo, split_repos

PROCESSED = Path("data/processed")


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 60, size=300)
    repos = np.repeat([f"r{i}" for i in range(len(sizes))], sizes)
    return pd.DataFrame({"repository_name": repos, "func_path_in_repository": "p",
                         "func_name": [f"f{i}" for i in range(len(repos))]})


def test_repo_cap_is_minimal():
    counts = np.array([1, 5, 50, 500])
    k = repo_cap(counts, 100)
    assert np.minimum(counts, k).sum() >= 100 > np.minimum(counts, k - 1).sum()


def test_toy_split_disjoint_and_test_id_in_train_repos():
    sample, k = sample_by_repo(_toy(), 4000, seed=0)
    assert len(sample) == 4000 and sample.groupby("repository_name").size().max() <= k
    df = split_repos(sample, 0.1, 0.05, 300, seed=0)
    repos = {s: set(df.loc[df.split == s, "repository_name"]) for s in ("train", "val", "test_ood", "test_id")}
    assert not repos["train"] & repos["val"] and not repos["train"] & repos["test_ood"] and not repos["val"] & repos["test_ood"]
    assert repos["test_id"] <= repos["train"]
    assert (df.split == "test_id").sum() == 300


@pytest.mark.skipif(not (PROCESSED / "train.parquet").exists(), reason="processed data not built")
def test_processed_splits_disjoint():
    cols = ["repository_name"]
    r = {s: set(pd.read_parquet(PROCESSED / f"{s}.parquet", columns=cols).repository_name)
         for s in ("train", "val", "test_id", "test_ood")}
    assert not r["train"] & r["val"]
    assert not r["train"] & r["test_ood"]
    assert not r["val"] & r["test_ood"]
    assert r["test_id"] <= r["train"]
