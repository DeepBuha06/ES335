"""Per-repo capped sampling and repository-level splits."""
import numpy as np
import pandas as pd


def repo_cap(counts, target):
    """Smallest K with sum(min(n_repo, K)) >= target."""
    lo, hi = 1, int(counts.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if np.minimum(counts, mid).sum() >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def sample_by_repo(df, target, seed):
    """Cap each repo at K functions, then subsample uniformly to `target`. Returns (sample, K)."""
    rng = np.random.default_rng(seed)
    k = repo_cap(df.groupby("repository_name").size().values, target)
    shuffled = df.iloc[rng.permutation(len(df))]
    capped = shuffled[shuffled.groupby("repository_name").cumcount() < k]
    sample = capped.iloc[rng.permutation(len(capped))[:target]]
    return sample.sort_values(["repository_name", "func_path_in_repository", "func_name"]), k


def split_repos(df, ood_frac, val_frac, test_id_size, seed):
    """Assign each row to train / val / test_id / test_ood (column `split`)."""
    rng = np.random.default_rng(seed)
    repos = np.array(sorted(df["repository_name"].unique()))
    repos = repos[rng.permutation(len(repos))]
    n_ood, n_val = round(ood_frac * len(repos)), round(val_frac * len(repos))
    role = pd.Series("train", index=repos)
    role.iloc[:n_ood] = "test_ood"
    role.iloc[n_ood:n_ood + n_val] = "val"
    df = df.assign(split=df["repository_name"].map(role).values)

    # test_id: random functions from train repos, never the last function of a repo
    train = df[df["split"] == "train"]
    order = train.iloc[rng.permutation(len(train))]
    rank = order.groupby("repository_name").cumcount()
    size = order["repository_name"].map(order["repository_name"].value_counts())
    eligible = order[rank < size - 1]
    chosen = eligible.index[:test_id_size]
    df.loc[chosen, "split"] = "test_id"
    return df
