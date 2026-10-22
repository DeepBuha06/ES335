import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import dataset
from src.data.windows import Windows

BOS, EOS = 2, 3


def test_windows_padding_and_targets():
    w = Windows([np.array([10, 11], dtype=np.uint16), np.array([12], dtype=np.uint16)], 3, BOS, EOS)
    assert len(w) == 3 + 2
    sl = slice(0, len(w))
    assert w.contexts(sl).tolist() == [[2, 2, 2], [2, 2, 10], [2, 10, 11], [2, 2, 2], [2, 2, 12]]
    assert w.targets(sl).tolist() == [10, 11, EOS, 12, EOS]
    assert w.seq_id.tolist() == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("split", ["test_id", "test_ood"])
def test_test_splits_are_guarded(split):
    with pytest.raises(PermissionError):
        dataset.load_split(split)
    with pytest.raises(PermissionError):
        dataset.load_ids(split)


def test_final_eval_script_may_load_tests(monkeypatch):
    seen = {}
    monkeypatch.setattr(dataset.pd, "read_parquet", lambda path, columns=None: seen.setdefault("path", path))
    monkeypatch.setattr(sys.modules["__main__"], "__file__", "/x/scripts/phase4_eval.py", raising=False)
    dataset.load_split("test_ood")
    assert Path(seen["path"]).name == "test_ood.parquet"


def test_train_val_loadable_anywhere(monkeypatch):
    monkeypatch.setattr(dataset.pd, "read_parquet", lambda path, columns=None: "ok")
    assert dataset.load_split("val") == "ok" and dataset.load_split("train") == "ok"
