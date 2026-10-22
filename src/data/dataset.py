"""Access to the processed splits. Test splits are guarded: only scripts/phase4_eval.py may load them."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
TEST_SPLITS = {"test_id", "test_ood"}
FINAL_EVAL_SCRIPT = "phase4_eval.py"


def _called_from_final_eval():
    main = sys.modules.get("__main__")
    return Path(getattr(main, "__file__", None) or "").name == FINAL_EVAL_SCRIPT


def load_split(name, columns=None):
    if name in TEST_SPLITS and not _called_from_final_eval():
        raise PermissionError(f"{name} may only be loaded by scripts/{FINAL_EVAL_SCRIPT}")
    return pd.read_parquet(PROCESSED / f"{name}.parquet", columns=columns)


def load_ids(name):
    """List of uint16 arrays, one per function (vocabulary ids, no BOS/EOS)."""
    return [np.asarray(x, dtype=np.uint16) for x in load_split(name, columns=["ids"])["ids"]]
