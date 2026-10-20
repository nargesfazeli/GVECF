"""Small helpers shared by the pre-training and training scripts."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def to_torch_sparse(mat: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a coalesced float32 torch sparse tensor."""
    coo = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, coo.shape, check_invariants=False).coalesce()


def sparse_dropout(x: torch.Tensor, rate: float) -> torch.Tensor:
    """Drop non-zero entries of a sparse tensor (node dropout in NGCF)."""
    if rate <= 0.0:
        return x
    indices, values = x.indices(), x.values()
    keep = torch.rand(values.shape[0], device=values.device) >= rate
    return torch.sparse_coo_tensor(
        indices[:, keep], values[keep] / (1.0 - rate), x.shape, check_invariants=False
    )


def get_logger(run_dir: Path, name: str = "gvecf") -> logging.Logger:
    logger = logging.getLogger(f"{name}:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "train.log"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def save_json(obj: dict, path: os.PathLike) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
