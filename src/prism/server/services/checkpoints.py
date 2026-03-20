from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} is not a checkpoint dictionary")
    return checkpoint


def validate_checkpoint_against_dataset(
    checkpoint: dict[str, Any],
    gene_names: np.ndarray,
) -> None:
    ckpt_gene_names = checkpoint.get("gene_names")
    if not isinstance(ckpt_gene_names, list):
        raise TypeError("checkpoint does not contain gene_names")

    gene_set = set(map(str, gene_names.tolist()))
    missing = [name for name in ckpt_gene_names if name not in gene_set]
    if missing:
        raise ValueError(
            f"checkpoint contains genes missing from the dataset, e.g. {missing[:5]}"
        )
