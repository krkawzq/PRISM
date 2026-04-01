"""Shared AnnData helpers used by CLI commands, server code, and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import anndata as ad
import numpy as np
from scipy import sparse


def select_matrix(adata: Any, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def ensure_dense_matrix(
    matrix: Any, *, dtype: np.dtype | type = np.float32
) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=dtype)
    return np.asarray(matrix, dtype=dtype)


def slice_gene_matrix(
    matrix: Any,
    gene_positions: list[int],
    *,
    cell_indices: np.ndarray | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
    if sparse.issparse(subset):
        return np.asarray(subset.toarray(), dtype=dtype)
    return np.asarray(subset, dtype=dtype)


def compute_reference_counts(
    matrix: Any,
    gene_positions: list[int],
    *,
    cell_indices: np.ndarray | None = None,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    subset = matrix[:, gene_positions]
    if cell_indices is not None:
        subset = subset[cell_indices, :]
    if sparse.issparse(subset):
        totals = np.asarray(subset.sum(axis=1)).reshape(-1)
    else:
        totals = np.asarray(subset, dtype=dtype).sum(axis=1)
    return np.asarray(totals, dtype=dtype).reshape(-1)


def write_h5ad_atomic(adata: ad.AnnData, output_path: str | Path) -> None:
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()
    adata.write_h5ad(temp_path)
    temp_path.replace(resolved)


__all__ = [
    "compute_reference_counts",
    "ensure_dense_matrix",
    "select_matrix",
    "slice_gene_matrix",
    "write_h5ad_atomic",
]
