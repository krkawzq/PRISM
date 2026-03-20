from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from prism.model._typing import DTYPE_NP


class GeneLookupError(Exception):
    pass


class GeneNotFoundError(GeneLookupError):
    pass


@dataclass(frozen=True, slots=True)
class GeneCandidate:
    gene_name: str
    gene_index: int
    total_umi: int
    detected_cells: int
    detected_fraction: float


def select_matrix(adata: Any, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} does not exist")
    return adata.layers[layer]


def compute_totals(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=1)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def compute_gene_totals(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        totals = np.asarray(matrix.sum(axis=0)).ravel()
    else:
        totals = np.asarray(matrix, dtype=DTYPE_NP).sum(axis=0)
    return np.asarray(totals, dtype=DTYPE_NP).reshape(-1)


def compute_detected_counts(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    else:
        counts = np.count_nonzero(np.asarray(matrix), axis=0)
    return np.asarray(counts, dtype=np.int64).reshape(-1)


def build_gene_to_idx(gene_names: np.ndarray) -> dict[str, int]:
    return {str(name): int(idx) for idx, name in enumerate(gene_names.tolist())}


def resolve_gene_query(
    query: str,
    gene_names: np.ndarray,
    gene_to_idx: dict[str, int],
) -> int:
    token = query.strip()
    if not token:
        raise GeneNotFoundError("empty gene query")

    if token in gene_to_idx:
        return gene_to_idx[token]

    if token.isdigit():
        idx = int(token)
        if 0 <= idx < len(gene_names):
            return idx

    lowered = token.lower()
    exact = [
        idx
        for idx, name in enumerate(gene_names.tolist())
        if str(name).lower() == lowered
    ]
    if exact:
        return int(exact[0])

    raise GeneNotFoundError(f"gene query {query!r} not found")


def search_gene_candidates(
    query: str,
    gene_names: np.ndarray,
    gene_total_counts: np.ndarray,
    gene_detected_counts: np.ndarray,
    n_cells: int,
    limit: int,
) -> list[GeneCandidate]:
    token = query.strip().lower()
    indices: list[int] = []
    for idx, name in enumerate(gene_names.tolist()):
        if not token or token in str(name).lower():
            indices.append(idx)

    indices = sorted(
        indices, key=lambda idx: float(gene_total_counts[idx]), reverse=True
    )[:limit]
    return [
        GeneCandidate(
            gene_name=str(gene_names[idx]),
            gene_index=int(idx),
            total_umi=int(round(float(gene_total_counts[idx]))),
            detected_cells=int(gene_detected_counts[idx]),
            detected_fraction=float(gene_detected_counts[idx]) / max(n_cells, 1),
        )
        for idx in indices
    ]


def slice_gene_counts(matrix: Any, gene_index: int) -> np.ndarray:
    subset = matrix[:, gene_index]
    if sparse.issparse(subset):
        values = subset.toarray().reshape(-1)
    else:
        values = np.asarray(subset).reshape(-1)
    return np.asarray(values, dtype=DTYPE_NP)
