from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from prism.io import (
    compute_reference_counts as compute_reference_counts_shared,
    select_matrix as select_matrix_shared,
    slice_gene_matrix as slice_gene_matrix_shared,
)
from prism.model import DTYPE_NP


class GeneLookupError(Exception):
    pass


class GeneNotFoundError(GeneLookupError):
    pass


@dataclass(frozen=True, slots=True)
class GeneCandidate:
    gene_name: str
    gene_index: int
    total_count: int
    detected_cells: int
    detected_fraction: float


def select_matrix(adata: Any, layer: str | None):
    return select_matrix_shared(adata, layer)


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


def compute_cell_zero_fraction(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        nonzero = np.asarray(matrix.getnnz(axis=1)).ravel()
        n_genes = int(matrix.shape[1])
    else:
        array = np.asarray(matrix)
        nonzero = np.count_nonzero(array, axis=1)
        n_genes = int(array.shape[1])
    zero_fraction = 1.0 - np.asarray(nonzero, dtype=DTYPE_NP) / max(n_genes, 1)
    return np.asarray(zero_fraction, dtype=DTYPE_NP).reshape(-1)


def build_gene_to_idx(gene_names: np.ndarray) -> dict[str, int]:
    return {str(name): int(idx) for idx, name in enumerate(gene_names.tolist())}


def resolve_gene_query(
    query: str,
    gene_names: np.ndarray,
    gene_names_lower: tuple[str, ...],
    gene_to_idx: dict[str, int],
    gene_lower_to_idx: dict[str, int],
) -> int:
    token = query.strip()
    if not token:
        raise GeneNotFoundError("empty gene query")
    if token in gene_to_idx:
        return gene_to_idx[token]
    if token.isdigit():
        index = int(token)
        if 0 <= index < len(gene_names):
            return index
    lowered = token.lower()
    exact = gene_lower_to_idx.get(lowered)
    if exact is not None:
        return int(exact)
    raise GeneNotFoundError(f"gene query {query!r} not found")


def search_gene_candidates(
    query: str,
    gene_names: np.ndarray,
    gene_names_lower: tuple[str, ...],
    gene_total_counts: np.ndarray,
    gene_detected_counts: np.ndarray,
    ranked_indices: np.ndarray,
    n_cells: int,
    limit: int,
) -> list[GeneCandidate]:
    token = query.strip().lower()
    indices: list[int] = []
    for idx in ranked_indices.tolist():
        resolved = int(idx)
        if not token or token in gene_names_lower[resolved]:
            indices.append(resolved)
            if len(indices) >= limit:
                break
    return [
        GeneCandidate(
            gene_name=str(gene_names[idx]),
            gene_index=int(idx),
            total_count=int(round(float(gene_total_counts[idx]))),
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


def slice_gene_matrix(
    matrix: Any, gene_positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    return slice_gene_matrix_shared(
        matrix,
        gene_positions,
        cell_indices=cell_indices,
        dtype=DTYPE_NP,
    )


def compute_reference_counts(
    matrix: Any, gene_positions: list[int], *, cell_indices: np.ndarray | None = None
) -> np.ndarray:
    return compute_reference_counts_shared(
        matrix,
        gene_positions,
        cell_indices=cell_indices,
        dtype=DTYPE_NP,
    )


def detect_label_columns(adata: Any) -> dict[str, np.ndarray]:
    preferred = {"label", "cell_type", "treatment", "group", "condition", "cluster"}
    ranked: list[tuple[int, str, np.ndarray]] = []
    for column in adata.obs.columns:
        values = np.asarray(adata.obs[column].astype(str)).reshape(-1)
        unique = np.unique(values)
        if unique.size < 2:
            continue
        if unique.size > min(64, max(int(adata.n_obs), 1)):
            continue
        rank = 0 if column in preferred else 1
        ranked.append((rank, str(column), values))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return {name: values for _, name, values in ranked}
