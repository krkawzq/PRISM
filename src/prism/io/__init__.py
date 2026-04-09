from __future__ import annotations

from .lists import (
    read_gene_list,
    read_string_list,
    write_gene_list,
    write_string_list,
)

__all__ = [
    "compute_reference_counts",
    "ensure_dense_matrix",
    "read_gene_list",
    "read_string_list",
    "select_matrix",
    "slice_gene_matrix",
    "write_gene_list",
    "write_h5ad",
    "write_string_list",
]


def __getattr__(name: str) -> object:
    if name in {
        "compute_reference_counts",
        "ensure_dense_matrix",
        "select_matrix",
        "slice_gene_matrix",
        "write_h5ad",
    }:
        from . import anndata as anndata_module

        return getattr(anndata_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
