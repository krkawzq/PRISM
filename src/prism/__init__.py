from __future__ import annotations

from importlib import import_module

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
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".io", __name__)
    return getattr(module, name)
